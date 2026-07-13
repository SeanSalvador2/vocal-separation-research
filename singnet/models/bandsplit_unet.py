r"""BandSplitUNet — a param-matched 3-band-split front-end (Direction 03 §3.1).

The band-split idea of the current SOTA family (BSRNN -> BS/Mel-RoFormer ->
SCNet -> Moises-Light): partition the spectrogram into frequency bands and give
each band its own encoder capacity. This module isolates *that* idea at compact
scale, matched to the SingNet-C1 baseline in parameters (:mod:`singnet.models.unet`).

Architecture (both the mel and uniform variants; only the band edges differ):

* The 2048 network bins are split into ``n_bands`` contiguous, disjoint,
  exhaustive bands (asserted in tests).
* **One encoder tower per band** — the same 5-level ``Conv(5x5, s2)-BN-LReLU``
  design as the baseline (blocks reused verbatim from :mod:`singnet.models.unet`),
  a base width ``c`` common to every tower, channels ``c -> 2c -> 4c -> 8c -> b5``
  (``b5`` = the bottleneck width, ``= 16c`` unless the param-match search bumps it,
  §3.1). Each tower pads its band's bin extent up to the next multiple of
  ``pad_multiple`` (32) internally so the five stride-2 layers halve cleanly, and
  the padding is cropped back off the final mask (``edges therefore need no
  divisibility``).
* **No cross-band mixing before the bottleneck** (the treatment). At each level
  the towers' outputs are concatenated along the **frequency axis** (channel
  counts already match by the shared width ``c``), forming full-spectrum skip
  tensors that the baseline-identical bottleneck + decoder consume exactly as the
  baseline consumes its own encoder features.

Shapes (mel edges ``[0, 142, 597, 2048]`` -> padded band widths ``[160, 480,
1472]`` -> padded total ``P = 2112``; base width ``c``, bottleneck ``b5``)::

    mix_mag                 (B, 2048, 256)
    featurize + split/pad   3 x (B, 1, p_b, 256)          p_b in {160,480,1472}
    tower level 1  Conv/2   3 x (c,  p_b/2,  128)  -concat-freq-> E1 (c,  1056, 128)
    tower level 2  Conv/2   3 x (2c, p_b/4,   64)  ----------->   E2 (2c,  528,  64)
    tower level 3  Conv/2   3 x (4c, p_b/8,   32)  ----------->   E3 (4c,  264,  32)
    tower level 4  Conv/2   3 x (8c, p_b/16,  16)  ----------->   E4 (8c,  132,  16)
    tower level 5  Conv/2   3 x (b5, p_b/32,   8)  ----------->   E5 (b5,   66,   8)  bottleneck
    dec1 ConvT/2  b5->8c     (8c, 132, 16)  (+ E4 -> 16c)  Dropout2d(0.5)
    dec2 ConvT/2 16c->4c     (4c, 264, 32)  (+ E3 ->  8c)  Dropout2d(0.5)
    dec3 ConvT/2  8c->2c     (2c, 528, 64)  (+ E2 ->  4c)  Dropout2d(0.5)
    dec4 ConvT/2  4c-> c     ( c,1056,128)  (+ E1 ->  2c)
    dec5 ConvT/2  2c-> c     ( c,2112,256)
    head Conv1x1   c->1       (1,2112,256)  -> sigmoid -> crop back -> (B,2048,256)

Featurization (``log(1+|X|)`` + per-chunk standardization over the **full** 2048
bins, *before* the band split), the Nyquist handling (2048 network bins in / out,
the Nyquist row re-appended at reconstruction), and the ``1x1`` sigmoid mask head
are **baseline-identical** — so :class:`BandSplitUNet` is a drop-in replacement
for :class:`~singnet.models.unet.SingNetC1` on the ``(B, 2048, 256) -> (B, 2048,
256)`` interface (same train loop, same overlap-add evaluator).

The chosen base width ``c`` and bottleneck ``b5`` are the deterministic output of
``scripts/match_params.py`` (the |P - 9,835,745| / 9,835,745 <= 2% search); they
are pinned here as :data:`MATCHED_BASE_WIDTH` / :data:`MATCHED_BOTTLENECK_WIDTH`
and the exact resulting parameter count is asserted in
``tests/test_bandsplit_model.py`` and ``tests/test_match_params.py`` — the
credibility crux (THEORY §4). The code is the single source of truth.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ..audio.stft import DEFAULT_N_FFT, N_FRAMES, N_MASK_BINS
from .unet import _DecoderBlock, _EncoderBlock, _cat_skip, _featurize

#: Default STFT pins that fix the bin<->Hz map (MASTER_PLAN §5). ``sr / n_fft`` Hz
#: per bin (``44100 / 4096 ~= 10.77`` Hz); the network sees the first 2048 bins.
DEFAULT_SR = 44100

#: Deterministic width-match result (``scripts/match_params.py``; THEORY §4):
#: the largest base width whose *pure* 3-tower variant stays under the 9,835,745
#: baseline, then a single ``+Delta`` bottleneck bump to close the gap to +0.06 %.
MATCHED_BASE_WIDTH = 23
#: Bottleneck (level-5) width for the matched variant: ``16 * 23 + 14 = 382``.
MATCHED_BOTTLENECK_WIDTH = 382
#: The baseline (SingNet-C1) parameter count both variants are matched to.
BASELINE_PARAM_COUNT = 9_835_745
#: The exact matched-variant count (mel and uniform are identical by construction).
MATCHED_VARIANT_PARAM_COUNT = 9_841_896


# --- band edges -------------------------------------------------------------

def hz_to_bin(freq_hz: float, sr: int = DEFAULT_SR, n_fft: int = DEFAULT_N_FFT) -> int:
    """Nearest STFT bin index for a frequency, ``round(f / (sr / n_fft))``."""
    return int(round(freq_hz * n_fft / sr))


def bin_to_hz(bin_index: int, sr: int = DEFAULT_SR, n_fft: int = DEFAULT_N_FFT) -> float:
    """Centre frequency (Hz) of an STFT bin, ``bin * sr / n_fft``."""
    return float(bin_index) * sr / n_fft


def _hz_to_mel_htk(freq_hz: float) -> float:
    r"""HTK mel scale :math:`m = 2595\,\log_{10}(1 + f/700)`."""
    return 2595.0 * math.log10(1.0 + freq_hz / 700.0)


def _mel_to_hz_htk(mel: float) -> float:
    r"""Inverse HTK mel :math:`f = 700\,(10^{m/2595} - 1)`."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_edges(
    n_bands: int = 3,
    sr: int = DEFAULT_SR,
    n_fft: int = DEFAULT_N_FFT,
    fmin: float = 0.0,
    n_mask_bins: int = N_MASK_BINS,
) -> list[int]:
    r"""Equal-mel band edges as **bin** indices (HTK mel; MASTER_PLAN §3.1, §9).

    Partition ``[fmin, sr/2]`` into ``n_bands`` equal-width intervals **in the HTK
    mel domain** (:math:`m = 2595\log_{10}(1+f/700)`), map the interior edges back
    to Hz and then to the nearest STFT bin. The first edge is pinned to bin ``0``
    and the last to ``n_mask_bins`` (the network-bin count, 2048) so the returned
    ``n_bands + 1`` edges partition the 2048 bins exactly, disjointly, and
    exhaustively.

    For the project pins (``sr=44100, n_fft=4096``, 2048 network bins) this yields
    ``[0, 142, 597, 2048]`` — interior edges at ~1.53 kHz and ~6.43 kHz, narrow
    low bands where vocal energy concentrates and a wide (partly AAC-dead) top
    band. Derived from the formula in code; the values are asserted for exact
    self-consistency and ±3-bin proximity to the plan's estimate in tests.
    """
    if n_bands < 1:
        raise ValueError(f"n_bands must be >= 1, got {n_bands}")
    fmax = sr / 2.0
    mel_lo, mel_hi = _hz_to_mel_htk(fmin), _hz_to_mel_htk(fmax)
    mels = [mel_lo + i / n_bands * (mel_hi - mel_lo) for i in range(n_bands + 1)]
    edges = [hz_to_bin(_mel_to_hz_htk(m), sr, n_fft) for m in mels]
    edges[0] = 0
    edges[-1] = int(n_mask_bins)
    _validate_edges(edges, n_mask_bins)
    return edges


def uniform_edges(n_bands: int = 3, n_mask_bins: int = N_MASK_BINS) -> list[int]:
    """Equal-**bin**-width band edges (the control that isolates mel spacing).

    Splits the 2048 network bins into ``n_bands`` intervals as equal in bin width
    as the integer count allows: ``[0, 683, 1365, 2048]`` for 3 bands (interior
    edges at ~7.35 kHz and ~14.70 kHz).
    """
    if n_bands < 1:
        raise ValueError(f"n_bands must be >= 1, got {n_bands}")
    edges = [round(i * n_mask_bins / n_bands) for i in range(n_bands + 1)]
    edges[0] = 0
    edges[-1] = int(n_mask_bins)
    _validate_edges(edges, n_mask_bins)
    return edges


def _validate_edges(edges: list[int], n_mask_bins: int) -> None:
    """Assert the edges partition ``[0, n_mask_bins)`` exactly/disjointly/exhaustively."""
    if edges[0] != 0 or edges[-1] != n_mask_bins:
        raise ValueError(f"edges must span [0, {n_mask_bins}]; got {edges}")
    for lo, hi in zip(edges, edges[1:]):
        if hi <= lo:
            raise ValueError(f"edges must be strictly increasing (non-empty bands); got {edges}")


# --- one per-band encoder tower ---------------------------------------------

class _EncoderTower(nn.Module):
    """A 5-level strided-conv encoder for one band (baseline block design).

    Channels ``1 -> c -> 2c -> 4c -> 8c -> b5``. :meth:`forward` returns the five
    level outputs (level 5 is the tower's bottleneck contribution), which the
    parent concatenates across bands along the frequency axis.
    """

    def __init__(self, base_width: int, bottleneck_width: int) -> None:
        super().__init__()
        c, b5 = base_width, bottleneck_width
        self.enc1 = _EncoderBlock(1, c)
        self.enc2 = _EncoderBlock(c, 2 * c)
        self.enc3 = _EncoderBlock(2 * c, 4 * c)
        self.enc4 = _EncoderBlock(4 * c, 8 * c)
        self.enc5 = _EncoderBlock(8 * c, b5)  # bottleneck level

    def forward(self, x: Tensor) -> list[Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        return [e1, e2, e3, e4, e5]


class BandSplitUNet(nn.Module):
    r"""Param-matched 3-band-split magnitude-mask U-Net (MASTER_PLAN §3.1, §9).

    Args:
        edges_bins: band boundaries as bin indices, length ``n_bands + 1``, with
            ``edges_bins[0] == 0`` and ``edges_bins[-1] == n_mask_bins`` (use the
            :meth:`from_mel_bands` / :meth:`from_uniform_bands` classmethods for
            the two studied layouts).
        base_width: the shared tower/decoder base width ``c`` (23 for the matched
            variant; :data:`MATCHED_BASE_WIDTH`).
        bottleneck_width: the level-5 width ``b5``; ``None`` -> ``16 * base_width``
            (the "pure" variant). The matched variant bumps it to
            :data:`MATCHED_BOTTLENECK_WIDTH` (382) to land within 2 % of the
            baseline (THEORY §4).
        n_mask_bins / n_frames: the network grid (2048 x 256, baseline-identical).
        pad_multiple: each band is zero-padded (high-frequency side) up to the
            next multiple of this (32 = ``2**5``) so five stride-2 layers halve
            cleanly; the padding is removed from the final mask.

    The forward interface is identical to :class:`~singnet.models.unet.SingNetC1`:
    ``(B, 2048, 256)`` mixture magnitude in, ``(B, 2048, 256)`` soft mask out.
    """

    def __init__(
        self,
        edges_bins: list[int],
        base_width: int,
        bottleneck_width: int | None = None,
        *,
        n_mask_bins: int = N_MASK_BINS,
        n_frames: int = N_FRAMES,
        pad_multiple: int = 32,
    ) -> None:
        super().__init__()
        edges_bins = [int(e) for e in edges_bins]
        _validate_edges(edges_bins, n_mask_bins)
        c = int(base_width)
        b5 = int(bottleneck_width) if bottleneck_width is not None else 16 * c

        self.edges_bins = edges_bins
        self.base_width = c
        self.bottleneck_width = b5
        self.n_bands = len(edges_bins) - 1
        self.n_mask_bins = int(n_mask_bins)
        self.n_frames = int(n_frames)
        self.pad_multiple = int(pad_multiple)

        # Per-band bin widths, the padded (multiple-of-32) widths, and the offsets
        # of each band's *real* bins inside the padded, concatenated frequency axis.
        self.band_widths = [hi - lo for lo, hi in zip(edges_bins, edges_bins[1:])]
        self.padded_widths = [
            int(math.ceil(w / self.pad_multiple) * self.pad_multiple) for w in self.band_widths
        ]
        self.padded_total = sum(self.padded_widths)
        offsets, running = [], 0
        for p in self.padded_widths:
            offsets.append(running)
            running += p
        self.band_offsets = offsets  # start of each band's region in the padded axis

        # One tower per band (all share width c / b5), then the baseline-identical
        # bottleneck + decoder + head at width c (dec1 consumes the b5 bottleneck).
        self.towers = nn.ModuleList(
            _EncoderTower(c, b5) for _ in range(self.n_bands)
        )
        self.dec1 = _DecoderBlock(b5, 8 * c, dropout=True)      # b5  -> 8c
        self.dec2 = _DecoderBlock(16 * c, 4 * c, dropout=True)  # 16c -> 4c (8c d + 8c skip)
        self.dec3 = _DecoderBlock(8 * c, 2 * c, dropout=True)   # 8c  -> 2c
        self.dec4 = _DecoderBlock(4 * c, c, dropout=False)      # 4c  -> c
        self.dec5 = _DecoderBlock(2 * c, c, dropout=False)      # 2c  -> c
        self.head = nn.Conv2d(c, 1, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming-normal (fan-in) for conv/transpose-conv; BN default (baseline §5)."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="leaky_relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # --- band routing (pure; unit-tested round-trip) ------------------------

    def split_and_pad(self, feat: Tensor) -> list[Tensor]:
        """Split a ``(B, C, 2048, T)`` tensor into per-band, zero-padded slices.

        Each band ``[lo, hi)`` is sliced along the frequency axis and padded on the
        **high-frequency side** up to its multiple-of-32 width, so a tower's five
        stride-2 layers halve exactly and the real bins stay contiguous at the
        start of the band's region (making :meth:`crop_back` a plain gather).
        """
        bands = []
        for (lo, hi), p in zip(
            zip(self.edges_bins, self.edges_bins[1:]), self.padded_widths
        ):
            band = feat[..., lo:hi, :]
            pad = p - (hi - lo)
            if pad:
                band = torch.nn.functional.pad(band, (0, 0, 0, pad))
            bands.append(band)
        return bands

    def crop_back(self, y_full: Tensor) -> Tensor:
        """Gather the real (unpadded) bins from a ``(B, C, padded_total, T)`` tensor.

        Inverse of the padding in :meth:`split_and_pad`: for each band take the
        first ``band_width`` frequency rows of its padded region and concatenate,
        recovering exactly ``n_mask_bins`` bins in ascending-frequency order.
        """
        pieces = [
            y_full[..., off : off + w, :]
            for off, w in zip(self.band_offsets, self.band_widths)
        ]
        return torch.cat(pieces, dim=-2)

    # --- forward ------------------------------------------------------------

    def forward(self, mix_mag: Tensor) -> Tensor:
        """Mixture magnitude ``(B, 2048, 256)`` -> soft mask ``(B, 2048, 256)``."""
        x = _featurize(mix_mag)  # (B, 1, 2048, 256) — full-spectrum standardization

        # Independent per-band towers (no cross-band mixing), then frequency-axis
        # concatenation of matched-channel level outputs into full-spectrum skips.
        band_inputs = self.split_and_pad(x)
        tower_levels = [tower(band) for tower, band in zip(self.towers, band_inputs)]
        skips = [
            torch.cat([levels[level] for levels in tower_levels], dim=-2)
            for level in range(5)
        ]
        e1, e2, e3, e4, e5 = skips  # e5 is the (concatenated) bottleneck

        d1 = self.dec1(e5)
        d2 = self.dec2(_cat_skip(d1, e4))
        d3 = self.dec3(_cat_skip(d2, e3))
        d4 = self.dec4(_cat_skip(d3, e2))
        d5 = self.dec5(_cat_skip(d4, e1))

        mask_full = self.head(d5).sigmoid()  # (B, 1, padded_total, 256)
        mask = self.crop_back(mask_full)     # (B, 1, 2048, 256)
        return mask.squeeze(1)

    @property
    def num_parameters(self) -> int:
        """Total trainable parameter count (asserted in tests; THEORY §4 table)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def edges_hz(self) -> list[float]:
        """Band edges in Hz (bin centres) — for figures and the banded grid."""
        return [bin_to_hz(b) for b in self.edges_bins]

    # --- constructors for the two studied layouts ---------------------------

    @classmethod
    def from_mel_bands(
        cls,
        n_bands: int = 3,
        base_width: int = MATCHED_BASE_WIDTH,
        bottleneck_width: int = MATCHED_BOTTLENECK_WIDTH,
        *,
        sr: int = DEFAULT_SR,
        n_fft: int = DEFAULT_N_FFT,
        fmin: float = 0.0,
        n_mask_bins: int = N_MASK_BINS,
    ) -> "BandSplitUNet":
        """Matched mel-spaced variant (edges via :func:`mel_edges`)."""
        edges = mel_edges(n_bands, sr=sr, n_fft=n_fft, fmin=fmin, n_mask_bins=n_mask_bins)
        return cls(edges, base_width, bottleneck_width, n_mask_bins=n_mask_bins)

    @classmethod
    def from_uniform_bands(
        cls,
        n_bands: int = 3,
        base_width: int = MATCHED_BASE_WIDTH,
        bottleneck_width: int = MATCHED_BOTTLENECK_WIDTH,
        *,
        n_mask_bins: int = N_MASK_BINS,
    ) -> "BandSplitUNet":
        """Matched equal-bin-width variant (edges via :func:`uniform_edges`)."""
        edges = uniform_edges(n_bands, n_mask_bins=n_mask_bins)
        return cls(edges, base_width, bottleneck_width, n_mask_bins=n_mask_bins)


def build_bandsplit(
    bands: str = "mel",
    base_width: int = MATCHED_BASE_WIDTH,
    bottleneck_width: int = MATCHED_BOTTLENECK_WIDTH,
    n_bands: int = 3,
) -> BandSplitUNet:
    """Factory used by the config-driven builder: ``bands`` in ``{mel, uniform}``."""
    if bands == "mel":
        return BandSplitUNet.from_mel_bands(n_bands, base_width, bottleneck_width)
    if bands == "uniform":
        return BandSplitUNet.from_uniform_bands(n_bands, base_width, bottleneck_width)
    raise ValueError(f"unknown band layout {bands!r}; expected 'mel' or 'uniform'")
