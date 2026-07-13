r"""SingNet-C1 — the fixed magnitude-mask U-Net (MASTER_PLAN §5).

Single-channel, Spleeter-family. Encoder of 5 strided 2-D conv blocks, symmetric
transposed-conv decoder with skip connections, ``1x1`` head + sigmoid producing a
soft ratio mask :math:`M \in [0,1]^{2048 \times 256}`.

Shapes (a 6 s / 44.1 kHz chunk, 2048 network bins x 256 frames)::

    input            1 x 2048 x 256
    enc1  Conv5x5/2 32 x 1024 x 128
    enc2  Conv5x5/2 64 x  512 x  64
    enc3  Conv5x5/2 128 x 256 x  32
    enc4  Conv5x5/2 256 x 128 x  16
    enc5  Conv5x5/2 512 x  64 x   8   (bottleneck)
    dec1  ConvT/2  256 x 128 x  16  (+ skip enc4 -> 512)  Dropout2d(0.5)
    dec2  ConvT/2  128 x 256 x  32  (+ skip enc3 -> 256)  Dropout2d(0.5)
    dec3  ConvT/2   64 x 512 x  64  (+ skip enc2 -> 128)  Dropout2d(0.5)
    dec4  ConvT/2   32 x 1024 x 128 (+ skip enc1 ->  64)
    dec5  ConvT/2   32 x 2048 x 256
    head  Conv1x1    1 x 2048 x 256  -> sigmoid

Input featurization (inside :meth:`forward`, identical across all loss arms):
:math:`\log(1+|X|)` followed by per-chunk standardization (scalar mean/std over
the chunk's featurized bins).

The exact parameter count is derived by hand in ``THEORY.md`` §5 and asserted in
``tests/test_model.py`` — the code is the single source of truth.
"""

from __future__ import annotations

from torch import Tensor, nn

from ..audio.stft import N_FRAMES, N_MASK_BINS


def _featurize(mag: Tensor) -> Tensor:
    r"""``log(1+|X|)`` then per-chunk standardization -> ``(B, 1, F, T)``.

    Accepts ``(B, F, T)`` or ``(B, 1, F, T)``. Standardization uses a single
    scalar mean/std per chunk (over channel/freq/frame), matching §5.
    """
    if mag.dim() == 3:
        mag = mag.unsqueeze(1)
    feat = mag.clamp_min(0.0).log1p()
    mean = feat.mean(dim=(1, 2, 3), keepdim=True)
    std = feat.std(dim=(1, 2, 3), keepdim=True)
    return (feat - mean) / (std + 1e-5)


class _EncoderBlock(nn.Module):
    """Conv2d 5x5 stride 2 pad 2 -> BatchNorm -> LeakyReLU(0.2)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=5, stride=2, padding=2)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class _DecoderBlock(nn.Module):
    """ConvTranspose2d 5x5 stride 2 -> BatchNorm -> ReLU (-> optional Dropout2d)."""

    def __init__(self, in_ch: int, out_ch: int, dropout: bool) -> None:
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=5, stride=2, padding=2, output_padding=1
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(0.5) if dropout else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.act(self.bn(self.deconv(x))))


class SingNetC1(nn.Module):
    """The fixed SingNet-C1 magnitude-mask U-Net.

    Args:
        base_channels: first encoder width (32 in the pinned config). Exposed so
            capacity sweeps in *other* directions can reuse the class; Direction
            01 always uses the default.
    """

    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        chans = [1, c, 2 * c, 4 * c, 8 * c, 16 * c]  # 1,32,64,128,256,512

        self.enc1 = _EncoderBlock(chans[0], chans[1])
        self.enc2 = _EncoderBlock(chans[1], chans[2])
        self.enc3 = _EncoderBlock(chans[2], chans[3])
        self.enc4 = _EncoderBlock(chans[3], chans[4])
        self.enc5 = _EncoderBlock(chans[4], chans[5])  # bottleneck 512

        # Decoder mirrors the encoder; skip concat doubles the next block's input.
        self.dec1 = _DecoderBlock(chans[5], chans[4], dropout=True)  # 512 -> 256
        self.dec2 = _DecoderBlock(chans[5], chans[3], dropout=True)  # 512 -> 128 (256+256 skip)
        self.dec3 = _DecoderBlock(chans[4], chans[2], dropout=True)  # 256 -> 64  (128+128 skip)
        self.dec4 = _DecoderBlock(chans[3], chans[1], dropout=False)  # 128 -> 32 (64+64 skip)
        self.dec5 = _DecoderBlock(chans[2], chans[1], dropout=False)  # 64 -> 32  (32+32 skip)
        self.head = nn.Conv2d(chans[1], 1, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming-normal (fan-in) for conv/transpose-conv; BN default."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="leaky_relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, mix_mag: Tensor) -> Tensor:
        """Mixture magnitude ``(B, 2048, 256)`` -> soft mask ``(B, 2048, 256)``."""
        x = _featurize(mix_mag)  # (B, 1, F, T)

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        d1 = self.dec1(e5)
        d2 = self.dec2(_cat_skip(d1, e4))
        d3 = self.dec3(_cat_skip(d2, e3))
        d4 = self.dec4(_cat_skip(d3, e2))
        d5 = self.dec5(_cat_skip(d4, e1))

        mask = self.head(d5).sigmoid()  # (B, 1, F, T)
        return mask.squeeze(1)

    @property
    def num_parameters(self) -> int:
        """Total trainable parameter count (asserted in tests, mirrored in THEORY §5)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _cat_skip(x: Tensor, skip: Tensor) -> Tensor:
    """Concatenate a decoder feature map with the mirrored encoder skip on channels."""
    import torch

    return torch.cat([x, skip], dim=1)


def build_model(base_channels: int = 32) -> SingNetC1:
    """Factory used by the train loop / configs."""
    return SingNetC1(base_channels=base_channels)
