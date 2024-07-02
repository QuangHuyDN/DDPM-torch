from typing import List
import torch
import torch.nn as nn


class SelfAttentionBlock(nn.Module):
    def __init__(self, channels: int, size: int):
        super().__init__()
        self.channels = channels
        self.size = size

        self.norm = nn.LayerNorm(channels)
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x: torch.Tensor):
        x = x.view(-1, self.channels, self.size * self.size).transpose(1, 2)
        x_norm = self.norm(x)
        attn, _ = self.mha(x_norm, x_norm, x_norm)
        attn = attn + x
        attn = self.ff(attn) + attn

        return attn.transpose(1, 2).view(
            -1, self.channels, self.size, self.size
        )


class DoubleConvWithResidual(nn.Module):
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        hidden_dim: int = None,
        kernel_size: int = 3,
        stride: int = 1,
        residual: bool = True,
    ) -> None:
        super().__init__()
        if not hidden_dim:
            hidden_dim = out_chans

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_chans,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=out_chans,
                kernel_size=kernel_size,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(1, out_chans),
        )

        self.residual = residual
        self.res_downsample = None
        if stride != 1 or in_chans != out_chans:
            self.res_downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_chans,
                    out_channels=out_chans,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_chans),
            )

    def forward(self, x: torch.Tensor):
        if self.residual:
            residual = x
            if self.res_downsample:
                residual = self.res_downsample(residual)
            return nn.functional.gelu(self.conv(x) + residual)
        else:
            return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, emb_dim: int = 256):
        super().__init__()

        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConvWithResidual(in_chans=in_chans, out_chans=in_chans),
            DoubleConvWithResidual(in_chans=in_chans, out_chans=out_chans),
        )

        self.emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=emb_dim, out_features=out_chans),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.block(x)
        emb = self.emb(t)[:, :, None, None].repeat(
            1, 1, x.shape[-2], x.shape[-1]
        )
        return x + emb


class Up(nn.Module):
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        emb_dim: int = 256,
        upsample: bool = True,
    ):
        super().__init__()

        if upsample:
            self.up = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels=in_chans,
                out_channels=in_chans // 2,
                kernel_size=2,
                stride=2,
            )

        self.conv = nn.Sequential(
            DoubleConvWithResidual(in_chans=in_chans, out_chans=in_chans),
            DoubleConvWithResidual(
                in_chans=in_chans, out_chans=out_chans, hidden_dim=in_chans // 2
            ),
        )

        self.emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=emb_dim, out_features=out_chans),
        )

    def forward(self, x: torch.Tensor, skip_x: torch.Tensor, t: torch.Tensor):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb(t)[:, :, None, None].repeat(
            1, 1, x.shape[-2], x.shape[-1]
        )
        return x + emb


class UNet(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        out_chans: int = 3,
        time_dim: int = 256,
        img_size: int = 64,
        dims: List[int] = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        self.img_size = img_size
        self.time_dim = time_dim
        if not dims:
            dims = [64, 128, 256, 512]
        self.dims = dims

        self.inc = DoubleConvWithResidual(in_chans, self.dims[0])

        self.down1 = Down(self.dims[0], self.dims[1])
        self.sa1 = SelfAttentionBlock(self.dims[1], img_size // 2)

        self.down2 = Down(self.dims[1], self.dims[2])
        self.sa2 = SelfAttentionBlock(self.dims[2], img_size // 4)

        self.down3 = Down(self.dims[2], self.dims[2])
        self.sa3 = SelfAttentionBlock(self.dims[2], img_size // 8)

        self.bot1 = DoubleConvWithResidual(self.dims[2], self.dims[3])
        self.bot2 = DoubleConvWithResidual(self.dims[3], self.dims[3])
        self.bot3 = DoubleConvWithResidual(self.dims[3], self.dims[2])

        self.up1 = Up(2 * self.dims[2], self.dims[1])
        self.sa4 = SelfAttentionBlock(self.dims[1], img_size // 4)

        self.up2 = Up(2 * self.dims[1], self.dims[0])
        self.sa5 = SelfAttentionBlock(self.dims[0], img_size // 2)

        self.up3 = Up(2 * self.dims[0], self.dims[0])
        self.sa6 = SelfAttentionBlock(self.dims[0], img_size)

        self.outc = nn.Conv2d(self.dims[0], out_chans, kernel_size=1)

    def pos_encoding(self, t: torch.Tensor, channels: int):
        inv_freq = 1.0 / (
            10000
            ** (
                torch.arange(0, channels, 2, device=self.device).float()
                / channels
            )
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)

        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)

        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)

        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)

        out = self.outc(x)
        return out


class ConditionalUNet(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        out_chans: int = 3,
        time_dim: int = 256,
        img_size: int = 64,
        dims: List[int] = None,
        num_classes: int = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        self.img_size = img_size
        self.time_dim = time_dim
        if not dims:
            dims = [64, 128, 256, 512]
        self.dims = dims

        self.inc = DoubleConvWithResidual(in_chans, self.dims[0])

        self.down1 = Down(self.dims[0], self.dims[1])
        self.sa1 = SelfAttentionBlock(self.dims[1], img_size // 2)

        self.down2 = Down(self.dims[1], self.dims[2])
        self.sa2 = SelfAttentionBlock(self.dims[2], img_size // 4)

        self.down3 = Down(self.dims[2], self.dims[2])
        self.sa3 = SelfAttentionBlock(self.dims[2], img_size // 8)

        self.bot1 = DoubleConvWithResidual(self.dims[2], self.dims[3])
        self.bot2 = DoubleConvWithResidual(self.dims[3], self.dims[3])
        self.bot3 = DoubleConvWithResidual(self.dims[3], self.dims[2])

        self.up1 = Up(2 * self.dims[2], self.dims[1])
        self.sa4 = SelfAttentionBlock(self.dims[1], img_size // 4)

        self.up2 = Up(2 * self.dims[1], self.dims[0])
        self.sa5 = SelfAttentionBlock(self.dims[0], img_size // 2)

        self.up3 = Up(2 * self.dims[0], self.dims[0])
        self.sa6 = SelfAttentionBlock(self.dims[0], img_size)

        self.outc = nn.Conv2d(self.dims[0], out_chans, kernel_size=1)

        self.num_classes = num_classes
        if self.num_classes:
            assert (
                self.num_classes > 0
            ), "Number of data classes must be positive"
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def pos_encoding(self, t: torch.Tensor, channels: int):
        inv_freq = 1.0 / (
            10000
            ** (
                torch.arange(0, channels, 2, device=self.device).float()
                / channels
            )
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor = None):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        if y:
            t += self.label_emb(y)

        x1 = self.inc(x)

        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)

        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)

        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)

        out = self.outc(x)
        return out
