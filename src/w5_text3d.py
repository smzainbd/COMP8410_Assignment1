import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NonCircularSemanticPrior(nn.Module):
    """Semantic encoder trained on classes 0 to 7."""
    def __init__(self, emb_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.AvgPool2d(kernel_size=4, stride=4),
            nn.Flatten(),
            nn.Linear(7 * 7, 64),
            nn.SiLU(),
            nn.Linear(64, emb_dim)
        )
        nn.init.constant_(self.conv[-1].bias, 0.25)
        self.cls_head = nn.Linear(emb_dim, 8)
        self.spatial_head = nn.Linear(emb_dim, 7 * 7)

    def embed(self, x):
        return F.normalize(self.conv(x), dim=-1)


class NonCircularScorePrior(nn.Module):
    """2D Denoising Score Prior trained strictly on garment classes 0..7 for DreamFusion SDS gradients."""
    def __init__(self):
        super().__init__()
        self.skip = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 1, 3, padding=1)
        )

    def predict_eps(self, z_t, cond_map):
        inp = torch.cat([z_t, cond_map], dim=1)
        return self.skip(inp) + self.net(inp)


class Small3DRadianceField(nn.Module):
    """Shared 3D Volume queried across camera azimuths theta in [0, 2*pi) at 28x28 resolution."""
    def __init__(self, hidden=64):
        super().__init__()
        # 3D coordinates (3) + 4 Fourier frequency bands (3 * 2 * 4 = 24) = 27 input features
        self.mlp = nn.Sequential(
            nn.Linear(27, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2)
        )

    def pos_enc(self, pts):
        enc = [pts]
        for f in (1.0, 2.0, 4.0, 8.0):
            enc.append((1.0 / f) * torch.sin(f * math.pi * pts))
            enc.append((1.0 / f) * torch.cos(f * math.pi * pts))
        return torch.cat(enc, dim=-1)

    def render_azimuth(self, azimuth: float, H: int = 28, W: int = 28, n_depth: int = 8, ray_jitter: float = 0.0, device="cpu"):
        gy, gx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=device),
            torch.linspace(-1.0, 1.0, W, device=device), indexing="ij"
        )
        zs = torch.linspace(-0.8, 0.8, n_depth, device=device)
        c, s = math.cos(azimuth), math.sin(azimuth)
        scale_x = 0.60 + 0.40 * abs(c)

        # Back-project camera screen rays at azimuth into canonical 3D object coordinates
        x_obj = (gx * scale_x + 0.15 * s).unsqueeze(-1) + 0.05 * s * zs.view(1, 1, n_depth)
        y_obj = gy.unsqueeze(-1).expand(H, W, n_depth)
        z_obj = zs.view(1, 1, n_depth).expand(H, W, n_depth)
        pts_w = torch.stack([x_obj, y_obj, z_obj], dim=-1)

        if ray_jitter > 0.0:
            pts_w = pts_w + (ray_jitter * 0.12) * torch.randn_like(pts_w)

        raw = self.mlp(self.pos_enc(pts_w.reshape(-1, 3))).view(H, W, n_depth, 2)
        shading = 0.80 + 0.20 * c
        intensity = torch.sigmoid(raw[..., 0] - 0.8)
        sigma = F.softplus(raw[..., 1] + 1.2)
        alpha = 1.0 - torch.exp(-sigma * (1.6 / n_depth))
        ones = torch.ones((H, W, 1), device=device)
        trans = torch.cumprod(torch.cat([ones, 1.0 - alpha + 1e-7], dim=-1), dim=-1)[..., :-1]
        weights = trans * alpha
        acc_alpha = weights.sum(dim=-1)
        img = torch.clamp((weights * intensity).sum(dim=-1) * shading, 0.0, 1.0).unsqueeze(0).unsqueeze(0)
        return img, acc_alpha