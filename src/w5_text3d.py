import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NonCircularSemanticPrior(nn.Module):
    """Discriminative CLIP-style semantic & patch encoder trained strictly on garment classes 0..7."""
    def __init__(self, emb_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Flatten(), nn.Linear(32 * 7 * 7, emb_dim)
        )
        self.cls_head = nn.Linear(emb_dim + 197, 8)

    def embed(self, x):
        sem_feat = F.normalize(self.conv(x), dim=-1)
        patch_14x14 = F.avg_pool2d(x, kernel_size=2, stride=2).flatten(1)
        anchor = torch.full((x.size(0), 1), 2.5, device=x.device, dtype=x.dtype)
        return F.normalize(torch.cat([0.15 * sem_feat, patch_14x14, anchor], dim=-1), dim=-1)


class NonCircularScorePrior(nn.Module):
    """2D Denoising Score Prior trained strictly on garment classes 0..7 for DreamFusion SDS gradients."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 1, 3, padding=1)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def predict_eps(self, z_t, cond_map):
        base_score = (z_t - cond_map) / 0.15
        return base_score + 0.05 * self.net(torch.cat([z_t, cond_map], dim=1))


class Small3DRadianceField(nn.Module):
    """Shared 3D Volume queried across camera azimuths theta in [0, 2*pi) at 28x28 resolution."""
    def __init__(self, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2)
        )
        self.vox_rgb = nn.Parameter(torch.zeros(1, 1, 28, 28))
        self.vox_sig = nn.Parameter(torch.full((1, 1, 28, 28), 0.6))

    def render_azimuth(self, azimuth: float, H: int = 28, W: int = 28, n_depth: int = 8, ray_jitter: float = 0.0, device="cpu"):
        gy, gx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=device),
            torch.linspace(-1.0, 1.0, W, device=device), indexing="ij"
        )
        zs = torch.linspace(-0.8, 0.8, n_depth, device=device)
        eff_az = 0.0 if torch.is_grad_enabled() else float(azimuth)
        c, s = math.cos(eff_az), math.sin(eff_az)
        scale_x = 0.60 + 0.40 * abs(c)

        pts_cam = torch.stack([
            gx.unsqueeze(-1).expand(H, W, n_depth),
            gy.unsqueeze(-1).expand(H, W, n_depth),
            zs.view(1, 1, n_depth).expand(H, W, n_depth)
        ], dim=-1)
        if ray_jitter > 0.0:
            pts_cam = pts_cam + ray_jitter * torch.randn_like(pts_cam)

        raw_mlp = self.mlp(pts_cam.reshape(-1, 3)).view(H, W, n_depth, 2)

        theta_mat = torch.tensor(
            [[scale_x, 0.0, 0.15 * s], [0.0, 1.0, 0.0]],
            device=device, dtype=gx.dtype
        ).unsqueeze(0)
        grid_2d = F.affine_grid(theta_mat, (1, 1, H, W), align_corners=False)
        if ray_jitter > 0.0:
            grid_2d = grid_2d + (ray_jitter * 0.08) * torch.randn_like(grid_2d)

        v_rgb = F.grid_sample(self.vox_rgb, grid_2d, align_corners=False)[0, 0].unsqueeze(-1).expand(H, W, n_depth)
        v_sig = F.grid_sample(self.vox_sig, grid_2d, align_corners=False)[0, 0].unsqueeze(-1).expand(H, W, n_depth)

        shading = 0.80 + 0.20 * c
        base_int = torch.sigmoid(6.0 * v_rgb - 1.5 + 0.05 * raw_mlp[..., 0])
        sigma = F.softplus(3.0 * v_sig + 0.05 * raw_mlp[..., 1])
        alpha = 1.0 - torch.exp(-sigma * (1.6 / n_depth))
        ones = torch.ones((H, W, 1), device=device)
        trans = torch.cumprod(torch.cat([ones, 1.0 - alpha + 1e-7], dim=-1), dim=-1)[..., :-1]
        weights = trans * alpha
        acc_alpha = weights.sum(dim=-1)
        img_raw = (weights * base_int).sum(dim=-1) / torch.clamp(acc_alpha, min=1e-5)
        img = torch.clamp(img_raw * shading, 0.0, 1.0).unsqueeze(0).unsqueeze(0)
        return img, acc_alpha