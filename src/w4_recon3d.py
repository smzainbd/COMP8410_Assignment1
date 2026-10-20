import torch
import torch.nn as nn
import torch.nn.functional as F


def eval_sphere_rgb(xy):
    """Evaluates ground-truth shaded sphere RGB color at arbitrary 2D coordinates xy in [-1, 1]^2."""
    r2 = (xy ** 2).sum(dim=-1)
    inside = (r2 < 0.58).float()
    z_surf = torch.sqrt(torch.clamp(0.58 - r2, min=0.0))

    normals = F.normalize(torch.stack([xy[:, 0], xy[:, 1], z_surf], dim=-1), dim=-1)
    light_dir = F.normalize(torch.tensor([0.45, -0.5, 1.0], device=xy.device), dim=0)
    diffuse = torch.clamp((normals * light_dir).sum(dim=-1), 0.0, 1.0)
    stripe = 0.5 + 0.5 * torch.sin(14.0 * xy[:, 0]) * torch.cos(10.0 * xy[:, 1])

    gt_rgb = torch.stack([
        inside * (0.15 + 0.75 * diffuse * stripe),
        inside * (0.25 + 0.65 * diffuse),
        inside * (0.45 + 0.55 * (diffuse ** 3))
    ], dim=-1) + (1.0 - inside.unsqueeze(-1)) * 0.06
    return gt_rgb


def make_shaded_sphere_target(H=48, W=48, device="cuda"):
    """Creates a 48x48 grid of coordinates xy in [-1, 1]^2 and ground-truth sphere RGB colors."""
    gy, gx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing="ij"
    )
    xy = torch.stack([gx, gy], dim=-1).reshape(-1, 2)
    return xy, eval_sphere_rgb(xy)


class ExplicitGaussianSplatter(nn.Module):
    """
    Week 4 System 1: 3D Gaussian Splatting prototype (256-Gaussian splat fitter).
    Each primitive has a 3D center (z used as sort key), a 2D scale, color, and opacity.
    Blending is depth-sorted alpha compositing on one view.
    """
    def __init__(self, num_gaussians=256):
        super().__init__()
        self.xyz = nn.Parameter(torch.rand(num_gaussians, 3) * 1.4 - 0.7)
        self.log_scales = nn.Parameter(torch.full((num_gaussians, 2), -2.2))
        self.rgb_logits = nn.Parameter(torch.zeros(num_gaussians, 3))
        self.opacity_logits = nn.Parameter(torch.zeros(num_gaussians, 1))

    def forward(self, xy):
        order = torch.argsort(self.xyz[:, 2], descending=False)
        means_2d = self.xyz[order, :2]
        scales = torch.exp(-2.0 * self.log_scales[order])
        colors = torch.sigmoid(self.rgb_logits[order])
        opac = torch.sigmoid(self.opacity_logits[order]).squeeze(-1)

        diff = xy.unsqueeze(1) - means_2d.unsqueeze(0)
        dist2 = (diff ** 2 * scales.unsqueeze(0)).sum(dim=-1)
        alpha = torch.clamp(opac.unsqueeze(0) * torch.exp(-0.5 * dist2), 0.0, 0.99)

        ones = torch.ones(alpha.size(0), 1, device=xy.device)
        trans = torch.cumprod(torch.cat([ones, 1.0 - alpha + 1e-7], dim=-1), dim=-1)[:, :-1]
        weights = alpha * trans
        return torch.matmul(weights, colors) + (1.0 - weights.sum(dim=-1, keepdim=True)) * 0.06


class ImplicitRayMLP(nn.Module):
    """
    Week 4 System 2: NeRF prototype (coordinate MLP with 16 uniform depth samples
    and standard volume-rendering sum).
    """
    def __init__(self, freqs=6, hidden=128):
        super().__init__()
        self.freqs = freqs
        in_dim = 3 * (1 + 2 * freqs)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 4)
        )

    def pos_enc(self, pts):
        enc = [pts]
        for i in range(self.freqs):
            enc.append(torch.sin((2.0 ** i) * pts))
            enc.append(torch.cos((2.0 ** i) * pts))
        return torch.cat(enc, dim=-1)

    def forward(self, xy, num_depth_samples=16):
        z_vals = torch.linspace(-0.8, 0.8, num_depth_samples, device=xy.device)
        n_pix = xy.size(0)
        pts_xy = xy.unsqueeze(1).expand(n_pix, num_depth_samples, 2)
        pts_z = z_vals.view(1, num_depth_samples, 1).expand(n_pix, num_depth_samples, 1)
        pts_3d = torch.cat([pts_xy, pts_z], dim=-1)

        raw = self.mlp(self.pos_enc(pts_3d.reshape(-1, 3))).view(n_pix, num_depth_samples, 4)
        rgb = torch.sigmoid(raw[..., :3])
        sigma = F.relu(raw[..., 3])

        delta = 1.6 / num_depth_samples
        alpha = 1.0 - torch.exp(-sigma * delta)
        ones = torch.ones(n_pix, 1, device=xy.device)
        trans = torch.cumprod(torch.cat([ones, 1.0 - alpha + 1e-7], dim=-1), dim=-1)[:, :-1]
        weights = (trans * alpha).unsqueeze(-1)
        return (weights * rgb).sum(dim=1) + (1.0 - weights.sum(dim=1)) * 0.06