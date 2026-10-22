import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_moving_garment_clips(x_static, T=8, shift_px=1):
    B, C, H, W = x_static.shape
    gy, gx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, H, device=x_static.device),
        torch.linspace(-1.0, 1.0, W, device=x_static.device),
        indexing="ij"
    )
    base_grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(B, H, W, 2)
    step_dx = (2.0 / W) * (shift_px * 1.5)
    frames = []
    for t in range(T):
        g = base_grid.clone()
        # Horizontal translation + subtle non-rigid fabric sway across frames
        sway = 0.07 * math.sin(t * 0.8) * gy.unsqueeze(0)
        g[..., 0] = g[..., 0] - t * step_dx + sway
        frames.append(F.grid_sample(x_static, g, mode="bilinear", padding_mode="zeros", align_corners=True))
    return torch.stack(frames, dim=1)  # (B, T, 1, 28, 28)


class VideoDiffusionBackbone(nn.Module):
    def __init__(self, dim=32, T=8):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, dim, 3, padding=1), nn.GroupNorm(4, dim), nn.SiLU(),
            nn.Conv2d(dim, dim, 3, stride=2, padding=1), nn.GroupNorm(4, dim), nn.SiLU()
        )
        self.q_sp = nn.Linear(dim, dim, bias=False)
        self.k_sp = nn.Linear(dim, dim, bias=False)
        self.v_sp = nn.Linear(dim, dim, bias=False)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, 4, stride=2, padding=1), nn.GroupNorm(4, dim), nn.SiLU(),
            nn.Conv2d(dim, 1, 3, padding=1)
        )
        # AnimateDiff plug-and-play 1D Temporal Transformer Attention module across T frames
        self.temp_pos = nn.Parameter(torch.randn(1, T, dim) * 0.02)
        self.temp_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
        self.temp_norm = nn.LayerNorm(dim)
        self.temp_proj = nn.Linear(dim, dim)

    def freeze_spatial_only(self):
        for name, p in self.named_parameters():
            if not name.startswith("temp_"):
                p.requires_grad = False

    def forward_2d_frame(self, x_frame):
        b = x_frame.size(0)
        h = self.enc(x_frame)
        tok = h.flatten(2).transpose(1, 2)
        q, k, v = self.q_sp(tok), self.k_sp(tok), self.v_sp(tok)
        attn = F.softmax(torch.bmm(q, k.transpose(1, 2)) / 6.0, dim=-1)
        h_out = (tok + torch.bmm(attn, v)).transpose(1, 2).view(b, -1, 14, 14)
        return torch.sigmoid(self.dec(h_out)), h_out

    def forward_t2v_zero(self, noisy_clips):
        B, T, C, H, W = noisy_clips.shape
        _, f0 = self.forward_2d_frame(noisy_clips[:, 0])
        _, D, H_lat, W_lat = f0.shape
        tok0 = f0.flatten(2).transpose(1, 2)
        k0, v0 = self.k_sp(tok0), self.v_sp(tok0)

        # Build 14x14 latent coordinate grid for Text2Video-Zero latent motion warping
        ly, lx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H_lat, device=noisy_clips.device),
            torch.linspace(-1.0, 1.0, W_lat, device=noisy_clips.device),
            indexing="ij"
        )
        lat_base = torch.stack([lx, ly], dim=-1).unsqueeze(0).expand(B, H_lat, W_lat, 2)
        lat_step_dx = (2.0 / W) * 1.5

        outs = []
        for t in range(T):
            # 1. Explicit Latent Motion Warping of Frame-0 latent f0 by translation vector t * lat_step_dx
            g_lat = lat_base.clone()
            g_lat[..., 0] = g_lat[..., 0] - t * lat_step_dx
            f0_warped = F.grid_sample(f0, g_lat, mode="bilinear", padding_mode="zeros", align_corners=True)

            # 2. Blend warped Frame-0 latent with current noisy frame encoding & apply Cross-Frame Attention
            h_t = 0.55 * f0_warped + 0.45 * self.enc(noisy_clips[:, t])
            tok_t = h_t.flatten(2).transpose(1, 2)
            q_t = self.q_sp(tok_t)
            attn_cross = F.softmax(torch.bmm(q_t, k0.transpose(1, 2)) / 6.0, dim=-1)
            h_aligned = (0.65 * tok_t + 0.35 * torch.bmm(attn_cross, v0)).transpose(1, 2).view(B, D, H_lat, W_lat)
            outs.append(torch.sigmoid(self.dec(h_aligned)))
        return torch.stack(outs, dim=1)

    def forward_animatediff(self, noisy_clips):
        B, T, C, H, W = noisy_clips.shape
        x_flat = noisy_clips.view(B * T, C, H, W)
        _, h_sp = self.forward_2d_frame(x_flat)
        _, D, H2, W2 = h_sp.shape
        z = h_sp.view(B, T, D, H2 * W2).permute(0, 3, 1, 2).reshape(B * H2 * W2, T, D)
        z_in = self.temp_norm(z + self.temp_pos[:, :T, :])
        z_attn, _ = self.temp_attn(z_in, z_in, z_in)
        z_out = (z + self.temp_proj(z_attn)).reshape(B, H2 * W2, T, D).permute(0, 2, 3, 1).reshape(B * T, D, H2, W2)
        return torch.sigmoid(self.dec(z_out)).view(B, T, C, H, W)
