import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionUNet2D(nn.Module):
    """
    Week 3 2D Convolutional U-Net with multi-token Cross-Attention at the 14x14 bottleneck (61,761 params):
    - Custom Diffusion (prototype): freezes spatial conv layers and tunes only token embeddings
      (cond_emb) and cross-attention key/value projections (w_k, w_v) -> 5,504 trainable parameters.
    - DreamBooth (prototype): tunes all 61,761 U-Net parameters and applies Classifier-Free Guidance
      using unconditional null token index 10.
    """
    def __init__(self, cond_dim=32, n_ctx=4):
        super().__init__()
        self.cond_dim = cond_dim
        self.n_ctx = n_ctx
        self.cond_emb = nn.Embedding(11, n_ctx * cond_dim)  # 0..9 classes, 10 = null token for CFG
        self.enc1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.SiLU())
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.SiLU())
        self.w_q = nn.Linear(64, 64, bias=False)
        self.w_k = nn.Linear(cond_dim, 64, bias=False)
        self.w_v = nn.Linear(cond_dim, 64, bias=False)
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.GroupNorm(4, 32), nn.SiLU())
        self.out_conv = nn.Conv2d(32, 1, 3, padding=1)

    def freeze_for_custom_diffusion(self):
        """Freezes all backbone weights except token embeddings (cond_emb) and cross-attention K, V (w_k, w_v)."""
        for name, p in self.named_parameters():
            if not ("w_k" in name or "w_v" in name or "cond_emb" in name):
                p.requires_grad = False

    def forward(self, x_n, y_idx):
        b = x_n.size(0)
        h1 = self.enc1(x_n)
        h2 = self.enc2(h1)
        tokens = h2.flatten(2).transpose(1, 2)

        c_seq = self.cond_emb(y_idx).view(b, self.n_ctx, self.cond_dim)  # (B, 4, 32)
        q = self.w_q(tokens)
        k = self.w_k(c_seq)
        v = self.w_v(c_seq)

        attn = F.softmax(torch.bmm(q, k.transpose(1, 2)) / 8.0, dim=-1)
        h_mid = (tokens + torch.bmm(attn, v)).transpose(1, 2).view(b, 64, 14, 14)
        return torch.sigmoid(self.out_conv(self.dec1(h_mid) + h1))

    def forward_cfg(self, x_n, y_idx, guidance_scale=1.5):
        """Classifier-Free Guidance combining conditional (y_idx) and unconditional (index 10) predictions."""
        cond_out = self.forward(x_n, y_idx)
        null_idx = torch.full_like(y_idx, 10)
        uncond_out = self.forward(x_n, null_idx)
        return torch.clamp(uncond_out + guidance_scale * (cond_out - uncond_out), 0.0, 1.0)