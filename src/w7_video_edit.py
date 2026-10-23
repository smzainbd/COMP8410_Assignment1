import torch
import torch.nn as nn
import torch.nn.functional as F


def make_video_edit_pair(x_base, T=8):
    B, C, H, W = x_base.shape
    gy, gx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, H, device=x_base.device),
        torch.linspace(-1.0, 1.0, W, device=x_base.device),
        indexing="ij"
    )
    base_grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(B, H, W, 2)
    step_dx = 2.0 / W
    stripe = (0.65 + 0.35 * torch.cos(torch.linspace(0, 5.0 * 3.14159, H, device=x_base.device))).view(1, 1, H, 1)
    src_frames, tgt_frames = [], []
    for t in range(T):
        g = base_grid.clone()
        g[..., 0] = g[..., 0] - t * step_dx
        f_src = F.grid_sample(x_base, g, mode="bilinear", padding_mode="border", align_corners=True)
        mask = (f_src > 0.15).float()
        f_tgt = mask * torch.clamp((1.0 - f_src) * 0.80 * stripe, 0.0, 1.0) + (1.0 - mask) * f_src
        src_frames.append(f_src)
        tgt_frames.append(f_tgt)
    flow_grid = base_grid.clone()
    flow_grid[..., 0] = flow_grid[..., 0] - step_dx
    return torch.stack(src_frames, dim=1), torch.stack(tgt_frames, dim=1), flow_grid


class SpatioTemporalEditNet(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(1, dim, 3, padding=1), nn.GroupNorm(4, dim), nn.SiLU())
        self.w_q = nn.Linear(dim, dim, bias=False)
        self.w_k = nn.Linear(dim, dim, bias=False)
        self.w_v = nn.Linear(dim, dim, bias=False)
        self.style_bias = nn.Parameter(torch.zeros(1, 1, dim))
        self.dec = nn.Conv2d(dim, 1, 3, padding=1)
        nn.init.constant_(self.dec.bias, -1.5)

    def freeze_for_tune_a_video(self):
        for name, p in self.named_parameters():
            if not ("w_q" in name or "w_k" in name or "w_v" in name or "style_bias" in name):
                p.requires_grad = False

    def edit_single_frame(self, f_src):
        B, C, H, W = f_src.shape
        tok = self.enc(f_src).flatten(2).transpose(1, 2) + self.style_bias
        q, k, v = self.w_q(tok), self.w_k(tok), self.w_v(tok)
        attn = F.softmax(torch.bmm(q, k.transpose(1, 2)) / 6.0, dim=-1)
        out_tok = tok + torch.bmm(attn, v)
        gate = torch.clamp(f_src * 2.5, 0.0, 1.0)
        out_img = gate * torch.sigmoid(self.dec(out_tok.transpose(1, 2).view(B, -1, H, W)))
        return out_img, out_tok

    def forward_tokenflow(self, src_clip):
        B, T, C, H, W = src_clip.shape
        _, tok_edit_0 = self.edit_single_frame(src_clip[:, 0])
        gy, gx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=src_clip.device),
            torch.linspace(-1.0, 1.0, W, device=src_clip.device),
            indexing="ij"
        )
        pos = torch.stack([gy, gx], dim=-1).view(1, H * W, 2).expand(B, -1, -1)
        f0 = F.normalize(self.enc(src_clip[:, 0]).flatten(2).transpose(1, 2), dim=-1)
        k0 = torch.cat([f0, pos * 1.5], dim=-1)
        outs = []
        for t in range(T):
            ft = F.normalize(self.enc(src_clip[:, t]).flatten(2).transpose(1, 2), dim=-1)
            qt = torch.cat([ft, pos * 1.5], dim=-1)
            sim = torch.bmm(qt, k0.transpose(1, 2))
            attn = F.softmax(sim * 50.0, dim=-1)
            prop_tok = torch.bmm(attn, tok_edit_0)
            _, cur_tok = self.edit_single_frame(src_clip[:, t])
            h_t = (0.55 * prop_tok + 0.45 * cur_tok).transpose(1, 2).view(B, -1, H, W)
            gate = torch.clamp(src_clip[:, t] * 2.5, 0.0, 1.0)
            outs.append(gate * torch.sigmoid(self.dec(h_t)))
        return torch.stack(outs, dim=1)

    def forward_tune_a_video(self, src_clip):
        B, T, C, H, W = src_clip.shape
        toks = [self.enc(src_clip[:, t]).flatten(2).transpose(1, 2) + self.style_bias for t in range(T)]
        outs = []
        for t in range(T):
            ctx = torch.cat([toks[0], toks[max(0, t - 1)]], dim=1)
            q, k, v = self.w_q(toks[t]), self.w_k(ctx), self.w_v(ctx)
            attn = F.softmax(torch.bmm(q, k.transpose(1, 2)) / 6.0, dim=-1)
            h_t = (toks[t] + torch.bmm(attn, v)).transpose(1, 2).view(B, -1, H, W)
            gate = torch.clamp(src_clip[:, t] * 2.5, 0.0, 1.0)
            outs.append(gate * torch.sigmoid(self.dec(h_t)))
        return torch.stack(outs, dim=1)