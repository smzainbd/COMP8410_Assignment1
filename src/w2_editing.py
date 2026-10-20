import torch
import torch.nn as nn
import torch.nn.functional as F


def make_style_edit_targets(x_clean):
    """
    Creates paired target-style images from clean 28x28 Fashion-MNIST inputs.
    The target style combines the 3x3 max-pool morphological outline with a
    toned-down interior fill, preserving the garment's spatial silhouette.
    """
    dilated = F.max_pool2d(x_clean, kernel_size=3, stride=1, padding=1)
    outline = torch.clamp(dilated - x_clean, 0.0, 1.0)
    styled = torch.clamp(0.50 * x_clean + 0.95 * outline, 0.0, 1.0)
    return styled


class SharedAttentionEditUNet(nn.Module):
    """
    Week 2 shared U-Net backbone (75,201 parameters) comparing two attention-editing mechanisms
    at the 14x14 bottleneck (196 spatial tokens):
    - Prompt-to-Prompt (prototype): runs the source prompt (0..9), records its 196x196
      self-attention map A_src, and injects A_src into the edited prompt pass (10..19).
    - StyleAligned (prototype): concatenates self-attention keys and values from a reference
      styled image with the current image ([K_ref, K_in], 392 keys) to share style across images.
    """
    def __init__(self, cond_dim=32, n_ctx=4):
        super().__init__()
        self.cond_dim = cond_dim
        self.n_ctx = n_ctx
        # 0..9 = source garment prompts; 10..19 = style-edited garment prompts
        self.cond_emb = nn.Embedding(20, n_ctx * cond_dim)

        self.enc1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.SiLU())
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.SiLU())

        # Prompt cross-attention (applied first so spatial tokens reflect the prompt)
        self.w_q = nn.Linear(64, 64, bias=False)
        self.w_k = nn.Linear(cond_dim, 64, bias=False)
        self.w_v = nn.Linear(cond_dim, 64, bias=False)

        # Spatial self-attention over 14x14 = 196 tokens
        self.q_self = nn.Linear(64, 64, bias=False)
        self.k_self = nn.Linear(64, 64, bias=False)
        self.v_self = nn.Linear(64, 64, bias=False)

        self.dec1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.GroupNorm(4, 32), nn.SiLU())
        self.out_conv = nn.Conv2d(32, 1, 3, padding=1)

    def _encode_and_cross_attend(self, x, prompt_idx):
        b = x.size(0)
        h1 = self.enc1(x)
        h2 = self.enc2(h1)
        tokens = h2.flatten(2).transpose(1, 2)

        c_seq = self.cond_emb(prompt_idx).view(b, self.n_ctx, self.cond_dim)  # (B, 4, 32)
        q = self.w_q(tokens)
        k = self.w_k(c_seq)
        v = self.w_v(c_seq)
        cross_attn = F.softmax(torch.bmm(q, k.transpose(1, 2)) / 8.0, dim=-1)
        cond_tokens = tokens + torch.bmm(cross_attn, v)
        return h1, cond_tokens

    def _decode(self, tokens, h1):
        b = tokens.size(0)
        h_mid = tokens.transpose(1, 2).view(b, 64, 14, 14)
        return torch.sigmoid(self.out_conv(self.dec1(h_mid) + h1))

    def forward_base(self, x, prompt_idx):
        h1, cond_tokens = self._encode_and_cross_attend(x, prompt_idx)
        qs = self.q_self(cond_tokens)
        ks = self.k_self(cond_tokens)
        vs = self.v_self(cond_tokens)
        self_attn = F.softmax(torch.bmm(qs, ks.transpose(1, 2)) / 8.0, dim=-1)
        out_tokens = cond_tokens + torch.bmm(self_attn, vs)
        return self._decode(out_tokens, h1), self_attn

    def forward_prompt_to_prompt(self, x_in, y_class, inject_weight=0.85):
        """Injects source prompt (y_class) self-attention into edited prompt (y_class + 10) pass."""
        _, attn_src = self.forward_base(x_in, y_class)
        h1, edit_tokens = self._encode_and_cross_attend(x_in, y_class + 10)
        qs = self.q_self(edit_tokens)
        ks = self.k_self(edit_tokens)
        vs = self.v_self(edit_tokens)
        attn_edit = F.softmax(torch.bmm(qs, ks.transpose(1, 2)) / 8.0, dim=-1)
        attn_injected = inject_weight * attn_src.detach() + (1.0 - inject_weight) * attn_edit
        out_tokens = edit_tokens + torch.bmm(attn_injected, vs)
        return self._decode(out_tokens, h1)

    def forward_style_aligned(self, x_in, y_class, x_style_ref, y_ref_class):
        """Shares self-attention keys/values with a reference styled image across [K_ref, K_in]."""
        b = x_in.size(0)
        h1, edit_tokens = self._encode_and_cross_attend(x_in, y_class + 10)
        ref_x = x_style_ref.expand(b, -1, -1, -1)
        ref_y = y_ref_class.expand(b) + 10
        _, ref_tokens = self._encode_and_cross_attend(ref_x, ref_y)

        qs = self.q_self(edit_tokens)
        k_shared = torch.cat([self.k_self(ref_tokens), self.k_self(edit_tokens)], dim=1)
        v_shared = torch.cat([self.v_self(ref_tokens), self.v_self(edit_tokens)], dim=1)

        attn_shared = F.softmax(torch.bmm(qs, k_shared.transpose(1, 2)) / 8.0, dim=-1)
        out_tokens = edit_tokens + torch.bmm(attn_shared, v_shared)
        return self._decode(out_tokens, h1)