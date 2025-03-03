import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils.token_masking import TokenMasking


class VITClassification(nn.Module):
    def __init__(
            self,
            img_size: int,
            model_name: str,
            num_classes: int,
            patch_size: int = 14,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_classes = num_classes

        self.encoder = torch.hub.load('facebookresearch/dinov2', model_name)

        self.out = nn.Sequential(
            nn.Linear(self.encoder.embed_dim, num_classes),
        )

        self.token_masking = TokenMasking(self.encoder.mask_token)

        self.param_defs_decoder = [
            ("out", self.out),
        ]

        self.param_defs_encoder_blocks = [
            ("encoder.blocks", self.encoder.blocks),
        ]

        self.param_defs_encoder_stems = [
            ("encoder.mask_token", self.encoder.mask_token),
            ("encoder.norm", self.encoder.norm),
            ("encoder.pos_embed", self.encoder.pos_embed),
            ("encoder.patch_embed.proj", self.encoder.patch_embed.proj),
            ("encoder.cls_token", self.encoder.cls_token)
            if hasattr(self.encoder, "cls_token")
            else (None, None),
        ]

        self.encoder_depth = len(self.encoder.blocks)

    def forward_features(self, img: torch.Tensor, mask_ratio: float):
        b, c, h, w = img.shape
        token_img_shape = (b, self.encoder.embed_dim, h // self.patch_size, w // self.patch_size)

        x_patch = self.encoder.patch_embed(img)
        if 0.0 < mask_ratio:
            x_patch = self.token_masking(x_patch, mask_ratio)
        x = torch.cat((self.encoder.cls_token.expand(x_patch.shape[0], -1, -1), x_patch), dim=1)
        x = x + self.encoder.interpolate_pos_encoding(x, w, h)
        x = x.contiguous()
        for i in range(self.encoder_depth):
            x = self.encoder.blocks[i](x)
        x = self.encoder.norm(x)
        x = x[:, 0]
        return x

    def forward(self, img, mask_ratio=0.0):
        b, c, h, w = img.shape
        assert h == w

        feats = self.forward_features(img, mask_ratio)
        logit = self.out(feats)

        return logit
