import torch
from torch import nn
import torchvision.models as models


class LightweightVisionEncoder(nn.Module):
    """ResNet-18 visual encoder used by Zeva's action-effect stream."""

    def __init__(self, output_dim: int = 256, *, pretrained: bool = True, num_views: int = 1):
        super().__init__()
        if num_views <= 0:
            raise ValueError("num_views must be positive.")
        self.num_views = num_views
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.proj = nn.Linear(512, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.view_fusion = (
            nn.Sequential(
                nn.Linear(output_dim * num_views, output_dim),
                nn.LayerNorm(output_dim),
                nn.SiLU(),
            )
            if num_views > 1
            else nn.Identity()
        )
        nn.init.orthogonal_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 4:
            if self.num_views != 1:
                raise ValueError(f"Expected [N,{self.num_views},3,H,W], got {tuple(images.shape)}.")
            features = self.backbone(images).flatten(1)
            return self.norm(self.proj(features))
        if images.ndim != 5 or images.shape[1:3] != (self.num_views, 3):
            raise ValueError(f"Expected [N,{self.num_views},3,H,W], got {tuple(images.shape)}.")
        batch_size = images.shape[0]
        features = self.backbone(images.flatten(0, 1)).flatten(1)
        features = self.norm(self.proj(features)).view(batch_size, self.num_views, -1)
        return self.view_fusion(features.flatten(1))
