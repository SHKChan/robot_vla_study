import torch
import torch.nn as nn
import torchvision.models as models

class BCPolicy(nn.Module):
    def __init__(self, action_dim: int = 7):
        super().__init__()
        # Vision encoder: pretrained on ResNet-18
        resnet: nn.Module = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet.fc = nn.Identity()
        self.vision_encoder : nn.Module = resnet # output: [B, 512]

        # State encoder: simple MLP
        self.state_encoder: nn.Module = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        ) # output: [B, 64]

        # Fusion layer: concat then map to action
        self.fusion = nn.Sequential(
            nn.Linear(512+64, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        vis_feat: torch.Tensor = self.vision_encoder(image)
        state_feat: torch.Tensor = self.state_encoder(state)
        combined: torch.Tensor = torch.cat([vis_feat, state_feat], dim=1)
        action: torch.Tensor = self.fusion(combined)
        return action

    def freeze_vision_encoder(self, freeze: bool=True):
        if freeze:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            self.vision_encoder.eval()
        else:
            for param in self.vision_encoder.parameters():
                param.requires_grad = True
            self.vision_encoder.train()


if __name__ == "__main__":
    model = BCPolicy(7)
    model.freeze_vision_encoder(True)
    image: torch.Tensor = torch.randn(4, 3, 224, 224) # batch_size, channels, height, width
    state: torch.Tensor = torch.randn(4, 7)
    action: torch.Tensor = model(image, state)
    print(action) # [B, 7]
