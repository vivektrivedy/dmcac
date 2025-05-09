import timm, torch
from dmcac.config import CFG

class ViTEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = timm.create_model(CFG.model_name, pretrained=True, num_classes=0)
        self.norm = torch.nn.functional.normalize

    @torch.no_grad()
    def forward(self, x):           # x (B,C,H,W)
        cls = self.vit.forward_features(x)[:, 0]   # CLS only
        return self.norm(cls, dim=-1)              # unit-norm
