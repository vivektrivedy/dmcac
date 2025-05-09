from torchvision.transforms import (
    RandAugment, Resize, RandomResizedCrop, ToTensor
)
from torchvision.transforms import Compose
from dmcac.config import CFG

# one global pipeline reused everywhere
VIEW_TFORM = Compose([
    Resize(256),
    RandomResizedCrop(CFG.img_size),
    RandAugment(num_ops=2, magnitude=9),
    ToTensor()
])
