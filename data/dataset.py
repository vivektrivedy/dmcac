import os, glob, random
from PIL import Image
from torch.utils.data import Dataset
from dmcac.data.augment import VIEW_TFORM
from dmcac.config import CFG

class DMCACDataset(Dataset):
    """
    Returns (views, label, path)
        views  : list[Tensor] length=CFG.num_views
        label  : int (class)
        path   : str (original file path)
    """
    def __init__(self, root):
        self.root = root
        self.classes = sorted(os.listdir(root))
        self.imgs = [(p, cls)
                     for cls in self.classes
                     for p in glob.glob(os.path.join(root, cls, "*"))]

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        path, cls = self.imgs[idx]
        img = Image.open(path).convert("RGB")
        views = [VIEW_TFORM(img) for _ in range(CFG.num_views)]
        return views, self.classes.index(cls), path
