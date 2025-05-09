import torch
from torch import nn
from torch.nn import functional as F
from dmcac.config import CFG

# ---------- Frobenius of pairwise JS ----------
def _js(p, q, eps=1e-8):
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (a.add(eps).log() - b.add(eps).log())).sum(-1)
    return 0.5 * (kl(p, m) + kl(q, m))

def frob_js(P):               # P (B,V,k) prob dists
    B, V, _ = P.shape
    loss = 0.
    for i in range(V):
        for j in range(i+1, V):
            loss += _js(P[:, i], P[:, j]).mean()
    return loss / (V * (V-1) / 2)

# ---------- classifiers ----------
class CLSClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(CFG.embed_dim, len(range(10000)))  # dummy large

    def forward(self, x, y):
        return F.cross_entropy(self.fc(x), y)

class CACLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = CFG.embed_dim ** -0.5
        self.fc = nn.Linear(CFG.embed_dim, len(range(10000)))

    def forward(self, q, db_vecs, db_labels):
        att = (q @ db_vecs.t()) * self.temperature
        w = att.softmax(-1)
        z = w @ db_vecs
        return F.cross_entropy(self.fc(z), db_labels[:, 0])
