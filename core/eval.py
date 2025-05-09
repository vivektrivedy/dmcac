import os, torch, tqdm, weaviate
from torch.utils.data import DataLoader
from dmcac.config import CFG
from dmcac.core.backbone import ViTEncoder
from dmcac.data.dataset import DMCACDataset
from dmcac.core.retrieval import ensure_schema, query_weaviate, upload_embeddings

def main():
    ensure_schema()
    device = CFG.device
    enc = ViTEncoder().to(device).eval()

    db_ds = DMCACDataset(CFG.test_root)
    db_loader = DataLoader(db_ds, batch_size=CFG.batch_size, num_workers=8)
    upload_embeddings(db_loader, device)          # push DB vectors

    q_loader = DataLoader(db_ds, batch_size=1, num_workers=4)
    hits = {k: 0 for k in [1,2,4,8]}
    total = 0
    for views, label, _ in tqdm.tqdm(q_loader, desc="eval"):
        q = enc(views[0].to(device).unsqueeze(0))[0]
        _, lbls = query_weaviate(q.cpu().numpy(), k=max(hits))
        total += 1
        for k in hits:
            hits[k] += int(label.item() in lbls[:k])
    for k,v in hits.items():
        print(f"Recall@{k}: {v/total:.4f}")
