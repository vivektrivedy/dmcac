import os, torch, random, itertools
from torch.utils.data import DataLoader
from dmcac.config import CFG
from dmcac.core.backbone import ViTEncoder
from dmcac.core.losses import frob_js, CLSClassifier, CACLayer
from dmcac.core.retrieval import ensure_schema, upload_embeddings, query_weaviate
from dmcac.data.dataset import DMCACDataset
from dmcac.data.augment import VIEW_TFORM
from dmcac.utils.timer import Timer

def main():
    # ---- prep ----
    ensure_schema()
    train_ds = DMCACDataset(CFG.train_root)
    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size,
                              shuffle=True, num_workers=8, drop_last=True)
    device = CFG.device
    enc = ViTEncoder().to(device).eval()     # frozen backbone
    cls_head = CLSClassifier().to(device)
    cac_head = CACLayer().to(device)
    opt = torch.optim.AdamW(
        itertools.chain(cls_head.parameters(), cac_head.parameters()),
        lr=CFG.lr, weight_decay=CFG.weight_decay
    )

    # ---- initial DB upload ----
    upload_embeddings(train_loader, device)

    for epoch in range(CFG.epochs):
        if epoch and epoch % CFG.refresh_epochs == 0:
            upload_embeddings(train_loader, device)   # refresh vectors

        for views, label, _ in train_loader:
            label = label.to(device)
            views_t = torch.stack([v.to(device) for v in views], dim=1)  # (B,V,C,H,W)
            B, V, _, _, _ = views_t.shape
            emb = enc(views_t.view(-1, *views_t.shape[2:]))    # (B*V,d)
            emb = emb.view(B, V, -1)

            # ---- self-supervised probs via Weaviate ----
            P = []
            db_vecs_all = []
            db_lbls_all = []
            for b in range(B):
                probs = []
                db_vecs_sample = []
                db_lbls_sample = []
                for v in range(V):
                    db_vecs, db_lbls = query_weaviate(
                        emb[b, v].cpu().numpy(), k=CFG.k_ann)
                    sims = (emb[b, v] @ db_vecs.T.to(device)).softmax(-1)
                    probs.append(sims)
                    db_vecs_sample.append(db_vecs)
                    db_lbls_sample.append(db_lbls)
                P.append(torch.stack(probs))                 # (V,k)
                db_vecs_all.append(torch.stack(db_vecs_sample))
                db_lbls_all.append(torch.stack(db_lbls_sample))
            P = torch.stack(P).to(device)                    # (B,V,k)

            loss = (
                CFG.w_frob * frob_js(P) +
                CFG.w_ce   * cls_head(emb[:, 0], label) +
                CFG.w_cac  * cac_head(emb[:, 0],
                                      torch.stack(db_vecs_all)[...,0,:].to(device),
                                      torch.stack(db_lbls_all).to(device))
            )

            opt.zero_grad(); loss.backward(); opt.step()
        print(f"[epoch {epoch}] loss={loss.item():.4f}")
