import os, json, torch, tqdm, weaviate
from dmcac.config import CFG
from dmcac.core.backbone import ViTEncoder

# ---------- Weaviate helpers ----------
def _client():
    return weaviate.Client(
        url=os.environ["WEAVIATE_URL"],
        auth_client_secret=weaviate.AuthApiKey(os.environ["WEAVIATE_API_KEY"])
    )

def ensure_schema():
    cl = os.environ["WEAVIATE_CLASS"]
    c = _client()
    if not c.schema.contains({"class": cl}):
        c.schema.create_class({
            "class": cl,
            "vectorizer": "none",
            "properties": [
                {"name": "label", "dataType": ["int"]},
                {"name": "path",  "dataType": ["text"]}
            ]
        })

def upload_embeddings(loader, device):
    """Encode full dataset and push to Weaviate (blocking)."""
    ensure_schema()
    cl = os.environ["WEAVIATE_CLASS"]
    enc = ViTEncoder().to(device).eval()
    cli = _client()
    with cli.batch as batch:
        for views, label, path in tqdm.tqdm(loader, desc="encode+upload"):
            img = views[0].to(device)  # first view only for DB
            vec = enc(img.unsqueeze(0)).cpu().numpy()[0]
            batch.add_data_object(
                {"label": int(label), "path": path},
                class_name=cl,
                vector=vec
            )

def query_weaviate(vec, k=CFG.k_ann):
    cl = os.environ["WEAVIATE_CLASS"]
    res = (_client().query
           .get(cl, ["label", "_additional { vector }"])
           .with_near_vector({"vector": vec, "certainty": 0.0})
           .with_limit(k)
           .do())
    hits = res["data"]["Get"].get(cl, [])
    labels = [int(h["label"]) for h in hits]
    vecs   = [h["_additional"]["vector"] for h in hits]
    return torch.tensor(vecs), torch.tensor(labels)
