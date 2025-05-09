DMCAC – Divergence Minimisation & Cross-Attention Retrieval
=========================================================

This repo contains the code for the paper "Divergence Minimisation & Cross-Attention Retrieval" from IJCAI 2024.

Every job is launched on **Modal A10G** GPUs and every vector lookup goes
through **Weaviate**.  Both of these approaches have generous free tiers and offer scalability to large scale datasets and performant retrieval.  Modal also scales to 0 and provides an easy to use API to launch jobs.

---

### 0 · Modal + Weaviate setup

```bash
pip install modal weaviate-client
modal token new                # authenticate

# create a Weaviate sandbox at cloud.weaviate.io, copy URL + API key
modal secret create weaviate-creds \
     WEAVIATE_URL=https://YOUR-CLUSTER.weaviate.network \
     WEAVIATE_API_KEY=KEY \
     WEAVIATE_CLASS=DMCACVector

## 1 · Install
pip install -r requirements.txt
modal deploy src/dmcac/pipeline/_modal_entry.py

## 2 Data
data/
├── train/
│   ├── class_a/ img1.jpg …
│   └── class_b/ …
└── test/
    ├── class_c/ …
    └── class_d/ …

## 3 Modal Launch for Train/Eval
modal run dmcac.train

### pushes DB vectors, then computes Recall@k
modal run dmcac.evaluate

#### Everything above is self-contained
Please see Weaviate/Modal latest documentation to keep up to date with backwards compatability.