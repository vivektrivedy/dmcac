DMCAC – Image Retrieval with Self-Supervised Divergence Minimization and Cross-Attention Classification
=========================================================

This repo contains the code for the paper [Image Retrieval with Self-Supervised Divergence Minimization and Cross-Attention Classification](https://www.ijcai.org/proceedings/2024/149) from IJCAI 2024.

## Abstract
Common approaches to image retrieval include contrastive methods and specialized loss functions such as ranking losses and entropy regularizers. We present DMCAC (Divergence Minimization with Cross-Attention Classification), a novel image retrieval method that offers a new perspective on this training paradigm. We use self-supervision with a novel divergence loss framework alongside a simple data flow adjustment that minimizes a distribution over a database directly during training. We show that jointly learning a query representation over a database is a competitive and often improved alternative to traditional contrastive methods for image retrieval. We evaluate our method across several model configurations and four datasets, achieving state-of-the-art performance in multiple settings. We also conduct a thorough set of ablations that show the robustness of our method across full vs. approximate retrieval and different hyperparameter configurations.

Every job is launched on **Modal Labs** GPUs and every vector lookup goes
through **Weaviate**.  Both of these approaches have generous free tiers and offer scalability to large scale datasets and performant retrieval.  Modal also scales to 0 and provides an easy to use API to launch jobs.

---


```bash
pip install modal weaviate-client
modal token new                # authenticate

## create a Weaviate sandbox at cloud.weaviate.io, copy URL + API key
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
