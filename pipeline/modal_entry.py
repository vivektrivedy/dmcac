import modal, os
from dmcac.pipeline import train_pipeline, eval_pipeline

image = (modal.Image.debian_slim()
         .pip_install_from_requirements("requirements.txt")
         .pip_install("torch==2.2.0", "torchvision==0.17.0"))

app    = modal.App("dmcac")
secret = modal.Secret.from_name("weaviate-creds")

@app.function(gpu="A10G", image=image, secret=secret, timeout=60*60*12)
def train():
    train_pipeline.run()

@app.function(image=image, secret=secret, timeout=60*60)
def evaluate():
    eval_pipeline.run()
