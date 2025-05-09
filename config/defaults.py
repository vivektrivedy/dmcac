from types import SimpleNamespace as NS

CFG = NS(
    # backbone
    model_name="vit_small_patch16_224.augreg_in21k",  # ViT-S
    embed_dim=384,

    # training
    batch_size=128,
    lr=3e-4,
    weight_decay=1e-2,
    epochs=60,
    refresh_epochs=15,          # DB re-encode
    num_views=6,                # A
    k_ann=12,                   # top-k Weaviate neighbours
    margin=0.2,
    device="cuda",

    # data
    img_size=224,
    train_root="data/train",
    test_root="data/test",

    # loss weights
    w_frob=1.0,
    w_ce=1.0,
    w_cac=1.0,
)
