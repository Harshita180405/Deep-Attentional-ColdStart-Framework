# ─────────────────────────────────────────────────────────────────────────────
#  config.py  —  All hyperparameters and device selection in one place
#  Edit these values before running to customise training behaviour.
# ─────────────────────────────────────────────────────────────────────────────

class Config:
    # ── Data ──────────────────────────────────────────────────────────────────
    data_dir        = "./data/ml-100k"   # where MovieLens 100K will be stored
    cold_start_tau  = 5                  # users/items with < tau interactions → cold

    # ── Model architecture ────────────────────────────────────────────────────
    latent_dim      = 64                 # shared embedding dimension d
    text_dim        = 128                # simulated BERT output width
    image_dim       = 256                # simulated ResNet output width
    cat_dim         = 32                 # (unused directly; kept for reference)
    demo_hidden     = [256, 128, 64]     # MLP hidden layer sizes
    num_item_mod    = 3                  # number of item modalities (text/img/cat)
    num_user_mod    = 2                  # number of user modalities (demo/hist)

    # ── Training ──────────────────────────────────────────────────────────────
    epochs          = 20
    batch_size      = 256
    lr              = 1e-3               # learning rate for fusion / AFFN layers
    lr_pretrained   = 1e-4               # fine-tune LR for encoder layers
    weight_decay    = 1e-4               # L2 regularisation coefficient λ
    dropout         = 0.2
    neg_sample_pow  = 0.75               # popularity-based negative sampling exponent
    top_k           = 10                 # K for NDCG@K / HR@K evaluation
    early_stop_pat  = 5                  # patience (epochs without val improvement)
    phase2_epochs   = 3                  # cold-start fine-tuning epochs
    seed            = 42

    # ── Device ────────────────────────────────────────────────────────────────
    # Priority order: AMD GPU (DirectML) → NVIDIA GPU (CUDA) → CPU
    @staticmethod
    def get_device():
        try:
            import torch_directml          # AMD GPU support on Windows
            dev = torch_directml.device()
            print("[Device] AMD GPU via DirectML")
            return dev
        except ImportError:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                print(f"[Device] CUDA GPU — {torch.cuda.get_device_name(0)}")
                return "cuda"
        except Exception:
            pass
        print("[Device] CPU")
        return "cpu"


# Singleton instance used everywhere else via `from config import CFG`
CFG = Config()
