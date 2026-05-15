from collections import defaultdict
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import CFG
from data_loader import load_data, split_data
from datasets import BPRDataset, RatingDataset
from features import item_feats, user_feats, build_hist_dict
from metrics import compute_mae_rmse, compute_ndcg
from model import DACSF

app_state = {}


def _log(app_state, msg):
    ts = datetime.now().strftime("%H:%M:%S")
    entry = f"[{ts}]  {msg}"
    app_state["log"].append(entry)
    if len(app_state["log"]) > 200:
        app_state["log"] = app_state["log"][-200:]
    print(msg)


def bpr_loss(s_pos, s_neg, model):
    """
    L_BPR = -log σ(s_pos - s_neg)  +  λ ||Θ||²
    """
    rank_loss = -F.logsigmoid(s_pos - s_neg).mean()
    l2_reg    = CFG.weight_decay * sum(p.pow(2).sum() for p in model.parameters())
    return rank_loss + l2_reg


# ─── Main training function ───────────────────────────────────────────────────

def run_training(app_state):
    """
    Full training pipeline. Designed to run in a background thread.
    Writes status, progress (0-100), per-epoch metrics and final results
    into app_state for the Flask API to stream to the frontend.
    """
    try:
        app_state.update({"status": "loading", "log": [],
                          "history": [], "results": {}, "progress": 0})

        log = lambda m: _log(app_state, m)
        log("=" * 50)
        log("  DACSF — Starting pipeline")
        log("=" * 50)

        device = torch.device(CFG.get_device())
        log(f"  Device: {device}")

        # ── 1. Load & split data ──────────────────────────────────────────────
        data = load_data(log_fn=log)
        train_df, val_df, test_df, cold_u, cold_i, freqs = split_data(
            data["ratings"], data["n_items"], log_fn=log)
        data.update(dict(train_df=train_df, val_df=val_df,
                         test_df=test_df, cold_users=cold_u, cold_items=cold_i))
        app_state.update({"data": data,
                          "n_users": data["n_users"],
                          "n_items": data["n_items"]})

        hist = build_hist_dict(train_df)

        # ── 2. Build model ────────────────────────────────────────────────────
        model = DACSF(CFG.text_dim, CFG.image_dim,
                      data["n_genres"], demo_dim=3, cfg=CFG).to(device)
        app_state["model"] = model
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f"  Model parameters: {n_params:,}")

        # ── 3. DataLoaders ────────────────────────────────────────────────────
        train_dataset = BPRDataset(train_df, data["n_items"], freqs)
        train_loader  = DataLoader(train_dataset, batch_size=CFG.batch_size,
                                   shuffle=True, num_workers=0)
        val_loader    = DataLoader(RatingDataset(val_df), batch_size=512)

        optimizer  = torch.optim.Adam(model.parameters(), lr=CFG.lr)
        best_ndcg  = 0.0
        patience   = 0
        best_state = None

        app_state["status"]       = "training"
        app_state["total_epochs"] = CFG.epochs

        # ── 4. Phase 1 — BPR training ─────────────────────────────────────────
        log("\n  Phase 1 — BPR Training")
        log("-" * 40)

        for epoch in range(1, CFG.epochs + 1):
            model.train()
            total_loss = 0.0

            for users, pos, neg in train_loader:
                u, p, n = users.numpy(), pos.numpy(), neg.numpy()
                uf  = user_feats(u, data, hist, device)
                pf  = item_feats(p, data, device)
                nf  = item_feats(n, data, device)

                s_pos, s_neg, _ = model(uf, pf, nf)
                loss = bpr_loss(s_pos, s_neg, model)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            app_state["epoch"]    = epoch
            app_state["progress"] = int(epoch / CFG.epochs * 85)

            # Validation metrics
            mae, rmse = compute_mae_rmse(model, val_loader, data, hist, device)
            prec, rec, ndcg, hr = compute_ndcg(
                model, val_df, train_df, data, hist, device, n_sample_users=80)

            m = dict(epoch=epoch,
                     loss=round(avg_loss, 4),
                     mae=round(mae, 4), rmse=round(rmse, 4),
                     precision=round(prec, 4), recall=round(rec, 4),
                     ndcg=round(ndcg, 4), hr=round(hr, 4))
            app_state["metrics"] = m
            app_state["history"].append(m)

            log(f"  Epoch {epoch:3d} | Loss:{avg_loss:.4f} | "
                f"MAE:{mae:.4f} | RMSE:{rmse:.4f} | "
                f"NDCG@{CFG.top_k}:{ndcg:.4f} | HR:{hr:.4f}")

            # Early stopping (NDCG@K on validation set)
            if ndcg > best_ndcg:
                best_ndcg  = ndcg
                patience   = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= CFG.early_stop_pat:
                    log(f"  Early stopping at epoch {epoch}.")
                    break

        # Restore best weights
        if best_state:
            model.load_state_dict(best_state)

        # ── 5. Phase 2 — Cold-start fine-tuning ──────────────────────────────
        log("\n  Phase 2 — Cold-Start Fine-tuning")
        cold_train = train_df[train_df["user_idx"].isin(cold_u)]

        if len(cold_train) > 0:
            cd_dataset = BPRDataset(cold_train, data["n_items"], freqs)
            cd_loader  = DataLoader(cd_dataset,
                                    batch_size=min(64, len(cold_train)),
                                    shuffle=True, num_workers=0)
            ft_opt = torch.optim.Adam(model.parameters(), lr=CFG.lr_pretrained)

            for ep in range(CFG.phase2_epochs):
                model.train()
                ep_loss = 0.0
                for u, p, n in cd_loader:
                    uf  = user_feats(u.numpy(), data, hist, device)
                    pf  = item_feats(p.numpy(), data, device)
                    nf  = item_feats(n.numpy(), data, device)
                    s_pos, s_neg, _ = model(uf, pf, nf)
                    loss = bpr_loss(s_pos, s_neg, model)
                    ft_opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    ft_opt.step()
                    ep_loss += loss.item()
                log(f"  Fine-tune {ep+1}/{CFG.phase2_epochs} "
                    f"| Loss:{ep_loss/len(cd_loader):.4f}")
        else:
            log("  No cold-start training samples found — skipping Phase 2.")

        # ── 6. Final evaluation ───────────────────────────────────────────────
        log("\n  Final Evaluation")
        log("-" * 40)
        app_state["progress"] = 90

        test_loader = DataLoader(RatingDataset(test_df), batch_size=512)
        mae, rmse   = compute_mae_rmse(model, test_loader, data, hist, device)
        p, r, nd, h = compute_ndcg(model, test_df, train_df,
                                    data, hist, device, n_sample_users=200)

        # Cold-start sub-evaluations
        cu_df = test_df[test_df["user_idx"].isin(cold_u)]
        ci_df = test_df[test_df["item_idx"].isin(cold_i)]

        pu, ru, nu, hu = (
            compute_ndcg(model, cu_df, train_df, data, hist, device, n_sample_users=100)
            if len(cu_df) else (0, 0, 0, 0)
        )
        pi, ri, ni, hi = (
            compute_ndcg(model, ci_df, train_df, data, hist, device, n_sample_users=100)
            if len(ci_df) else (0, 0, 0, 0)
        )

        results = dict(
            mae=round(mae, 4), rmse=round(rmse, 4),
            precision=round(p, 4), recall=round(r, 4),
            ndcg=round(nd, 4), hr=round(h, 4),
            cold_user_ndcg=round(nu, 4), cold_user_hr=round(hu, 4),
            cold_item_ndcg=round(ni, 4), cold_item_hr=round(hi, 4),
        )
        app_state["results"]  = results
        app_state["progress"] = 100
        app_state["status"]   = "ready"

        log(f"\n  All-data  → MAE:{mae:.4f} | RMSE:{rmse:.4f} | NDCG@10:{nd:.4f}")
        log(f"  Cold User → NDCG@10:{nu:.4f} | HR@10:{hu:.4f}")
        log(f"  Cold Item → NDCG@10:{ni:.4f} | HR@10:{hi:.4f}")
        log("\n  ✅ Training complete — model ready for recommendations")

    except Exception as e:
        import traceback
        app_state["status"] = "error"
        app_state["error"]  = str(e)
        _log(app_state, f"  ❌ Error: {e}")
        traceback.print_exc()
