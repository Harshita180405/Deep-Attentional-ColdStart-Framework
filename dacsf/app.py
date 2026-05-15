import threading
from flask import Flask, render_template, jsonify, request

from config import CFG
from trainer import run_training
from recommender import recommend_for_user

app = Flask(__name__)

# ─── Shared state (written by trainer thread, read by Flask routes) ───────────
app_state = {
    "status":       "idle",   # idle | loading | training | ready | error
    "log":          [],
    "progress":     0,
    "epoch":        0,
    "total_epochs": CFG.epochs,
    "metrics":      {},       # latest epoch metrics dict
    "history":      [],       # list of per-epoch metric dicts
    "results":      {},       # final evaluation results
    "model":        None,
    "data":         None,
    "n_users":      0,
    "n_items":      0,
    "error":        "",
}


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/train", methods=["POST"])
def api_train():
    if app_state["status"] in ("loading", "training"):
        return jsonify({"ok": False, "msg": "Training already in progress."})

    # Allow the frontend to override hyperparameters at runtime
    cfg_data = request.json or {}
    CFG.epochs         = int(cfg_data.get("epochs",  CFG.epochs))
    CFG.latent_dim     = int(cfg_data.get("dim",     CFG.latent_dim))
    CFG.lr             = float(cfg_data.get("lr",    CFG.lr))
    CFG.dropout        = float(cfg_data.get("dropout", CFG.dropout))
    CFG.cold_start_tau = int(cfg_data.get("tau",     CFG.cold_start_tau))
    CFG.top_k          = int(cfg_data.get("k",       CFG.top_k))

    thread = threading.Thread(target=run_training, args=(app_state,), daemon=True)
    thread.start()
    return jsonify({"ok": True})


@app.route("/api/state")
def api_state():
    return jsonify({
        "status":       app_state["status"],
        "progress":     app_state["progress"],
        "epoch":        app_state["epoch"],
        "total_epochs": app_state["total_epochs"],
        "metrics":      app_state["metrics"],
        "history":      app_state["history"],
        "results":      app_state["results"],
        "log":          app_state["log"][-80:],
        "n_users":      app_state["n_users"],
        "n_items":      app_state["n_items"],
        "error":        app_state["error"],
    })


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    if app_state["status"] != "ready":
        return jsonify({"error": "Model not ready. Please train first."})

    body    = request.json or {}
    uid     = int(body.get("user_idx", 0))
    top_k   = int(body.get("top_k", CFG.top_k))
    n_users = app_state["n_users"]

    if uid < 0 or uid >= n_users:
        return jsonify({"error": f"User ID must be in range [0, {n_users - 1}]."})

    recos = recommend_for_user(uid, app_state, top_k=top_k)
    return jsonify({"recommendations": recos})


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  DACSF Dashboard")
    print("=" * 60)
    print(f"  Device : {CFG.get_device()}")
    print(f"  Open   : http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    app.run(debug=False, host="127.0.0.1", port=5000, threaded=True)
