# DACSF — Deep Attentional Cold-Start Framework

A full implementation of the DACSF recommendation model with a live
training dashboard served by Flask.

---

## Project Structure

```
dacsf/
├── app.py            ← Flask server + API routes          (entry point)
├── config.py         ← All hyperparameters (edit here)
├── model.py          ← DACSF architecture (encoders, AFFN, scoring)
├── trainer.py        ← Training loop (Phase 1 BPR + Phase 2 fine-tune)
├── data_loader.py    ← MovieLens 100K download + preprocessing
├── datasets.py       ← PyTorch Dataset classes (BPR + Rating)
├── features.py       ← Batch tensor builders for items and users
├── metrics.py        ← MAE, RMSE, NDCG@K, HR@K evaluation
├── recommender.py    ← Inference: Top-K recommendation generation
├── requirements.txt
├── templates/
│   └── index.html    ← Dashboard HTML
└── static/
    ├── css/style.css ← Dashboard styles
    └── js/dashboard.js ← Frontend JS (chart, polling, recos)
```

---

## Setup

### 1. Install PyTorch for your hardware

**CPU only (simplest):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**NVIDIA GPU (CUDA 12.1):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**AMD GPU on Windows (DirectML):**
```bash
pip install torch-directml
```

### 2. Install remaining dependencies
```bash
pip install -r requirements.txt
```

### 3. Run
```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

The MovieLens 100K dataset downloads automatically on first run (~5 MB).

---

## Usage

1. **Adjust hyperparameters** in the left panel (or edit `config.py` for defaults).
2. Click **▶ Train DACSF** — the training log, live metrics, and charts
   update in real time.
3. Once training completes, the **Evaluation Results** card shows
   overall and cold-start NDCG / HR scores.
4. Enter a **User ID** and click **✦ Recommend** to generate Top-K
   personalized recommendations with per-modality attention weights.

---

## Customising

| What to change | Where |
|---|---|
| Hyperparameters | `config.py` → `Config` class |
| Model architecture | `model.py` |
| Real BERT / ResNet features | `data_loader.py` → replace `text_mat` / `img_mat` |
| New dataset | `data_loader.py` → replace `load_data()` |
| Evaluation protocol | `metrics.py` |
| Frontend look | `static/css/style.css` |

---

## Notes

- Simulated BERT and ResNet features are used (random unit-norm vectors).
  Swap these for real encoder outputs in `data_loader.py` for full fidelity.
- `torch_directml` (AMD GPU) is auto-detected; falls back to CUDA then CPU.
- The data directory `./data/ml-100k` is created relative to wherever
  you run `python app.py`.
