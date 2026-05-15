import torch
import numpy as np
import shap

from trainer import app_state

model = app_state["model"]
model.eval()

train_df = app_state["data"]["train_df"]

user_ids = train_df["user_idx"].values
item_ids = train_df["item_idx"].values


# 🔹 SHAP wrapper (IMPORTANT: your model takes features, not raw IDs)
from features import user_feats, item_feats, build_hist_dict

data = app_state["data"]
hist = build_hist_dict(train_df)
device = torch.device("cpu")


def model_predict(x):
    x = torch.tensor(x, dtype=torch.long)

    users = x[:, 0].numpy()
    items = x[:, 1].numpy()

    uf = user_feats(users, data, hist, device)
    pf = item_feats(items, data, device)

    with torch.no_grad():
        scores = model(uf, pf)   

    return scores.cpu().numpy()


# 🔹 Prepare SHAP input
X = []
for u, i in zip(user_ids[:300], item_ids[:300]):
    X.append([u, i])

X = np.array(X)


# 🔹 SHAP
background = X[:50]
explainer = shap.KernelExplainer(model_predict, background)

shap_values = explainer.shap_values(X[:100])


# 🔹 Plot
shap.summary_plot(shap_values, X[:100], feature_names=["User", "Item"])