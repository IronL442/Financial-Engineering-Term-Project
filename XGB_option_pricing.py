"""
Option Pricing: B&S vs ANN1/2/3 vs XGB1/2/3
Based on: D'Uggento et al. (2025), Big Data Research
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# ── 0. Config ──────────────────────────────────────────────────────────────────
RISK_FREE_RATE = 0.0425
SEED           = 42
BATCH_SIZE     = 32
EPOCHS         = 100
LR             = 1e-3
PATIENCE       = 200        # Early stopping patience
WEIGHT_DECAY   = 1e-4       # L2 regularization

# Dropout per model — ANN3가 변수 많아서 dropout 더 강하게
DROPOUT = {
    "ANN1": 0.1,
    "ANN2": 0.2,
    "ANN3": 0.3,
}

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── 1. Load Data ───────────────────────────────────────────────────────────────
df = pd.read_csv(Path(__file__).parent / "data" / "nasdaq100_options_preprocessed.csv")
print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} cols")

# ── 2. Feature Sets ────────────────────────────────────────────────────────────
BS_FEATURES  = ["S", "Strike", "Tau", "Sigma", "Call"]

DIV_FEATURES = [
    "dividendRate", "dividendYield", "payoutRatio",
    "lastDividendValue", "fiveYearAvgDividendYield",
    "trailingAnnualDividendRate", "trailingAnnualDividendYield",
]

ALL_FEATURES = [c for c in df.columns if c not in ["ticker", "Price"]]

ANN1_FEATURES = BS_FEATURES
ANN2_FEATURES = BS_FEATURES + DIV_FEATURES
ANN3_FEATURES = ALL_FEATURES

TARGET = "Price"

# ── 3. Train / Val / Test Split (Random 8:1:1) ────────────────────────────────
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=SEED)
val_df, test_df   = train_test_split(temp_df, test_size=0.5, random_state=SEED)

train_df = train_df.reset_index(drop=True)
val_df   = val_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)

print(f"Train: {len(train_df)} rows")
print(f"Val:   {len(val_df)} rows")
print(f"Test:  {len(test_df)} rows")

call_mask = test_df["Call"].values == 1
put_mask  = test_df["Call"].values == 0

# ── 4. Metrics ─────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, name=""):
    mae   = mean_absolute_error(y_true, y_pred)
    mse   = mean_squared_error(y_true, y_pred)
    rmse  = np.sqrt(mse)
    mape  = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-6))) * 100
    smape = np.mean(
        2 * np.abs(y_true - y_pred) /
        (np.abs(y_true) + np.abs(y_pred) + 1e-6)
    ) * 100
    r2    = r2_score(y_true, y_pred)
    print(f"  {name:8s} | MAE={mae:.4f}  MSE={mse:.4f}  RMSE={rmse:.4f}  MAPE={mape:.2f}%  sMAPE={smape:.2f}%  R²={r2:.4f}")
    return {"Model": name, "MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape, "sMAPE": smape, "R2": r2}

# ── 5. Black-Scholes ───────────────────────────────────────────────────────────
def bs_price(S, K, tau, sigma, r, is_call):
    tau   = np.maximum(tau, 1e-6)
    sigma = np.maximum(sigma, 1e-6)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    call = S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
    put  = K * np.exp(-r * tau) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return np.where(is_call, call, put)

# ── 6. ANN ─────────────────────────────────────────────────────────────────────
class FeedforwardNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 64, 32), dropout=0.0):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.Sigmoid()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def run_ann(name, features):
    dropout = DROPOUT.get(name, 0.0)
    print(f"\n  Training {name} (input_dim={len(features)}, dropout={dropout})...")

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(train_df[features].values.astype(np.float32))
    X_val   = scaler.transform(val_df[features].values.astype(np.float32))
    X_test  = scaler.transform(test_df[features].values.astype(np.float32))

    y_train = train_df[TARGET].values.astype(np.float32)
    y_val   = val_df[TARGET].values.astype(np.float32)
    y_test  = test_df[TARGET].values.astype(np.float32)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=BATCH_SIZE
    )

    model     = FeedforwardNet(len(features), dropout=dropout)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,   # L2 regularization
    )
    criterion = nn.MSELoss()

    best_val     = float("inf")
    best_state   = None
    patience_cnt = 0

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            criterion(model(X_b), y_b).backward()
            optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_loss = np.mean([
                criterion(model(X_b), y_b).item()
                for X_b, y_b in val_loader
            ])

        # Early stopping
        if val_loss < best_val:
            best_val     = val_loss
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1

        if epoch % 100 == 0:
            print(f"    Epoch {epoch:4d}/{EPOCHS} | val_loss={val_loss:.4f}  best={best_val:.4f}  patience={patience_cnt}/{PATIENCE}")

        if patience_cnt >= PATIENCE:
            print(f"    Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(X_test)).numpy()

    return y_pred, y_test


# ── 7. XGBoost ─────────────────────────────────────────────────────────────────
def run_xgb(name, features):
    print(f"\n  Training {name} (input_dim={len(features)})...")

    X_train = train_df[features].values
    y_train = train_df[TARGET].values
    X_val   = val_df[features].values
    y_val   = val_df[TARGET].values
    X_test  = test_df[features].values
    y_test  = test_df[TARGET].values

    model = xgb.XGBRegressor(
        n_estimators          = 500,
        max_depth             = 6,
        learning_rate         = 0.05,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        min_child_weight      = 5,
        reg_alpha             = 0.1,
        reg_lambda            = 1.0,
        random_state          = SEED,
        early_stopping_rounds = 20,
        eval_metric           = "rmse",
        verbosity             = 0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_pred = model.predict(X_test)

    importance = pd.Series(
        model.feature_importances_, index=features
    ).sort_values(ascending=False)
    print(f"    Top 10 features:\n{importance.head(10).to_string()}")

    return y_pred, y_test, model


# ── 8. Run All Models ──────────────────────────────────────────────────────────
results = []

print("\n" + "="*70)
print("RESULTS ON TEST SET")
print("="*70)

# --- B&S ---
print("\n[B&S]")
bs_pred   = bs_price(
    S       = test_df["S"].values,
    K       = test_df["Strike"].values,
    tau     = test_df["Tau"].values,
    sigma   = test_df["Sigma"].values,
    r       = RISK_FREE_RATE,
    is_call = test_df["Call"].values.astype(bool),
)
y_true_bs = test_df[TARGET].values
results.append(compute_metrics(y_true_bs[call_mask], bs_pred[call_mask], "B&S-C"))
results.append(compute_metrics(y_true_bs[put_mask],  bs_pred[put_mask],  "B&S-P"))

# --- ANN1 ---
print("\n[ANN1]")
ann1_pred, ann1_true = run_ann("ANN1", ANN1_FEATURES)
results.append(compute_metrics(ann1_true[call_mask], ann1_pred[call_mask], "ANN1-C"))
results.append(compute_metrics(ann1_true[put_mask],  ann1_pred[put_mask],  "ANN1-P"))

# --- ANN2 ---
print("\n[ANN2]")
ann2_pred, ann2_true = run_ann("ANN2", ANN2_FEATURES)
results.append(compute_metrics(ann2_true[call_mask], ann2_pred[call_mask], "ANN2-C"))
results.append(compute_metrics(ann2_true[put_mask],  ann2_pred[put_mask],  "ANN2-P"))

# --- ANN3 ---
print("\n[ANN3]")
ann3_pred, ann3_true = run_ann("ANN3", ANN3_FEATURES)
results.append(compute_metrics(ann3_true[call_mask], ann3_pred[call_mask], "ANN3-C"))
results.append(compute_metrics(ann3_true[put_mask],  ann3_pred[put_mask],  "ANN3-P"))

# --- XGB1 ---
print("\n[XGB1]")
xgb1_pred, xgb1_true, xgb1_model = run_xgb("XGB1", ANN1_FEATURES)
results.append(compute_metrics(xgb1_true[call_mask], xgb1_pred[call_mask], "XGB1-C"))
results.append(compute_metrics(xgb1_true[put_mask],  xgb1_pred[put_mask],  "XGB1-P"))

# --- XGB2 ---
print("\n[XGB2]")
xgb2_pred, xgb2_true, xgb2_model = run_xgb("XGB2", ANN2_FEATURES)
results.append(compute_metrics(xgb2_true[call_mask], xgb2_pred[call_mask], "XGB2-C"))
results.append(compute_metrics(xgb2_true[put_mask],  xgb2_pred[put_mask],  "XGB2-P"))

# --- XGB3 ---
print("\n[XGB3]")
xgb3_pred, xgb3_true, xgb3_model = run_xgb("XGB3", ANN3_FEATURES)
results.append(compute_metrics(xgb3_true[call_mask], xgb3_pred[call_mask], "XGB3-C"))
results.append(compute_metrics(xgb3_true[put_mask],  xgb3_pred[put_mask],  "XGB3-P"))

# ── 9. Summary Table ───────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SUMMARY TABLE")
print("="*70)
results_df = pd.DataFrame(results).round(4)
print(results_df.to_string(index=False))
results_df.to_csv("results.csv", index=False)
print("\nSaved → results.csv")

# ── 10. Feature Importance (XGB1, XGB2, XGB3) ────────────────────────────────
print("\n[Feature Importance]")

for name, model, features in [
    ("XGB1", xgb1_model, ANN1_FEATURES),
    ("XGB2", xgb2_model, ANN2_FEATURES),
    ("XGB3", xgb3_model, ANN3_FEATURES),
]:
    imp_df = pd.Series(
        model.feature_importances_, index=features
    ).sort_values(ascending=False).reset_index()
    imp_df.columns = ["feature", "importance"]
    imp_df["model"] = name
    imp_df.to_csv(f"{name.lower()}_feature_importance.csv", index=False)
    print(f"  Saved → {name.lower()}_feature_importance.csv  ({len(features)} features)")

# ── 11. SHAP Values (XGB3) ────────────────────────────────────────────────────
print("\n[SHAP Values]")
try:
    import shap

    X_test_arr = test_df[ANN3_FEATURES].values

    explainer   = shap.TreeExplainer(xgb3_model)
    shap_values = explainer.shap_values(X_test_arr)   # (n_test, n_features)

    # Save raw SHAP values
    shap_df = pd.DataFrame(shap_values, columns=ANN3_FEATURES)
    shap_df.to_csv("xgb3_shap_values.csv", index=False)
    print("  Saved → xgb3_shap_values.csv")

    # Mean absolute SHAP per feature (global importance)
    mean_shap = pd.DataFrame({
        "feature":    ANN3_FEATURES,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    mean_shap.to_csv("xgb3_shap_importance.csv", index=False)
    print("  Saved → xgb3_shap_importance.csv")
    print(f"  Top 10 SHAP features:\n{mean_shap.head(10).to_string(index=False)}")

    # SHAP for Call / Put separately
    for label, mask, fname in [
        ("Call", call_mask, "xgb3_shap_call.csv"),
        ("Put",  put_mask,  "xgb3_shap_put.csv"),
    ]:
        shap_sub = shap_values[mask]
        mean_sub = pd.DataFrame({
            "feature":       ANN3_FEATURES,
            "mean_abs_shap": np.abs(shap_sub).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)
        mean_sub.to_csv(fname, index=False)
        print(f"  Saved → {fname}  ({label}: {mask.sum()} samples)")

except ImportError:
    print("  ⚠️  shap not installed. Run: pip install shap")

# ── 12. Predictions + Residuals ───────────────────────────────────────────────
print("\n[Predictions & Residuals]")

# Moneyness = S / Strike  (>1: ITM call / OTM put, <1: OTM call / ITM put)
result_test = test_df.copy()
result_test["moneyness"] = result_test["S"] / result_test["Strike"]
result_test["moneyness_cat"] = pd.cut(
    result_test["moneyness"],
    bins=[0, 0.95, 1.05, 99],
    labels=["OTM", "ATM", "ITM"]
)

# Add predictions from every model
result_test["pred_BS"]   = bs_pred
result_test["pred_ANN1"] = ann1_pred
result_test["pred_ANN2"] = ann2_pred
result_test["pred_ANN3"] = ann3_pred
result_test["pred_XGB1"] = xgb1_pred
result_test["pred_XGB2"] = xgb2_pred
result_test["pred_XGB3"] = xgb3_pred

# Residuals (actual - predicted)
for model_name in ["BS", "ANN1", "ANN2", "ANN3", "XGB1", "XGB2", "XGB3"]:
    result_test[f"resid_{model_name}"] = (
        result_test["Price"] - result_test[f"pred_{model_name}"]
    )

result_test.to_csv("test_predictions_residuals.csv", index=False)
print("  Saved → test_predictions_residuals.csv")
print(f"  Columns: {result_test.columns.tolist()}")

# ── 13. Residual Summary by Moneyness / Tau / Sector ─────────────────────────
print("\n[Residual Analysis]")

# By Moneyness
print("\n  MAE by Moneyness (XGB3):")
mono_analysis = result_test.groupby("moneyness_cat").apply(
    lambda g: pd.Series({
        "n":            len(g),
        "MAE_BS":       np.abs(g["resid_BS"]).mean(),
        "MAE_ANN3":     np.abs(g["resid_ANN3"]).mean(),
        "MAE_XGB3":     np.abs(g["resid_XGB3"]).mean(),
    })
).reset_index()
print(mono_analysis.to_string(index=False))
mono_analysis.to_csv("residual_by_moneyness.csv", index=False)

# By Tau bucket
result_test["tau_cat"] = pd.cut(
    result_test["Tau"],
    bins=[0, 60/365, 1, 9999],
    labels=["Short(<2mo)", "Mid(2mo-1yr)", "Long(>1yr)"]
)
print("\n  MAE by Tau (XGB3):")
tau_analysis = result_test.groupby("tau_cat").apply(
    lambda g: pd.Series({
        "n":        len(g),
        "MAE_BS":   np.abs(g["resid_BS"]).mean(),
        "MAE_ANN3": np.abs(g["resid_ANN3"]).mean(),
        "MAE_XGB3": np.abs(g["resid_XGB3"]).mean(),
    })
).reset_index()
print(tau_analysis.to_string(index=False))
tau_analysis.to_csv("residual_by_tau.csv", index=False)

# By Sector
if "sector" in result_test.columns:
    print("\n  MAE by Sector (XGB3):")
    sector_analysis = result_test.groupby("sector").apply(
        lambda g: pd.Series({
            "n":        len(g),
            "MAE_BS":   np.abs(g["resid_BS"]).mean(),
            "MAE_ANN3": np.abs(g["resid_ANN3"]).mean(),
            "MAE_XGB3": np.abs(g["resid_XGB3"]).mean(),
        })
    ).reset_index().sort_values("MAE_XGB3", ascending=False)
    print(sector_analysis.to_string(index=False))
    sector_analysis.to_csv("residual_by_sector.csv", index=False)

print("\n✅ All interpretability files saved!")
print("Files for Joy:")
print("  - xgb1/2/3_feature_importance.csv")
print("  - xgb3_shap_values.csv")
print("  - xgb3_shap_importance.csv")
print("  - xgb3_shap_call.csv / xgb3_shap_put.csv")
print("  - test_predictions_residuals.csv")
print("  - residual_by_moneyness.csv")
print("  - residual_by_tau.csv")
print("  - residual_by_sector.csv")