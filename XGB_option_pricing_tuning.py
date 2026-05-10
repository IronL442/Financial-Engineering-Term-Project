"""
Option Pricing: B&S vs ANN1/2/3 vs XGB1/2/3
- Optuna hyperparameter tuning for both ANN and XGBoost
- SHAP interpretability analysis
- Residual analysis by moneyness / tau / sector
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
import optuna
import warnings
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# ── 0. Config ──────────────────────────────────────────────────────────────────
RISK_FREE_RATE  = 0.0425
SEED            = 42
TUNING_EPOCHS   = 500       # epoch limit during Optuna trials (speed)
FINAL_EPOCHS    = 3000      # epoch limit for final training
PATIENCE        = 200       # early stopping patience
BATCH_SIZE_DEF  = 32
N_TRIALS        = 30        # Optuna trials per model

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
val_df,   test_df = train_test_split(temp_df, test_size=0.5, random_state=SEED)

train_df = train_df.reset_index(drop=True)
val_df   = val_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

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
    print(f"  {name:8s} | MAE={mae:.4f}  MSE={mse:.4f}  RMSE={rmse:.4f}  "
          f"MAPE={mape:.2f}%  sMAPE={smape:.2f}%  R²={r2:.4f}")
    return {"Model": name, "MAE": mae, "MSE": mse, "RMSE": rmse,
            "MAPE": mape, "sMAPE": smape, "R2": r2}

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
    def __init__(self, input_dim, hidden_dims, dropout):
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


def prepare_data(features, scaler=None, fit=False):
    """Scale features and return tensors."""
    if scaler is None:
        scaler = StandardScaler()
    X_train = train_df[features].values.astype(np.float32)
    X_val   = val_df[features].values.astype(np.float32)
    X_test  = test_df[features].values.astype(np.float32)
    if fit:
        scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)
    y_train = train_df[TARGET].values.astype(np.float32)
    y_val   = val_df[TARGET].values.astype(np.float32)
    y_test  = test_df[TARGET].values.astype(np.float32)
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def train_ann(model, X_train, y_train, X_val, y_val,
              lr, weight_decay, batch_size, max_epochs, patience):
    """Train ANN with early stopping. Returns best val_loss."""
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=batch_size
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    best_val, best_state, patience_cnt = float("inf"), None, 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            criterion(model(X_b), y_b).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = np.mean([criterion(model(X_b), y_b).item()
                                for X_b, y_b in val_loader])
        if val_loss < best_val:
            best_val     = val_loss
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
        if patience_cnt >= patience:
            break

    model.load_state_dict(best_state)
    return best_val


def optuna_ann(name, features):
    """Optuna tuning for ANN."""
    print(f"\n  [Optuna] Tuning {name} ({N_TRIALS} trials)...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = \
        prepare_data(features, fit=True)

    def objective(trial):
        # Hyperparameter search space
        n_layers    = trial.suggest_int("n_layers", 1, 4)
        hidden_dims = [
            trial.suggest_categorical(f"units_{i}", [32, 64, 128, 256])
            for i in range(n_layers)
        ]
        dropout      = trial.suggest_float("dropout", 0.0, 0.4, step=0.05)
        lr           = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        batch_size   = trial.suggest_categorical("batch_size", [16, 32, 64])

        model = FeedforwardNet(len(features), hidden_dims, dropout)
        val_loss = train_ann(
            model, X_train, y_train, X_val, y_val,
            lr, weight_decay, batch_size,
            max_epochs=TUNING_EPOCHS, patience=50
        )
        return val_loss

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best = study.best_params
    print(f"    Best params: {best}")
    print(f"    Best val_loss: {study.best_value:.4f}")

    # Final training with best params
    n_layers    = best["n_layers"]
    hidden_dims = [best[f"units_{i}"] for i in range(n_layers)]
    dropout      = best["dropout"]
    lr           = best["lr"]
    weight_decay = best["weight_decay"]
    batch_size   = best["batch_size"]

    print(f"    Final training (up to {FINAL_EPOCHS} epochs)...")
    model = FeedforwardNet(len(features), hidden_dims, dropout)
    train_ann(model, X_train, y_train, X_val, y_val,
              lr, weight_decay, batch_size,
              max_epochs=FINAL_EPOCHS, patience=PATIENCE)

    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(X_test)).numpy()

    return y_pred, y_test, best, model, scaler


# ── 7. XGBoost ─────────────────────────────────────────────────────────────────
def optuna_xgb(name, features):
    """Optuna tuning for XGBoost."""
    print(f"\n  [Optuna] Tuning {name} ({N_TRIALS} trials)...")
    X_train = train_df[features].values
    y_train = train_df[TARGET].values
    X_val   = val_df[features].values
    y_val   = val_df[TARGET].values
    X_test  = test_df[features].values
    y_test  = test_df[TARGET].values

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 1000, step=100),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0, step=0.1),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0, step=0.1),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":         trial.suggest_float("reg_alpha", 0.0, 1.0, step=0.1),
            "reg_lambda":        trial.suggest_float("reg_lambda", 0.0, 2.0, step=0.1),
            "random_state":      SEED,
            "early_stopping_rounds": 20,
            "eval_metric":       "rmse",
            "verbosity":         0,
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  verbose=False)
        y_pred = model.predict(X_val)
        return mean_squared_error(y_val, y_pred)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best = study.best_params
    print(f"    Best params: {best}")
    print(f"    Best val_MSE: {study.best_value:.4f}")

    # Final model with best params
    final_model = xgb.XGBRegressor(
        **{k: v for k, v in best.items()},
        random_state          = SEED,
        early_stopping_rounds = 20,
        eval_metric           = "rmse",
        verbosity             = 0,
    )
    final_model.fit(X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False)

    y_pred = final_model.predict(X_test)

    importance = pd.Series(
        final_model.feature_importances_, index=features
    ).sort_values(ascending=False)
    print(f"    Top 10 features:\n{importance.head(10).to_string()}")

    return y_pred, y_test, best, final_model

# ── 8. Run All Models ──────────────────────────────────────────────────────────
results  = []
models   = {}   # store trained models for interpretability
scalers  = {}   # store scalers
best_params = {}

print("\n" + "="*70)
print("RESULTS ON TEST SET")
print("="*70)

# --- B&S ---
print("\n[B&S]")
bs_pred = bs_price(
    S       = test_df["S"].values,
    K       = test_df["Strike"].values,
    tau     = test_df["Tau"].values,
    sigma   = test_df["Sigma"].values,
    r       = RISK_FREE_RATE,
    is_call = test_df["Call"].values.astype(bool),
)
y_true_test = test_df[TARGET].values
results.append(compute_metrics(y_true_test[call_mask], bs_pred[call_mask], "B&S-C"))
results.append(compute_metrics(y_true_test[put_mask],  bs_pred[put_mask],  "B&S-P"))

# --- ANN1 ---
print("\n[ANN1]")
ann1_pred, ann1_true, ann1_params, ann1_model, ann1_scaler = \
    optuna_ann("ANN1", ANN1_FEATURES)
results.append(compute_metrics(ann1_true[call_mask], ann1_pred[call_mask], "ANN1-C"))
results.append(compute_metrics(ann1_true[put_mask],  ann1_pred[put_mask],  "ANN1-P"))
models["ANN1"] = ann1_model
scalers["ANN1"] = ann1_scaler
best_params["ANN1"] = ann1_params

# --- ANN2 ---
print("\n[ANN2]")
ann2_pred, ann2_true, ann2_params, ann2_model, ann2_scaler = \
    optuna_ann("ANN2", ANN2_FEATURES)
results.append(compute_metrics(ann2_true[call_mask], ann2_pred[call_mask], "ANN2-C"))
results.append(compute_metrics(ann2_true[put_mask],  ann2_pred[put_mask],  "ANN2-P"))
models["ANN2"] = ann2_model
scalers["ANN2"] = ann2_scaler
best_params["ANN2"] = ann2_params

# --- ANN3 ---
print("\n[ANN3]")
ann3_pred, ann3_true, ann3_params, ann3_model, ann3_scaler = \
    optuna_ann("ANN3", ANN3_FEATURES)
results.append(compute_metrics(ann3_true[call_mask], ann3_pred[call_mask], "ANN3-C"))
results.append(compute_metrics(ann3_true[put_mask],  ann3_pred[put_mask],  "ANN3-P"))
models["ANN3"] = ann3_model
scalers["ANN3"] = ann3_scaler
best_params["ANN3"] = ann3_params

# --- XGB1 ---
print("\n[XGB1]")
xgb1_pred, xgb1_true, xgb1_params, xgb1_model = \
    optuna_xgb("XGB1", ANN1_FEATURES)
results.append(compute_metrics(xgb1_true[call_mask], xgb1_pred[call_mask], "XGB1-C"))
results.append(compute_metrics(xgb1_true[put_mask],  xgb1_pred[put_mask],  "XGB1-P"))
models["XGB1"] = xgb1_model
best_params["XGB1"] = xgb1_params

# --- XGB2 ---
print("\n[XGB2]")
xgb2_pred, xgb2_true, xgb2_params, xgb2_model = \
    optuna_xgb("XGB2", ANN2_FEATURES)
results.append(compute_metrics(xgb2_true[call_mask], xgb2_pred[call_mask], "XGB2-C"))
results.append(compute_metrics(xgb2_true[put_mask],  xgb2_pred[put_mask],  "XGB2-P"))
models["XGB2"] = xgb2_model
best_params["XGB2"] = xgb2_params

# --- XGB3 ---
print("\n[XGB3]")
xgb3_pred, xgb3_true, xgb3_params, xgb3_model = \
    optuna_xgb("XGB3", ANN3_FEATURES)
results.append(compute_metrics(xgb3_true[call_mask], xgb3_pred[call_mask], "XGB3-C"))
results.append(compute_metrics(xgb3_true[put_mask],  xgb3_pred[put_mask],  "XGB3-P"))
models["XGB3"] = xgb3_model
best_params["XGB3"] = xgb3_params

# ── 9. Summary Table ───────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SUMMARY TABLE")
print("="*70)
results_df = pd.DataFrame(results).round(4)
print(results_df.to_string(index=False))
results_df.to_csv("results.csv", index=False)
print("Saved → results.csv")

# Best params summary
params_df = pd.DataFrame(best_params).T
params_df.to_csv("best_params.csv")
print("Saved → best_params.csv")

# ── 10. Feature Importance (XGB1/2/3) ─────────────────────────────────────────
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
    print(f"  Saved → {name.lower()}_feature_importance.csv")

# ── 11. SHAP Values (XGB3) ────────────────────────────────────────────────────
print("\n[SHAP Values]")
try:
    import shap

    X_test_arr = test_df[ANN3_FEATURES].values
    explainer   = shap.TreeExplainer(xgb3_model)
    shap_values = explainer.shap_values(X_test_arr)

    # Raw SHAP values
    shap_df = pd.DataFrame(shap_values, columns=ANN3_FEATURES)
    shap_df.to_csv("xgb3_shap_values.csv", index=False)
    print("  Saved → xgb3_shap_values.csv")

    # Global mean absolute SHAP
    mean_shap = pd.DataFrame({
        "feature":       ANN3_FEATURES,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    mean_shap.to_csv("xgb3_shap_importance.csv", index=False)
    print("  Saved → xgb3_shap_importance.csv")
    print(f"  Top 10 SHAP:\n{mean_shap.head(10).to_string(index=False)}")

    # Call / Put separate SHAP
    for label, mask, fname in [
        ("Call", call_mask, "xgb3_shap_call.csv"),
        ("Put",  put_mask,  "xgb3_shap_put.csv"),
    ]:
        sub = shap_values[mask]
        pd.DataFrame({
            "feature":       ANN3_FEATURES,
            "mean_abs_shap": np.abs(sub).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False).to_csv(fname, index=False)
        print(f"  Saved → {fname}  ({label}: {mask.sum()} samples)")

except ImportError:
    print("  ⚠️  shap not installed: pip install shap")

# ── 12. Predictions + Residuals ───────────────────────────────────────────────
print("\n[Predictions & Residuals]")

result_test = test_df.copy()

# Moneyness
result_test["moneyness"]     = result_test["S"] / result_test["Strike"]
result_test["moneyness_cat"] = pd.cut(
    result_test["moneyness"],
    bins=[0, 0.95, 1.05, 99],
    labels=["OTM", "ATM", "ITM"]
)

# Tau category
result_test["tau_cat"] = pd.cut(
    result_test["Tau"],
    bins=[0, 60/365, 1, 9999],
    labels=["Short(<60d)", "Mid(60-365d)", "Long(>365d)"]
)

# All predictions
result_test["pred_BS"]   = bs_pred
result_test["pred_ANN1"] = ann1_pred
result_test["pred_ANN2"] = ann2_pred
result_test["pred_ANN3"] = ann3_pred
result_test["pred_XGB1"] = xgb1_pred
result_test["pred_XGB2"] = xgb2_pred
result_test["pred_XGB3"] = xgb3_pred

# Residuals
for m in ["BS", "ANN1", "ANN2", "ANN3", "XGB1", "XGB2", "XGB3"]:
    result_test[f"resid_{m}"] = result_test["Price"] - result_test[f"pred_{m}"]

result_test.to_csv("test_predictions_residuals.csv", index=False)
print("  Saved → test_predictions_residuals.csv")

# ── 13. Residual Analysis ──────────────────────────────────────────────────────
print("\n[Residual Analysis]")

def residual_summary(group_col):
    return result_test.groupby(group_col, observed=True).apply(
        lambda g: pd.Series({
            "n":         len(g),
            "MAE_BS":    np.abs(g["resid_BS"]).mean(),
            "MAE_ANN3":  np.abs(g["resid_ANN3"]).mean(),
            "MAE_XGB3":  np.abs(g["resid_XGB3"]).mean(),
        })
    ).reset_index()

# By Moneyness
mono = residual_summary("moneyness_cat")
mono.to_csv("residual_by_moneyness.csv", index=False)
print(f"\n  By Moneyness:\n{mono.to_string(index=False)}")

# By Tau
tau = residual_summary("tau_cat")
tau.to_csv("residual_by_tau.csv", index=False)
print(f"\n  By Tau:\n{tau.to_string(index=False)}")

# By Sector
if "sector" in result_test.columns:
    # decode one-hot back to sector name
    sector_cols = [c for c in result_test.columns if c.startswith("sector_")]
    if sector_cols:
        result_test["sector_name"] = result_test[sector_cols].idxmax(axis=1).str.replace("sector_", "")
        sec = residual_summary("sector_name").sort_values("MAE_XGB3", ascending=False)
        sec.to_csv("residual_by_sector.csv", index=False)
        print(f"\n  By Sector:\n{sec.to_string(index=False)}")

print("\n" + "="*70)
print("✅ ALL DONE — Files for Joy:")
print("  results.csv")
print("  best_params.csv")
print("  xgb1/2/3_feature_importance.csv")
print("  xgb3_shap_values.csv")
print("  xgb3_shap_importance.csv")
print("  xgb3_shap_call.csv / xgb3_shap_put.csv")
print("  test_predictions_residuals.csv")
print("  residual_by_moneyness.csv")
print("  residual_by_tau.csv")
print("  residual_by_sector.csv")
print("="*70)