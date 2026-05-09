# Financial-Engineering-Term-Project

# Project Overview: Extending D'Uggento et al. (2025)

---

## What the Original Paper Did

> *"ML outperforms B&S, more variables lead to better performance, and RNN achieves the best results."*

**Model progression:**
B&S → ANN1 → ANN2 → ANN3 → RNN

**Limitations of the original paper:**
- Large, heterogeneous dataset (73,154 options across 4,004 companies)
- RNN architecture details not disclosed
- Hyperparameter tuning methodology not reported
- Constant volatility (σ) assumption in B&S left unchallenged
- No interpretability analysis

---

## Our Research Questions

---

### RQ0. Replication — Shared Task
> *"Does the same pattern hold for large-cap stocks in the Nasdaq-100?"*

**Models:** B&S → ANN1 → ANN2 → ANN3

**Key differences from the original paper:**
- Restricted to Nasdaq-100 large-cap stocks
- Smaller dataset (1,780 options)
- Fair comparison via Optuna hyperparameter tuning for all models

**Expected findings:**
- Higher B&S R² than the original paper, as large-cap stocks better satisfy B&S assumptions (efficient markets, lognormal returns)
- Similar pattern of improvement as variables increase

---

### RQ1. Contribution 1 — XGBoost (Woohyuk)
> *"Does a tree-based model (XGBoost) outperform neural networks (ANN) on the same dataset?"*

**Model comparison:**
- ANN1 vs XGB1 (5 features: B&S variables)
- ANN2 vs XGB2 (12 features: B&S + dividend variables)
- ANN3 vs XGB3 (124 features: all variables)

**Advancement over the original paper:**
- The original paper only compared ANN and RNN
- We introduce a tree-based ensemble model as an additional benchmark
- We investigate which model architecture is more robust on small datasets (1,780 rows)
- Optuna tuning applied equally to both ANN and XGBoost for fair comparison

**Expected findings:**
- XGBoost outperforms ANN, particularly for put options
- Tree-based models are more robust than neural networks on small datasets
- Adding more variables yields clearer performance gains in XGBoost than in ANN

---

### RQ2. Contribution 2 — Neural SDE (Inha)
> *"Does relaxing the constant volatility assumption of B&S improve pricing accuracy?"*

**Model comparison:**
B&S (σ fixed) vs Neural SDE (σ learned by neural network)

**Model structure:**
```
dS = r · S dt + σ_net(S, t; θ) · S dW
```
- Drift term r is fixed (same as B&S)
- Only the diffusion term σ is replaced by a neural network
- Option price = E[payoff] estimated via Monte Carlo simulation

**Advancement over the original paper:**
- The original paper treats B&S as a black-box baseline
- We directly challenge the constant σ assumption by learning it from data
- We answer: *"Which specific assumption of B&S is most violated in practice?"*

**Additional analysis:**
- How does learned σ_net vary with moneyness (S/K)? → Volatility smile
- How does learned σ_net vary with time to expiry (Tau)? → Term structure of volatility
- Comparison of learned σ vs original implied Sigma

---

### RQ3. Contribution 3 — Interpretability (Joy)
> *"How does XGBoost make its predictions, and what drives option pricing in practice?"*

**Analysis using SHAP:**
- Which variables matter most overall? → SHAP summary plot
- Do important variables differ between call and put options?
- How does each variable individually affect predicted price? → SHAP dependence plot
- Can we explain a single option's predicted price? → SHAP waterfall plot

**Error pattern analysis:**
- Does model accuracy differ by moneyness (OTM / ATM / ITM)?
- Does model accuracy differ by time to expiry (short / mid / long)?
- Does model accuracy differ by sector (Technology / Healthcare / etc.)?

**Advancement over the original paper:**
- The original paper provides no interpretability analysis
- We use SHAP to open the black box and explain model decisions
- We identify which market conditions are hardest to price accurately

---

## Overall Research Narrative

```
[Replication]
B&S < ANN1 < ANN2 < ANN3
→ Confirms: ML > B&S, more variables = better performance

              ↓ Three directions of advancement

[Contribution 1]          [Contribution 2]            [Contribution 3]
XGBoost vs ANN            B&S vs Neural SDE            Interpretability
"Are tree-based models    "Does relaxing the           "Why does XGBoost
 better than neural nets   constant σ assumption        work well, and
 on small datasets?"       improve pricing?"            where does it fail?"
```

---

## One-Sentence Summary

> *"We replicate the finding that ML outperforms B&S for option pricing, and extend it in three directions: (1) demonstrating that XGBoost is more robust than ANN on small datasets, particularly for put options; (2) showing that relaxing the constant volatility assumption of B&S via a Neural SDE captures volatility smile and term structure effects; and (3) providing SHAP-based interpretability to explain which variables and market conditions drive model performance."*