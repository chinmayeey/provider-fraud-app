<div align="center">

# 🏥 Healthcare Provider Fraud Detection

**Predicting fraudulent Medicare providers from claims data — end-to-end ML pipeline + live Streamlit app**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-2ea44f?style=for-the-badge)

**[🚀 Try the Live App](#)** &nbsp;|&nbsp; **[📓 View the Notebook](./Healthcare_Fraud_Detection.ipynb)**

</div>

---

## 📌 Problem Statement

Healthcare fraud costs the U.S. insurance system billions every year. Providers commit fraud by:

- Billing for services never rendered
- Submitting duplicate claims for the same service
- Misrepresenting the service actually provided
- Charging for a more complex/expensive procedure than was performed

This project builds a **binary classifier** that predicts whether a **provider** (not a single claim) is likely to be flagged as potentially fraudulent — based purely on patterns in their claims and patient population.

> **Input:** provider claims (inpatient + outpatient) and beneficiary details
> **Output:** `Fraud` / `Not Fraud` + a probability score, per provider

---

## 🧭 Approach

| Stage | What happens |
|---|---|
| **A — Merge** | Combine Inpatient + Outpatient + Beneficiary + labels into one claim-level table |
| **B — Feature Engineering** | Collapse claim-level data into ~21 provider-level "fingerprints" (billing volume, network size, length of stay, diagnosis diversity, etc.) |
| **C — Modelling & Evaluation** | Train + compare models with metrics suited to rare-event detection (Precision, Recall, F1, ROC-AUC — not accuracy) |
| **D — Deployment** | Wrap the trained model in a Streamlit app so anyone can upload data and get instant predictions |

---

## 📂 Dataset

| File | Grain | Contents |
|---|---|---|
| `Train-*.csv` | 1 row = 1 provider | `PotentialFraud` label (Yes/No) |
| `Train_Beneficiarydata-*.csv` | 1 row = 1 patient | Age, chronic conditions, region |
| `Train_Inpatientdata-*.csv` | 1 row = 1 hospital admission | Admit/discharge dates, diagnosis codes, amount billed |
| `Train_Outpatientdata-*.csv` | 1 row = 1 clinic visit | Same idea, no admission |

> ⚠️ **Raw data is not included in this repo** per the case study's privacy guidelines. Only code and the trained model are published here.

**Class balance:** 5,410 providers in training, only **9.35% fraudulent** — a genuinely imbalanced, rare-event problem.

---

## 🔑 Engineered Features (highlights)

| Feature | Signal |
|---|---|
| `total_reimbursed` / `max_reimbursed` | Unusually large billing amounts |
| `total_deductible` | Billing pattern irregularities |
| `inpatient_ratio` | Different fraud "styles" between inpatient/outpatient |
| `avg_length_of_stay` | Padding admission length to bill more |
| `unique_beneficiaries` / `claims_per_bene` | Providers "serving" implausibly many patients |
| `avg_num_diag_codes` | Diversity vs. templated fraudulent billing |

---

## 🤖 Model Comparison

| Model | ROC-AUC | F1 (Fraud) | Precision (Fraud) | Recall (Fraud) |
|---|---|---|---|---|
| Logistic Regression | 0.9616 | 0.5679 | 0.41 | **0.91** |
| **Random Forest** ✅ | 0.9599 | **0.6308** | 0.52 | 0.81 |
| Gradient Boosting | 0.9606 | 0.6102 | **0.71** | 0.53 |
| XGBoost | **0.9642** | 0.5928 | 0.44 | 0.90 |

- All four models separate fraud from non-fraud strongly — ROC-AUC is tightly clustered between **0.960–0.964**, so the real decision is about the **precision/recall trade-off**, not raw ranking power.
- **Logistic Regression & XGBoost** cast the widest net (~90% recall) but with more false positives (precision ~0.41–0.44) — best suited to a "flag everything, let humans triage" strategy.
- **Gradient Boosting** is the most precise (71%) but misses nearly half of actual fraud cases (53% recall) — best if manual investigation capacity is very limited.
- **Random Forest** gives the best overall balance (**highest F1, 0.6308**) and is deployed in the live app.
- In practice, the **probability score** (not the default 0.5 cutoff) is what the fraud team should tune — raising or lowering the threshold moves each model along its own precision/recall curve depending on how many cases they can actually investigate.

**Top predictive features:** total/max reimbursement amount, total deductible paid, and inpatient claim share — consistent with known fraud patterns (billing unusually large or unusually complex care).

---

## 💡 Business Recommendations

1. **Prioritize manual audits** for providers in the top 5% of total reimbursement and total deductible billed.
2. **Flag inpatient-heavy providers with unusually long average stays** for review — a known padding tactic.
3. **Use the probability score, not just Yes/No**, to build a prioritized review queue sized to the fraud team's actual capacity.
4. **Re-train periodically** — billing behavior and fraud tactics shift over time.

---

## 🗂️ Repository Structure
provider-fraud-app/
├── Healthcare_Fraud_Detection.ipynb   # Full pipeline: merge → features → model → evaluation
├── app.py                              # Streamlit app (loads model, predicts on uploaded CSV)
├── fraud_model.pkl                     # Trained Random Forest model
├── model_features.pkl                  # Exact feature list/order the model expects
└── requirements.txt                    # streamlit, pandas, scikit-learn, joblib

---

## ▶️ Run It Yourself

```bash
git clone https:https://provider-fraud-app-aztkki9vxtfzcseqtkpkb9.streamlit.app/
cd provider-fraud-app
pip install -r requirements.txt
streamlit run app.py
```

Then open `http://localhost:8501`, upload a CSV of provider-level features, and get instant fraud predictions.

---

## 🛠️ Tech Stack

`Python` · `pandas` · `scikit-learn` · `XGBoost` · `Streamlit` · `joblib`

---

## 🔒 Data Privacy

This repository contains **only code and the trained model artifact**. The underlying Medicare claims dataset is not published or redistributed here, per the case study's data-usage terms.

---

<div align="center">

*Built as part of a Healthcare Provider Fraud Detection case study.*

</div>
