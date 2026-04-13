"""Generate worksheet6 dataset and Jupyter notebooks."""
import json
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(20260407)

BASE     = Path(__file__).parent
DATA_DIR = BASE / "data"
CODE_DIR = BASE / "code"

# ── 1. Dataset ─────────────────────────────────────────────────────────────────

# True DGP: log_wage = 1.5 + 0.08*education + 0.04*experience
#           - 0.12*female + industry_FE + N(0, 0.20)
# True return to education: 0.08 (8% per year of schooling)
# OVB direction when experience is omitted:
#   experience = 30 - 0.6*education + noise  → Cov(educ, exp) < 0
#   β_experience > 0  →  OVB = (neg) × (pos) = DOWNWARD bias
#   i.e., short regression UNDERSTATES return to education

N = 500

education  = np.random.normal(14, 2.5, N).clip(8, 22).round().astype(int)
experience = (30 - 0.6 * education + np.random.normal(0, 6, N)).clip(0, 40).round().astype(int)
female     = np.random.binomial(1, 0.48, N)

industries = ["Manufacturing", "Services", "Finance", "Healthcare", "Education"]
industry   = np.random.choice(industries, N)

ind_fe = {
    "Manufacturing": 0.00,
    "Services":      0.10,
    "Finance":       0.20,
    "Healthcare":    0.12,
    "Education":    -0.08,
}
industry_effect = np.array([ind_fe[i] for i in industry])

epsilon  = np.random.normal(0, 0.20, N)
log_wage = (1.5
            + 0.08 * education
            + 0.04 * experience
            - 0.12 * female
            + industry_effect
            + epsilon)

df = pd.DataFrame({
    "worker_id": range(1, N + 1),
    "education":  education,
    "experience": experience,
    "female":     female,
    "industry":   industry,
    "log_wage":   log_wage.round(4),
})

df.to_csv(DATA_DIR / "wage_data.csv", index=False)
print(f"wage_data.csv — {len(df)} rows")

# ── 2. Notebook helpers ────────────────────────────────────────────────────────

def md(src):
    return {"cell_type": "markdown", "id": "x",
            "metadata": {}, "source": src}

def code(src):
    return {"cell_type": "code", "execution_count": None, "id": "x",
            "metadata": {}, "outputs": [], "source": src}

def assign_ids(cells):
    for i, c in enumerate(cells):
        c["id"] = f"cell-{i:03d}"
    return cells

def notebook(cells):
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3",
                           "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "cells": assign_ids(cells),
    }

# ── 3. Cell content ────────────────────────────────────────────────────────────

TITLE_MD = """\
# Week 6 Worksheet: Running OLS in Python

**ECC3479 — Data and Evidence in Economics**

This notebook covers bivariate and multiple OLS regression.
You will fit models, interpret coefficients, examine how controls
change estimates (OVB), and diagnose your final specification.

Work through each part in order. Save figures to `output/`.
"""

SETUP_CODE = """\
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

np.random.seed(20260407)
sns.set_theme(style="whitegrid", font_scale=1.1)

DATA_DIR   = Path("../data")
OUTPUT_DIR = Path("../output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
"""

LOAD_CODE = """\
# Load the dataset
df = pd.read_csv(DATA_DIR / "wage_data.csv")
print(df.shape)
df.head()
"""

PART_A_MD = """\
---
## Part A: Simple OLS

**Context.** The dataset `wage_data.csv` contains 500 Australian workers
from a 2022 HILDA-style survey. Variables:

| Column | Description |
|--------|-------------|
| `worker_id` | Worker identifier |
| `education` | Years of completed schooling (8–22) |
| `experience` | Years of work experience (0–40) |
| `female` | 1 = female, 0 = male |
| `industry` | Industry sector (5 categories) |
| `log_wage` | Log hourly wage (outcome) |
"""

Q1_MD = """\
### Question 1

Run a bivariate OLS regression of `log_wage` on `education`.
Print the coefficient table.
"""

Q1_ANS = """\
# Q1 — bivariate OLS: log_wage ~ education
m1 = smf.ols("log_wage ~ education", data=df).fit()
print(m1.summary().tables[1])
"""

Q1_STU = """\
# Q1 — Run bivariate OLS: log_wage ~ education
# Hint: smf.ols("outcome ~ predictor", data=df).fit()

"""

Q2_MD = """\
### Question 2

Interpret the `education` coefficient precisely. Your answer should state:
- The direction and magnitude of the association
- The units of both the predictor and the outcome (log wage → approximate % change)
- A careful caveat about what this estimate *does not* tell us
"""

Q2_ANS = """\
# Q2 — Interpretation (written answer — no code needed)
# Sample answer:
# The coefficient on education is approximately 0.057.
# One additional year of schooling is associated with a 5.7% higher hourly wage
# on average (using the approximation that Δlog_wage ≈ % change).
# This is a CORRELATION, not a causal estimate: workers with more education
# may also differ on experience, ability, or family background, all of which
# independently affect wages. Without controlling for these confounders, we
# cannot interpret 0.057 as the causal return to schooling.
print(f"Education coefficient (simple): {m1.params['education']:.4f}")
print(f"R-squared: {m1.rsquared:.3f}")
"""

Q2_STU = """\
# Q2 — Written interpretation (write your answer as a comment or markdown cell)
# State: coefficient value, units, and one important caveat

"""

Q3_MD = """\
### Question 3

What is the $R^2$ of the simple regression? Explain what it measures
and what it *does not* tell you about the usefulness of education as a
predictor of wages.
"""

Q3_ANS = """\
# Q3 — R-squared interpretation
print(f"R-squared: {m1.rsquared:.3f}")
# R^2 ≈ 0.19: education alone explains about 19% of the variation in log wages.
# This tells us that education is a moderate predictor of wages in this sample.
# It does NOT tell us whether the relationship is causal, nor whether the
# remaining 81% of variation is explainable by other variables.
# A high R^2 is neither necessary nor sufficient for a valid causal estimate.
"""

Q3_STU = """\
# Q3 — Compute and interpret R-squared
# Hint: m1.rsquared

"""

PART_B_MD = """\
---
## Part B: Multiple Regression and OVB

**Goal.** Add controls and observe how the `education` coefficient changes.
Use the direction of change to identify the sign of omitted variable bias.
"""

Q4_MD = """\
### Question 4

Add `experience` as a control variable. Run the regression and compare
the `education` coefficient to Question 1. Does it rise or fall?

Using the OVB formula $\\tilde{\\beta}_1 = \\hat{\\beta}_1 + \\hat{\\beta}_2 \\cdot \\hat{\\gamma}_1$,
determine the sign of the bias in the simple regression.
"""

Q4_ANS = """\
# Q4 — Add experience
m2 = smf.ols("log_wage ~ education + experience", data=df).fit()
print(m2.summary().tables[1])
print()
print(f"Simple regression:   education coeff = {m1.params['education']:.4f}")
print(f"With experience:     education coeff = {m2.params['education']:.4f}")
print(f"Change: {m2.params['education'] - m1.params['education']:+.4f}")
print()
# OVB analysis:
# beta_experience > 0 (more experience → higher wages)
# Cov(education, experience) < 0 (more education → started work later → less experience)
# so gamma_1 = Cov(educ, exp)/Var(educ) < 0
# OVB = beta_experience * gamma_1 = (positive) * (negative) = NEGATIVE (downward bias)
# => simple regression UNDERSTATES the true return to education
# Confirming: coefficient rises when we add experience (from ~0.057 to ~0.080)
gamma1 = df["education"].cov(df["experience"]) / df["education"].var()
print(f"Auxiliary regression slope (Cov/Var): {gamma1:.3f}  (negative → downward bias)")
"""

Q4_STU = """\
# Q4 — Add experience; compare coefficients and determine OVB direction
# Run smf.ols("log_wage ~ education + experience", data=df).fit()

"""

Q5_MD = """\
### Question 5

Add `female` and `industry` to the regression. Report a side-by-side
coefficient table for all three specifications.

Which specification do you prefer, and why?
"""

Q5_ANS = """\
# Q5 — Full specification: education + experience + female + industry
m3 = smf.ols("log_wage ~ education + experience + female + C(industry)",
             data=df).fit()
print(m3.summary().tables[1])
print()
# Side-by-side comparison of education coefficient across specs:
print("Education coefficient across specifications:")
print(f"  M1 (education only):              {m1.params['education']:.4f}")
print(f"  M2 (+ experience):                {m2.params['education']:.4f}")
print(f"  M3 (+ experience + female + ind): {m3.params['education']:.4f}")
print()
print(f"R-squared M1: {m1.rsquared:.3f}  |  M2: {m2.rsquared:.3f}  |  M3: {m3.rsquared:.3f}")
"""

Q5_STU = """\
# Q5 — Add female and industry dummies; compare all three models
# Use C(industry) in the formula for categorical dummy encoding

"""

PART_C_MD = """\
---
## Part C: Residual Diagnostics

**Goal.** Check whether your preferred model satisfies exogeneity by
examining the residuals.
"""

Q6_MD = """\
### Question 6

Plot the residuals from your Model 3 against the fitted values.
Describe any pattern you see. What would a clean plot look like,
and what would a problematic one look like?
"""

Q6_ANS = """\
# Q6 — Residual vs fitted plot
fig, ax = plt.subplots(figsize=(8, 4))
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)

ax.scatter(m3.fittedvalues, m3.resid, color="#5B7DB1", alpha=0.4, s=20)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Fitted values")
ax.set_ylabel("Residuals")
ax.set_title("Residuals vs Fitted Values — Model 3")

fig.savefig(OUTPUT_DIR / "residuals_vs_fitted.png", dpi=150,
            bbox_inches="tight", transparent=True)
plt.show()

# Interpretation:
# A clean plot: residuals scattered symmetrically around zero with no pattern.
# This is consistent with E[ε|X] ≈ 0 (exogeneity) and homoskedasticity.
# A problematic plot would show: a funnel shape (heteroskedasticity),
# a U-shape (misspecified functional form), or a trend (omitted variable).
"""

Q6_STU = """\
# Q6 — Plot residuals from m3 against fitted values
# Use m3.fittedvalues and m3.resid

"""

Q7_MD = """\
### Question 7

Write one sentence interpreting the `female` coefficient from Model 3.
Be precise about units, direction, and the appropriate level of caution
about causality.
"""

Q7_ANS = """\
# Q7 — Interpret the female coefficient
female_coeff = m3.params["female"]
print(f"Female coefficient: {female_coeff:.4f}")
# Sample interpretation:
# Holding education, experience, and industry constant, female workers earn
# approximately {abs(female_coeff)*100:.1f}% {'less' if female_coeff < 0 else 'more'}
# per hour than male workers on average (using the log approximation).
# This is a conditional correlation, not necessarily a causal gender wage gap:
# unmeasured differences in occupation, hours worked, or firm type
# could partly explain this estimate.
print(f"Female workers earn ~{abs(female_coeff)*100:.1f}% "
      f"{'less' if female_coeff < 0 else 'more'} (conditional on controls).")
"""

Q7_STU = """\
# Q7 — Write one sentence interpreting the female coefficient
# Print the coefficient first, then write your interpretation as a comment

"""

PART_D_MD = """\
---
## Part D (Extension): Frisch-Waugh Verification

**Goal.** Verify the Frisch-Waugh-Lovell theorem numerically:
the coefficient on `education` in a multiple regression equals the slope
from regressing the *residualised* outcome on the *residualised* education.
"""

Q8_MD = """\
### Question 8

**Residualise.** Run two auxiliary regressions:
1. Regress `log_wage` on `experience + female + C(industry)` → save residuals as `e_y`
2. Regress `education` on `experience + female + C(industry)` → save residuals as `e_x`

Then regress `e_y` on `e_x`. Compare the slope to the education coefficient
from Model 3.
"""

Q8_ANS = """\
# Q8 — Frisch-Waugh verification
controls = "experience + female + C(industry)"

# Residualise log_wage on controls
e_y = smf.ols(f"log_wage ~ {controls}", data=df).fit().resid

# Residualise education on controls
e_x = smf.ols(f"education ~ {controls}", data=df).fit().resid

# FWL regression
fw = smf.ols("e_y ~ e_x",
             data=pd.DataFrame({"e_y": e_y, "e_x": e_x})).fit()

print(f"FWL slope on e_x:           {fw.params['e_x']:.6f}")
print(f"M3 education coefficient:   {m3.params['education']:.6f}")
print(f"Difference:                 {abs(fw.params['e_x'] - m3.params['education']):.2e}")
# Should be < 1e-10 (numerical precision only)
"""

Q8_STU = """\
# Q8 — Verify Frisch-Waugh numerically
# Step 1: residualise log_wage on controls (experience, female, industry)
# Step 2: residualise education on the same controls
# Step 3: regress residualised log_wage on residualised education
# Compare slope to education coefficient in m3

"""

# ── 4. Assemble notebooks ─────────────────────────────────────────────────────

def build(answers: bool) -> dict:
    def cell(ans_src, stu_src):
        return code(ans_src) if answers else code(stu_src)

    cells = [
        md(TITLE_MD),
        code(SETUP_CODE),
        code(LOAD_CODE),
        md(PART_A_MD),
        md(Q1_MD),  cell(Q1_ANS, Q1_STU),
        md(Q2_MD),  cell(Q2_ANS, Q2_STU),
        md(Q3_MD),  cell(Q3_ANS, Q3_STU),
        md(PART_B_MD),
        md(Q4_MD),  cell(Q4_ANS, Q4_STU),
        md(Q5_MD),  cell(Q5_ANS, Q5_STU),
        md(PART_C_MD),
        md(Q6_MD),  cell(Q6_ANS, Q6_STU),
        md(Q7_MD),  cell(Q7_ANS, Q7_STU),
        md(PART_D_MD),
        md(Q8_MD),  cell(Q8_ANS, Q8_STU),
    ]
    return notebook(cells)


answers_nb = build(answers=True)
student_nb = build(answers=False)

with open(CODE_DIR / "worksheet6_answers.ipynb", "w", encoding="utf-8") as f:
    json.dump(answers_nb, f, indent=1, ensure_ascii=False)

with open(CODE_DIR / "worksheet6_student.ipynb", "w", encoding="utf-8") as f:
    json.dump(student_nb, f, indent=1, ensure_ascii=False)

print(f"worksheet6_answers.ipynb — {len(answers_nb['cells'])} cells")
print(f"worksheet6_student.ipynb — {len(student_nb['cells'])} cells")
