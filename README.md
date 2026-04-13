# Week 6 Worksheet: Running OLS in Python

ECC3479 — Data and Evidence in Economics

This folder contains everything needed to run the Week 6 worksheet on
ordinary least squares regression. Students fit bivariate and multiple
regressions, interpret coefficients, examine how adding controls changes
estimates (demonstrating omitted variable bias), and check residual diagnostics.

---

## Folder structure

```
worksheet6/
├── code/
│   ├── worksheet6_student.ipynb   ← student copy (empty code cells)
│   └── worksheet6_answers.ipynb   ← instructor copy (completed code + written answers)
├── data/
│   └── wage_data.csv              ← 500-worker synthetic dataset
├── output/                        ← save figures here (gitkeep placeholder)
├── _generate.py                   ← reproduces dataset and notebooks from scratch
├── requirements.txt
└── README.md
```

---

## Getting started

1. Fork this repository on GitHub and clone your fork:
   ```bash
   git clone https://github.com/<your-username>/worksheet6.git
   cd worksheet6
   ```

2. Create a Python environment and install packages:
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. Open the student notebook in VS Code:
   ```
   code/worksheet6_student.ipynb
   ```

4. Select the correct Python kernel when prompted, then work through the tasks.

---

## Dataset

| File | Rows | Description |
|------|------|-------------|
| `wage_data.csv` | 500 | Synthetic 2022 HILDA-style survey. Columns: `worker_id`, `education` (years, 8–22), `experience` (years, 0–40), `female` (0/1), `industry` (5 categories), `log_wage` (outcome). |

The dataset is **synthetic** with a known true data-generating process:

```
log_wage = 1.5 + 0.08·education + 0.04·experience
         − 0.12·female + industry_FE + ε,   ε ~ N(0, 0.04)
```

True return to education: **8% per year of schooling**.

A simple regression of `log_wage` on `education` alone gives a **downward-biased**
coefficient (~0.064) because `experience` is omitted: more educated workers
tend to have less experience (started work later), and experience has a
positive effect on wages. Controlling for experience recovers a coefficient
closer to the true 0.08.

---

## Reminder

AI can help you move faster, but you are still responsible for checking
whether the code runs, whether the plots make sense, and whether the
written interpretations are defensible.
