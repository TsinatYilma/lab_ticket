# Estimating software ticket resolution time

Lab on predicting how long a support ticket takes to close. The csv is synthetic, so this is about the modelling choices: which columns are allowed, how to split, and whether a fancier model beats a straight line. It is not a description of a real team.

The work is in [`Lab2_Ticket_Resolution.ipynb`](Lab2_Ticket_Resolution.ipynb). That file has the code, the plots, and the written answers.

## Dataset

`lab_ticket_resolution.csv`: 240 tickets, no missing values.

| Column | Meaning | Used? |
|---|---|---|
| `ticket_id` | Row id | No, it is only an identifier |
| `complexity_score` | Complexity at triage, 1-10 | Yes |
| `lines_changed` | Lines edited during the fix | No, not known yet (leakage) |
| `team_experience_years` | Experience of the assigned team | Yes |
| `reopened` | Whether the ticket came back (0/1) | No, only known after resolution (leakage) |
| `resolution_hours` | Time to resolve | Target |

The prediction is meant to happen when the ticket arrives. `lines_changed` correlates about 0.85 with the target, which is another reason to leave it out.

## What I did

1. Loaded the csv and checked the shape, types, summary stats, missing values, and the first rows.
2. Set `resolution_hours` as y. Kept `complexity_score` and `team_experience_years`.
3. 80/20 split with `random_state=42` (192 train, 48 test). Degree and alpha were chosen with 5-fold cross-validation on the training set only.
4. Mean baseline (`DummyRegressor`).
5. Linear regression.
6. Polynomial features of degree 2 and 3, scaled, compared on the training folds.
7. Ridge and Lasso, with alpha chosen on those same folds. I also ran them on the degree-3 model, because that is the one with enough terms for a penalty to matter.
8. Actual-vs-predicted and residual plots for linear regression and the degree-3 Lasso.
9. Picked a model.

## Test set

| Model | MAE (hours) | RMSE (hours) | R2 |
|---|---|---|---|
| Mean baseline | 9.114 | 10.598 | -0.021 |
| Linear regression | 2.906 | 3.572 | 0.884 |
| Polynomial (degree 2) | 3.020 | 3.741 | 0.873 |
| Polynomial (degree 3) | 3.140 | 3.949 | 0.858 |
| Ridge (alpha 0.1) | 2.906 | 3.572 | 0.884 |
| Lasso (alpha 0.001) | 2.906 | 3.572 | 0.884 |
| Ridge, degree 3 (alpha 1) | 3.009 | 3.732 | 0.873 |
| Lasso, degree 3 (alpha 0.1) | 2.942 | 3.644 | 0.879 |

## What I would use

Linear regression on `complexity_score` and `team_experience_years`.

It takes the average error from 9.1 hours down to 2.9, and R2 is 0.88. Each complexity point adds about 3.08 hours. Each year of team experience takes off about 0.32 hours.

Degree 2 was only about 0.04 hours better in cross-validation, and worse on the test set. Degree 3 was worse than that. The residuals don't show a curve.

Ridge (alpha 0.1) and Lasso (alpha 0.001) match the linear model on these two columns. On the degree-3 polynomial they do something: Ridge pulls the largest coefficient from about 19.3 down to about 5.6, and Lasso sets 5 of the 9 coefficients to zero. The test error improves, and it still doesn't beat the straight line.

## Limits

- The data is synthetic.
- The test set is 48 tickets, so a gap of a tenth of an hour is not a real result.
- The fit leans on `complexity_score`, and that score is someone's estimate.
- The split is random. Tickets collected over years, or from the same teams, would need a time split or a group split. This file has neither a date nor a team id.
- A few tickets are off by 7 to 9 hours. I would not use the prediction as a hard deadline.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook Lab2_Ticket_Resolution.ipynb
```

Run it from this folder so it can find `lab_ticket_resolution.csv`.
