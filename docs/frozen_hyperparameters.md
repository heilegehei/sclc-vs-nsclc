# Frozen hyperparameters for the GAM and rotation forest

This document describes the two laboratory classifiers in Section 2.4 and Supplementary Table S2. Both use the same ordered panel of 18 laboratory predictors.

## Generalized additive model

The project wrapper calls `pygam.LogisticGAM` (pyGAM 0.12.0).

| Setting | Value used |
|---|---|
| Inputs | Center each predictor on its training mean and divide by its training standard deviation (population SD). Apply the stored transformation when predicting. |
| Additive terms | One cubic spline per predictor, with 10 spline basis functions per term |
| Smoothing penalty | λ = 0.6 |
| Intercept | Fitted |
| Convergence tolerance | 0.0001 |
| Iteration limit | The factory sets `max_iter=100`. The wrapper calls pyGAM with `max(max_iter, 200)`, giving an effective limit of 200 iterations. |
| Output used for comparison | Predicted probability of SCLC |

## Custom rotation forest

The implementation uses PCA and decision trees from scikit-learn 1.7.2.

| Setting | Value used |
|---|---|
| Ensemble size | 100 decision trees |
| Feature grouping | For each tree, randomly permute the 18 features and split them into three disjoint groups of six. |
| Rotation | Fit a separate PCA to each group using all training observations, the full SVD solver, and all six components. Apply the fitted group transformations in order and concatenate their outputs. |
| Tree | Gini split criterion, unrestricted depth, minimum of one observation per leaf |
| Prediction | Arithmetic mean of the 100 trees’ SCLC probabilities |

### Random seeds by fit

Both seeds derive from the project master seed (20260902). The arguments shown below are passed to the project’s `derive_seed` function.

| Fit | Seed derivation | Result |
|---|---|---:|
| Rotation forest reproduced for the Table 2 comparison | `derive_seed(20260902, "full_primary", "I_M_N", "rotation_forest")` | 504097906 |
| Rotation forest member of the frozen deployment ensemble | `derive_seed(20260902, "holdout_fit", "model", "rotation_forest")` | 1947554071 |

## Reference implementation checks

The checks used the same 775 Center A development records and compared predictions on the same 213 evaluation records. The GAM reference was a direct call to pyGAM using the settings above. For the rotation forest comparison, aeon 1.6.0 used `RotationForestClassifier` with 100 estimators, `min_group=max_group=6`, a Gini decision tree, `remove_proportion=0.5`, `n_jobs=1`, and `random_state=504097906`. The aeon implementation applies min–max scaling and samples classes and observations for group rotations.

The GAM wrapper and direct pyGAM calls yielded identical probabilities for all 213 records (ROC AUC 0.702, Brier score 0.154). ROC AUCs for the custom and aeon rotation forests were 0.728 and 0.731, respectively. Their predicted probabilities had a Spearman rank correlation of 0.902 and a mean absolute difference of 0.063.

[Model factory and frozen parameter definitions](../01_methods/models/src/model_factory.py)
