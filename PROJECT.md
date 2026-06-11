# Obesity Classification — Project Documentation

This document explains the algorithms used in `FinalCodingDataMining.ipynb`, how each one is implemented, and walks through a concrete example case. It is written as a study guide: if you understand everything here, you understand the project.

**The task:** given a person's lifestyle and physical attributes (age, height, weight, eating habits, activity, etc.), predict their obesity category (`NObeyesdad` — e.g. `Normal_Weight`, `Overweight_Level_I`, `Obesity_Type_I`, ...). We train on `obesity_train.csv` and predict labels for `obesity_test.csv`.

**The twist:** the test set contains classes the training set does not (`Obesity_Type_II` and `Obesity_Type_III`). A supervised classifier can never predict labels it didn't see in training, so the project uses a **classification-then-clustering** design: an XGBoost classifier handles the known classes, then a **two-stage KMeans clustering** splits the model's "ceiling" predictions into the missing higher classes.

**Current results:** validation accuracy **0.9874** (macro F1 0.9869), known-label test accuracy **0.9630**, end-to-end full-label-set test accuracy **0.9161** (841/918). Per-class test recall: Type III 1.000, Type II 0.808, Type I 0.857 — the Type I/II boundary in stage 2 is the main remaining error source.

---

## 1. Algorithms Used

| # | Algorithm / Technique | Role in the project |
|---|----------------------|---------------------|
| 1 | **XGBoost (gradient-boosted trees)** | Primary classifier over the known training classes |
| 2 | **BMI feature engineering** | Domain-knowledge feature (`Weight / Height²`) |
| 3 | **Two-stage KMeans clustering override** | Splits rows predicted as `Obesity_Type_I` into Types I/II/III — recovers classes missing from training |
| 4 | **Stratified train/validation split** | Honest model evaluation |
| 5 | **Confidence threshold tuning** | Distinguishing confident vs. uncertain predictions |
| 6 | **SHAP (SHapley Additive exPlanations)** | Model explainability — *why* the model predicts what it predicts |

Supporting techniques: schema/required-column checks, exploratory data summaries, native categorical encoding for XGBoost, label encoding for the target, `StandardScaler` + ordinal category codes for the clustering step.

---

## 2. How Each Algorithm Is Implemented

### 2.0 Data loading, schema checks, and EDA

Before any modeling, the notebook:

- Verifies the training data contains the target column and that both files contain the required numeric columns `Height` and `Weight` (raising a clear `ValueError` otherwise).
- Summarizes both dataframes with `summarize_dataframe` — dtype, missing count/percentage, and unique-value count per column.
- Explicitly identifies the **unknown classes**: labels present in the test target but absent from training:

```python
UnknownClass = list(set(test_df['NObeyesdad'].unique()).difference(set(train_df['NObeyesdad'].unique())))
```

This is what reveals that `Obesity_Type_II` and `Obesity_Type_III` exist only in the test set — the structural problem the rest of the pipeline is built around.

### 2.1 BMI Feature Engineering

Body Mass Index is the standard clinical measure of obesity, so we compute it explicitly rather than hoping the model rediscovers `Weight / Height²` from raw columns:

```python
def compute_bmi(df):
    height = pd.to_numeric(df["Height"], errors="coerce")
    weight = pd.to_numeric(df["Weight"], errors="coerce")
    return weight / (height ** 2)        # height in meters → kg/m²
```

BMI is added to both train and test dataframes, and its distribution is plotted with reference lines at 35 (clinical minimum for Obesity Type II) and 40 (Type III). BMI serves two purposes: (a) a strong input feature for the classifier, and (b) the variable used to *order* the clusters in the override step (§2.4).

### 2.2 Preprocessing and XGBoost (primary model)

**Idea:** gradient boosting builds trees *sequentially*, not in parallel. Each new tree is fitted to the errors (gradients of the loss) of the ensemble so far — so every tree corrects what the previous trees got wrong. This usually outperforms simpler ensembles on tabular data, at the cost of more hyperparameters.

Features and target are separated with `split_features`, and `align_feature` guarantees the test set has exactly the training feature columns (erroring on missing columns, dropping extras with a warning).

**Implementation:**

```python
def build_model():
    return xgb.XGBClassifier(
        n_estimators=300,          # up to 300 boosting rounds (trees)
        max_depth=6,               # tree depth — controls model complexity
        learning_rate=0.05,        # shrink each tree's contribution (small steps = less overfitting)
        subsample=0.8,             # each tree sees 80% of rows (randomness fights overfitting)
        colsample_bytree=0.8,      # each tree sees 80% of columns
        enable_categorical=True,   # XGBoost handles categorical dtype natively — no one-hot needed
        tree_method="hist",        # fast histogram-based split finding
        eval_metric="mlogloss",    # multiclass log-loss
        verbosity=0,
        random_state=RANDOM_STATE,
    )
```

XGBoost gets **no one-hot encoding**. Instead, categorical columns are converted to pandas `category` dtype, and XGBoost splits on categories directly:

```python
def prepare_xgb_features(df, category_levels=None):
    prepared = df.copy()
    category_cols = prepared.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_cols = [col for col in prepared.columns if col not in category_cols]

    for col in numeric_cols:
        prepared[col] = pd.to_numeric(prepared[col], errors="coerce")

    if category_levels is None:                      # fitting on training data:
        category_levels = {}
        for col in category_cols:
            prepared[col] = prepared[col].astype("category")
            category_levels[col] = prepared[col].cat.categories   # remember the levels
    else:                                            # transforming test data:
        for col, levels in category_levels.items():
            if col in prepared.columns:
                prepared[col] = pd.Categorical(prepared[col], categories=levels)
    return prepared, category_levels
```

**Crucial concept — `category_levels`:** the category-to-code mapping is learned on the *training* data and then *reused* for the test data. If the test set were encoded independently, `"Female"` might map to code 1 in training but code 0 in test, silently corrupting every prediction. Any category value never seen in training becomes `NaN`. This is the same principle as "fit the scaler on train, only transform on test."

The target labels (strings) are converted to integers with `LabelEncoder`, because XGBoost expects classes `0..K-1`:

```python
label_encoder = LabelEncoder()
y_all = label_encoder.fit_transform(y_all_labels)   # "Normal_Weight" -> 1, etc.
class_names = label_encoder.classes_                # used to decode predictions later
```

### 2.3 Stratified Validation Split

Before training the final model on all data, we hold out 20% of the training set to measure performance honestly. **Stratified** means each class keeps the same proportion in both splits — without it, a rare class could end up entirely in one split. The notebook wraps this in `stratified_validation_split`, which first validates that there are at least two classes and at least two rows per class (otherwise stratification is impossible), then calls:

```python
train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
```

The validation model is trained on the 80% and scored on the 20% with accuracy, macro F1 (treats every class equally — important with imbalance), weighted F1, macro precision and recall, plus a classification report and confusion matrix to see *which classes get confused with which*.

### 2.4 The KMeans Clustering Override ("classification then clustering")

**The problem this solves — important exam-level insight:** a supervised classifier can only ever predict labels *it saw during training*. Here `Obesity_Type_II` and `Obesity_Type_III` are missing from the training labels, so the model labels very heavy people as `Obesity_Type_I` — the highest class it knows (the "ceiling"). No amount of tuning fixes this; it's structural.

**The fix:** instead of a hard BMI rule, the notebook uses **two stages of unsupervised learning** on the ceiling rows (`two_stage_cluster_ceiling_rows`):

**Stage 1 — peel off Type III (k=3, full feature set):**

1. Collect all test rows the classifier predicted as `Obesity_Type_I` (`HIGHEST_KNOWN_CLASS`).
2. Build a numeric matrix from the full classifier feature set plus BMI (`CLUSTER_FEATURES`) — categorical columns are converted to their ordinal category codes, NaNs imputed with column means, everything standardized with `StandardScaler` (KMeans is distance-based, so features must be on comparable scales), and the standardized **BMI column is multiplied by `BMI_WEIGHT = 1.5`** so BMI dominates the geometry slightly.
3. Run `KMeans(n_clusters=3, n_init=10, random_state=RANDOM_STATE)` and sort the clusters by **ascending mean BMI**.
4. **Only the highest-mean-BMI cluster is kept** from this stage: its rows become `Obesity_Type_III` with `prediction_source = "clustering_override"`. The other two clusters are *not* trusted for the I/II split.

**Stage 2 — split Type I vs Type II (k=2, BMI-dominant features):**

5. Re-cluster the remaining ceiling rows with `KMeans(n_clusters=2)` using only `BMI_CLUSTER_FEATURES = ["BMI", "Weight", "Height"]` (BMI weight 1.0). The I/II/III boundaries are clinically defined by BMI alone, so only body-size features enter this geometry.
6. The cluster with the **lower mean BMI** stays `Obesity_Type_I`; the other becomes `Obesity_Type_II` with `prediction_source = "clustering_override_stage2"`.

```python
stage1_map = {
    sorted_clusters[0]: "Obesity_Type_I",     # ignored — re-clustered in stage 2
    sorted_clusters[1]: "Obesity_Type_II",    # ignored — re-clustered in stage 2
    sorted_clusters[2]: "Obesity_Type_III",   # kept
}
```

If fewer ceiling rows exist than clusters, the override is skipped with a warning (and stage 2 is skipped if fewer than 2 rows remain). Each stage prints a cluster → class mapping with the min/mean/max BMI per cluster, so the split is inspectable. Every prediction is tagged with a `prediction_source` (`classifier`, `classifier_low_confidence`, `clustering_override`, or `clustering_override_stage2`) so the export is fully auditable — you can always see *which mechanism* produced each label. This hybrid "supervised model + unsupervised post-processing" design is the project's namesake.

**Why two stages?** A single k=3 clustering must separate I, II, and III simultaneously in the full feature space, where lifestyle features can blur the BMI boundaries. Splitting the problem lets stage 1 isolate the most distinctive group (Type III, very high BMI) with rich features, then stage 2 draw the harder I-vs-II line using only the clinically relevant body-size variables.

**Sanity check against the clinical rule:** after the override, the notebook compares the cluster assignments on the ceiling rows against the hard clinical BMI rule (Type II = BMI 35–40, Type III = BMI ≥ 40) and prints the agreement percentage.

**Trade-off vs. a fixed BMI rule:** clustering uses more than just BMI and needs no hard-coded thresholds — but the quality of Type I/II/III predictions now depends on how cleanly the clusters separate, which the agreement diagnostic makes visible.

### 2.5 Confidence Threshold Tuning

`predict_proba` gives a probability per class; the maximum is the model's *confidence*. We search thresholds 0.50–0.97 on the validation set and pick the one maximizing accuracy *among confident predictions*, while requiring at least 60% of rows to stay above the threshold (coverage):

```python
for threshold in np.round(np.arange(0.50, 0.971, 0.01), 2):
    confident = confidence >= threshold
    coverage = confident.mean()                       # fraction of rows kept
    confident_accuracy = accuracy_score(y_true[confident], predicted[confident])
    ...
candidates = threshold_df[threshold_df["coverage"] >= min_coverage]
```

Ties are broken by confident macro F1, then coverage, then the lower threshold. If no threshold meets the coverage floor, the best-accuracy threshold is used regardless. The trade-off: a higher threshold means the kept predictions are more accurate, but fewer rows qualify. The chosen threshold only labels rows as `classifier_low_confidence` in the output — useful metadata for anyone consuming the predictions.

### 2.6 Test-Set Evaluation (two stages)

Unlike a pure prediction task, this test set *has* labels, so the notebook evaluates twice:

1. **Known-label evaluation (classifier only):** rows whose true label exists in the training classes are scored with accuracy, a classification report, and a confusion matrix. Rows with the unknown Type II/III labels are excluded (and listed explicitly), since the raw classifier cannot get them right.
2. **Full-label-set evaluation (after the clustering override):** all labeled rows are scored against the complete label set including `Obesity_Type_II` and `Obesity_Type_III`. This measures the end-to-end hybrid system — and how well the cluster→class mapping recovered the missing classes.

### 2.7 SHAP Explainability

SHAP assigns each feature, *for each individual prediction*, a contribution value rooted in game theory (Shapley values): how much did this feature push the prediction toward or away from each class? For tree models, `TreeExplainer` computes this exactly and fast:

```python
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_all_xgb)
shap.summary_plot(shap_values, X_all_xgb, plot_type="bar")   # global importance
shap.summary_plot(shap_values, X_all_xgb)                    # per-feature value/direction beeswarm
```

The mean absolute SHAP value per feature gives a global importance ranking, which the notebook compares against XGBoost's built-in `feature_importances_` (a helper, `mean_abs_shap_by_feature`, normalizes SHAP's different output shapes for multiclass models). Expect `BMI`, `Weight`, and `Height` to dominate — which makes sense, and *that sanity check is the point of explainability*.

### 2.8 Export

The final `obesity_predictions.csv` contains all original test columns plus `BMI`, `predicted_label`, `confidence`, `prediction_source`, and one `probability_<class>` column per training class.

---

## 3. Worked Example Case

Let's trace one fictional test row through the entire pipeline.

**Input row from `obesity_test.csv`:**

| Gender | Age | Height | Weight | FAVC | FCVC | NCP | CAEC | SMOKE | CH2O | FAF | CALC | MTRANS |
|--------|-----|--------|--------|------|------|-----|------|-------|------|-----|------|--------|
| Female | 24 | 1.62 m | 110 kg | yes | 2.0 | 3.0 | Sometimes | no | 2.0 | 0.0 | Sometimes | Public_Transportation |

**Step 1 — Feature engineering.** BMI is computed:

```
BMI = 110 / 1.62² = 110 / 2.6244 ≈ 41.9
```

**Step 2 — Encoding.** Categorical columns (`Gender`, `FAVC`, `CAEC`, ...) are cast to the `category` dtype using the *training* category levels; numeric columns pass through.

**Step 3 — XGBoost prediction.** The model outputs a probability per training class, e.g.:

```
Insufficient_Weight: 0.00   Normal_Weight: 0.01   Overweight_Level_I: 0.02
Overweight_Level_II: 0.05   Obesity_Type_I: 0.92
```

Predicted label: `Obesity_Type_I`, confidence = 0.92. Note the model **cannot** say `Obesity_Type_III` — that label was never in its training data. This is the structural limitation from §2.4 in action: the model is *confidently wrong*.

**Step 4 — Confidence check.** Suppose the tuned threshold is 0.74. Since 0.92 ≥ 0.74, the row is provisionally tagged `classifier`.

**Step 5 — Clustering override.** The row was predicted `Obesity_Type_I`, so it joins the ceiling group. In **stage 1**, its features (plus BMI ≈ 41.9 × 1.5 weighting, one of the highest) are standardized and clustered with k=3; KMeans places it in the highest-mean-BMI cluster, which maps to `Obesity_Type_III`. The label is replaced and the source becomes `clustering_override`. (Had it landed in one of the two lower clusters instead, **stage 2** would re-cluster it with k=2 on BMI/Weight/Height to decide between Type I and Type II — Type II rows get source `clustering_override_stage2`.)

**Step 6 — Export.** The row in `obesity_predictions.csv` looks like:

| ...original columns... | BMI | predicted_label | confidence | prediction_source | probability_Obesity_Type_I | ... |
|---|-----|-----------------|------------|-------------------|------|---|
| ... | 41.9 | **Obesity_Type_III** | 0.92 | **clustering_override** | 0.92 | ... |

**Contrast case:** a 1.75 m, 70 kg person has BMI ≈ 22.9. The classifier likely predicts `Normal_Weight`, which is not the ceiling class, so the row never enters the clustering step — its prediction stands with source `classifier` (or `classifier_low_confidence` if its confidence fell below the threshold).

---

## 4. Key Takeaways for Studying

1. **A supervised model cannot predict unseen classes.** When classes are missing from training, post-processing must fill the gap — here, unsupervised clustering of the ceiling predictions, ordered by a domain variable (BMI).
2. **Fit encoders/category levels on training data only**, then reuse them on test data — otherwise codes silently misalign.
3. **Stratify** validation splits when classes are imbalanced.
4. **Scale features before KMeans.** KMeans is distance-based; unscaled features (Weight in kg vs. binary categories) would dominate the geometry.
5. **Tag every prediction's source** (`classifier` / `classifier_low_confidence` / `clustering_override` / `clustering_override_stage2`) so a hybrid system stays auditable.
6. **Confidence ≠ correctness** (the model was 92% sure of a wrong label in the example), which is exactly why thresholds and post-processing matter.
7. **Evaluate the system you ship, not just the model:** known-label accuracy measures the classifier; full-label-set accuracy measures the classifier *plus* the clustering override.
8. **Explainability (SHAP) is a sanity check**: if the top features didn't make domain sense, you'd suspect data leakage or a bug.
