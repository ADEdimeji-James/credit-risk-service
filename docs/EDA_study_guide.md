# EDA Study Guide: Home Credit Default Risk

A companion to [notebooks/01_eda.ipynb](../notebooks/01_eda.ipynb). It explains **what** was done in each section, **why**, and **how to defend it** in an interview or exam.

**How to use it:** read one section of this guide, then the matching notebook section, then cover the page and answer that section's "Explain it in 30 seconds" prompt out loud. Part 11 holds practice questions with model answers.

---

## Part 1 · The big picture

### 1.1 The project in five sentences
1. A lender must decide, for each loan application, whether to approve, review or decline.
2. We build a model that outputs the **probability of default (PD)**, a number from 0 to 1.
3. Errors cost different amounts. Approving a defaulter loses about ₦350,000 (C_FN). Declining a good borrower loses about ₦125,000 in fees (C_FP).
4. So the cost-optimal decline cut-off is **p\* = C_FP / (C_FP + C_FN) = 125 / 475 ≈ 0.26**, not 0.5.
5. The model is served as an API that returns the PD, a risk band and the reasons for each score.

### 1.2 What EDA is *for*
EDA (exploratory data analysis) is not "making charts". It is **answering questions whose answers decide later engineering choices**. Every section in the notebook has a *Question* and a *Decides* line. In an interview, always connect a finding to the decision it drove:

> "I found X, so in Stage N I did Y."

| Section | Question | Decides |
|---|---|---|
| 1 Shape and target | How big is the data, and how rare is default? | Metrics, stratification, cost-model check |
| 2 Data quality | What is missing, fake or extreme? | Imputation, flags, transformations |
| 3 Signal | Which features separate defaulters from repayers? | Feature-engineering priorities |
| 4 Redundancy | Which features duplicate each other? | Pruning for logistic regression |
| 5 Fairness | How do groups differ, and what stands in for protected attributes? | The fairness audit design |
| 6 Bureau | Can the history table be used safely? | Aggregation plan, leakage rule |
| 7 Findings | What matters most? | The summary carried forward |

### 1.3 Vocabulary you must own

| Term | Meaning in this project |
|---|---|
| **TARGET** | 1 = the client had payment difficulties (late payment beyond a set number of days on an early instalment), 0 = repaid. This is "payment difficulty", not strictly "default". |
| **Positive class** | Default (TARGET = 1). The rare class we care about catching. |
| **Class imbalance** | 8.07% positives, so 11.4 repayers for every defaulter. |
| **Leakage** | Using information during training that would not exist at prediction time. It makes offline scores look great and the live model fail. |
| **Stratified split** | A split that keeps the same default rate (8.07%) in every part. |
| **PR-AUC** | Area under the precision–recall curve. It focuses on the positive class. A random model scores about the prevalence (0.08), not 0.5. |
| **ROC-AUC / Gini** | Ranking quality. Gini = 2 × AUC − 1, the form credit teams use. |
| **Calibration** | Whether a PD of 0.20 really means 20% of such applicants default. The p\* rule needs it. |
| **Row-wise transform** | Computed from one row only (e.g. credit / income). Learns nothing from other rows, so it **cannot leak**. |
| **Fitted transform** | Learns a value from the data (a median, a clipping cap, bin edges, an encoder). **Must be fitted on training folds only**, inside the pipeline. |

The last two rows are the single most important idea in the whole EDA. Most "leakage" interview questions reduce to: *is this transform row-wise or fitted?*

---

## Part 2 · Section 0: Setup

**What:** imports, display options, paths built with `pathlib` from the repo root, and the printed pandas and numpy versions.

**Why:**
- **`pathlib` and relative paths** mean the notebook runs on any machine where it sits in `notebooks/`. Hardcoded `C:\Users\...` paths break on CI (Linux) and on anyone else's machine.
- **Printing versions** matters because pandas 3.x reads text as a `str` dtype while pandas 2.x uses `object`. A result can change between versions, and the printed version makes it reproducible.
- **Licence rule (no `head()`, no `sample()` of raw rows):** Kaggle competition data may not be redistributed. A notebook pushed to GitHub with raw rows in its outputs would redistribute it. Aggregates and plots are fine.

**Explain it in 30 seconds:** "Setup makes the notebook portable and reproducible: relative paths, printed library versions, and a rule never to display raw rows because of the dataset licence."

---

## Part 3 · Section 1: Shape and target

### 1.1–1.3 Choosing dtypes and measuring memory
**What:** read 1,000 rows to see the inferred types, then load the full file with smaller types: `int64 → int32`, `float64 → float32`, text → `category`. Memory fell from 325 MB to 129 MB (−60%).

**Why each choice is safe, and how it was *proven* safe:**
- **int32** holds about ±2.1 billion. The observed range was −25,229 to 456,255. Checked after loading.
- **float32** has a 24-bit mantissa, so it stores whole numbers exactly only up to **2²⁴ = 16,777,216**. Only 2 incomes exceed that, and the worst rounding error on any money column was 0.0125, which is negligible. Checked against a float64 reload.
- **category** stores each distinct text value once, plus small integer codes per row. Columns with few unique values (e.g. gender) shrink a lot.
- **The risk with a 1,000-row sample:** a column that is integer in the first 1,000 rows might have a NaN later, and `int32` cannot hold NaN. The full load would then crash. It didn't, and that clean load is the proof.

**Caveats you should volunteer:**
- `category` is an **EDA convenience only**. The API receives plain strings, and a value never seen in training would break category codes. In the model, encoding is done by `OneHotEncoder(handle_unknown="ignore")` inside the pipeline.
- Money *calculations* (profit, costs, large sums) use float64.

### 1.4 Structure
307,511 rows × 122 columns, and `SK_ID_CURR` is unique. **One row = one application.** Consequence: every history table (bureau, previous applications) has many rows per applicant and must be **aggregated to one row per `SK_ID_CURR` before joining**. Otherwise the join duplicates applicants and inflates the data.

### 1.5 Target and class balance
24,825 defaults out of 307,511 = **8.07%**, imbalance **11.39 : 1**.

**Why it matters, in three consequences:**
1. **Accuracy is useless.** Predicting "repaid" for everyone scores 91.9% accuracy and catches zero defaulters.
2. **PR-AUC's random baseline is 0.08, not 0.5.** A PR-AUC of 0.25 sounds low but is about 3× better than random. Always quote the baseline next to the score.
3. **Stratify every split.** With 8% positives, a random split could easily give one fold 7.5% and another 8.6%, making fold scores incomparable. A 20% stratified hold-out still contains about 4,965 defaults, enough for stable estimates.

**Cost-model check:** the Problem Statement assumed 8%. With 8.07%, the approve-all profit per application is 0.9193 × 125,000 − 0.0807 × 350,000 ≈ **₦86,670** (assumed: ₦87,000). The assumption holds, and the model's value is the profit it adds *above* this floor.

**Imbalance handling decision:** 11 : 1 is moderate. Try **class weights** first (simple, no synthetic data). SMOTE is only an experiment, and if used it goes **inside** the CV folds via an imblearn Pipeline. Resampling before the split would leak synthetic copies of validation rows into training.

### 1.6 Train vs test columns
`application_test.csv` has no `TARGET`; it was Kaggle's scoring file. So **our test set is carved out of `application_train`** (Stage 4). The check confirmed that the only column difference is `TARGET`: every feature available in training is also available at prediction time.

**Explain it in 30 seconds:** "One row per applicant, 8.07% defaults, 11:1 imbalance. That kills accuracy as a metric, so I use PR-AUC against a 0.08 baseline plus expected profit, stratify every split, and start with class weights rather than SMOTE. The 8% assumption in my cost model held."

---

## Part 4 · Section 2: Data quality

### 2.1 Missingness overview
67 of 122 columns have missing values, and 41 are more than 50% missing.

**The beginner mistake:** "drop every column that is more than 50% missing". That rule ignores the key question: **does missingness itself carry information?**

**Theory you can quote:** statisticians classify missing data as:
- **MCAR** (Missing Completely At Random): missingness unrelated to anything. Dropping or imputing is harmless.
- **MAR** (Missing At Random): related to other observed columns.
- **MNAR** (Missing Not At Random): related to the unobserved value or the outcome itself.

If the default rate differs between rows with and without a value, the data is **not MCAR**, and the *fact that it is missing* is a usable feature.

### 2.2 Families and "missing together"
Columns were grouped into families by name pattern (housing `_AVG/_MODE/_MEDI`, `EXT_SOURCE_*`, `AMT_REQ_CREDIT_BUREAU_*`, social circle).

**The test:** compare, per family, the share of rows with *any* column missing against the share with *all* columns missing.
- any = all means the columns go missing on exactly the same rows, so **one indicator covers the family**: bureau requests (13.5% = 13.5%) and social circle (0.33% = 0.33%).
- any ≫ all means the columns go missing independently: `EXT_SOURCE` (64% vs 0.06%), so **each needs its own indicator**.
- Housing (74% vs 47%) is in between: a core block missing together, plus extra column-level gaps.

**Why not just compare missing shares?** Equal missing *shares* do not prove the *same rows* are missing. Two columns can each be 50% missing on completely different halves of the data.

### 2.3 Does missingness predict default?
| What is missing | Default when missing | Default when present |
|---|---|---|
| All housing data | 9.26% | 6.96% (complete); partial: 7.04% |
| `EXT_SOURCE_3` | 9.31% | 7.77% |
| `EXT_SOURCE_1` | 8.52% | 7.50% |
| `OWN_CAR_AGE` | 8.50% | 7.24% |

Decisions:
- **Housing:** partial and complete behave alike; only "nothing at all" stands out. So **one flag, `HAS_HOUSING_DATA`** (false only if all 47 columns are missing), captures the signal.
- **Car age:** missing for all 202,924 applicants without a car and only 5 owners, so "missing car age" = "no car". `FLAG_OWN_CAR` already says this. **No new indicator**, because it would duplicate an existing feature.
- **`EXT_SOURCE_*`:** one missing-indicator each.

**Careful wording for interviews:** "missingness is *associated with* higher default", not "*causes*". It is a descriptive gap.

### 2.4 Sign of `DAYS_*` columns
All `DAYS_*` values count days **before** the application, so they should be ≤ 0. Converted `DAYS_BIRTH` to `AGE_YEARS` (20.5–69.1, plausible). Only `DAYS_EMPLOYED` broke the rule, with a maximum of 365,243 days (about 1,000 years).

### 2.5 The `DAYS_EMPLOYED` sentinel
- 55,374 rows (18%) hold exactly 365243, the only positive value.
- 99.96% are pensioners, and the other 22 are unemployed. **The sentinel means "not currently employed"**; it is a code, not an error.
- That group defaults at **5.4% vs 8.7%**, so it is real signal.

**Decision:** keep a flag `EMPLOYED_PLACEHOLDER`, and set `YEARS_EMPLOYED` to NaN for those rows. The NaN is later imputed inside the pipeline (LR) or left as NaN (LightGBM).

**Why not fill it with the minimum (a tempting idea)?** The minimum `DAYS_EMPLOYED` is the *most negative* value, which is the *longest* tenure (about 49 years). Filling pensioners with it tells the model they are the most stably employed people in the data, the opposite of the truth. **A sentinel is a category in disguise; give it its own flag.**

### 2.6 Hidden placeholders in text columns
Home Credit writes `'XNA'` or `'Unknown'` instead of leaving a blank, so `isna()` misses them.
- `ORGANIZATION_TYPE = 'XNA'`: exactly the 55,374 sentinel rows. Same fact, recorded again.
- `CODE_GENDER = 'XNA'`: 4 rows. `NAME_FAMILY_STATUS = 'Unknown'`: 2 rows. Too few to matter. The one-hot encoder with `handle_unknown="ignore"` absorbs them, and the 4 gender rows are excluded from the fairness comparison.

### 2.7 Extreme money values
`AMT_INCOME_TOTAL` has a maximum of **117,000,000**, 248× its 99th percentile. That applicant was a regular worker asking for about 560k (income ≈ 208× the loan, against a median ratio of 0.31) who defaulted. **Almost certainly a typo** (extra zeros).

**Handling depends on the model:**
- **Logistic regression** fits a weighted sum, so one gigantic value stretches the scale and bends the coefficient. Fix: `log1p(income)` and use ratios (credit/income) rather than raw amounts.
- **Trees (LightGBM)** split on *order* ("income > 200k?"), not on distance, so an outlier just sits at the end of the order and barely matters.
- **A clipping cap is a fitted value.** The 99th percentile must be learned on the training fold, inside the pipeline, not copied from this notebook.

### 2.8 Imputation strategy, the key argument
| Model | Strategy | Why |
|---|---|---|
| Logistic regression | `SimpleImputer(strategy="median", add_indicator=True)` inside the pipeline | Median is robust to outliers; the indicator keeps the "was missing" signal. Fitted on training folds only. |
| LightGBM | Leave NaN | It learns at each split which side missing values should go to. |

**Why not −999?** For a linear model, −999 is a *real number*, far below every real value. It drags the fitted line toward a fake extreme and wrecks feature scaling. Trees tolerate it (it is just "very low"), but there is no reason to use it when trees handle NaN natively.

**Explain it in 30 seconds:** "Missingness wasn't random: applicants with no housing data or no external score defaulted more, so I kept those columns and added indicators instead of dropping them. `DAYS_EMPLOYED` used 365243 as a code for pensioners, so I turned it into a flag rather than imputing a number. Imputation is model-specific: median plus indicators fitted inside the pipeline for LR, native NaN handling for LightGBM. And any fitted value is learned per fold."

---

## Part 5 · Section 3: Signal

### 3.1 Method
- **Rates, not counts:** 92% of people repay, so a count chart of almost anything is mostly repayers. The useful question is "*within* this group, what share defaulted?"
- **Equal-count bins (`pd.qcut`), not equal-width bins (`pd.cut`):** with skewed data, equal-width bins put almost everyone in one bin and leave a handful in the rest, so those rates are noise. Deciles give each bin about 30,000 people.
- **Lift** = group default rate ÷ overall rate. A lift of 2.0 means twice the average risk.
- **Missing values get their own group** (`dropna=False`), because their rate is part of the signal (2.3).

**How noisy is a rate? (worth memorising)** The standard error of a proportion is √(p(1−p)/n).
- n = 30,000, p = 0.08 → SE ≈ 0.16 percentage points. Very stable.
- n = 22, p = 0.36 → SE ≈ 10 percentage points. Essentially noise.

This is why groups under 500 rows were left out of the plots, and why "unemployed applicants default at 36%" is *not* a finding.

### 3.2 Numeric features (results)
| Feature | Shape | Range of default rate |
|---|---|---|
| `EXT_SOURCE_3` | Monotonic ↓ | 20.0% → 3.2% |
| `EXT_SOURCE_2` | Monotonic ↓ | 18.4% → 3.0% |
| `EXT_SOURCE_1` | Monotonic ↓ | 17.6% → 2.6% |
| `AGE_YEARS` | Monotonic ↓ | 11.6% → 4.9% |
| `YEARS_EMPLOYED` | Monotonic ↓ | ~11% → 4.6% |
| `CREDIT_TO_GOODS` | Rising | 5–7% → 9–13% |
| `CREDIT_TO_INCOME` | **Hump** | 6.9% → ~9% → 7.1% |
| `ANNUITY_TO_INCOME` | Weak rise | 7.1% → 8.9% |

**Interpreting shapes (a favourite interview topic):**
- **Monotonic** (a steady staircase): logistic regression captures it with one coefficient.
- **Non-monotonic** (hump or U): a single linear coefficient cannot represent "risk goes up, then down". LR would fit a flat-ish line and miss it. Fixes for LR: binning, splines, or a squared term. **Trees capture it automatically.** This is a concrete, evidence-based argument for trying LightGBM.

**`CREDIT_TO_GOODS` story:** when the loan is larger than the price of the goods it buys, the difference is usually financed fees or insurance. Applicants borrowing well above the goods price default more. This is a domain-sensible feature, which matters for explainability.

**Ratios are row-wise.** `AMT_CREDIT / AMT_INCOME_TOTAL` uses one row only, so it can be computed anywhere without leakage.

### 3.3 Categorical features
- Contract type: cash loans 8.35% vs revolving 5.48%.
- Income type: working 9.6% → pensioner 5.4%. The tiny groups (unemployed 22, maternity leave 5, student 18, businessman 10) are noise.
- Education: a clean gradient from higher education (5.4%) to lower secondary (10.9%).

### 3.4 Ranking (spread)
Spread = riskiest group's rate − safest group's rate, in percentage points, over groups with ≥ 500 rows.

Order: `EXT_SOURCE_3` 16.8 > `EXT_SOURCE_2` 15.4 > `EXT_SOURCE_1` 15.0 > `CREDIT_TO_GOODS` 8.0 > `YEARS_EMPLOYED` 6.9 > `AGE_YEARS` 6.7 > education 5.6 > income type 4.2 > contract type 2.9 > `CREDIT_TO_INCOME` 2.3 > `ANNUITY_TO_INCOME` 1.7.

**Limitations to state honestly:**
- It is **univariate**: it ignores overlap (age and tenure move together).
- It ignores group size and curve shape.
- It is used to **prioritise** work, not to select features. Selection is done by cross-validated model performance.

**Explain it in 30 seconds:** "I compared default *rates* across equal-count bins. The three external scores are by far the strongest and monotonic. Credit-to-goods was the best engineered ratio. Credit-to-income turned out hump-shaped, which a linear model can't capture without binning, an argument for gradient boosting. I treated tiny groups as noise, because a rate from 22 people has a ±10-point standard error."

---

## Part 6 · Section 4: Redundancy

### Why redundancy matters
- **Multicollinearity in logistic regression:** if two inputs are near-copies, many combinations of their coefficients fit equally well. Coefficients become unstable, can flip sign between folds, and stop being interpretable. L2 regularisation softens this but doesn't fix interpretation.
- **Explanations:** SHAP splits credit between twins, so one real reason shows as two weak ones.
- **Trees:** accuracy is barely affected, but importance is split.

### Method
- **Spearman** (rank) correlation: catches any monotonic relation and is robust to outliers like the 117M income. Pearson only measures *linear* relations and is outlier-sensitive.
- **50,000-row sample** with a fixed seed: the standard error of a correlation is about (1 − ρ²)/√n ≤ 0.0045, so it is accurate to about ±0.01 at a fraction of the runtime.
- **|ρ| > 0.9** threshold: a common convention, not a law.

### Results and decisions
- 74 pairs above 0.9, of which **42 are housing `_AVG/_MODE/_MEDI` triplets** (≥ 0.92). Keep `_AVG` (14 columns) + `TOTALAREA_MODE` + 4 text `_MODE` columns; drop 28.
- `AMT_CREDIT` ≈ `AMT_GOODS_PRICE` (0.985): keep `AMT_CREDIT` + the `CREDIT_TO_GOODS` ratio.
- `OBS_30` ≈ `OBS_60` social circle (0.997): keep one.
- `REGION_RATING_CLIENT` ≈ `_W_CITY` (0.95): keep one.
- **"Not employed" is encoded four times:** `DAYS_EMPLOYED = 365243`, `ORGANIZATION_TYPE = 'XNA'` (exact), `FLAG_EMP_PHONE = 0` (all 55,374 plus 12 others), and `NAME_INCOME_TYPE = 'Pensioner'` (all but 22). Keep the flag and income type; drop `FLAG_EMP_PHONE`.
- The drop list lives in **YAML config** (project rule: settings in config, not hardcoded), and applies to the LR baseline. LightGBM can take the full set.

**Explain it in 30 seconds:** "I used Spearman correlation on a seeded 50k sample to find near-duplicates above 0.9. Most were housing columns reported three ways, so I kept one version. I also found that four different columns encode 'not employed'. Pruning matters for logistic regression, where collinearity makes coefficients unstable, and for clean per-applicant explanations."

---

## Part 7 · Section 5: Fairness groundwork

### Why it exists
The Problem Statement (§7) commits to: protected attributes (**gender, age, family status**) are **excluded from the model** but **kept aside for an audit**. Credit scoring is a high-risk use case under the EU AI Act (Annex III 5(b)), and Nigeria's NDPA 2023 s.37 gives applicants rights around automated decisions. EDA provides the **base rates** the audit needs.

### Base rates
| Group | Default rate |
|---|---|
| Women (66%) / Men (34%) | 7.0% / **10.1%** |
| Age 18–24 / 25–34 / 35–44 / 45–54 / 55+ | 12.3% / 10.7% / 8.4% / 7.0% / **5.2%** |
| Civil marriage / single / married / widow | 9.9% / 9.8% / 7.6% / 5.8% |
| Region rating 1 / 2 / 3 | 4.8% / 7.9% / 11.4% |

### The key conceptual point: which fairness metric?
- **Demographic parity** = equal *approval rates* across groups. Since the groups genuinely default at different rates, a *well-calibrated* model will **not** approve them equally. Forcing parity means knowingly mispricing risk.
- **Equalised odds / error-rate balance** = equal *false-negative and false-positive rates* across groups. The model may approve groups at different rates but should not be *wrong more often* for one group.
- The Problem Statement chooses **error rates** (with approval rates reported for context). The base rates above are why.

### Proxies
Measured as rank correlation with `IS_MALE` and `AGE_YEARS`:
- **Age:** `EXT_SOURCE_1` ≈ 0.61, employment flag ≈ 0.60, `FLAG_DOCUMENT_6` ≈ 0.39, `CNT_CHILDREN` ≈ 0.37. **Strong proxies.**
- **Gender:** strongest is `EXT_SOURCE_1` ≈ 0.31. Moderate.
- **Region vs ethnicity:** untestable, because there is no ethnicity column. Documented as a risk.

(Technical note: `Series.corr(method="spearman")` needs SciPy, which isn't installed. Pearson correlation of ranks *is* Spearman correlation, so ranks were computed first.)

### The trade-off you must be able to discuss
- Excluding age **costs signal**: age alone spreads default rates by about 7 points.
- Excluding age **does not remove age**: `EXT_SOURCE_1` and the employment flag carry much of it.
- Dropping the proxies too would destroy the strongest predictors and still leave weaker proxies.
- **Defensible position (industry norm):** exclude the protected attribute itself, *flag* the proxies, and **measure outcomes** (error-rate gaps per group) in the audit. Report the accuracy cost of excluding age in the README; it is a policy decision, not a modelling accident.

**Explain it in 30 seconds:** "Groups really do default at different rates: men at 10% vs women at 7%, under-25s at 12% vs over-55s at 5%. So equal approval rates would be the wrong target; I audit error rates per group instead. Age is excluded from the model, but external score 1 correlates 0.6 with age, so the model can partly recover it. I don't pretend exclusion solves fairness; I measure the outcome."

---

## Part 8 · Section 6: Bureau history

### What the table is
`bureau.csv`: one row per **previous credit at another lender**, 1.7M rows, median 4 per applicant (max 116). It is relational data, so it must be **aggregated to one row per applicant, then left-joined**.

### Why a *left* join?
14.3% of applicants have no bureau record. An inner join would silently drop them, and they are a distinct, riskier group: **10.1% vs 7.7%** default ("thin file" applicants). A left join keeps them with NaN aggregates plus a `HAS_BUREAU_RECORD` flag.

### Leakage check (time)
All bureau dates are relative to the application. A positive day count means *after* the application, which is information the lender couldn't have had.
| Column | Positive values | Verdict |
|---|---|---|
| `DAYS_CREDIT` (credit opened) | 0 | ✅ Clean |
| `DAYS_ENDDATE_FACT` (credit actually closed) | 0 | ✅ Clean |
| `DAYS_CREDIT_ENDDATE` (*scheduled* end) | 602,603 | ✅ Not leakage: a contract term known on application day |
| `DAYS_CREDIT_UPDATE` (record last updated) | **17** | ⚠️ Possibly post-application state. Filter out. |

**The subtle lesson:** a positive date is not automatically leakage. Ask *when was this fact knowable?* A loan's scheduled end date is known when the loan starts. Its *actual* end date is only known when it ends.

### Signal
- Currently overdue on another credit: **15.9%** vs 7.6%.
- Number of credits still active: 0 → 5.7%, 1 → 7.1%, 3 → 8.3%, 5+ → **12.1%**.

### Plan
Filter `DAYS_CREDIT ≤ 0` and `DAYS_CREDIT_UPDATE ≤ 0`, then aggregate per applicant: count, active count, any overdue, total debt, total credit, most recent credit. These aggregates use only *that applicant's own* records, so they are row-wise per applicant and safe outside the CV pipeline. They are added in Stage 5, **after** the application-only baseline, so their value is measured by CV PR-AUC rather than assumed.

**Explain it in 30 seconds:** "The bureau table has many rows per applicant, so I aggregate then left-join, keeping the 14% with no history as their own group, since they default more. I checked every date column for post-application information. Opening and closing dates were clean; a scheduled end date in the future is a contract term, not leakage; and 17 records updated after the application get filtered out by rule."

---

## Part 9 · Section 7: The ten findings (memorise this table)

| # | Finding | Decision |
|---|---|---|
| 1 | 8.07% default, 11.4 : 1 | PR-AUC + expected profit, stratified splits, class weights first |
| 2 | Missingness predicts default | Keep columns; indicators; median + indicator for LR; NaN for LightGBM |
| 3 | 365243 = "not employed" (pensioners) | Flag + NaN; never impute with a number |
| 4 | `EXT_SOURCE_*` dominate | Core features; combinations first in Stage 5 |
| 5 | `CREDIT_TO_GOODS` strong; `CREDIT_TO_INCOME` hump-shaped | Build ratios; non-monotonic shapes favour trees |
| 6 | One implausible 117M income | `log1p` for LR; fit any cap in-fold |
| 7 | 74 near-duplicate pairs | Prune for LR (housing 47 → 19 + flag, etc.), drop list in YAML |
| 8 | Group base rates differ | Audit error rates per group, not just approval rates |
| 9 | Age has strong proxies | Exclude attribute, flag proxies, measure outcome; state the cost |
| 10 | Bureau dates clean; strong signal | Filter, aggregate, left-join in Stage 5 |

---

## Part 10 · Cross-cutting principles

1. **Row-wise vs fitted.** Row-wise transforms (ratios, flags, unit conversions, per-applicant bureau aggregates) can be computed anywhere. Fitted transforms (imputation medians, scalers, encoders, clipping caps, bin edges, SMOTE) are learned **inside the pipeline on training folds only**.
2. **EDA on the full training file is description, not fitting.** I looked at all 307k rows to *understand* the data. That is standard, but every number that becomes a model parameter is re-learned per fold. Strictly, even *choices* made from the full data are a mild form of snooping; using a hold-out that is never examined in detail limits it.
3. **Model-dependent preprocessing.** Imputation, outliers, non-monotonic shapes and collinearity all matter more for logistic regression than for trees. Always say *which model* a decision is for.
4. **Small groups are noise.** Check n before trusting a rate (standard error √(p(1−p)/n)).
5. **Association is not causation.** All EDA findings are descriptive.
6. **Selection bias:** we only see outcomes for loans that *were approved*. The model never learns how rejected applicants would have behaved. This is documented, not solved (reject inference is out of scope).
7. **`TARGET` is "payment difficulty", not outright default**, so C_FN = ₦350k is an upper bound.

---

## Part 11 · Practice questions with model answers

**Q1. Why not use accuracy?**
With 8% defaults, approving everyone gives 92% accuracy and catches no defaulters. I use PR-AUC (baseline 0.08), recall at the cost-optimal threshold, and expected profit.

**Q2. Your PR-AUC is 0.27. Isn't that bad?**
The random baseline for PR-AUC is the prevalence, 0.08. So 0.27 is over 3× better than random. ROC-AUC's baseline is 0.5, which is why the two numbers look so different.

**Q3. Why is the decision threshold 0.26 and not 0.5?**
Approve when the expected gain is positive: (1 − p)·C_FP > p·C_FN, which gives p\* = C_FP / (C_FP + C_FN) = 125 / 475 ≈ 0.26. Missing a defaulter costs 2.8× more than wrongly declining, so the cut-off is low. It only works if the probabilities are calibrated.

**Q4. You had columns 70% missing. Why didn't you drop them?**
Missingness was informative: applicants with no housing data defaulted at 9.3% vs 7.0%. Dropping the columns discards that. I keep them with missing-indicators, and for housing I showed partial and complete rows behave alike, so one "has any housing data" flag captures it.

**Q5. How did you impute?**
It depends on the model. For logistic regression: median plus missing-indicator, fitted inside the pipeline on training folds. For LightGBM: native NaN handling. No −999 sentinels, because a linear model treats −999 as a real extreme value.

**Q6. What's wrong with imputing before the train/test split?**
The imputer's median would be computed using test rows, so information from the test set leaks into training and the evaluation becomes optimistic. Everything fitted goes inside a Pipeline so cross-validation refits it per fold.

**Q7. What was the strangest thing in the data?**
`DAYS_EMPLOYED = 365243`, about 1,000 years, on 18% of rows. It's a code for "not employed": 99.96% are pensioners, and they default *less* (5.4%). I turned it into a flag and set tenure to missing. Filling it with a number like the minimum would have labelled pensioners as the longest-employed people in the data.

**Q8. How do you know a feature is useful before modelling?**
Default rate across equal-count bins, compared with the 8.07% average. The external scores ranged from about 20% to 3% across deciles. But univariate views ignore overlap, so the final test is CV performance.

**Q9. Why `qcut` and not `cut`?**
Skewed data in equal-width bins leaves most bins nearly empty and their rates noisy. Equal-count bins give every rate the same reliability.

**Q10. Why might gradient boosting beat logistic regression here?**
Some relationships aren't monotonic: credit-to-income is hump-shaped. LR fits one slope per feature unless I hand-engineer bins or splines; trees find the shape on their own. LightGBM also handles NaN and outliers natively. The Problem Statement's adoption rule still requires it to beat LR on profit and PR-AUC consistently across folds.

**Q11. What is multicollinearity and did you have it?**
Near-duplicate inputs, which make linear coefficients unstable and uninterpretable. Yes: 74 pairs above |ρ| 0.9, mostly housing columns reported as average, mode and median. I kept one version per measure for the LR baseline.

**Q12. Why Spearman instead of Pearson?**
Spearman uses ranks, so it detects any monotonic relationship and isn't distorted by outliers like the 117M income.

**Q13. You removed gender and age from the model. Is it fair now?**
Not automatically. Age has strong proxies: external score 1 correlates 0.6 with age. Removing the attribute doesn't remove the information. So I measure outcomes: false-negative and false-positive rates per group in the audit.

**Q14. Should approval rates be equal across men and women?**
Not necessarily. Their base default rates differ (10.1% vs 7.0%), so a calibrated model will approve them at different rates. The stronger fairness test is whether the model makes *errors* at similar rates for each group.

**Q15. How did you check for leakage in the bureau data?**
All dates are relative to the application. I checked every `DAYS_*` column for positive values. Opening and closing dates were never in the future. Scheduled end dates are, but those are contract terms known at application. 17 records were updated after the application, and the pipeline filters them out by rule.

**Q16. Why a left join for bureau?**
14% of applicants have no bureau history and they default more (10.1%). An inner join would silently drop them. A left join keeps them, with an explicit no-history flag.

**Q17. Why did you load with float32? Isn't that risky for money?**
float32 stores integers exactly up to 2²⁴ ≈ 16.8M. Only 2 incomes exceed that, and the maximum error was 1.25 kobo. I verified that against a float64 reload. Profit calculations use float64.

**Q18. Why class weights before SMOTE?**
They're simpler, create no synthetic data, and 11:1 is only moderate imbalance. If I try SMOTE, it goes inside the CV folds via an imblearn Pipeline, because oversampling before splitting puts copies of validation information into training.

**Q19. What can't your model know?**
How rejected applicants would have behaved (selection bias), and whether patterns from a different market and period transfer to a Nigerian lender (domain shift). Both are documented limitations.

**Q20. What would you do with more time?**
Use `bureau_balance` and `previous_application` (monthly histories), test target-encoding for `ORGANIZATION_TYPE`, and do a stricter time-based validation if application dates were available.

---

## Part 12 · Glossary

- **Aggregation:** collapsing many rows per key into one (count, sum, max).
- **Base rate:** a group's actual default rate in the data.
- **Calibration:** agreement between predicted probabilities and observed frequencies.
- **Category dtype:** pandas storage of text as integer codes plus a lookup table.
- **Cost-optimal threshold (p\*):** the PD above which declining is cheaper in expectation.
- **Decile:** one of 10 equal-sized groups after sorting.
- **Equalised odds:** fairness criterion of equal error rates across groups.
- **Imputation:** filling missing values.
- **Leakage:** training on information unavailable at prediction time.
- **Lift:** group rate ÷ overall rate.
- **MCAR / MAR / MNAR:** missing completely at random / at random (given observed data) / not at random.
- **Missing-indicator:** a 0/1 column marking where a value was missing.
- **Monotonic:** always moving in one direction.
- **Multicollinearity:** strong correlation between model inputs.
- **Proxy:** a permitted feature that closely tracks a protected attribute.
- **Sentinel / placeholder:** a special value standing for "not available" (365243, `'XNA'`).
- **Spearman correlation:** correlation of ranks.
- **Standard error of a proportion:** √(p(1−p)/n), the typical sampling noise in a rate.
- **Stratification:** keeping class proportions equal across splits.
- **Thin file:** an applicant with little or no credit history.
