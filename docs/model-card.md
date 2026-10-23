# Model card

**Model:** Random Forest near-full parking classifier. **Status:** Course project benchmark. **Published run:** October 23, 2026. **Course:** Clemson CPSC 4300/6300 Applied Data Science.

## Intended use

Inspect and reproduce a bounded tabular ML experiment. It estimates a contemporaneous near-capacity label from calendar, pricing, geography, capacity and weather. It is not validated for directing drivers, changing parking prices, forecasting future availability or making policy decisions.

## Training and loss

100 trees; `max_depth=12`; `min_samples_leaf=10`; `class_weight=balanced_subsample`; seed 6300. Trees select splits by Gini impurity reduction. The classifier does not optimize the same global cross-entropy objective as logistic regression. The logistic baseline uses weighted log loss plus L2 regularization (`C=1`); histogram gradient boosting uses log loss. No reward model, RLHF, preference optimization or generative-model training is claimed.

## Data and target

Three shards of Kaggle's version 1 Seattle parking dataset. 10,366,111 rows scanned, 51,463 sampled, 47,990 valid sampled rows retained. Target is paid occupancy divided by stated capacity at least 0.95. Inputs explicitly exclude occupancy and target-derived variables. Numeric missing values use training medians; categorical missing values use training modes. One-hot encoding ignores unseen categories.

Training: 27,424 rows over 15 observed dates. Validation: 9,697 rows over four dates. Test: 2,852 rows over four later dates. Two boundary dates are purged. Dates have calendar gaps. Sampled coverage is January through July 2015, not a continuous annual series.

## Selection and performance

Select the model by validation macro F1 at 0.5. Select its threshold on validation positive F1, yielding 0.51. At this threshold the test confusion matrix is TN 2,155; FP 399; FN 153; TP 145. Positive prevalence is 10.45%. Accuracy 80.65%, macro F1 0.6154, positive F1 0.3444, precision 26.65%, recall 48.66%, ROC AUC 0.7639, average precision 0.2969. AP is distinct from a trapezoidal PR AUC. The threshold's operational cost trade-off has not been validated.

The majority baseline achieves 89.55% accuracy and zero positive recall. Higher accuracy alone therefore would select a model that detects no near-full cases. Boosting has higher test ROC AUC and average precision than the forest; the forest is selected by the predeclared validation macro-F1 rule, not by claiming it wins every metric. Three expanding-window training reviews document temporal sensitivity.

## Limitations

Natural prevalence, missing rates, limited dates, repeating locations, weather availability and label semantics all restrict interpretation. No probability calibration, prospective testing, unseen-blockface evaluation or business-impact study has been completed. Compare metrics only when the target definition, sample and evaluation partitions match.
