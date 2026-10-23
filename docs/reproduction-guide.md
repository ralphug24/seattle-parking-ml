# Seattle parking project walkthrough

Use this guide beside the executed notebook. The report and PowerPoint describe the original 2025 assignment. The code and results in this repository come from our October 2026 reconstruction with AI coding assistance. We changed the target, sample and evaluation design, so the new numbers are not the old results reproduced exactly.

## 1 Understand the assignment before choosing a model

CP1 asks us to explain the dataset, clean it, describe the outcome and inspect predictors. CP2 asks for a justified model, test error, interpretation and three predictions. CP3 asks for alternative models evaluated consistently and a comparison of their predictions.

Our unit of analysis is one parking blockface at one recorded minute. A blockface is one side of a street segment. One row is an observation, not one unique parking block. The same location appears repeatedly over time.

Our question is whether the payment-derived occupancy count is near the stated capacity at that observation time. We are not yet predicting availability thirty minutes into the future.

## 2 Retrieve and bound the data

The original source is Kaggle's Seattle parking dataset, version 1. We downloaded source shards 0, 27 and 55, containing 10,366,111 rows in total. We read them in chunks of 250,000 rather than loading all raw records into memory.

With random seed 6300, each row has a 0.5% chance of entering the working sample. This yielded 51,463 sampled rows before cleaning. We did not sample an equal number from each class. Preserving the natural imbalance makes the evaluation closer to the sampled data distribution.

This is a resource decision with a limitation: three shards are not a representative annual Seattle sample. The cleaned sample contains 25 observed dates between January and July 2015 with calendar gaps. The repository includes the exact sample, so reproducing the model run does not require downloading the roughly 1.9 GB of raw CSVs.

## 3 Clean the rows and define the label

We parse timestamps and remove records with missing identifiers, invalid timestamps, zero or negative capacity, negative occupancy, or occupancy exceeding capacity. We deduplicate the sampled location/timestamp keys. The sampled data contained no duplicate keys to remove in this run. After all cleaning, 47,990 rows remain.

The target is:

```python
occupancy_ratio = PaidOccupancy / ParkingSpaceCount
near_full = (occupancy_ratio >= 0.95).astype(int)
```

A block with eight occupied spaces and capacity twenty is 40% occupied, so its new label is zero. A block with three occupied spaces and capacity three gets label one. The old count threshold of eight would label those examples differently. The two definitions disagree on 12.2% of the cleaned sample.

The 95% threshold is our declared modeling assumption. Payment-based counts are not a direct census of physical availability. We need to validate the meaning of the source before making operational claims.

## 4 Build inputs without leaking the answer

Inputs include capacity, time limit, price, latitude, longitude, temperature, precipitation, day of week, month, weekend and holiday flags, and parking area/category fields. We transform the hour into sine and cosine values so that the representation respects a daily cycle.

```python
hour_sin = sin(2 * pi * fractional_hour / 24)
hour_cos = cos(2 * pi * fractional_hour / 24)
```

We explicitly exclude PaidOccupancy, occupancy_ratio, near_full and the old occupancy-derived label. Those variables reveal or directly construct the answer. Using them as predictors would create target leakage.

SourceElementKey identifies blockfaces for data checks and example inspection; it is not a predictor in this run.

## 5 Separate training, validation and testing

We order whole dates chronologically, allocate approximately 60%/20%/20% of dates, and purge one entire date at each boundary. This reduces mixing of adjacent-time records between partitions.

- Training: 27,424 rows over 15 observed dates, January 7 through June 29.
- Purged boundary date: July 3.
- Validation: 9,697 rows over four observed dates, July 7 through July 18.
- Purged boundary date: July 20.
- Test: 2,852 rows over four observed dates, July 21 through July 25.

The counts do not add to 47,990 because 8,017 rows fall on the two purged dates. Training fits models and preprocessing. Validation chooses the model and threshold. Testing provides the final later-date comparison. The same blockfaces can appear in multiple partitions, so this is not unseen-location validation.

## 6 Fit preprocessing only on training rows

Numeric missing values use training-set medians. Categorical missing values use training-set modes. Numeric features are standardized, and categories are one-hot encoded. Unseen categories at prediction time are ignored by the encoder.

All of this lives inside a scikit-learn Pipeline. When fitting a model, the pipeline learns its preprocessing from that model's training rows. Validation and test rows only receive the learned transformations.

About 48% of parking rates are missing. Imputation lets the experiment run, but it does not recover the true prices or remove the need to investigate this missingness.

## 7 Train a baseline and alternatives

The majority baseline always classifies observations as not near-full. It tells us what high accuracy can look like without useful positive detection.

Logistic regression uses weighted binary log loss with L2 regularization. Binary log loss is -[y log(p) + (1-y) log(1-p)]; it penalizes confident incorrect probabilities. L2 regularization penalizes large coefficients. We use C=1 and balanced class weights.

Random Forest fits 100 decision trees with maximum depth twelve and at least ten training samples per leaf. Each split seeks to reduce Gini impurity, defined as 1 minus the sum of squared class proportions. Balanced subsample weights adjust class influence within the trees. Forest predictions aggregate the trees' probability estimates. We did not train the forest with Adam or a reinforcement-learning reward.

Histogram gradient boosting uses log loss, 100 boosting iterations, a maximum of fifteen leaves and L2 regularization of one. It sequentially improves an ensemble. This is a histogram implementation of boosting, a specific reconstruction choice.

These settings are fixed comparison configurations, not the result of an exhaustive hyperparameter search.

## 8 Choose the model and threshold using validation

At threshold 0.5, validation macro F1 is 0.474 for logistic regression, 0.582 for Random Forest and 0.532 for histogram boosting. Our declared selection rule chooses the highest validation macro F1, so it selects Random Forest.

We then examine thresholds from 0.05 to 0.95 in steps of 0.01 for that selected model. Positive-class F1 on validation selects 0.51. This threshold becomes fixed before reporting its test confusion matrix.

Model selection and threshold selection are separate steps with different scoring rules in this experiment. A future study could choose one unified operational cost objective. We do not choose a new winner or threshold from the final test results.

## 9 Interpret the test results with actual counts

At threshold 0.51, the selected forest produces:

- True negatives: 2,155 correctly classified not-near-full observations.
- False positives: 399 near-full alerts when the label is zero.
- False negatives: 153 missed near-full observations.
- True positives: 145 correctly detected near-full observations.

Precision is 145 / (145 + 399) = 26.65%. It asks: of our near-full alerts, how many were correct?

Recall is 145 / (145 + 153) = 48.66%. It asks: of the actual near-full observations, how many did we detect?

Positive-class F1 is the harmonic mean of that precision and recall, about 0.344. Macro F1 averages the F1 scores for both classes equally, about 0.615. They are different quantities.

Accuracy is (2,155 + 145) / 2,852 = 80.65%. Error rate is 19.35%. The majority baseline reaches 89.55% accuracy but detects no positive cases. This is why higher accuracy is not sufficient evidence that the model serves the intended purpose.

ROC AUC is 0.764. It summarizes ranking of positives above negatives across thresholds; it is not 76.4% classification accuracy. Average precision is 0.297 versus positive prevalence of 0.104. It summarizes precision and recall behavior, not the precision at our selected threshold.

The boosting model has slightly stronger test ranking measures. That does not change which model our validation selection rule chose. The forest is not best on every measure, and its errors remain substantial.

## 10 Inspect examples and stability

We compare all models on three observed test examples chosen by low, median and high forest probability:

- July 21, 08:15, First Hill, capacity eleven: forest score 0.012, true label zero.
- July 24, 10:49, South Lake Union, capacity eight: forest score 0.390, true label zero.
- July 24, 12:09, South Lake Union, capacity three: forest score 0.819, true label one.

All three have a missing recorded rate. Scores have not been calibrated, so do not interpret 0.819 as a validated 81.9% operational risk. These examples illustrate predictions; they are not a separate performance study or proof of causal drivers.

Three expanding-window reviews use only training dates to inspect sensitivity to time windows. The forest's default-threshold macro F1 varies from approximately 0.556 to 0.570 across these reviews. These are descriptive checks, not confidence intervals or proof of year-round stability.

## 11 Reproduce and inspect the artifacts

Use Python 3.12, create an isolated environment, and install the recorded dependency lock. The commands below assume your terminal is in the repository root.

```bash
python -m venv .venv
```

On Windows, use the environment's Python directly:

```powershell
.venv/Scripts/python.exe -m pip install -r requirements-lock.txt
.venv/Scripts/python.exe experiment.py
.venv/Scripts/python.exe tools/package_artifacts.py
.venv/Scripts/python.exe tools/verify_artifacts.py
```

The experiment writes fitted pipelines into models/ and metrics, predictions, cases and plots into reports/. Verification reloads the selected model and checks that its predictions reproduce the recorded results. Artifact checksums detect changes to the published files. Re-running packaging refreshes the manifest after an intentional rerun.

To execute the notebook on Windows:

```powershell
.venv/Scripts/python.exe -m ipykernel install --prefix .jupyter --name seattle-parking
.venv/Scripts/python.exe tools/execute_notebook.py
.venv/Scripts/python.exe tools/package_artifacts.py
.venv/Scripts/python.exe tools/verify_artifacts.py
```

The portable commands in README.md cover activated environments. Timing can vary by machine; it is not a promised performance benchmark. The same locked environment and exact sample are needed to closely reproduce numerical results. Different platforms can introduce minor numerical differences.

## 12 Explain the research program relevance

The project connects an ambiguous problem to a data contract, fixed evaluation criteria, bounded compute, experiment comparisons and a documented handoff. The current decision log and milestone table make dependencies and open risks visible. They are reconstruction artifacts, not a claim about management routines used in the original class.

Before expanding the experiment, prioritize the source and label semantics, broader seasonal coverage, unseen-location testing, missing-price analysis, calibration and stakeholder error costs. A future-availability product also needs a future target and features that would actually be available at prediction time.

## Interview rehearsal

- What is one row, and what do the input and output represent?
- Why did we change the old count-based target?
- How could occupancy itself leak the answer?
- What does each model optimize during training?
- Which choices used validation data, and what role did test data play?
- Why can the dummy model beat the forest on accuracy?
- Explain precision and recall using the actual confusion matrix.
- What does ROC AUC tell us, and what does it not tell us?
- Which result would you investigate next, and why?
- Which parts belong to your original coursework, and which belong to this AI-assisted reconstruction?
