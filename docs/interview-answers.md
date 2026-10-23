# Interview discussion draft

Use these drafts beside the notebook and study guide. Explain the work in your own words, describe your personal role accurately, and tie numerical claims to the published experiment configuration.

## Model training

My example is a Seattle parking occupancy project from Applied Data Science coursework. The inputs include time of day, day of week, parking capacity and price, location and weather. The output is a binary indicator of whether paid occupancy reaches 95% of the blockface's stated capacity.

The project compares logistic regression, Random Forest and histogram gradient boosting with a majority-class baseline. Logistic regression minimizes weighted binary log loss with L2 regularization. For the Random Forest, individual trees use Gini impurity reduction to choose splits. These are standard library objectives; there is no reinforcement-learning reward or custom loss. A forest is suitable for exploring nonlinear relationships and feature interactions in tabular data.

The selected forest has 100 trees, maximum depth 12 and a minimum of ten samples per leaf. It was selected on validation macro F1 at probability 0.5. A separate validation step selected a threshold of 0.51 by positive-class F1.

The modeling workflow makes several choices explicit: the target accounts for capacity rather than using a count threshold of eight; occupancy-derived features are excluded; preprocessing is fitted inside the training pipeline; later whole dates are held out; and the run preserves configurations, model files and predictions. The notebook and saved artifacts make those choices inspectable.

## Model evaluation

I would explain evaluation through the class imbalance. On the later-date test set, only 10.45% of observations are near-full. A model that always predicts not-near-full achieves 89.55% accuracy and zero positive recall, so accuracy alone would give an incomplete assessment.

The evaluation reports macro F1, positive precision and recall, ROC AUC, average precision, log loss and confusion matrices. Macro F1 weights the two classes equally. Precision describes how many near-full alerts are correct; recall describes how many actual near-full cases are detected. ROC AUC measures ranking across thresholds, and average precision focuses on positive-class retrieval against a prevalence baseline.

The validation-selected forest achieves test ROC AUC of 0.764 and average precision of 0.297. At the validation-selected threshold, precision is 0.267 and recall is 0.487. That is 145 true positives, 399 false positives and 153 false negatives. These results show signal and substantial errors, so I would not claim production readiness.

The dataset sample covers a limited set of dates. I would next expand seasonal coverage, test unseen locations, assess calibration and define error costs with stakeholders before recommending an operational threshold. I would also define a separate future-horizon task before calling the model a parking-availability forecast.

## Relevance to an AI research technical program role

The project makes requirements, data and compute dependencies, selection rules, experiment outcomes and open risks inspectable. It includes decision records, milestone acceptance criteria, environment versions and model handoff instructions. These artifacts connect model work to experiment coordination and technical handoff. CPU-only training was sufficient at this scale; expanding compute should follow a measured experiment need.

## Explain these distinctions before the interview

- Training objective versus evaluation metric: Gini/log loss guide learning; macro F1 and AP assess behavior.
- Model selection versus threshold selection: both use validation data; neither uses test outcomes.
- Macro F1 versus positive F1: averaging equally across classes differs from focusing on near-full cases.
- ROC AUC versus average precision: both assess ranking, but AP emphasizes the rare positive class.
- Current-state classification versus future forecasting: the present task has no forecast horizon.
- Experiment comparability: different targets, samples and splits mean different experiments.
