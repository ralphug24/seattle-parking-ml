"""Create a readable notebook that executes the same versioned experiment module."""
from pathlib import Path
import nbformat as nb

root=Path(__file__).resolve().parents[1]
md=nb.v4.new_markdown_cell
code=nb.v4.new_code_cell
book=nb.v4.new_notebook(cells=[
md("""# Seattle parking occupancy reconstruction

Rebuilt October 23, 2026 from a 2025 Applied Data Science group project. This notebook records new experiments with AI coding assistance. It does not recover the missing original training notebook or reproduce the old report's numbers.

**Question:** Can calendar, pricing, capacity, geographic and contemporaneous weather variables classify whether paid occupancy is at least 95% of a blockface's stated capacity? This is same-time classification, not a future-availability forecast. Payment-derived occupancy is not a direct sensor census of occupied spaces.

**Source:** Kaggle `arundhativt/seattle-parking-dataset`, version 1, shards 0, 27 and 55. The source metadata lists CC0. The repository includes a deterministic cleaned sample, source hashes and download instructions. Three shards are not a representative annual sample.
"""),
code("""from pathlib import Path
import sys, json
root = Path.cwd()
if not (root / 'experiment.py').exists():
    root = root.parent
sys.path.insert(0, str(root))
from experiment import prepare, plot_eda, split_data, train_and_evaluate, stability, FEATURES, CONFIG
import pandas as pd
print(json.dumps(CONFIG, indent=2))"""),
md("""## CP1 Data preparation and target definition

Scan source CSVs in chunks and retain a seeded 0.5% Bernoulli sample. Remove invalid timestamps, nonpositive capacities, negative occupancy, and occupancy greater than stated capacity. Deduplicate sampled blockface/timestamp pairs. Engineer cyclic time features. Preserve natural class prevalence.

The old report's `PaidOccupancy >= 8` target is retained only for a definition comparison. It is not used to train the new models. Occupancy, occupancy ratio and target-derived columns are excluded from model inputs.

Inspect missingness, particularly parking rates. Imputation and category encoding are fitted inside each training pipeline, not on the complete dataset."""),
code("""df = prepare()
plot_eda(df)
audit = json.loads((root / 'reports/data_audit.json').read_text())
print(json.dumps(audit, indent=2))
df[FEATURES + ['near_full']].head()"""),
md("""## CP2 Baseline and evaluation contract

Split whole dates chronologically into approximately 60% training, 20% validation and 20% test, purging one whole date at each boundary. Keep later-date test data out of model and threshold selection. Calendar gaps in the three source shards limit what this split establishes. The same locations can occur in all partitions; this is not a test of unseen-blockface generalization.

Compare a majority baseline, L2-regularized logistic regression with balanced class weights, a Random Forest, and histogram gradient boosting. Logistic regression uses weighted binary log loss plus L2 regularization. Forest trees use Gini impurity reduction. Histogram gradient boosting uses log loss. These are distinct training objectives; reported test error is an evaluation metric.

Use validation macro F1 at probability 0.5 to choose the model. Then choose its threshold on validation positive-class F1. This small, predetermined comparison is not an exhaustive hyperparameter search."""),
code("""splits = split_data(df)
print(json.dumps(json.loads((root / 'reports/splits.json').read_text()), indent=2))"""),
md("""## CP3 Benchmark training and held-out evaluation

Accuracy can hide failure on a rare class. Report macro F1, positive-class precision/recall/F1, balanced accuracy, ROC AUC, average precision, log loss, error rate and confusion matrices. Average precision should be read against test positive prevalence. Class weighting can affect probability calibration, so probability scores should not be treated as calibrated operational risk estimates."""),
code("""results, winner = train_and_evaluate(splits)
pd.read_csv(root / 'reports/test_metrics.csv').set_index('model').round(4)"""),
code("""benchmark = json.loads((root / 'reports/benchmark.json').read_text())
print('Selected on validation:', winner)
print('Threshold selected on validation:', benchmark['selected_threshold'])
print(json.dumps(benchmark['selected_test'], indent=2))"""),
md("""## Experiment stability

Three expanding-window reviews stay within training dates. The first date of each validation window is purged. These are descriptive checks of sensitivity to time windows, not independent randomized trials or evidence of year-round reliability."""),
code("""folds = pd.DataFrame(stability(splits['train']))
folds[['fold', 'model', 'macro_f1', 'roc_auc', 'average_precision']].round(4)"""),
md("""## Three cases and model behavior

Select observed test examples with low, median and high predicted probability and compare all models on the same rows. These examples illustrate behavior; they do not constitute a separate validation set or a causal explanation."""),
code("""pd.read_csv(root / 'reports/three_cases.csv').round(4)"""),
code("""from IPython.display import Image, display
display(Image(filename=str(root / 'reports/data_overview.png')))
display(Image(filename=str(root / 'reports/evaluation.png')))"""),
md("""## Handoff and next experiments

The repository saves preprocessing and model together, metrics, predictions, sample and source checksums, environment lock, decision records and project requirements. CPU-only training is sufficient for this bounded tabular benchmark. No GPU run or GPU performance claim is made.

Before operational use: expand to all seasons and locations, verify capacity and paid-occupancy semantics, evaluate unseen blockfaces, build a distinct future-horizon target, assess probability calibration, and choose thresholds using measured false-positive and false-negative costs. Current results do not establish reduced traffic, better driver outcomes or production readiness.
""")])
book.metadata['kernelspec']={'display_name':'Seattle Parking Python','language':'python','name':'seattle-parking'}
out=root/'notebooks/Seattle_Parking_Reconstruction.ipynb'
out.parent.mkdir(exist_ok=True)
nb.write(book,out)
print(out)
