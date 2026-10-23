"""Seattle parking classification benchmark with reproducible training and evaluation."""
from pathlib import Path
import hashlib
import json
import os
import platform
import time
import warnings

import joblib
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "tmp/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, log_loss, precision_score, recall_score, roc_auc_score,
    precision_recall_curve, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
SEED = 6300
NUMERIC = ["ParkingTimeLimitCategory", "ParkingSpaceCount", "PaidParkingRate", "Latitude",
           "Longitude", "temp", "precip", "hour_sin", "hour_cos", "dow", "month", "is_weekend", "is_us_holiday"]
CATEGORICAL = ["PaidParkingArea", "PaidParkingSubArea", "ParkingCategory"]
FEATURES = NUMERIC + CATEGORICAL
FORBIDDEN = {"PaidOccupancy", "occupancy_ratio", "near_full", "is_full", "legacy_high_occupancy"}
CONFIG = {"seed": SEED, "sample_fraction": 0.005, "parts": [0, 27, 55],
          "target": "PaidOccupancy / ParkingSpaceCount >= 0.95", "task": "same-time classification, not future forecasting",
          "split": "chronological whole dates, 60/20/20; one entire date purged at each boundary",
          "selection": "validation macro F1 at threshold 0.5; positive-class threshold selected on validation only",
          "cpu_threads": 2, "gpu": False}

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8")

def prepare():
    cached = ROOT / "data/processed/sample.csv.gz"
    if cached.exists():
        return pd.read_csv(cached, parse_dates=["OccupancyDateTime"])
    rng = np.random.default_rng(SEED)
    frames, scan = [], []
    for part in CONFIG["parts"]:
        path = ROOT / f"data/raw/PaidParkingOccupancyWeatherData_2015_{part}.csv"
        count = 0
        for chunk in pd.read_csv(path, chunksize=250_000, low_memory=False):
            count += len(chunk)
            frames.append(chunk.loc[rng.random(len(chunk)) < CONFIG["sample_fraction"]].assign(source_part=part))
        scan.append({"part": part, "rows_scanned": count})
        print("Scanned", part, count, flush=True)
    df = pd.concat(frames, ignore_index=True)
    sampled = len(df)
    df["OccupancyDateTime"] = pd.to_datetime(df["OccupancyDateTime"], errors="coerce")
    df = df.dropna(subset=["OccupancyDateTime", "SourceElementKey", "PaidOccupancy", "ParkingSpaceCount"])
    df = df[(df.ParkingSpaceCount > 0) & (df.PaidOccupancy >= 0) & (df.PaidOccupancy <= df.ParkingSpaceCount)].copy()
    before = len(df)
    df = df.drop_duplicates(["SourceElementKey", "OccupancyDateTime"])
    duplicates = before - len(df)
    dt = df.OccupancyDateTime
    df["hour_sin"] = np.sin(2 * np.pi * (dt.dt.hour + dt.dt.minute / 60) / 24)
    df["hour_cos"] = np.cos(2 * np.pi * (dt.dt.hour + dt.dt.minute / 60) / 24)
    df["dow"], df["month"] = dt.dt.dayofweek, dt.dt.month
    for col in ["is_weekend", "is_us_holiday"]:
        df[col] = df[col].astype(int)
    df["occupancy_ratio"] = df.PaidOccupancy / df.ParkingSpaceCount
    df["near_full"] = (df.occupancy_ratio >= 0.95).astype(int)
    df["legacy_high_occupancy"] = (df.PaidOccupancy >= 8).astype(int)
    df = df.sort_values(["OccupancyDateTime", "SourceElementKey"]).reset_index(drop=True)
    cached.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cached, index=False)
    write_json(ROOT / "reports/data_audit.json", {"scanned_sources": scan, "sampled_rows": sampled,
        "clean_rows": len(df), "duplicate_sampled_keys_removed": duplicates,
        "date_min": str(dt.min()), "date_max": str(dt.max()), "blockfaces": int(df.SourceElementKey.nunique()),
        "near_full_prevalence": float(df.near_full.mean()), "legacy_prevalence": float(df.legacy_high_occupancy.mean()),
        "label_disagreement": float((df.near_full != df.legacy_high_occupancy).mean()),
        "missing_fraction": df[FEATURES].isna().mean().to_dict(),
        "limitation": "Deterministic Bernoulli sample from three source shards, not a representative annual sample. Deduplication is within sampled rows."})
    return df

def split_data(df):
    dates = np.array(sorted(df.OccupancyDateTime.dt.date.unique()))
    if len(dates) < 10:
        raise ValueError("Too few distinct dates for train/validation/test with purged boundaries")
    a, b = int(len(dates) * 0.6), int(len(dates) * 0.8)
    day = df.OccupancyDateTime.dt.date
    splits = {"train": df[day.isin(dates[:a])].copy(),
              "validation": df[day.isin(dates[a+1:b])].copy(), "test": df[day.isin(dates[b+1:])].copy()}
    assert not FORBIDDEN.intersection(FEATURES)
    assert splits["train"].OccupancyDateTime.max() < splits["validation"].OccupancyDateTime.min()
    assert splits["validation"].OccupancyDateTime.max() < splits["test"].OccupancyDateTime.min()
    for split in splits.values():
        assert split.near_full.nunique() == 2
    summary = {k: {"rows": len(v), "start": str(v.OccupancyDateTime.min()), "end": str(v.OccupancyDateTime.max()),
                      "positive_rate": float(v.near_full.mean()), "dates": int(v.OccupancyDateTime.dt.date.nunique())} for k,v in splits.items()}
    write_json(ROOT / "reports/splits.json", {"splits": summary, "purged_dates": [str(dates[a]), str(dates[b])]})
    return splits

def plot_eda(df):
    fig, axs = plt.subplots(2, 2, figsize=(11, 7))
    df.near_full.value_counts().sort_index().plot.bar(ax=axs[0,0], color=['#607d8b','#b7791f'])
    axs[0,0].set(title='Natural class counts', xlabel='Near full (0/1)', ylabel='Sampled observations')
    df.groupby(df.OccupancyDateTime.dt.hour).near_full.mean().plot(ax=axs[0,1], marker='o')
    axs[0,1].set(title='Sampled hourly pattern',xlabel='Hour',ylabel='Near-full proportion')
    df.groupby('PaidParkingRate').near_full.mean().plot.bar(ax=axs[1,0])
    axs[1,0].set(title='Observed rates only (missing rates excluded)',xlabel='Recorded parking rate',ylabel='Near-full proportion')
    df.groupby(pd.cut(df.ParkingSpaceCount,[0,5,10,15,20,50]), observed=True).near_full.mean().plot.bar(ax=axs[1,1])
    axs[1,1].set(title='Near-full proportion by stated capacity',xlabel='Capacity bin',ylabel='Near-full proportion')
    fig.suptitle('Exploratory sample descriptions, not causal effects',fontsize=12)
    fig.tight_layout();fig.savefig(ROOT/'reports/data_overview.png',dpi=180);plt.close(fig)

def pipelines():
    pre = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
        ("category", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                               ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL)])
    models = {"dummy": DummyClassifier(strategy="prior"),
              "logistic": LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=SEED),
              "random_forest": RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=10,
                                      class_weight="balanced_subsample", n_jobs=2, random_state=SEED),
              "hist_gradient_boosting": HistGradientBoostingClassifier(loss="log_loss", max_iter=100,
                            max_leaf_nodes=15, l2_regularization=1.0, early_stopping=False, random_state=SEED)}
    return {k: Pipeline([("preprocess", clone(pre)), ("model", m)]) for k,m in models.items()}

def metrics(y, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    return {"accuracy": accuracy_score(y,pred), "error_rate": 1-accuracy_score(y,pred),
        "balanced_accuracy": balanced_accuracy_score(y,pred), "macro_f1": f1_score(y,pred,average="macro",zero_division=0),
        "positive_f1": f1_score(y,pred,zero_division=0), "precision": precision_score(y,pred,zero_division=0),
        "recall": recall_score(y,pred,zero_division=0), "roc_auc": roc_auc_score(y,prob),
        "average_precision": average_precision_score(y,prob), "log_loss": log_loss(y,prob),
        "confusion_matrix": confusion_matrix(y,pred,labels=[0,1]).tolist(), "threshold": float(threshold)}

def train_and_evaluate(splits):
    from threadpoolctl import threadpool_limits
    train, val, test = [splits[k] for k in ["train","validation","test"]]
    results, fitted, probs = {}, {}, {}
    (ROOT / "models").mkdir(exist_ok=True)
    for name, pipe in pipelines().items():
        start=time.perf_counter()
        with threadpool_limits(limits=2):
            pipe.fit(train[FEATURES], train.near_full)
            vp=pipe.predict_proba(val[FEATURES])[:,1]
            tp=pipe.predict_proba(test[FEATURES])[:,1]
        results[name]={"fit_and_predict_seconds":time.perf_counter()-start,
                       "validation":metrics(val.near_full,vp), "test":metrics(test.near_full,tp)}
        fitted[name],probs[name]=pipe,{"validation":vp,"test":tp}
        joblib.dump(pipe,ROOT / f"models/{name}.joblib",compress=3)
        print(name, 'validation macro F1:', round(results[name]['validation']['macro_f1'], 4),
              'test ROC AUC:', round(results[name]['test']['roc_auc'], 4), flush=True)
    winner=max((n for n in results if n!='dummy'), key=lambda n:results[n]['validation']['macro_f1'])
    thresholds=np.linspace(.05,.95,91)
    threshold=float(max(thresholds,key=lambda t:f1_score(val.near_full,probs[winner]['validation']>=t,zero_division=0)))
    write_json(ROOT / "reports/benchmark.json", {"config":CONFIG,"features":FEATURES,"results":results,
        "selected_model":winner,"selected_threshold":threshold,
        "selected_validation":metrics(val.near_full,probs[winner]['validation'],threshold),
        "selected_test":metrics(test.near_full,probs[winner]['test'],threshold),
        "environment":{"python":platform.python_version(),"sklearn":sklearn.__version__,"pandas":pd.__version__,"numpy":np.__version__},
        "code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    table=pd.DataFrame([{**{k:v for k,v in r['test'].items() if k!='confusion_matrix'},'model':n} for n,r in results.items()])
    table.to_csv(ROOT / "reports/test_metrics.csv", index=False)
    pred=test[["OccupancyDateTime","SourceElementKey","near_full"]].copy()
    pred['probability']=probs[winner]['test'];pred['prediction']=(pred.probability>=threshold).astype(int)
    pred.to_csv(ROOT / 'reports/test_predictions.csv.gz',index=False)
    # Three observed test examples: low, median and high predicted probability. These are descriptive, not independent validation.
    order=np.argsort(probs[winner]['test'])
    positions=[int(order[0]),int(order[len(order)//2]),int(order[-1])]
    cases=test.iloc[positions][['OccupancyDateTime','SourceElementKey']+FEATURES+['near_full']].copy()
    for name in fitted:
        cases[name+'_probability']=probs[name]['test'][positions]
    cases.to_csv(ROOT / 'reports/three_cases.csv',index=False)
    fig,axs=plt.subplots(1,2,figsize=(11,4.4))
    for name in fitted:
        fpr,tpr,_=roc_curve(test.near_full,probs[name]['test'])
        precision,recall,_=precision_recall_curve(test.near_full,probs[name]['test'])
        axs[0].plot(fpr,tpr,label=name.replace('_',' '))
        if name != 'dummy':
            axs[1].plot(recall,precision,label=name.replace('_',' '))
    axs[0].plot([0,1],[0,1],'k--',alpha=.3)
    axs[0].set(xlabel='False positive rate',ylabel='True positive rate',title='Later-date test: ROC')
    axs[1].axhline(test.near_full.mean(),color='k',linestyle='--',alpha=.5,label='No-skill prevalence')
    axs[1].set(xlabel='Recall',ylabel='Precision',title='Later-date test: precision and recall')
    axs[0].legend(fontsize=8);axs[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(ROOT / 'reports/evaluation.png',dpi=180);plt.close(fig)
    return results, winner

def stability(train):
    """Three expanding-window reviews within the training dates only."""
    from threadpoolctl import threadpool_limits
    dates=np.array(sorted(train.OccupancyDateTime.dt.date.unique()))
    chunks=np.array_split(dates,4)
    day=train.OccupancyDateTime.dt.date
    rows=[]
    for fold in range(1,4):
        tr=train[day.isin(np.concatenate(chunks[:fold]))]
        va=train[day.isin(chunks[fold][1:])] # purge first validation date
        if tr.near_full.nunique()<2 or va.near_full.nunique()<2:
            continue
        for name,pipe in pipelines().items():
            if name=='dummy':continue
            with threadpool_limits(limits=2):
                pipe.fit(tr[FEATURES],tr.near_full)
                pr=pipe.predict_proba(va[FEATURES])[:,1]
            rows.append({'fold':fold,'model':name,'train_rows':len(tr),'validation_rows':len(va),
                         'validation_start':str(va.OccupancyDateTime.min()),
                         **{k:v for k,v in metrics(va.near_full,pr).items() if k!='confusion_matrix'}})
    pd.DataFrame(rows).to_csv(ROOT/'reports/temporal_stability.csv',index=False)
    return rows

def main():
    warnings.filterwarnings('error',category=sklearn.exceptions.ConvergenceWarning)
    start=time.perf_counter()
    df=prepare(); plot_eda(df); splits=split_data(df)
    train_and_evaluate(splits);stability(splits['train'])
    write_json(ROOT/'reports/runtime.json',{'total_seconds':time.perf_counter()-start})
    print('Completed experiment; see reports/benchmark.json',flush=True)

if __name__=='__main__':main()
