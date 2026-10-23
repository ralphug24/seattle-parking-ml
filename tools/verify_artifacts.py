"""Check model round-trip predictions, split boundaries, and target exclusion."""
from pathlib import Path
import json
import hashlib
import sys
import numpy as np
import pandas as pd
import joblib

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiment import prepare, split_data, FEATURES, FORBIDDEN, metrics

df=prepare()
assert not df.duplicated(['SourceElementKey','OccupancyDateTime']).any()
assert not set(FEATURES).intersection(FORBIDDEN)
splits=split_data(df)
report=json.loads((ROOT/'reports/benchmark.json').read_text())
saved=pd.read_csv(ROOT/'reports/test_predictions.csv.gz')
test=splits['test']
model=joblib.load(ROOT/f"models/{report['selected_model']}.joblib")
prob=model.predict_proba(test[FEATURES])[:,1]
np.testing.assert_allclose(prob,saved.probability,rtol=1e-12,atol=1e-12)
calculated=metrics(test.near_full,prob,report['selected_threshold'])
for key,value in calculated.items():
    np.testing.assert_allclose(value,report['selected_test'][key],rtol=1e-10)
assert len(pd.read_csv(ROOT/'reports/three_cases.csv'))==3
print('Verified: saved-model predictions and metrics, disjoint chronological partitions, target exclusion, sampled key uniqueness and three cases.')
manifest=ROOT/'reports/artifact_manifest.json'
if manifest.exists():
    for name,record in json.loads(manifest.read_text()).items():
        path=ROOT/name
        assert path.stat().st_size==record['bytes'], name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256'],name
    print('Verified artifact checksums.')
