# Experiment plan and research handoff

These records organize the Seattle parking project requirements, published experiment run and research handoff. Record date: October 23, 2026.

## Requirements mapped to Canvas

| Deliverable | Course requirement | Project evidence |
| --- | --- | --- |
| CP1 | Dataset unit, observations, coverage, cleaning, outcome and predictor analysis | Notebook data audit; source manifest; explicit sample limitations |
| CP2 | Justify a model, estimate test error, discuss fit, predict at least three cases | Logistic regression baseline; chronological split; recorded metrics; three observed test cases |
| CP3 | Compare alternative models, justify error estimation and compare predictions | Random Forest and histogram gradient boosting; majority baseline; consistent partitions and metrics |

Canvas assignment references: [CP1](https://clemson.instructure.com/courses/266220/assignments/2194699), [CP2](https://clemson.instructure.com/courses/266220/assignments/2194700), [CP3](https://clemson.instructure.com/courses/266220/assignments/2194701). The public notebook packages the modeling and evaluation workflow for study and reproduction.

## Milestones and dependencies

| Milestone | Depends on | Evidence / acceptance |
| --- | --- | --- |
| Recover requirements and source | Canvas and surviving coursework | Requirements table; original artifacts retained |
| Define target and data contract | Dataset schema and capacity semantics | Capacity-based target; explicit feature allowlist; sample checksum |
| Freeze evaluation design | Time coverage and both classes per split | Ordered date partitions; purged boundaries; validation-only selection |
| Run benchmark | Environment and processed sample | Four saved pipelines; benchmark JSON; three temporal reviews |
| Review limitations | Test metrics and source audit | Model card; missingness and false-positive/negative counts |
| Package handoff | Verified saved models | Notebook, dependency lock, predictions, manifests and runnable commands |

## Decision records

1. **Capacity-based label.** Replace the absolute count threshold of eight with 95% of stated capacity. The threshold is a declared modeling assumption, not a measured driver tolerance. Compare definitions in the audit.
2. **Bounded sampling.** Scan three source shards and sample uniformly per row with a fixed RNG seed. This makes CPU experimentation feasible but limits external validity. Do not imply the sample covers every season.
3. **No class resampling.** Preserve natural prevalence. Use balanced weights for logistic regression and forest trees, while comparing an unweighted boosting implementation.
4. **Time separation.** Use ordered whole dates, purge boundary dates and fit preprocessing only on training rows. Repeated locations remain a limitation. Random row splits could mix adjacent minutes.
5. **Small fixed search.** Compare three model families with declared parameters. Select by validation macro F1 at 0.5, then tune the selected model's threshold by validation positive F1. This has not established an optimal operational cost trade-off.
6. **CPU first.** Use two CPU threads for this sample. Do not reserve GPUs for tabular models that do not require them. Reassess compute only if expanded-data timing or a justified neural experiment creates a bottleneck.
7. **Evidence before claims.** Tie published metrics to their recorded target, sample and evaluation configuration. Do not report traffic reduction, production use, or validated driver impact.

## Risks and follow-up priorities

| Risk | Consequence | Next action |
| --- | --- | --- |
| Payment-derived occupancy differs from physical availability | The label may not match a driver's experience | Validate dataset semantics against SDOT documentation and actual occupancy |
| Three shards, sparse calendar coverage | Season and location bias | Add representative months and blockfaces before stronger generalization claims |
| About 48% of rates missing | Imputation may mask source drift | Inspect missingness by source/date and run a price-feature ablation |
| Repeated blockfaces | Results do not prove unseen-location generalization | Add grouped location holdout |
| Uncalibrated weighted model | Predicted probability may misstate risk | Add calibration with a separate validation allocation |
| False positives and false negatives lack measured costs | F1 threshold may be operationally unsuitable | Establish costs with stakeholders and predefine decision criteria |
| Contemporaneous inputs and labels | No future availability guarantee | Define a separate future-horizon target and use only information available at prediction time |

## Handoff requirements

Input schema is the allowlist in `experiment.FEATURES`. The selected joblib pipeline includes imputation, scaling, category encoding and model weights. Use the recorded threshold from `reports/benchmark.json`. Preserve Python/dependency versions and verify sample/model hashes before loading. Schema checks, calibration monitoring, a serving API and ongoing monitoring are future engineering work, not delivered production capabilities.
