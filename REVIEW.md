# Project review: four viewpoints, ten improvement rounds

A review of the program against the hackathon brief. Scoring in the brief: model quality 85 points
(classification, regression and clustering count equally), data profiling 10, speed 5 (credited only
when model performance is useful). 18 private datasets, 60 s each on one CPU environment, and a public
repository with pinned dependencies, the standard entry point, team names, and a README with the
assumptions.

## Findings and what was done

### Programming expert
| # | Finding | Status (verified) |
|---|---------|-------------------|
| P1 | No hard stop at 60 s: a library call slower than predicted could overrun | **Fixed:** watchdog thread ends the process 3 s before the limit. A simulated 120 s hang under a 20 s limit stopped at 17 s with valid outputs |
| P2 | Fallback predictions written only *after* profiling, so a profiling crash meant no output | **Fixed:** fallback is written first; profiling errors fall back to simple column roles |
| P3 | No top-level guard in `main.py` | **Fixed** |
| P4 | Unreadable line in `api._nonlinearity` | **Fixed** |
| P5 | Unused `clustering.assign_new` | **Removed** |
| P6 | Fixed 12 s safety margin made short time limits useless | **Fixed:** margin = min(12 s, 20% of the limit) |
| P7 | `?` missing markers (UCI style) turned numeric columns into categorical ones | **Fixed** (found by `tools/stress_test2.py`) |

### Application-architecture expert
| # | Finding | Status (verified) |
|---|---------|-------------------|
| A1 | `requirements.txt` **could not be installed on Python 3.10 or 3.14**, yet the README claimed 3.10 support | **Fixed:** environment markers; pip resolution checked for 3.10, 3.11, 3.12, 3.13 and 3.14; clean venv install passes the smoke test |
| A2 | Output ids were a row index even when the test table has an id column | **Fixed:** strict id detection (`Id`, `row_id`, `customerID`; not `paid` / `valid`). Row order is unchanged |
| A3 | No simple Python API | **Added** `automl.run(...)` |
| A4 | No quick self-test for a judge | **Added** `tools/smoke_test.py` (about 30 s) |
| A5 | Run environment not recorded | **Added** `environment` to `profile.json` |
| A6 | Thread counts used all machine cores even inside containers / CPU limits | **Fixed:** usable CPUs detected; pinned to 2 cores the heaviest test takes 41 s |

### AI / machine-learning engineer
| # | Finding | Status (measured) |
|---|---------|-------------------|
| M1 | No count-specific model for count targets (Bike Sharing) | **Added** Poisson boosting: bike-style RMSE 13.86 → 13.11 |
| M2 | Search always ran to the soft budget even when nothing improved | **Fixed:** stop after 3 non-improving candidates (5 under 1,000 rows). Classification 16.0 → 13.2 s mean, same quality |
| M3 | Datetime parts given to linear / distance models as plain numbers | **Added** cyclical sin/cos for hour, weekday, month |
| M4 | Irrelevant-feature detection (mutual information alone) missed interaction effects: 3 false alarms on Friedman, 3 on machine failure | **Fixed:** MI **or** permutation importance. Now exact on all three datasets with known truth |
| M5 | Clustering selection, target encoding, cheap-first search | Done in the previous version; verified again here |

### Hackathon judge: risks that depend on kickoff information
| # | Risk | What was done | Left for kickoff |
|---|------|---------------|------------------|
| J1 | **GitHub still holds version 1.** The brief freezes the submitted commit at the deadline | n/a (uploading is the owner's decision) | Upload this folder before the deadline |
| J2 | The input/output contract arrives at kickoff; our format is an assumption | **Mitigated:** `--predictions-file`, `--prediction-column`, `--output-id-column`, `--profile-file`, `--metric` change the format without code edits (tested) | Run the official checker on the 9 official practice sets |
| J3 | The brief says "Three students per team"; one member listed | Documented in the README assumptions | Confirm with the organisers |
| J4 | Scoring metric unknown | **Measured:** if graded by accuracy, macro-F1 tuning costs at most 0.021 (once +0.007); the reverse choice costs up to 0.041 F1. Macro-F1 stays the default | Set `--metric` once the scoring rules are public |
| J5 | Profile key names may differ from the checker's | **Mitigated:** a `summary` block under common names (`column_types`, `missing_values`, `class_distribution`, `variations`, ...) next to the detailed keys | Map to the official names if they differ |
| J6 | Clustering with a test table outputs test rows only | **Mitigated:** `predictions_all_rows.csv` also has labels for every row (tested: 50 test rows + 150 all rows) | Pick the file the contract wants |

## The ten rounds

| Round | Change | Result |
|---|---|---|
| 0 | Baseline rerun of every suite | 40/40 practice + 11/11 stress pass, no bugs |
| 1 | Watchdog, fallback before profiling, top-level guard, id detection, code clean-up | Hang test stops cleanly at limit − 3 s |
| 2 | Profile: `feature_types`, `missing_by_column`, `variations_detected`, `environment` | All 9 variations injected into the noisy multiclass set detected, no false ones |
| 3 | Early stop of the model search | −20% time on classification; patience tuned so small tables keep their ensembles |
| 4 | Poisson boosting, cyclical dates | Bike RMSE −5% |
| 5 | Real CPU detection; 2-core test | Every stress test under 60 s on 2 cores |
| 6 | `automl.run()` API, smoke test | Pass |
| 7 | Messy-encoding sweep (`stress_test2.py`) | Found and fixed the `?` marker bug; 6/6 pass |
| 8 | Irrelevant-feature detection with permutation importance | Exact on Friedman, machine failure, high-cardinality |
| 9 | Full retest | F1 0.883 → 0.887, R² 0.854 → 0.857, ARI 0.913, max 27 s, all suites pass |
| 10 | Clean-venv install, README / review update | Pass |
| 11 | Judge risks J2, J4, J5, J6: contract options, metric trade-off study, `summary` block, all-rows clustering file | Options tested; full retest below |

## Second set of ten rounds

Each round reread the brief, changed one thing, re-ran the program and kept the change only if the
measurements improved.

| Round | Change | Measured result |
|---|---|---|
| 1 | Profile: real count of training rows without a label (was always 0), missing values in the test table, duplicate columns detected even when stored as `1` vs `1.0` | All three verified on a built case with known answers |
| 2 | New size tests (`tools/stress_test3.py`). **Bug found: a 1,000-column table timed out** (fallback, F1 0.33). Fixed with wide-table screening (top 150 columns by Extra-Trees importance) and a time budget for the feature-relevance analysis; relevance model scaled for many classes | 1,000 columns: F1 **0.33 → 0.93**, 25 s; 20 classes: F1 0.72 → **0.90**, 45 s → 23 s |
| 3 | Full practice retest; a side-effect of round 2 on ≤ 5-class tables reverted | Noisy multiclass recovered |
| 4 | Per-phase `timings` in the profile; relevance model sized to the table; lazy imports | Relevance on small tables 4.7 s → 1.4 s |
| 5 | Second SVM / SVR strength and a local k-NN | Classification F1 0.885 → **0.888**, regression R² 0.857 → **0.866**, heavy-tailed target +0.071 |
| 6 | Tried 5 instead of 3 stability subsamples for clustering | No change on any dataset, so reverted. Clustering left as is (0.913, best possible 0.94) |
| 7 | All 22 robustness tests | 22/22 pass |
| 8 | Hard time budget for permutation importance; numeric duplicate hashing | Relevance ≤ 5 s (was 10 s on wide / many-class tables); 300k-row profile 7.4 s → 1.9 s |
| 9 | Full retest; found the strong SVM being cut off on a busy CPU | Diagnosed: CPU at 64–100% from other programs |
| 10 | Strong SVM without internal calibration above 2,500 rows (2.6× faster, same predictions); minimum 12 s search window when early phases were slow; Poisson model protected for count targets | Hard nonlinear 0.889 → **0.906**; bike counts kept at 0.988 under load |
| 10b | Final check on a heavily loaded laptop (free RAM 0.6–1.3 GB) exposed: watchdog started only after the heavy imports, no profile when stopped early, only the constant fallback when no model finished. Fixed: watchdog starts first, minimal profile written at start, a linear-model answer saved before the search | Simulated overload: saved answer R² 0.30 instead of −0.06. Final: F1 **0.888**, R² **0.866**, ARI 0.913, 40/40 practice, 17/17 robustness, smoke test pass |

Note on timings: the laptop used for testing was often at 60–100% CPU from other programs (browser,
Windows audio service), which slows every run. On the dedicated assessment machine, times are lower.
