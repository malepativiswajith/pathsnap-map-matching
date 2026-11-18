# Final Model Decision

## Choice

Use a `StandardScaler` plus Logistic Regression (`C=1`, `max_iter=300`, `lbfgs`) to score each candidate road segment. The Viterbi decoder then chooses a route using the candidate probabilities as emission scores and directed road-arc connectivity as transition scores.

## Why this model

The primary selection metric is observation-level Top-1: for each GPS observation, rank its candidate road segments and count whether the highest-scoring candidate matches the aligned route arc.

Five-fold GroupKFold kept entire trips together. Logistic Regression achieved 75.63% +/- 2.87 percentage points mean Top-1. Extra Trees was effectively tied at 75.61% +/- 2.61 points. Logistic Regression was selected because it has materially higher candidate-level macro F1 than Extra Trees (78.83% vs. 75.55%), lower grouped-CV log loss (0.263 vs. 0.363), and much lower inference time (about 0.0008 vs. 0.0073 ms per observation in the recorded environment).

Hist Gradient Boosting had the best candidate macro F1 (81.13%) and log loss (0.227), but lower observation Top-1 (73.94%). It is a reasonable alternative if candidate-level classification or probability calibration becomes the main objective.

## Five-model grouped-CV comparison

| Model | Observation Top-1 mean +/- std | Candidate macro F1 mean | Candidate log loss mean |
| --- | ---: | ---: | ---: |
| Logistic Regression | 75.63% +/- 2.87 pp | 0.7883 | 0.2628 |
| Extra Trees | 75.61% +/- 2.61 pp | 0.7555 | 0.3627 |
| Hist Gradient Boosting | 73.94% +/- 2.68 pp | 0.8113 | 0.2274 |
| Random Forest | 73.89% +/- 2.85 pp | 0.7877 | 0.2944 |
| Decision Tree | 70.42% +/- 3.40 pp | 0.7518 | 0.6472 |

## Hyperparameter search

Gaussian-process Bayesian optimization searched `C` over `10^-3` to `10^3`, using three grouped folds on training trips. The best reported setting was `C=1`, with 75.62% +/- 3.08 points mean observation Top-1. The tested values were nearly indistinguishable, so tuning did not demonstrate a measurable improvement over the default. The final model keeps `C=1` for a simple, reproducible choice.

## Student-test development evaluation

After selecting the model, it was refit using all 80 training trips and evaluated on the permitted 10-trip student-test partition:

- 76.33% Top-1 over all 18,846 route-labeled observations.
- 190 observations (1.01% of route-labeled observations) had no positive candidate road segment. These are counted as misses in the all-observation Top-1 score.
- Candidate-level accuracy was 89.24%, macro F1 was 0.7955, and log loss was 0.2706.
- Route alignment labeled 18,846 of 19,160 GPS observations (98.36%).

On the same trips, the current HMM baseline increased weighted Top-1 from 76.33% to 81.55% across all route-labeled observations. Counting candidate misses per trip, the macro mean over the 10 trips increased from 79.06% to 84.45%. The decoder's mean continuity score increased from 86.79% to 99.47% on decoded adjacent observation pairs. The HMM transition penalties are simple baseline values and have not been selected by a separate nested tuning experiment.

The student-test data was not used to fit the classifier or choose `C`. It is a development check, not an unbiased final benchmark. No evaluator-test coordinates or labels were used for any modeling decision.

## Training artifact and input controls

`models/logistic_candidate_model.joblib` contains the StandardScaler and Logistic Regression pipeline fitted on all 80 training trips. Training retained positive route candidates and sampled up to four negative candidates for each labeled observation. The seven model features are listed in `train_models.py`. Reference route arc, route-alignment distance, and label columns are excluded from model input.

## Limitations

Training labels come from a monotone spatial route-alignment heuristic with a 50 m distance threshold. The heuristic can miss observations or assign an imperfect route arc, especially around crossings, long gaps, and repeated road segments. Some training trips have large unlabeled portions. All reported scores measure agreement with the heuristic labels and should be described as provisional.
