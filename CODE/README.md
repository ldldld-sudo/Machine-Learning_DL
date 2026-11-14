# Activity Recognition from Smartphone Sensors

This repository processes phyphox accelerometer and gyroscope recordings and compares logistic regression with a random forest for five activities: SIT, STAND, WALK, UP and DOWN.

## 1. Repository contents

- `ML_Assignment1_Activity_Pipeline.ipynb`: data cleaning, feature extraction, validation, experiments and prediction workflow.
- `make_paper_figure.py`: creates the paper's across-person confusion-matrix figure from the notebook output.
- `Data/Dan_Data/`: personal phyphox ZIP recordings.
- `Data/Pool Data/S01/` through `Data/Pool Data/S04/`: anonymised pooled recordings.
- `results/`: generated CSV files, saved models and figures.

Keep the ZIP files in these folders and preserve their exported contents. The notebook expects to be run with this directory as the working directory.

## 2. Requirements

Use Python with the packages listed in `requirements.txt`. The notebook also requires Jupyter; install it if it is not already available:

```bash
python -m pip install -r requirements.txt
python -m pip install jupyter
```

Package versions are not pinned in `requirements.txt`, so small numerical or display differences may occur across environments.

## 3. Reproduce the analysis

From this `CODE` directory, launch the notebook:

```bash
jupyter lab ML_Assignment1_Activity_Pipeline.ipynb
```

Run all notebook cells from top to bottom. This reads the personal and pooled ZIP files, performs cleaning and feature extraction, creates the recording-level splits, tunes and evaluates the classifiers, and writes the result CSVs and `results/trained_models.joblib`.

## 4. Reproduce the paper tables and figures

The current paper contains two tables and two figures.

| Paper item | How to reproduce it |
|---|---|
| Table 1: same-person test metrics | Use `results/within_person_metrics.csv`. Include both models and both decision policies; round the metric columns as in the paper. The notebook displays summary metrics in Cell 93 and class-wise metrics in Cell 95. |
| Table 2: across-person results | Use `results/across_person_metrics.csv`, filtering to `feature_set == "axes_and_magnitudes"`. Report accuracy and macro F1 from `policy == "argmax"`; report cost per 1,000 windows from `policy == "minimum_expected_cost"`. S04 is the held-out same-person reference. |
| Figure 1: WALK_01 transition cleaning | Run notebook Cell 29. It displays two before/after plots; the first plot, for Dan's `WALK_01.zip`, is the paper figure. The cell displays the plot but does not save it as a PNG. |
| Figure 2: across-person confusion matrices | First run all notebook cells so `results/confusion_matrices.csv` exists. Then, from this directory, run the script below. |

```bash
python make_paper_figure.py
```

The script writes `results/figures/paper_across_person_confusions.png`. It reads the argmax confusion counts for S01–S04; S04 is the held-out same-person reference.

The feature-group comparison discussed in the paper can be checked using Cell 101's output. It is also reproducible from `results/across_person_metrics.csv` by selecting S01–S03 with `policy == "argmax"` and averaging accuracy by model and `feature_set`.

## 5. Generated result files

The notebook writes the following files under `results/`:

- `data_quality_log.csv`, `label_sanity_summary.csv` and `label_review_candidates.csv`: cleaning and label-screening outputs.
- `window_features.csv`: one row per generated window and its extracted features.
- `recording_splits.csv` and `self_identity_matches.csv`: split assignments and identity-match evidence.
- `validation_tuning_and_features.csv`: validation scores and selected configurations.
- `within_person_metrics.csv`, `across_person_metrics.csv` and `confusion_matrices.csv`: evaluation results.
- `trained_models.joblib`: fitted models used by the prediction function.
- `predictions.csv`: generated when predicting a new recording.

## 6. Feature and model notes

The notebook tests one-, two- and three-second windows with the overlap settings defined in `WINDOW_CONFIGS`, and compares axes, magnitudes, and combined feature groups. All model and feature choices are selected using validation macro F1; test data are used only for final evaluation. The notebook keeps all windows from a recording in the same split.

## 7. Predict a new recording

Run the notebook through Cell 105 so the models are fitted and `predict_recording` is defined. Then call this in a notebook cell:

```python
predict_recording("path/to/new_recording.zip")
```

The prediction function writes window-level predictions to `results/predictions.csv`.

## 8. AI-use statement

AI was used to assist with reviewing the analysis workflow, coding and editing documentation. The author checked the reported results against the notebook outputs and CSV files and is responsible for the submitted work.
