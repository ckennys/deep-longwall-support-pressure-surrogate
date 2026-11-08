# Supplementary Data v1

This package supports the manuscript on boundary-aware surrogate modelling for rapid screening of minimum equivalent support pressure in deep longwall mining.

## Scope

The package contains anonymised FLAC3D-derived labels, model evaluation tables, learning-curve data, stratified error analysis, field-pressure conversion summaries and Python scripts used to reproduce the main analytical tables and figures.

Raw field mine-pressure reports are not included because they contain site-specific operational information. Only anonymised statistics, extracted support-parameter status and conversion summaries are provided.

## Folder Structure

- `data/`: FLAC3D parameter combinations, required support-pressure labels and support-pressure search records.
- `tables/`: manuscript-ready supplementary tables and model evaluation summaries.
- `figures/`: key figures derived from the supplied tables.
- `scripts/`: Python scripts for model evaluation, learning curves, boundary classification, field conversion and stratified error analysis.
- `metadata/`: feature lists, pressure levels and repeated-validation settings.

## Key Data Files

- `data/SData1_FLAC3D_160_parameter_combinations_and_labels.csv`: 160 geological and mining-parameter combinations with finite and boundary labels.
- `data/SData2_controllable_regression_labels_139.csv`: 139 controllable cases used for finite-pressure regression.
- `data/SData3_boundary_labels_21.csv`: 21 boundary cases labelled as `>2.6 MPa`.
- `data/SData4_support_pressure_search_records_1229.csv`: 1229 FLAC3D support-pressure search records.

## Key Result Tables

- `tables/STable3_candidate_model_metrics.csv`: cross-validated candidate regression model metrics.
- `tables/STable4_repeated_stage1_classification_summary.csv`: repeated boundary-classification summary.
- `tables/STable5_repeated_stage2_regression_summary.csv`: repeated finite-pressure regression summary.
- `tables/STable6_learning_curve_regression_summary.csv`: regression learning-curve summary.
- `tables/STable7_learning_curve_classification_summary.csv`: boundary-classification learning-curve summary.
- `tables/STable8_stratified_error_by_pressure_level.csv`: MAE, RMSE, bias and level accuracy by support-pressure level.
- `tables/STable9_nearest_level_confusion_matrix.csv`: nearest-level confusion matrix for pressure-level prediction.
- `tables/STable10_field_support_statistics_anonymised.csv`: anonymised field support-column hydraulic-pressure statistics.
- `tables/STable11_field_surrogate_interval_validation.csv`: comparison between field-derived equivalent pressure and near-field surrogate prediction interval.
- `tables/STable12_field_conversion_parameter_status.csv`: status of support-pressure conversion parameters.

## Reproducibility Notes

The main cross-validation scripts use shuffled 5-fold splits with `random_state=42`. Learning-curve scripts use repeated random splits, with seed `42 + training_sample_size` for each sample-size setting. CatBoost, RandomForest, LogisticRegression, MLP and TabPFN calls receive the seed where supported.

The supplied scripts assume the project folder structure used in the manuscript workspace. When running outside this workspace, update input paths in the script headers.

Suggested reproduction order:

1. `scripts/run_advanced_tabular_models.py`
2. `scripts/run_exceedance_classifier.py`
3. `scripts/run_learning_curve_evidence.py`
4. `scripts/make_stratified_error_analysis.py`
5. `scripts/map_field_pressure_to_equiv_support.py`

## Field Data Boundary

Field monitoring files are excluded from this package. The included field tables contain derived statistics and source-status notes only. The field validation in the manuscript should be interpreted as parameter-constrained interval validation and engineering-scale validation.

## Version

Prepared from the local manuscript workspace on 2026-06-24.
