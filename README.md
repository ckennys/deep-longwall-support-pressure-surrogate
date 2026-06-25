# Boundary-aware surrogate modelling for rapid screening of equivalent support pressure in deep longwall mining

This repository contains the anonymised supplementary data and source code required to reproduce the main machine-learning analyses, learning curves, stratified error analysis, field-pressure conversion and selected FLAC3D case-generation steps for the manuscript:

**Boundary-aware surrogate modelling for rapid screening of minimum equivalent support pressure in deep longwall mining**

## Repository Contents

- `data/`: anonymised FLAC3D-derived input combinations, support-pressure search records and labels.
- `tables/`: model metrics, repeated cross-validation summaries, learning-curve summaries, stratified error tables and field-conversion summaries.
- `figures/`: key manuscript and supplementary figures derived from the supplied tables.
- `source_code/ml/`: scripts for candidate model evaluation, boundary classification, learning curves and stratified error analysis.
- `source_code/flac3d/`: scripts for FLAC3D support-search case generation and result collection, plus one example `.dat` case file.
- `source_code/field_validation/`: scripts for field-pressure processing and equivalent-pressure conversion.
- `metadata/`: metadata and SHA256 file hashes.
- `manifest.csv`: file-level description and confidentiality status.

## Data Availability Boundary

Raw mine monitoring reports are not included because they contain site-specific operational information. This repository provides only anonymised statistics, derived pressure-conversion tables, parameter-status notes and processing scripts.

Large FLAC3D binary save files are excluded. The repository includes labels, support-pressure search summaries, scripts and a representative `.dat` example to support reproducibility without uploading large proprietary model states.

## Reproducibility

The main cross-validation scripts use fixed random seeds or explicit repeated random-split settings:

- Candidate regression model comparison: shuffled 5-fold cross-validation with `random_state=42`.
- Boundary classification: stratified 5-fold cross-validation with `random_state=42`.
- Learning curves: repeated random splits with seed `42 + training_sample_size`.

The scripts assume the folder structure used in this repository. Some optional models require additional packages such as CatBoost or TabPFN.

Suggested reproduction order:

1. `source_code/ml/run_advanced_tabular_models.py`
2. `source_code/ml/run_exceedance_classifier.py`
3. `source_code/ml/run_learning_curve_evidence.py`
4. `source_code/ml/make_stratified_error_analysis.py`
5. `source_code/field_validation/map_field_pressure_to_equiv_support.py`

## Citation

If this repository is used, please cite the associated manuscript and the archived Zenodo version:

https://doi.org/10.5281/zenodo.20839925

## License

Code is released under the MIT License. Data tables are released under the Creative Commons Attribution 4.0 International License, subject to the field-data confidentiality boundary described above.
