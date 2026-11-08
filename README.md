# Constraint-aware multi-module surrogates for longwall support assessment

This repository provides the code for predicting support demand, displacement fields and normal stress fields from parameterized FLAC3D simulations of layered coal-measure strata. The current source version is **2.0.0**. The complete data/model archive has been prepared and verified locally; **its new Zenodo version is awaiting publication**. [中文说明](README.zh-CN.md)

| Module | Frozen architecture | Output |
|---|---|---|
| Support demand | TabPFN-anchored CMCRNet, with four configurations and five outer folds | No-pass probability and finite demand decoded on the tested pressure grid |
| Displacement | 64-mode PCA and two-member MLP; step-5 coefficient range constraints (Guard64) | `ux_m`, `uz_m` |
| Normal stress | PCA with component-specific tree ensembles and step-specific field-combination weights | `sxx_MPa`, `syy_MPa`, `szz_MPa` |

The three modules have separate representations, regressors and objectives. The field modules are data-driven. The framework does not include a mechanics-residual training path or a validated automatic feedback loop from demand prediction to field prediction.

## Code and complete artifacts

GitHub contains source code, small examples, input/output contracts, environment records and verification results. The full datasets and 25 frozen model files are in the separate archive:

* File: `longwall-surrogate-v2.0.0_20261108.zip`
* Size: 3,048,576,124 bytes, approximately 3.05 GB
* SHA256: `b56acdc8f97d8e6e97a3813dab3b4a800ac34ce872946a6fcdb9b32c14dfb5e7`
* New-version DOI/download: **awaiting Zenodo publication**; see [archive metadata](metadata/archive.json).

The existing [Zenodo v1.0.0 record](https://zenodo.org/records/20839925) contains the earlier support-only study. Its DOI identifies those historical files. The [project concept DOI](https://doi.org/10.5281/zenodo.20839924) links versions; a fixed new-version DOI will be added when the complete v2 archive is published.

The complete archive contains:

| Data batch | Size and role |
|---|---|
| Support development | 960 combinations, 639 finite labels, 321 no-pass labels, 7,629 search records |
| Archived support evaluation | 64 combinations, 48 finite and 16 no-pass |
| Historical field development | 190 physical groups and 2,970 snapshots; fit/model-selection/internal-confirmation groups: 133/28/29 |
| Guard64 references | 21 groups, 53 pressure cases, 265 corrected large-strain development snapshots |
| New-case evaluation | All 50 planned snapshots retained with quality status; 41 qualified; six qualified step-5 snapshots from five physical groups |

Historical development, corrected references and new-case evaluation retain separate roles. The original research acceptance results, including failed conditions, are preserved in the archive. Software replay checks concern reproducibility of deposited predictions.

## Quick start

Use Python 3.13. The pinned environment was tested with Python 3.13.5 on 2026-11-08. It documents the replay environment, without inferring all historical training versions.

```bash
git clone https://github.com/ckennys/deep-longwall-support-pressure-surrogate.git
cd deep-longwall-support-pressure-surrogate
python -m pip install -r requirements.txt
```

After obtaining the complete ZIP, verify and install it under the ignored `artifacts/` directory:

```bash
python scripts/prepare_archive.py --archive /path/to/longwall-surrogate-v2.0.0_20261108.zip
python verify_files.py
python predict_support.py --dataset archived --output outputs/support.csv
python predict_fields.py --input examples/field_inputs_10.npz --output outputs/fields.npz
python evaluate.py --output-dir outputs/evaluation
python verify_replay.py --output outputs/replay.json
```

The installer checks the ZIP identity and all deposited file checksums. It leaves the repository source files separate from the extracted archive. If the complete archive is already extracted elsewhere, set `LONGWALL_SURROGATE_ARTIFACTS` to the directory containing its `data/`, `models/` and `SHA256SUMS.txt`. [Installation and data access](docs/data_access.md)

The supplied field example contains ten actual inputs and valid masks. Selected expected field values and eight support inputs/outputs are supplied in `examples/`. Frozen model inference requires the complete artifact archive. Lightweight contract checks can run without it:

```bash
python -m unittest discover -s tests -v
```

## Outputs and interpretation

Support input comprises eight channels. `q_top_MPa = 0.024 * (H_m - 85)` is derived from depth. The candidate pressures are `[0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.3, 2.6]` MPa. A no-pass probability of at least 0.5 denotes predicted no pass within this grid. Finite demand is clipped to the grid and mapped to the nearest tested pressure, with lower-pressure tie breaking. For predicted no-pass cases, finite demand and decoded pressure remain undefined. [Support inference](docs/support_inference.md)

The frozen student uses a teacher anchor. Historical teacher caches are deposited for verified replay. Arbitrary new support inputs require a separately supplied `teacher_anchor_MPa`; TabPFN foundation weights are obtained under their original licence and are not included.

Field input comprises 14 raw channels, including geometry, specified support pressure and step fraction. Displacement uses 39 additional engineered features. The reference section grid is 64 by 86 in C order. Displacement is in metres; normal stress is in MPa, with compression negative. Invalid output positions are NaN under the valid mask. Only step-5 displacement coefficients use Guard64; stress members reconstruct and combine fields at each step. [Field contract](docs/field_inference.md)

Historical field training sampled 1.7, 2.0, 2.3 and 2.6 MPa. Continuous or lower-pressure use requires additional validation. The MLP evaluates a continuous learned mapping, while tree ensembles evaluate their paths and leaf values.

## Reproducibility and source organization

* `longwall_surrogate/`: portable inference, exact archived feature/prediction implementations and artifact path resolution.
* `predict_support.py`, `predict_fields.py`: command-line inference.
* `evaluate.py`, `verify_replay.py`, `verify_files.py`: evaluation and frozen-prediction/checksum replay.
* `scripts/train_cmcrnet.py`: student training using the deposited historical teacher caches and original objectives; writes to a new `runs/` directory.
* `archived_code/`: historical training, selection and figure scripts, with historical workspace dependencies documented in their source.
* `metadata/`, `docs/validation/`: model specifications, contracts, artifact identity and actual verification reports.
* `legacy/v1.0.0/`: previous support-only repository contents, preserved with their original file identities.

The complete archive includes numerical source data for Figs. 5.1 to 5.7. Figure 5.7 is a held-group development diagnostic for g012, 2.3 MPa, step 5; it is separate from the new-case evaluation. [Reproduction scope and evidence](docs/reproducibility.md)

## Citation and licences

Use the specific published Zenodo version supporting the results being cited. The v2 archive DOI and actual creator metadata will be synchronized after publication. [Citation guidance](CITATION.md)

Project source code is MIT. Project-generated data, metadata, result tables, figure source data and trained surrogate artifacts use CC BY 4.0; see [file-level licences](LICENSES.md). Third-party runtime libraries, TabPFN foundation weights and FLAC3D retain their own licences. Native solver reruns require a separately licensed FLAC3D installation. No commercial solver executable or native binary save file is redistributed.
