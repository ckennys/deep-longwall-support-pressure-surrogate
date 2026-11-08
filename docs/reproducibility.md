# Reproduction scope and actual checks

The original complete archive was verified on 2026-11-08. `docs/validation/archive_packaging_verification.json` records the archive identity, CRC and checksum checks, and numerical replay. It is a local verification record, not a claim of a GitHub-hosted workflow run.

Fifteen frozen-inference checks cover all 50 planned field snapshots, unguarded displacement and the unchanged steps 1 to 4, and all four support configurations on the 960 development and 64 archived combinations. Field values replayed from the original complete archive matched the deposited predictions exactly on valid positions. Support differences were within the recorded numerical tolerance. The subsequently extracted archive was also exercised from an unrelated current directory; changing batch size can introduce small floating-point differences.

The GitHub entry scripts use artifact-directory resolution. Their separate validation report is written after local checks against the complete archive, so code-path and artifact-path verification are distinguishable from the earlier archive-only checks.

## Data roles

* Historical support and field development retain their original numerical reference settings and split identities.
* The 265 corrected Guard64 reference snapshots are a development batch with physical-group held-out folds. They are not a new blind test.
* All 50 planned new-case snapshots are retained with their quality status; accuracy metrics use only the 41 qualified references.
* Step-5 before/after comparison uses six qualified pressure cases from five physical groups.
* Supplemental normal stress inference was computed after the native references with previously frozen weights. Displacement prediction freeze and this supplemental inference retain separate temporal records.

## Evaluation

```bash
python evaluate.py --output-dir outputs/evaluation
python verify_replay.py --output outputs/replay.json
```

The evaluation script derives pooled and per-step errors from qualified valid positions and emits per-snapshot statistics. Pooled R2 is not the mean snapshot R2. Units and masks are preserved. Original acceptance records remain unchanged: two of 17 original conditions failed. A successful software replay establishes agreement with deposited predictions and does not reverse the research acceptance result.

The complete archive maps numerical source files to Figs. 5.1 to 5.7. Figure 5.7's g012 development example uses held-group coefficient bounds. Final deployment model bounds use the final reference fit. The displayed unguarded divergence and subsequent reduction belong to that development diagnostic.

## Environment and historical source

`requirements-inference.txt` and `metadata/tested_environment.json` document the current tested replay configuration. Current installed TabPFN versions are not substituted for the historical teacher and relabeled as historically identical. Historical teacher caches support exact observation-level replay; arbitrary teacher inference remains a separately supplied dependency.

The portable student training copy uses deposited caches and keeps the original objectives and optimization choices. Archived model/figure runners can reference their original workspace organization and optional helpers; they are retained for provenance. The verified public paths are the portable inference, evaluation, artifact import and frozen replay scripts.

Commercial solver binaries and `.f3sav` states are excluded. Native re-simulation requires a separately licensed FLAC3D installation. Replaying the deposited predictions and evaluating against deposited reference fields does not call FLAC3D.
