# Field input and reconstruction contract

The 14 raw features and units are listed in `metadata/feature_dictionary.csv`; the ordered names are also in `metadata/input_output_contract.json`. `q_top_MPa = 0.024*(H_m - 85)`. Pressure input is in MPa, length in metres, friction angle in degrees and step fraction is `k/5`, for `k = 1, 2, 3, 4, 5`.

Custom NPZ inputs need `features` with shape `(N,14)`, `feature_names`, and `valid_mask` with shape `(N,5504)` or `(N,64,86)`. The reference grid has 64 aligned vertical points and 86 horizontal points. Horizontal coordinates are `linspace(1,171,86)` m and aligned vertical coordinates are `linspace(0.5,99.5,64)` m. Flattening is C order. The displayed reference section is 172 m by 100 m.

```bash
python predict_fields.py --input examples/field_inputs_10.npz --output outputs/fields.npz
python predict_fields.py --targets ux_m uz_m --no-guard --output outputs/displacement_before_guard.npz
```

Displacement prediction preserves the original 14 plus 39 engineered features, frozen standardization, two-member `53 to 192 to 256 to 128 to 64` MLP with SiLU, coefficient scaling, 64-mode PCA and signed square-root inverse. Restored coefficients are averaged and only step 5 applies the stored per-mode bounds. Steps 1 to 4 do not clip coefficients. The complete bundles contain the PCA, MLP states, scales and bounds together.

Normal stress members use 14 or 29 features according to their saved configurations. Each member reconstructs its field before the fields are combined with the current step's stored nonnegative weights. Step-specific ET members occur in the yy and zz ensembles. Stress does not pass through Guard64. The member order, mode counts and weights are recorded in `metadata/model_specifications.json` and the frozen bundles.

The five outputs are `ux_m`, `uz_m`, `sxx_MPa`, `syy_MPa`, `szz_MPa`. Invalid positions are NaN and excluded from evaluation using the valid mask. Compression is negative. Shear stresses, mechanics residuals and solver certification are outside the current interface.

The historical field training pressures were 1.7, 2.0, 2.3 and 2.6 MPa. At an unsampled pressure, an MLP evaluates its learned continuous mapping, while a tree ensemble evaluates its split paths and leaf values. Such predictions need reference validation; a wrong pressure prediction is an input error and does not by itself prove that a sampled pressure is outside the training distribution. No automatic joint support/field decision rule is validated by placing the three modules in one repository.
