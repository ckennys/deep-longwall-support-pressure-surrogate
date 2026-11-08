# Support inference and output semantics

Eight inputs in order: `H_m`, `lambda`, `M_m`, `q_top_MPa`, `alpha_E_roof`, `alpha_c_roof`, `phi_roof_deg`, `alpha_t_roof`.

The student architecture is 8 to 64 to 64 to 32, with SiLU and a LayerNorm after the first activation. Separate scalar heads supply a bounded demand residual and a risk residual. The demand residual is added to the teacher anchor. Configurations using coupling include the positive demand-risk link. There are four configurations and five outer student folds for each.

```bash
python predict_support.py --dataset development --output outputs/support_oof.csv
python predict_support.py --dataset archived --output outputs/support_archived.csv
python predict_support.py --input examples/support_inputs_8.csv --output outputs/support_example.csv
```

Development replay uses each row's recorded outer fold and teacher anchor. Archived and custom inference average the five student demand values and probabilities. Custom inputs require a supplied `teacher_anchor_MPa`; the historical caches are used only for the deposited observations. Teacher foundation weights are not redistributed.

The tested pressure grid is `[0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.3, 2.6]` MPa. Predicted no-pass probability at least 0.5 denotes a no-pass prediction within this grid. The probability threshold is the equal-probability split, not an optimized misclassification-cost threshold. Finite demand is clipped to the pressure range and mapped to the nearest candidate, with ties resolved toward the lower pressure.

`clipped_demand_MPa` is diagnostic. For predicted no-pass rows, `finite_demand_MPa` and `decoded_pressure_level_MPa` are NaN. A diagnostic clip at 2.6 MPa does not establish passing support at that pressure.

The original support training code includes classification, finite regression, teacher anchoring, censoring, demand-risk consistency, ranking and local monotonicity. `scripts/train_cmcrnet.py` uses the deposited caches and writes new training outputs under `runs/`; replay of the frozen checkpoints does not retrain the model. Install `requirements-training-extra.txt` for optional training and plotting dependencies. The archive's original acceptance and evaluation scope is retained.
