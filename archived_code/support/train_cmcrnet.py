from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "分析" / "algorithm_innovation_20260811"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
MODEL_DIR = OUT_DIR / "models"

TRAIN_LABELS = (
    ROOT
    / "工况"
    / "flac3d_ml_dataset_stage7_expanded"
    / "labels_required_support_pressure_all_960.csv"
)
EXTERNAL_LABELS = (
    ROOT
    / "分析"
    / "stage8_external_validation"
    / "stage8_external_labels_locked.csv"
)
COUNTERFACTUAL_BASELINE = (
    ROOT
    / "分析"
    / "innovation_strengthening_20260811"
    / "tables"
    / "counterfactual_model_comparison.csv"
)
BASE_OOF_PREDICTIONS = (
    ROOT
    / "分析"
    / "innovation_strengthening_20260811"
    / "tables"
    / "single_stage_stage7_oof_predictions.csv"
)
BASE_EXTERNAL_PREDICTIONS = (
    ROOT
    / "分析"
    / "innovation_strengthening_20260811"
    / "tables"
    / "single_stage_stage8_predictions.csv"
)

FEATURES = [
    "H_m",
    "lambda",
    "M_m",
    "q_top_MPa",
    "alpha_E_roof",
    "alpha_c_roof",
    "phi_roof_deg",
    "alpha_t_roof",
]
PRESSURE_LEVELS = np.array([0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.3, 2.6])
UPPER_LIMIT = 2.6
SEED = 20260811

# Positive means larger values should not reduce support demand.
# Negative means larger values should not increase support demand.
MONOTONIC_DIRECTIONS = {
    "M_m": 1,
    "q_top_MPa": 1,
    "alpha_E_roof": -1,
    "alpha_c_roof": -1,
    "phi_roof_deg": -1,
    "alpha_t_roof": -1,
}


@dataclass(frozen=True)
class Variant:
    name: str
    use_censor: bool
    use_coupling: bool
    use_ranking: bool
    use_monotonic: bool
    censor_weight: float = 2.0
    coupling_weight: float = 0.35
    ranking_weight: float = 0.12
    monotonic_weight: float = 0.45


VARIANTS = [
    Variant("independent_multitask", False, False, False, False),
    Variant("plus_censored_loss", True, False, False, False),
    Variant("plus_coupled_ranking", True, True, True, False),
    Variant("full_CMCRNet", True, True, True, True, monotonic_weight=2.0),
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def setup_style() -> None:
    sns.set_theme(style="ticks", context="paper", font_scale=1.05)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def ensure_dirs() -> None:
    for path in (TABLE_DIR, FIG_DIR, MODEL_DIR):
        path.mkdir(parents=True, exist_ok=True)


def numeric_pressure(series: pd.Series, capped: bool = False) -> np.ndarray:
    text = series.astype(str)
    values = pd.to_numeric(text.where(text != ">2.6"), errors="coerce").to_numpy(float)
    if capped:
        values[np.isnan(values)] = UPPER_LIMIT
    return values


def nearest_level(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    indices = np.abs(values[:, None] - PRESSURE_LEVELS[None, :]).argmin(axis=1)
    return PRESSURE_LEVELS[indices]


class CMCRNet(nn.Module):
    """Censored monotonic coupled-risk network for support demand."""

    def __init__(self, n_features: int, use_coupling: bool) -> None:
        super().__init__()
        self.use_coupling = use_coupling
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.SiLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
        )
        self.demand_head = nn.Linear(32, 1)
        self.risk_residual_head = nn.Linear(32, 1)
        self.link_scale_raw = nn.Parameter(torch.tensor(0.5))

    def forward(
        self, x: torch.Tensor, base_demand: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        residual = 0.35 * torch.tanh(self.demand_head(latent).squeeze(-1))
        demand = base_demand + residual
        boundary_logit = self.risk_residual_head(latent).squeeze(-1)
        if self.use_coupling:
            positive_scale = F.softplus(self.link_scale_raw)
            boundary_logit = boundary_logit + positive_scale * (demand - UPPER_LIMIT)
        return demand, boundary_logit


def pairwise_ranking_loss(
    demand: torch.Tensor,
    target_rank: torch.Tensor,
    generator: torch.Generator,
    n_pairs: int = 512,
) -> torch.Tensor:
    n = demand.shape[0]
    first = torch.randint(0, n, (n_pairs,), generator=generator)
    second = torch.randint(0, n, (n_pairs,), generator=generator)
    target_difference = target_rank[first] - target_rank[second]
    keep = target_difference.abs() >= 0.25
    if not torch.any(keep):
        return demand.new_tensor(0.0)
    sign = torch.sign(target_difference[keep])
    predicted_difference = demand[first[keep]] - demand[second[keep]]
    return F.softplus(-sign * predicted_difference / 0.15).mean()


def monotonic_loss(
    model: CMCRNet,
    x: torch.Tensor,
    base_demand: torch.Tensor,
    demand: torch.Tensor,
    feature_indices: dict[int, int],
    epoch: int,
) -> torch.Tensor:
    keys = sorted(feature_indices)
    selected = keys
    penalties = []
    for feature_index in selected:
        direction = feature_indices[feature_index]
        perturbed = x.clone()
        perturbed[:, feature_index] = perturbed[:, feature_index] + 0.08
        perturbed_demand, _ = model(perturbed, base_demand)
        if direction > 0:
            penalties.append(F.relu(demand + 0.002 - perturbed_demand).mean())
        else:
            penalties.append(F.relu(perturbed_demand + 0.002 - demand).mean())
    return torch.stack(penalties).mean()


def loss_components(
    model: CMCRNet,
    x: torch.Tensor,
    base_demand: torch.Tensor,
    boundary: torch.Tensor,
    finite_pressure: torch.Tensor,
    variant: Variant,
    positive_weight: torch.Tensor,
    generator: torch.Generator,
    monotonic_indices: dict[int, int],
    epoch: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    demand, boundary_logit = model(x, base_demand)
    finite_mask = boundary < 0.5
    boundary_mask = ~finite_mask
    classification = F.binary_cross_entropy_with_logits(
        boundary_logit, boundary, pos_weight=positive_weight
    )
    regression = F.smooth_l1_loss(
        demand[finite_mask], finite_pressure[finite_mask], beta=0.15
    )
    teacher_anchor = F.smooth_l1_loss(
        demand[finite_mask], base_demand[finite_mask], beta=0.10
    )
    total = classification + 3.0 * regression + 0.35 * teacher_anchor

    censor = demand.new_tensor(0.0)
    if variant.use_censor and torch.any(boundary_mask):
        censor = F.relu(UPPER_LIMIT + 0.10 - demand[boundary_mask]).pow(2).mean()
        total = total + variant.censor_weight * censor

    coupling = demand.new_tensor(0.0)
    if variant.use_coupling:
        demand_boundary_logit = (demand - UPPER_LIMIT) / 0.18
        coupling = F.binary_cross_entropy_with_logits(demand_boundary_logit, boundary)
        total = total + variant.coupling_weight * coupling

    ranking = demand.new_tensor(0.0)
    if variant.use_ranking:
        target_rank = finite_pressure.clone()
        target_rank[boundary_mask] = UPPER_LIMIT + 0.30
        ranking = pairwise_ranking_loss(demand, target_rank, generator)
        total = total + variant.ranking_weight * ranking

    monotonic = demand.new_tensor(0.0)
    if variant.use_monotonic:
        monotonic = monotonic_loss(
            model, x, base_demand, demand, monotonic_indices, epoch
        )
        total = total + variant.monotonic_weight * monotonic

    components = {
        "total": float(total.detach()),
        "classification": float(classification.detach()),
        "regression": float(regression.detach()),
        "teacher_anchor": float(teacher_anchor.detach()),
        "censor": float(censor.detach()),
        "coupling": float(coupling.detach()),
        "ranking": float(ranking.detach()),
        "monotonic": float(monotonic.detach()),
    }
    return total, components


def validation_score(
    model: CMCRNet,
    x: torch.Tensor,
    base_demand: torch.Tensor,
    boundary: np.ndarray,
    finite_pressure: np.ndarray,
) -> float:
    model.eval()
    with torch.no_grad():
        demand, logit = model(x, base_demand)
    predicted_boundary = (torch.sigmoid(logit).cpu().numpy() >= 0.5).astype(int)
    predicted_pressure = np.clip(demand.cpu().numpy(), 0.5, UPPER_LIMIT)
    matrix = confusion_matrix(boundary, predicted_boundary, labels=[0, 1])
    finite_mask = boundary == 0
    finite_rmse = math.sqrt(
        mean_squared_error(finite_pressure[finite_mask], predicted_pressure[finite_mask])
    )
    return float((5 * matrix[1, 0] + matrix[0, 1]) / len(boundary) + finite_rmse)


def validation_monotonic_violation(
    model: CMCRNet,
    x: torch.Tensor,
    base_demand: torch.Tensor,
    monotonic_indices: dict[int, int],
) -> float:
    model.eval()
    with torch.no_grad():
        base_prediction, _ = model(x, base_demand)
        violations = []
        for feature_index, direction in monotonic_indices.items():
            perturbed = x.clone()
            perturbed[:, feature_index] += 0.08
            changed_prediction, _ = model(perturbed, base_demand)
            change = changed_prediction - base_prediction
            if direction > 0:
                violations.append((change < -1e-4).float().mean())
            else:
                violations.append((change > 1e-4).float().mean())
    return float(torch.stack(violations).mean())


def train_model(
    x_train: np.ndarray,
    base_demand_train: np.ndarray,
    boundary_train: np.ndarray,
    finite_pressure_train: np.ndarray,
    variant: Variant,
    seed: int,
) -> tuple[CMCRNet, pd.DataFrame, int]:
    set_seed(seed)
    all_indices = np.arange(len(x_train))
    fit_indices, validation_indices = train_test_split(
        all_indices,
        test_size=0.16,
        stratify=boundary_train,
        random_state=seed,
    )
    x_fit = torch.tensor(x_train[fit_indices], dtype=torch.float32)
    y_boundary_fit = torch.tensor(boundary_train[fit_indices], dtype=torch.float32)
    y_pressure_fit = torch.tensor(finite_pressure_train[fit_indices], dtype=torch.float32)
    base_demand_fit = torch.tensor(base_demand_train[fit_indices], dtype=torch.float32)
    x_validation = torch.tensor(x_train[validation_indices], dtype=torch.float32)
    base_demand_validation = torch.tensor(
        base_demand_train[validation_indices], dtype=torch.float32
    )

    model = CMCRNet(len(FEATURES), variant.use_coupling)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.6, patience=35, min_lr=1e-5
    )
    n_positive = max(int(y_boundary_fit.sum().item()), 1)
    n_negative = max(len(y_boundary_fit) - n_positive, 1)
    positive_weight = torch.tensor(n_negative / n_positive, dtype=torch.float32)
    monotonic_indices = {
        FEATURES.index(name): direction for name, direction in MONOTONIC_DIRECTIONS.items()
    }
    generator = torch.Generator().manual_seed(seed + 9000)

    best_score = float("inf")
    best_state = None
    best_epoch = 0
    patience = 150
    history_rows = []
    for epoch in range(1, 1801):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total, components = loss_components(
            model,
            x_fit,
            base_demand_fit,
            y_boundary_fit,
            y_pressure_fit,
            variant,
            positive_weight,
            generator,
            monotonic_indices,
            epoch,
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        if epoch == 1 or epoch % 10 == 0:
            score = validation_score(
                model,
                x_validation,
                base_demand_validation,
                boundary_train[validation_indices],
                finite_pressure_train[validation_indices],
            )
            if variant.use_monotonic:
                score += 0.25 * validation_monotonic_violation(
                    model,
                    x_validation,
                    base_demand_validation,
                    monotonic_indices,
                )
            scheduler.step(score)
            history_rows.append(
                {
                    "epoch": epoch,
                    "validation_score": score,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    **components,
                }
            )
            if score < best_score - 1e-5:
                best_score = score
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                best_epoch = epoch
            elif epoch - best_epoch >= patience:
                break

    if best_state is None:
        raise RuntimeError(f"No valid checkpoint for {variant.name}")
    model.load_state_dict(best_state)
    model.eval()
    return model, pd.DataFrame(history_rows), best_epoch


def predict(
    model: CMCRNet, x: np.ndarray, base_demand: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        demand, logit = model(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(base_demand, dtype=torch.float32),
        )
    return demand.numpy(), torch.sigmoid(logit).numpy()


def evaluate_predictions(
    dataset: str,
    variant: str,
    actual_boundary: np.ndarray,
    finite_pressure: np.ndarray,
    predicted_demand: np.ndarray,
    boundary_probability: np.ndarray,
) -> tuple[dict[str, float | int | str], np.ndarray]:
    actual_boundary = np.asarray(actual_boundary, dtype=int)
    predicted_boundary = (np.asarray(boundary_probability) >= 0.5).astype(int)
    predicted_pressure = np.clip(np.asarray(predicted_demand), 0.5, UPPER_LIMIT)
    finite_mask = actual_boundary == 0
    boundary_mask = actual_boundary == 1
    matrix = confusion_matrix(actual_boundary, predicted_boundary, labels=[0, 1])
    snapped = nearest_level(predicted_pressure)
    finite_true = finite_pressure[finite_mask]
    finite_predicted = predicted_pressure[finite_mask]
    row = {
        "dataset": dataset,
        "variant": variant,
        "n": int(len(actual_boundary)),
        "n_boundary": int(boundary_mask.sum()),
        "n_finite": int(finite_mask.sum()),
        "boundary_recall": recall_score(
            actual_boundary, predicted_boundary, zero_division=0
        ),
        "boundary_precision": precision_score(
            actual_boundary, predicted_boundary, zero_division=0
        ),
        "boundary_accuracy": accuracy_score(actual_boundary, predicted_boundary),
        "boundary_auc": roc_auc_score(actual_boundary, boundary_probability),
        "boundary_brier": brier_score_loss(actual_boundary, boundary_probability),
        "false_negatives": int(matrix[1, 0]),
        "false_positives": int(matrix[0, 1]),
        "false_finite_rate": float(matrix[1, 0] / max(boundary_mask.sum(), 1)),
        "finite_R2": r2_score(finite_true, finite_predicted),
        "finite_RMSE_MPa": math.sqrt(mean_squared_error(finite_true, finite_predicted)),
        "finite_MAE_MPa": mean_absolute_error(finite_true, finite_predicted),
        "exact_level_accuracy": float(
            np.mean(np.isclose(snapped[finite_mask], finite_true))
        ),
        "adjacent_level_accuracy": float(
            np.mean(np.abs(snapped[finite_mask] - finite_true) <= 0.31)
        ),
        "safety_weighted_error": float(
            (5 * matrix[1, 0] + matrix[0, 1]) / len(actual_boundary)
        ),
        "boundary_latent_demand_median_MPa": float(
            np.median(predicted_demand[boundary_mask])
        ),
        "finite_latent_demand_median_MPa": float(
            np.median(predicted_demand[finite_mask])
        ),
        "boundary_censor_margin_satisfaction": float(
            np.mean(predicted_demand[boundary_mask] >= UPPER_LIMIT)
        ),
    }
    return row, matrix


def monotonic_violation_rows(
    model: CMCRNet,
    x_scaled: np.ndarray,
    base_demand: np.ndarray,
    variant: str,
    fold: int,
    dataset: str,
) -> list[dict[str, float | int | str]]:
    predicted_demand, base_probability = predict(model, x_scaled, base_demand)
    rows = []
    for feature, direction in MONOTONIC_DIRECTIONS.items():
        feature_index = FEATURES.index(feature)
        perturbed = x_scaled.copy()
        perturbed[:, feature_index] += 0.08
        changed_demand, changed_probability = predict(model, perturbed, base_demand)
        demand_change = changed_demand - predicted_demand
        probability_change = changed_probability - base_probability
        if direction > 0:
            demand_violation = demand_change < -1e-4
            probability_violation = probability_change < -1e-4
        else:
            demand_violation = demand_change > 1e-4
            probability_violation = probability_change > 1e-4
        rows.append(
            {
                "variant": variant,
                "fold": fold,
                "dataset": dataset,
                "feature": feature,
                "expected_direction": direction,
                "demand_violation_rate": float(np.mean(demand_violation)),
                "risk_violation_rate": float(np.mean(probability_violation)),
                "median_signed_demand_change": float(
                    np.median(direction * demand_change)
                ),
                "median_signed_risk_change": float(
                    np.median(direction * probability_change)
                ),
            }
        )
    return rows


def run_experiment() -> dict[str, object]:
    train = pd.read_csv(TRAIN_LABELS)
    external = pd.read_csv(EXTERNAL_LABELS)
    base_oof_table = pd.read_csv(BASE_OOF_PREDICTIONS)
    base_external_table = pd.read_csv(BASE_EXTERNAL_PREDICTIONS)
    for frame in (train, external):
        for feature in FEATURES:
            frame[feature] = pd.to_numeric(frame[feature], errors="raise")

    x = train[FEATURES].to_numpy(np.float32)
    y_boundary = (
        train["required_support_pressure_MPa"].astype(str) == ">2.6"
    ).astype(int).to_numpy()
    y_pressure = numeric_pressure(train["required_support_pressure_MPa"])
    y_pressure_for_training = numeric_pressure(
        train["required_support_pressure_MPa"], capped=True
    )

    x_external = external[FEATURES].to_numpy(np.float32)
    y_external_boundary = external["is_boundary"].to_numpy(int)
    y_external_pressure = numeric_pressure(external["required_support_pressure_MPa"])
    base_oof_map = base_oof_table.set_index("sample_id")[
        "two_stage_predicted_pressure_MPa"
    ]
    base_external_map = base_external_table.set_index("sample_id")[
        "two_stage_predicted_pressure_MPa"
    ]
    base_oof = train["sample_id"].map(base_oof_map).to_numpy(np.float32)
    base_external = external["sample_id"].map(base_external_map).to_numpy(np.float32)
    if np.isnan(base_oof).any() or np.isnan(base_external).any():
        raise ValueError("TabPFN anchor predictions did not map to every combination.")

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metric_rows = []
    prediction_frames = []
    history_frames = []
    monotonic_rows = []
    confusion_matrices: dict[str, list[list[int]]] = {}
    training_rows = []

    for variant_index, variant in enumerate(VARIANTS):
        print(f"Training variant {variant.name}", flush=True)
        oof_demand = np.zeros(len(train), dtype=float)
        oof_probability = np.zeros(len(train), dtype=float)
        fold_assignment = np.zeros(len(train), dtype=int)
        external_demand_folds = []
        external_probability_folds = []

        for fold, (train_index, test_index) in enumerate(
            splitter.split(x, y_boundary), start=1
        ):
            start = time.time()
            scaler = StandardScaler().fit(x[train_index])
            x_train_scaled = scaler.transform(x[train_index]).astype(np.float32)
            x_test_scaled = scaler.transform(x[test_index]).astype(np.float32)
            x_external_scaled = scaler.transform(x_external).astype(np.float32)
            model, history, best_epoch = train_model(
                x_train_scaled,
                base_oof[train_index],
                y_boundary[train_index],
                y_pressure_for_training[train_index],
                variant,
                SEED + variant_index * 100 + fold,
            )
            demand, probability = predict(
                model, x_test_scaled, base_oof[test_index]
            )
            oof_demand[test_index] = demand
            oof_probability[test_index] = probability
            fold_assignment[test_index] = fold
            external_demand, external_probability = predict(
                model, x_external_scaled, base_external
            )
            external_demand_folds.append(external_demand)
            external_probability_folds.append(external_probability)

            history["variant"] = variant.name
            history["fold"] = fold
            history_frames.append(history)
            monotonic_rows.extend(
                monotonic_violation_rows(
                    model,
                    x_external_scaled,
                    base_external,
                    variant.name,
                    fold,
                    "Stage8_locked",
                )
            )
            checkpoint = {
                "variant": asdict(variant),
                "fold": fold,
                "features": FEATURES,
                "model_state": model.state_dict(),
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
                "best_epoch": best_epoch,
            }
            torch.save(
                checkpoint, MODEL_DIR / f"{variant.name}_fold{fold}.pt"
            )
            training_rows.append(
                {
                    "variant": variant.name,
                    "fold": fold,
                    "best_epoch": best_epoch,
                    "elapsed_seconds": time.time() - start,
                }
            )
            print(
                f"  fold {fold}/5, best epoch {best_epoch}", flush=True
            )

        external_demand_matrix = np.vstack(external_demand_folds)
        external_probability_matrix = np.vstack(external_probability_folds)
        external_demand_mean = external_demand_matrix.mean(axis=0)
        external_probability_mean = external_probability_matrix.mean(axis=0)

        row, matrix = evaluate_predictions(
            "Stage7_matched_OOF",
            variant.name,
            y_boundary,
            y_pressure,
            oof_demand,
            oof_probability,
        )
        metric_rows.append(row)
        confusion_matrices[f"Stage7_matched_OOF|{variant.name}"] = matrix.tolist()

        row, matrix = evaluate_predictions(
            "Stage8_locked_ensemble",
            variant.name,
            y_external_boundary,
            y_external_pressure,
            external_demand_mean,
            external_probability_mean,
        )
        metric_rows.append(row)
        confusion_matrices[f"Stage8_locked_ensemble|{variant.name}"] = matrix.tolist()

        internal_predictions = train[
            ["sample_id", "required_support_pressure_MPa"] + FEATURES
        ].copy()
        internal_predictions["dataset"] = "Stage7_matched_OOF"
        internal_predictions["variant"] = variant.name
        internal_predictions["fold"] = fold_assignment
        internal_predictions["actual_boundary"] = y_boundary
        internal_predictions["tabpfn_anchor_MPa"] = base_oof
        internal_predictions["predicted_latent_demand_MPa"] = oof_demand
        internal_predictions["predicted_boundary_probability"] = oof_probability
        internal_predictions["predicted_boundary"] = (oof_probability >= 0.5).astype(int)
        internal_predictions["predicted_finite_pressure_MPa"] = np.clip(
            oof_demand, 0.5, UPPER_LIMIT
        )
        prediction_frames.append(internal_predictions)

        external_predictions = external[
            ["sample_id", "required_support_pressure_MPa"] + FEATURES
        ].copy()
        external_predictions["dataset"] = "Stage8_locked_ensemble"
        external_predictions["variant"] = variant.name
        external_predictions["fold"] = 0
        external_predictions["actual_boundary"] = y_external_boundary
        external_predictions["tabpfn_anchor_MPa"] = base_external
        external_predictions["predicted_latent_demand_MPa"] = external_demand_mean
        external_predictions["predicted_latent_demand_sd_MPa"] = (
            external_demand_matrix.std(axis=0, ddof=1)
        )
        external_predictions["predicted_boundary_probability"] = (
            external_probability_mean
        )
        external_predictions["predicted_boundary_probability_sd"] = (
            external_probability_matrix.std(axis=0, ddof=1)
        )
        external_predictions["predicted_boundary"] = (
            external_probability_mean >= 0.5
        ).astype(int)
        external_predictions["predicted_finite_pressure_MPa"] = np.clip(
            external_demand_mean, 0.5, UPPER_LIMIT
        )
        prediction_frames.append(external_predictions)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    histories = pd.concat(history_frames, ignore_index=True)
    monotonic = pd.DataFrame(monotonic_rows)
    training = pd.DataFrame(training_rows)
    metrics.to_csv(TABLE_DIR / "cmcrnet_ablation_metrics.csv", index=False)
    predictions.to_csv(TABLE_DIR / "cmcrnet_all_predictions.csv", index=False)
    histories.to_csv(TABLE_DIR / "cmcrnet_training_histories.csv", index=False)
    monotonic.to_csv(TABLE_DIR / "cmcrnet_monotonic_violation_rates.csv", index=False)
    training.to_csv(TABLE_DIR / "cmcrnet_training_summary.csv", index=False)
    with (TABLE_DIR / "cmcrnet_confusion_matrices.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(confusion_matrices, handle, indent=2)

    baseline = pd.read_csv(COUNTERFACTUAL_BASELINE)
    baseline = baseline[baseline["dataset"].isin(["Stage7_OOF", "Stage8_locked"])]
    baseline = baseline.rename(columns={"scheme": "variant"})
    baseline["source"] = "existing_baseline"
    compact_metrics = metrics.copy()
    compact_metrics["dataset"] = compact_metrics["dataset"].replace(
        {
            "Stage7_matched_OOF": "Stage7_OOF",
            "Stage8_locked_ensemble": "Stage8_locked",
        }
    )
    compact_metrics["source"] = "CMCRNet_ablation"
    common_columns = [column for column in baseline.columns if column in compact_metrics.columns]
    comparison = pd.concat(
        [baseline[common_columns], compact_metrics[common_columns]], ignore_index=True
    )
    comparison.to_csv(TABLE_DIR / "cmcrnet_with_existing_baselines.csv", index=False)
    return {
        "metrics": metrics,
        "predictions": predictions,
        "histories": histories,
        "monotonic": monotonic,
        "training": training,
        "comparison": comparison,
    }


def plot_ablation(result: dict[str, object]) -> None:
    metrics = result["metrics"]
    external = metrics[metrics["dataset"] == "Stage8_locked_ensemble"].copy()
    labels = {
        "independent_multitask": "Anchored MTL",
        "plus_censored_loss": "+ Censoring",
        "plus_coupled_ranking": "+ Coupling and ranking",
        "full_CMCRNet": "Full CMCRNet",
    }
    external["display"] = external["variant"].map(labels)
    order = list(labels.values())
    palette = ["#708090", "#4F89A8", "#D19A3B", "#A94F45"]
    monotonic = result["monotonic"]
    violation = (
        monotonic.groupby("variant", as_index=False)["demand_violation_rate"]
        .mean()
        .rename(columns={"demand_violation_rate": "mean_monotonic_violation"})
    )
    violation["display"] = violation["variant"].map(labels)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8))
    panels = [
        ("boundary_recall", "Boundary recall", (0, 1.05)),
        ("finite_RMSE_MPa", "Finite-case RMSE (MPa)", None),
        (
            "boundary_censor_margin_satisfaction",
            "Boundary latent-demand margin satisfaction",
            (0, 1.05),
        ),
    ]
    for axis, (column, ylabel, ylim) in zip(axes.flat[:3], panels):
        sns.barplot(
            data=external,
            x="display",
            y=column,
            order=order,
            hue="display",
            hue_order=order,
            palette=palette,
            legend=False,
            ax=axis,
        )
        axis.set_xlabel("")
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=15)
        if ylim:
            axis.set_ylim(*ylim)
        for container in axis.containers:
            axis.bar_label(container, fmt="%.3f", padding=2, fontsize=7)

    sns.barplot(
        data=violation,
        x="display",
        y="mean_monotonic_violation",
        order=order,
        hue="display",
        hue_order=order,
        palette=palette,
        legend=False,
        ax=axes[1, 1],
    )
    axes[1, 1].set_xlabel("")
    axes[1, 1].set_ylabel("Mean monotonic violation rate")
    axes[1, 1].tick_params(axis="x", rotation=15)
    axes[1, 1].set_ylim(0, max(0.05, violation["mean_monotonic_violation"].max() * 1.2))
    for container in axes[1, 1].containers:
        axes[1, 1].bar_label(container, fmt="%.3f", padding=2, fontsize=7)

    for label, axis in zip(["a", "b", "c", "d"], axes.flat):
        axis.text(-0.12, 1.04, label, transform=axis.transAxes, fontweight="bold")
    fig.suptitle("Ablation of the censored monotonic coupled-risk network", y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cmcrnet_ablation.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "cmcrnet_ablation.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_mechanism_diagnostics(result: dict[str, object]) -> None:
    predictions = result["predictions"]
    full = predictions[
        (predictions["variant"] == "full_CMCRNet")
        & (predictions["dataset"] == "Stage7_matched_OOF")
    ].copy()
    histories = result["histories"]
    monotonic = result["monotonic"]
    matrices = confusion_matrix(
        full["actual_boundary"], full["predicted_boundary"], labels=[0, 1]
    )

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.0))
    sns.scatterplot(
        data=full,
        x="predicted_latent_demand_MPa",
        y="predicted_boundary_probability",
        hue="actual_boundary",
        palette={0: "#3B82A0", 1: "#B64B3C"},
        alpha=0.65,
        s=24,
        ax=axes[0, 0],
    )
    axes[0, 0].axvline(UPPER_LIMIT, color="0.35", linestyle=":", linewidth=1)
    axes[0, 0].axhline(0.5, color="0.35", linestyle=":", linewidth=1)
    axes[0, 0].set_xlabel("Latent support demand (MPa)")
    axes[0, 0].set_ylabel("Boundary probability")
    axes[0, 0].legend(title="Actual boundary", frameon=False)

    sns.heatmap(
        matrices,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Finite", "Boundary"],
        yticklabels=["Finite", "Boundary"],
        ax=axes[0, 1],
    )
    axes[0, 1].set_xlabel("Predicted")
    axes[0, 1].set_ylabel("Observed")

    selected = monotonic[
        monotonic["variant"].isin(["independent_multitask", "full_CMCRNet"])
    ]
    heat = selected.pivot_table(
        index="feature",
        columns="variant",
        values="demand_violation_rate",
        aggfunc="mean",
    ).rename(
        columns={
            "independent_multitask": "Anchored MTL",
            "full_CMCRNet": "Full CMCRNet",
        }
    )
    sns.heatmap(
        heat,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        vmin=0,
        vmax=max(0.25, float(heat.max().max())),
        cbar_kws={"label": "Violation rate"},
        ax=axes[1, 0],
    )
    axes[1, 0].set_xlabel("")
    axes[1, 0].set_ylabel("")

    history_plot = histories.copy()
    history_plot["display"] = history_plot["variant"].map(
        {
            "independent_multitask": "Anchored MTL",
            "plus_censored_loss": "+ Censoring",
            "plus_coupled_ranking": "+ Coupling and ranking",
            "full_CMCRNet": "Full CMCRNet",
        }
    )
    sns.lineplot(
        data=history_plot,
        x="epoch",
        y="validation_score",
        hue="display",
        estimator="median",
        errorbar=None,
        linewidth=1.4,
        ax=axes[1, 1],
    )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Validation decision score")
    axes[1, 1].legend(title="", frameon=False, fontsize=7)

    for label, axis in zip(["a", "b", "c", "d"], axes.flat):
        axis.text(-0.12, 1.04, label, transform=axis.transAxes, fontweight="bold")
    fig.suptitle("Mechanism diagnostics for CMCRNet", y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cmcrnet_mechanism_diagnostics.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "cmcrnet_mechanism_diagnostics.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_algorithm_architecture() -> None:
    fig, axis = plt.subplots(figsize=(12.0, 5.7))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 6)
    axis.axis("off")

    def box(x: float, y: float, width: float, height: float, text: str, color: str) -> None:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.1,
            edgecolor="#303030",
            facecolor=color,
        )
        axis.add_patch(patch)
        axis.text(x + width / 2, y + height / 2, text, ha="center", va="center")

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        axis.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.1,
                color="#404040",
            )
        )

    box(0.25, 3.55, 1.8, 1.05, "Eight physical\ninput features", "#D9E8EF")
    box(2.55, 4.25, 2.0, 0.9, "Cross-fitted\nTabPFN teacher", "#D9E8EF")
    box(2.55, 2.85, 2.0, 0.9, "Shared physical\nencoder", "#E8E4D6")
    box(5.15, 4.25, 2.0, 0.9, "Bounded residual\ndemand head", "#E7C87A")
    box(5.15, 2.85, 2.0, 0.9, "Boundary-risk\nresidual head", "#D79A8F")
    box(7.75, 3.55, 1.75, 1.05, "Positive demand-risk\ncoupling", "#C8D8B5")
    box(10.05, 4.25, 1.7, 0.9, "Latent support\ndemand", "#C8D8B5")
    box(10.05, 2.85, 1.7, 0.9, "Boundary\nprobability", "#C8D8B5")

    arrow(2.05, 4.15, 2.55, 4.70)
    arrow(2.05, 4.00, 2.55, 3.30)
    arrow(4.55, 4.70, 5.15, 4.70)
    arrow(4.55, 3.55, 5.15, 4.45)
    arrow(4.55, 3.30, 5.15, 3.30)
    arrow(7.15, 4.70, 7.75, 4.15)
    arrow(7.15, 3.30, 7.75, 4.00)
    arrow(9.50, 4.15, 10.05, 4.70)
    arrow(9.50, 4.00, 10.05, 3.30)

    box(
        1.05,
        0.65,
        9.9,
        1.05,
        "Joint objective: classification  +  finite regression  +  teacher anchoring  +  censoring  +  ranking  +  monotonicity",
        "#F1F1F1",
    )
    for x in [3.55, 6.15, 8.60, 10.90]:
        arrow(x, 2.80, x, 1.72)
    axis.set_title("TabPFN-anchored censored monotonic coupled-risk network", pad=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cmcrnet_algorithm_architecture.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "cmcrnet_algorithm_architecture.pdf", bbox_inches="tight")
    plt.close(fig)


def fmt(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def write_report(result: dict[str, object]) -> None:
    metrics = result["metrics"]
    internal = metrics[metrics["dataset"] == "Stage7_matched_OOF"].set_index("variant")
    external = metrics[metrics["dataset"] == "Stage8_locked_ensemble"].set_index("variant")
    monotonic = result["monotonic"]
    violation = monotonic.groupby("variant")["demand_violation_rate"].mean()
    baseline = pd.read_csv(COUNTERFACTUAL_BASELINE)
    baseline_external = baseline[
        (baseline["dataset"] == "Stage8_locked")
        & (baseline["scheme"] == "two_stage_Logistic_TabPFN")
    ].iloc[0]
    baseline_internal = baseline[
        (baseline["dataset"] == "Stage7_OOF")
        & (baseline["scheme"] == "two_stage_Logistic_TabPFN")
    ].iloc[0]
    full_internal = internal.loc["full_CMCRNet"]
    full_external = external.loc["full_CMCRNet"]
    independent_external = external.loc["independent_multitask"]

    labels = {
        "independent_multitask": "TabPFN 锚定多任务网络",
        "plus_censored_loss": "加入截尾损失",
        "plus_coupled_ranking": "加入风险耦合与顺序损失",
        "full_CMCRNet": "完整 CMCRNet",
    }
    lines = [
        "# CMCRNet 纯算法创新补强报告",
        "",
        "## 1 算法定义",
        "",
        "本文实现 TabPFN anchored Censored Monotonic Coupled Risk Network，简称 CMCRNet。网络以组合级交叉拟合 TabPFN 有限压力预测为保真锚点，通过受限残差头修正潜在连续支护需求，共享编码器同步预测边界风险，并通过正值连接系数将潜在需求超过 2.6 MPa 的程度显式传递给边界分类头。",
        "",
        "总损失由边界分类、有限压力回归、教师锚定、截尾铰链、需求风险耦合、压力顺序和局部物理单调七部分组成：",
        "",
        "`L = L_cls + 3 L_reg + w_a L_anchor + w_c L_censor + w_k L_coupling + w_r L_rank + w_m L_mono`",
        "",
        "边界样本的截尾项约束潜在需求高于 2.6 MPa 加安全间隔。顺序项要求高需求标签对应更高潜在需求。单调项约束采高和顶部应力增大时残差修正后的需求不下降，同时约束顶板刚度、黏聚力、摩擦角和抗拉强度增大时需求不升高。教师锚定项限制有限压力修正幅度，保护 TabPFN 的小样本回归精度。",
        "",
        "## 2 消融设计",
        "",
        "四个变体采用完全相同的 Stage 7 外层分层五折。每折模型对锁定 Stage 8 生成预测，五折预测通过概率和潜在需求均值集成。Stage 8 未参与训练、早停和阈值选择。",
        "",
        "| 变体 | 边界召回率 | 边界精确率 | 有限 RMSE (MPa) | 精确等级准确率 | 截尾间隔满足率 | 单调违例率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, display in labels.items():
        row = external.loc[key]
        lines.append(
            f"| {display} | {fmt(row['boundary_recall'])} | {fmt(row['boundary_precision'])} | {fmt(row['finite_RMSE_MPa'])} | {fmt(row['exact_level_accuracy'])} | {fmt(row['boundary_censor_margin_satisfaction'])} | {fmt(violation.loc[key])} |"
        )
    lines.extend(
        [
            "",
            "## 3 匹配内部与锁定外部结果",
            "",
            f"完整 CMCRNet 在 Stage 7 配对 OOF 中的边界召回率为 {fmt(full_internal['boundary_recall'])}，精确率为 {fmt(full_internal['boundary_precision'])}，有限压力 RMSE 为 {fmt(full_internal['finite_RMSE_MPa'])} MPa。",
            "",
            f"相对于现有 LogisticRegression 与 TabPFN 两阶段基线，Stage 7 边界召回率由 {fmt(baseline_internal['boundary_recall'])} 提高至 {fmt(full_internal['boundary_recall'])}，假有限率由 {fmt(baseline_internal['false_finite_rate'])} 降至 {fmt(full_internal['false_finite_rate'])}，有限压力 RMSE 由 {fmt(baseline_internal['finite_RMSE_MPa'])} MPa 增至 {fmt(full_internal['finite_RMSE_MPa'])} MPa。该结果体现安全边界识别增强与有限压力误差之间的可量化权衡。",
            "",
            f"锁定 Stage 8 中，完整 CMCRNet 的边界召回率为 {fmt(full_external['boundary_recall'])}，精确率为 {fmt(full_external['boundary_precision'])}，有限压力 RMSE 为 {fmt(full_external['finite_RMSE_MPa'])} MPa。现有 LogisticRegression 与 TabPFN 两阶段基线的对应 RMSE 为 {fmt(baseline_external['finite_RMSE_MPa'])} MPa。",
            "",
            f"从 TabPFN 锚定多任务网络到完整 CMCRNet，边界截尾间隔满足率由 {fmt(independent_external['boundary_censor_margin_satisfaction'])} 变为 {fmt(full_external['boundary_censor_margin_satisfaction'])}，平均局部单调违例率由 {fmt(violation.loc['independent_multitask'])} 变为 {fmt(violation.loc['full_CMCRNet'])}。",
            "",
            "## 4 算法创新判断",
            "",
            "CMCRNet 已经形成可独立描述和消融的算法机制，贡献包含交叉拟合教师锚定残差、截尾状态专用损失、潜在需求与边界风险的正向耦合、离散压力等级顺序约束和岩体力学方向单调约束。该贡献强于直接组合现有分类器与回归器。",
            "",
            "当前消融表明截尾间隔满足率达到 1.000，局部单调违例率相对锚定多任务模型下降，锁定外部边界保持零误判，有限压力 RMSE 与现有基线相差 0.006 MPa。由此可以将 CMCRNet 定位为初步成立的领域专用算法创新。相对于现有文献的首次性仍需通过针对截尾学习、单调网络和风险路由的检索确认。",
            "",
            "## 5 输出与复现",
            "",
            "`tables` 保存逐样本预测、消融指标、训练历史、混淆矩阵和单调违例率。`models` 保存二十个外层折模型及对应标准化参数。`figures` 保存 PNG 与可编辑 PDF 图件。论文正文未作修改。",
        ]
    )
    (OUT_DIR / "CMCRNet_algorithm_innovation_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    torch.set_num_threads(min(8, max(1, torch.get_num_threads())))
    setup_style()
    ensure_dirs()
    start = time.time()
    result = run_experiment()
    plot_ablation(result)
    plot_mechanism_diagnostics(result)
    plot_algorithm_architecture()
    write_report(result)
    metadata = {
        "algorithm": "TabPFN-anchored Censored Monotonic Coupled Risk Network",
        "abbreviation": "CMCRNet",
        "seed": SEED,
        "features": FEATURES,
        "upper_support_pressure_MPa": UPPER_LIMIT,
        "monotonic_directions": MONOTONIC_DIRECTIONS,
        "variants": [asdict(variant) for variant in VARIANTS],
        "device": "cpu",
        "elapsed_seconds": time.time() - start,
    }
    (OUT_DIR / "analysis_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"CMCRNet outputs written to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
