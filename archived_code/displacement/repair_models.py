"""Clean PCA plus MLP heads; every fit and bound uses its training groups only."""
import numpy as np
import torch
from torch import nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from common import inverse, old_module


class CoefficientNet(nn.Module):
    def __init__(self, inputs=53, outputs=64):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(inputs, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(),
                                    nn.Dropout(.02), nn.Linear(64, outputs))

    def forward(self, x):
        return self.layers(x)


def prepare(x, truth, valid, seed=20261025):
    x, truth, valid = np.asarray(x, float), np.asarray(truth, float), np.asarray(valid, bool)
    assert len(x) > 64 and truth.shape == valid.shape and np.isfinite(truth[valid]).all()
    engineered = old_module().engineered(x)
    scaler = StandardScaler().fit(engineered)
    z = np.where(valid, np.sign(truth)*np.sqrt(np.abs(truth)), 0.).astype(np.float32)
    with threadpool_limits(limits=2):
        pca = PCA(n_components=64, svd_solver="randomized", random_state=seed).fit(z)
        coefficients = pca.transform(z).astype(np.float32)
    coefficient_mean = coefficients.mean(0)
    output_scale = np.maximum(coefficients.std(0), max(float(coefficients.std())*.015, 1e-6))
    standardized = (coefficients-coefficient_mean)/output_scale
    bound = np.maximum(1.25*np.max(np.abs(standardized), axis=0), 2.5)
    rms = np.sqrt(np.where(valid, truth**2, 0.).sum(1)/valid.sum(1))
    physical_scale = max(float(np.median(rms)), .001)
    return {"input_scaler": scaler, "pca": pca, "coefficient_mean": coefficient_mean,
            "coefficient_scale": output_scale, "normalized_bound": bound, "physical_scale_m": physical_scale,
            "modes": 64, "fit_only": True}


def fit_members(common, x, truth, valid, bounded, seeds, epochs=600, progress=None):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    x = np.clip(common["input_scaler"].transform(old_module().engineered(x)), -8, 8).astype(np.float32)
    truth = np.where(valid, truth, 0.).astype(np.float32)
    valid = np.asarray(valid, bool)
    z = np.where(valid, np.sign(truth)*np.sqrt(np.abs(truth)), 0.)
    with threadpool_limits(limits=2):
        c = common["pca"].transform(z)
    y = (c-common["coefficient_mean"])/common["coefficient_scale"]
    tx, ty, field = torch.from_numpy(x), torch.from_numpy(y.astype(np.float32)), torch.from_numpy(truth)
    mask = torch.from_numpy(valid)
    count = mask.sum(1)
    mean = torch.from_numpy(common["coefficient_mean"])
    scale = torch.from_numpy(common["coefficient_scale"])
    bound = torch.from_numpy(common["normalized_bound"])
    basis = torch.from_numpy(common["pca"].components_)
    pca_mean = torch.from_numpy(common["pca"].mean_)
    weights = scale.square()/scale.square().mean()
    physical_scale = common["physical_scale_m"]
    states, curves = [], []
    for seed in seeds:
        torch.manual_seed(seed)
        net = CoefficientNet(x.shape[1], y.shape[1])
        optimizer = torch.optim.AdamW(net.parameters(), lr=.002, weight_decay=.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=.00008)
        net.train()
        for epoch in range(epochs):
            raw = net(tx)
            standardized = bound*torch.tanh(raw/bound) if bounded else raw
            predicted_c = standardized*scale+mean
            transformed = predicted_c@basis+pca_mean
            reconstructed = transformed*transformed.abs()
            coefficient_huber = torch.nn.functional.smooth_l1_loss(standardized, ty, beta=1.)
            coefficient_mse = ((standardized-ty).square()*weights).mean()
            physical_huber = torch.nn.functional.huber_loss(reconstructed, field, reduction="none", delta=physical_scale)
            physical_loss = (torch.where(mask, physical_huber, 0.).sum(1)/count).mean()/physical_scale**2
            loss = .5*coefficient_huber+.25*coefficient_mse+.25*physical_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite clean-reference training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(net.parameters(), 8., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            if epoch % 25 == 0 or epoch == epochs-1:
                curves.append({"seed": seed, "epoch": epoch+1, "loss": float(loss.detach()),
                    "coefficient_huber": float(coefficient_huber.detach()), "coefficient_mse": float(coefficient_mse.detach()),
                    "physical_huber_scaled": float(physical_loss.detach()), "gradient_norm_before_clip": float(gradient)})
            if progress is not None and epoch % 100 == 0:
                progress(seed, epoch+1)
        states.append({k: v.detach().cpu().numpy() for k, v in net.state_dict().items()})
    return {"common": common, "bounded": bool(bounded), "states": states,
            "seeds": list(seeds), "epochs": epochs, "architecture": "53_64_64_64"}, curves


def predict_coefficients(package, features):
    x = np.clip(package["common"]["input_scaler"].transform(old_module().engineered(features)), -8, 8).astype(np.float32)
    common = package["common"]
    members = []
    with torch.no_grad():
        for state in package["states"]:
            net = CoefficientNet(x.shape[1], common["modes"])
            net.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()})
            net.eval()
            raw = net(torch.from_numpy(x)).numpy()
            if package["bounded"]:
                raw = common["normalized_bound"]*np.tanh(raw/common["normalized_bound"])
            members.append(raw*common["coefficient_scale"]+common["coefficient_mean"])
    return np.mean(members, axis=0)


def predict(package, features):
    common = package["common"]
    coefficients = predict_coefficients(package, features)
    with threadpool_limits(limits=2):
        z = coefficients@common["pca"].components_+common["pca"].mean_
    return inverse(z).astype(np.float32)


def predict_repaired(bundle, features, valid):
    """Use observable excavation fraction alone to route the specialized head."""
    features = np.asarray(features)
    old = old_module()
    result = old.predict_field(bundle["archived_head"], features, valid)
    fifth = np.isclose(features[:, 13], 1., atol=1e-7, rtol=0)
    if fifth.any():
        if "step5_guard" in bundle:
            head, guard = bundle["archived_head"], bundle["step5_guard"]
            coefficients = old.predict_regressor(head["regressor"], old.engineered(features[fifth]))*head["scale"]
            active = guard["active_modes"]
            coefficients[:, active] = np.clip(coefficients[:, active], guard["lower"][active], guard["upper"][active])
            with threadpool_limits(limits=2):
                z = coefficients@head["pca"].components_[:64]+head["pca"].mean_
            result[fifth] = inverse(z).astype(np.float32)
        else:
            result[fifth] = predict(bundle["step5_head"], features[fifth])
    return result
