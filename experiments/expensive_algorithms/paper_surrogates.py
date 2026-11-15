"""Recent-paper surrogate-assisted algorithms exposed through EvoX.

These baselines are independent of XDE-SINDy. Each owns its own model,
archive, infill rule and FE counter. The implementations follow the central
mechanisms described by the cited 2025-2026 papers and record the citation in
the experiment audit.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EVOX_SOURCE = _PROJECT_ROOT / "evox" / "src"
if str(_EVOX_SOURCE) not in sys.path:
    sys.path.insert(0, str(_EVOX_SOURCE))

from evox.core import Algorithm, Problem

from .paper_backend import ComputeBackend


PAPER_METADATA = {
    "sgdlco": {
        "name": "SGDLCO",
        "paper": "Liu et al. (2025), Surrogate-assisted global and distributed local collaborative optimization algorithm for expensive constrained optimization problems",
        "doi": "10.1038/s41598-025-85233-6",
        "rule": "global surrogate pool plus distributed local surrogate search",
    },
    "sagpe": {
        "name": "SAGPE",
        "paper": "Huang et al. (2025), A Surrogate-Assisted Gray Prediction Evolution Algorithm for High-Dimensional Expensive Optimization Problems",
        "doi": "10.3390/math13061007",
        "rule": "gray prediction of incumbent trend combined with surrogate evolution",
    },
    "bgp_saea": {
        "name": "BGP-SAEA",
        "paper": "Zou et al. (2025), Uncertainty estimation by boosting with Gaussian process for expensive optimization problems",
        "doi": "10.1016/j.asoc.2025.113320",
        "rule": "boosted Gaussian-process ensemble with uncertainty-driven infill",
    },
    "sade_ss": {
        # ``sade_ss`` is the canonical configuration acronym. Configuration
        # keys and newly generated result folders use the publication acronym
        # directly.
        "name": "SADE-SS",
        "paper": "Zhou et al. (2025), Surrogate-Assisted Differential Evolution With Search Space Tightening for High-Dimensional Expensive Optimization Problems",
        "doi": "10.1109/TSMC.2025.3582897",
        "rule": "surrogate-assisted differential evolution with adaptive search-space tightening",
    },
    "rddsp_saea": {
        "name": "RDDSP-SAEA",
        "paper": "Negreiros and Batista (2026), A surrogate-assisted algorithm based on principal component analysis and partial least squares regression space projection strategies for high-dimensional expensive problems",
        "doi": "10.1007/s12293-026-00518-1",
        "rule": "multi-decision-space partitioning with PCA/PLS reduced-space RBF surrogates",
    },
}

# The cited comparison implementations use conventional CPU surrogate
# execution. XDE-SINDy is the CUDA method under study and is routed separately.
PAPER_CPU_ALGORITHMS = frozenset({
    "sgdlco",
    "sagpe",
    "bgp_saea",
    "sade_ss",
})


def _normal_cdf(values: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(values / math.sqrt(2.0)))


def _gray_forecast(values: list[float]) -> float:
    """Forecast the next incumbent with a numerically guarded GM(1,1) fit."""
    series = np.asarray(values, dtype=float).reshape(-1)
    if series.size < 4 or not np.all(np.isfinite(series)):
        return float(series[-1]) if series.size else 0.0
    offset = max(0.0, -float(np.min(series))) + 1e-8
    x0 = series + offset
    accumulated = np.cumsum(x0)
    z1 = 0.5 * (accumulated[1:] + accumulated[:-1])
    design = np.column_stack((-z1, np.ones_like(z1)))
    try:
        a, b = np.linalg.lstsq(design, x0[1:], rcond=None)[0]
    except np.linalg.LinAlgError:
        return float(series[-1])
    if abs(float(a)) < 1e-10 or not np.isfinite(a + b):
        return float(series[-1])
    next_accumulated = (x0[0] - b / a) * math.exp(-a * series.size) + b / a
    previous_accumulated = accumulated[-1]
    forecast = next_accumulated - previous_accumulated - offset
    if not np.isfinite(forecast):
        return float(series[-1])
    return float(forecast)


class _RBFModel:
    def __init__(self, bounds: np.ndarray, backend: ComputeBackend, gaussian: bool = False) -> None:
        self.bounds = np.asarray(bounds, dtype=float)
        self.backend = backend
        self.gaussian = bool(gaussian)
        self.centers = np.empty((0, self.bounds.shape[0]), dtype=float)
        self.values = np.empty(0, dtype=float)
        self.coefficients: Any = None
        self.location = 0.0
        self.scale = 1.0
        self.spread = 1.0
        self.fit_count = 0

    def _normalize(self, points: np.ndarray) -> np.ndarray:
        return 2.0 * (np.asarray(points, dtype=float) - self.bounds[:, 0]) / (
            self.bounds[:, 1] - self.bounds[:, 0]
        ) - 1.0

    def _distance(self, points: np.ndarray) -> Any:
        xp = self.backend.array_module
        query = self.backend.to_device(self._normalize(points)).astype(xp.float64, copy=False)
        centers = self.backend.to_device(self._normalize(self.centers)).astype(xp.float64, copy=False)
        d = max(query.shape[1], 1)
        squared = (
            xp.sum(query * query, axis=1, keepdims=True)
            + xp.sum(centers * centers, axis=1, keepdims=True).T
            - 2.0 * (query @ centers.T)
        ) / d
        return xp.sqrt(xp.maximum(squared, 0.0))

    def fit(self, points: np.ndarray, values: np.ndarray) -> None:
        self.centers = np.asarray(points, dtype=float).copy()
        self.values = np.asarray(values, dtype=float).reshape(-1).copy()
        self.location = float(np.median(self.values))
        self.scale = max(1e-8, 1.4826 * float(np.median(np.abs(self.values - self.location))))
        target = (self.values - self.location) / self.scale
        distances = self._distance(self.centers)
        host = np.asarray(self.backend.to_numpy(distances), dtype=float)
        positive = host[host > 1e-12]
        self.spread = float(np.median(positive)) if positive.size else 1.0
        xp = self.backend.array_module
        basis = xp.exp(-0.5 * (distances / max(self.spread, 1e-8)) ** 2) if self.gaussian else distances**3
        design = xp.concatenate((xp.ones((len(self.centers), 1), dtype=xp.float64), basis), axis=1)
        self.coefficients = self.backend.ridge_fit(
            design,
            self.backend.to_device(target),
            1e-3,
            return_device=True,
        )
        self.fit_count += 1

    def predict(self, points: np.ndarray) -> np.ndarray:
        distances = self._distance(points)
        xp = self.backend.array_module
        basis = xp.exp(-0.5 * (distances / max(self.spread, 1e-8)) ** 2) if self.gaussian else distances**3
        design = xp.concatenate((xp.ones((len(points), 1), dtype=xp.float64), basis), axis=1)
        values = self.backend.matmul(design, self.coefficients, operation="paper_surrogate_prediction")
        return self.location + self.scale * np.asarray(values, dtype=float).reshape(-1)

    def uncertainty(self, points: np.ndarray) -> np.ndarray:
        distances = np.asarray(self.backend.to_numpy(self._distance(points)), dtype=float)
        return np.min(distances, axis=1)

    def audit(self) -> dict[str, Any]:
        return {"model": "gaussian_rbf" if self.gaussian else "cubic_rbf", "fit_count": self.fit_count, "center_count": len(self.centers)}


class _GaussianProcessModel:
    """Exact squared-exponential GP posterior evaluated on the CUDA backend."""

    def __init__(self, bounds: np.ndarray, backend: ComputeBackend) -> None:
        self.bounds = np.asarray(bounds, dtype=float)
        self.backend = backend
        self.centers = np.empty((0, self.bounds.shape[0]), dtype=float)
        self.location = 0.0
        self.scale = 1.0
        self.lengthscale = 1.0
        self.noise = 1e-6
        self.kernel_matrix: Any = None
        self.alpha: Any = None
        self.fit_count = 0

    def _torch_state(self) -> tuple[Any, Any, Any]:
        import torch

        device = torch.device(
            "cpu"
            if not self.backend.cuda_enabled
            else f"cuda:{self.backend.device_index}"
        )
        lower = torch.as_tensor(self.bounds[:, 0], dtype=torch.float64, device=device)
        span = torch.as_tensor(self.bounds[:, 1] - self.bounds[:, 0], dtype=torch.float64, device=device)
        return torch, device, (lower, span)

    def _distance(self, points: np.ndarray) -> Any:
        torch, device, (lower, span) = self._torch_state()
        query = torch.as_tensor(np.asarray(points, dtype=float), dtype=torch.float64, device=device)
        centers = torch.as_tensor(self.centers, dtype=torch.float64, device=device)
        query = 2.0 * (query - lower) / span - 1.0
        centers = 2.0 * (centers - lower) / span - 1.0
        dimension = max(query.shape[1], 1)
        squared = (
            torch.sum(query * query, dim=1, keepdim=True)
            + torch.sum(centers * centers, dim=1, keepdim=True).T
            - 2.0 * (query @ centers.T)
        ) / dimension
        return torch.sqrt(torch.clamp(squared, min=0.0))

    def _normalize(self, points: np.ndarray) -> np.ndarray:
        return 2.0 * (np.asarray(points, dtype=float) - self.bounds[:, 0]) / (
            self.bounds[:, 1] - self.bounds[:, 0]
        ) - 1.0

    def _kernel(self, distances: Any) -> Any:
        import torch

        return torch.exp(-0.5 * (distances / max(self.lengthscale, 1e-8)) ** 2)

    def fit(self, points: np.ndarray, values: np.ndarray) -> None:
        self.centers = np.asarray(points, dtype=float).copy()
        values = np.asarray(values, dtype=float).reshape(-1)
        self.location = float(np.mean(values))
        self.scale = max(1e-8, float(np.std(values)))
        target = (values - self.location) / self.scale
        distances = self._distance(self.centers)
        host = distances.detach().cpu().numpy()
        positive = host[host > 1e-12]
        self.lengthscale = float(np.median(positive)) if positive.size else 1.0
        torch, device, _ = self._torch_state()
        kernel = self._kernel(distances)
        kernel = kernel + self.noise * torch.eye(
            kernel.shape[0], dtype=kernel.dtype, device=device
        )
        try:
            self.alpha = torch.linalg.solve(
                kernel, torch.as_tensor(target, dtype=torch.float64, device=device)
            )
        except torch.linalg.LinAlgError:
            self.alpha = torch.linalg.pinv(kernel) @ torch.as_tensor(target, dtype=torch.float64, device=device)
        self.kernel_matrix = kernel
        self.fit_count += 1

    def predict_uncertainty(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.alpha is None or self.kernel_matrix is None:
            raise RuntimeError("Gaussian process is not fitted")
        import torch

        cross = self._kernel(self._distance(points))
        mean = cross @ self.alpha
        try:
            solved = torch.linalg.solve(self.kernel_matrix, cross.T)
        except torch.linalg.LinAlgError:
            solved = torch.linalg.pinv(self.kernel_matrix) @ cross.T
        variance = torch.clamp(1.0 - torch.sum(cross * solved.T, dim=1), min=1e-10)
        return (
            self.location + self.scale * mean.detach().cpu().numpy(),
            self.scale * torch.sqrt(variance).detach().cpu().numpy(),
        )

    def audit(self) -> dict[str, Any]:
        return {
            "model": "exact_gaussian_process",
            "kernel": "squared_exponential",
            "fit_count": self.fit_count,
            "center_count": len(self.centers),
            "noise": self.noise,
        }


class _GPEnsemble:
    def __init__(self, bounds: np.ndarray, backend: ComputeBackend, members: int = 3) -> None:
        self.bounds = np.asarray(bounds, dtype=float)
        self.backend = backend
        self.members = int(members)
        self.models = [_GaussianProcessModel(bounds, backend) for _ in range(self.members)]
        self.noise = 1e-6
        self.learning_rate = 0.5
        self.weak_coef: np.ndarray | None = None
        self.weak_location = np.zeros(self.bounds.shape[0], dtype=float)
        self.weak_scale = np.ones(self.bounds.shape[0], dtype=float)

    def _weak_features(self, points: np.ndarray) -> np.ndarray:
        x = np.asarray(points, dtype=float)
        z = 2.0 * (x - self.bounds[:, 0]) / np.maximum(self.bounds[:, 1] - self.bounds[:, 0], 1e-12) - 1.0
        return np.column_stack((np.ones(len(z)), z))

    def _weak_predict(self, points: np.ndarray) -> np.ndarray:
        if self.weak_coef is None:
            return np.zeros(len(points), dtype=float)
        return self._weak_features(points) @ self.weak_coef

    def fit(self, points: np.ndarray, values: np.ndarray) -> None:
        rng = np.random.default_rng(1009 + len(values))
        values = np.asarray(values, dtype=float).reshape(-1)
        design = self._weak_features(points)
        # The weak learner captures the global trend; GP members model only
        # residuals, which is the boosting decomposition used by BGP-SAEA.
        # Use the sample-space solve when d is larger than the archive. This
        # keeps BGP-SAEA usable at D=500/2000 without forming an O(d^2)
        # normal-equation factorization for every surrogate refit.
        if design.shape[0] <= design.shape[1]:
            gram = design @ design.T + 1e-8 * np.eye(design.shape[0], dtype=float)
            try:
                alpha = np.linalg.solve(gram, values)
            except np.linalg.LinAlgError:
                alpha = np.linalg.pinv(gram, rcond=1e-10) @ values
            self.weak_coef = design.T @ alpha
        else:
            ridge = 1e-8 * np.eye(design.shape[1], dtype=float)
            ridge[0, 0] = 0.0
            try:
                self.weak_coef = np.linalg.solve(design.T @ design + ridge, design.T @ values)
            except np.linalg.LinAlgError:
                self.weak_coef = np.linalg.pinv(design, rcond=1e-10) @ values
        residual = values - self._weak_predict(points)
        for model in self.models:
            indices = rng.integers(0, len(values), size=len(values))
            model.fit(points[indices], residual[indices])

    def predict_uncertainty(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        corrections = []
        posterior_sigmas = []
        for model in self.models:
            mean, sigma = model.predict_uncertainty(points)
            corrections.append(mean)
            posterior_sigmas.append(sigma)
        correction = np.vstack(corrections)
        mean = self._weak_predict(points) + self.learning_rate * np.mean(correction, axis=0)
        # Migrate GP posterior variance into the ensemble uncertainty and add
        # member disagreement as the finite-sample boosting component.
        posterior_var = np.mean(np.vstack(posterior_sigmas) ** 2, axis=0)
        disagreement = np.var(correction, axis=0)
        return mean, np.sqrt(np.maximum(posterior_var + disagreement, 1e-12))

    def audit(self) -> dict[str, Any]:
        return {
            "model": "boosted_gaussian_process_ensemble",
            "kernel": "squared_exponential",
            "members": self.members,
            "learning_rate": self.learning_rate,
            "weak_learner": "linear_global_trend",
            "uncertainty": "migrated_gp_posterior_variance_plus_ensemble_disagreement",
        }


class _ReducedSpaceRBFModel:
    """GPU PCA/PLS projection followed by an independent reduced-space RBF."""

    def __init__(self, bounds: np.ndarray, backend: ComputeBackend, latent_dim: int = 8) -> None:
        self.bounds = np.asarray(bounds, dtype=float)
        self.backend = backend
        self.latent_dim = int(max(1, latent_dim))
        self.mean: Any = None
        self.components: Any = None
        self.base: _RBFModel | None = None
        self.fit_count = 0
        self.pca_components = 0
        self.pls_components = 0

    def _project(self, points: np.ndarray) -> np.ndarray:
        if self.mean is None or self.components is None:
            raise RuntimeError("reduced-space model is not fitted")
        import torch

        device = torch.device(
            "cpu"
            if not self.backend.cuda_enabled
            else f"cuda:{self.backend.device_index}"
        )
        values = torch.as_tensor(np.asarray(points, dtype=float), dtype=torch.float64, device=device)
        lower = torch.as_tensor(self.bounds[:, 0], dtype=torch.float64, device=device)
        span = torch.as_tensor(self.bounds[:, 1] - self.bounds[:, 0], dtype=torch.float64, device=device)
        normalized = 2.0 * (values - lower) / span - 1.0
        reduced = (normalized - self.mean) @ self.components
        return reduced.detach().cpu().numpy()

    def fit(self, points: np.ndarray, values: np.ndarray) -> None:
        import torch

        device = torch.device(
            "cpu"
            if not self.backend.cuda_enabled
            else f"cuda:{self.backend.device_index}"
        )
        x = torch.as_tensor(np.asarray(points, dtype=float), dtype=torch.float64, device=device)
        y = torch.as_tensor(np.asarray(values, dtype=float).reshape(-1), dtype=torch.float64, device=device)
        lower = torch.as_tensor(self.bounds[:, 0], dtype=torch.float64, device=device)
        span = torch.as_tensor(self.bounds[:, 1] - self.bounds[:, 0], dtype=torch.float64, device=device)
        normalized = 2.0 * (x - lower) / span - 1.0
        mean = torch.mean(normalized, dim=0)
        centered = normalized - mean
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        pca_count = min(self.latent_dim, int(vh.shape[0]), max(1, int(x.shape[0] - 1)))
        pca = vh[:pca_count].T
        centered_y = y - torch.mean(y)
        pls = centered.T @ centered_y
        pls_norm = torch.linalg.norm(pls)
        if float(pls_norm.detach().cpu()) > 1e-10:
            pls = (pls / pls_norm).reshape(-1, 1)
            directions = torch.cat((pca, pls), dim=1)
            components = torch.linalg.qr(directions, mode="reduced").Q
        else:
            components = pca
        latent = min(self.latent_dim, int(components.shape[1]))
        self.mean = mean
        self.components = components[:, :latent]
        projected = (centered @ self.components).detach().cpu().numpy()
        reduced_bounds = np.tile(np.asarray([[-1.0, 1.0]], dtype=float), (latent, 1))
        self.base = _RBFModel(reduced_bounds, self.backend, gaussian=False)
        self.base.fit(projected, np.asarray(values, dtype=float))
        self.pca_components = pca_count
        self.pls_components = int(latent > pca_count)
        self.fit_count += 1

    def predict(self, points: np.ndarray) -> np.ndarray:
        if self.base is None:
            raise RuntimeError("reduced-space model is not fitted")
        return self.base.predict(self._project(points))

    def uncertainty(self, points: np.ndarray) -> np.ndarray:
        if self.base is None:
            raise RuntimeError("reduced-space model is not fitted")
        return self.base.uncertainty(self._project(points))

    def audit(self) -> dict[str, Any]:
        return {
            "model": "reduced_space_cubic_rbf",
            "projection": "GPU PCA plus supervised PLS direction",
            "latent_dimension": self.latent_dim,
            "pca_components": self.pca_components,
            "pls_components": self.pls_components,
            "fit_count": self.fit_count,
        }


class RecentPaperController:
    def __init__(self, benchmark: Any, config: Any, method: str, seed: int) -> None:
        self.benchmark = benchmark
        self.config = config
        self.method = method
        self.metadata = PAPER_METADATA[method]
        self.execution_device = "cpu" if method in PAPER_CPU_ALGORITHMS else "cuda"
        self.budget = config.max_evaluations_for(benchmark.dimension)
        self.batch_size = config.effective_real_batch_size
        self.backend = ComputeBackend(
            "cpu" if self.execution_device == "cpu" else "cuda",
            config.gpu_device,
            config.gpu_min_elements,
        )
        self.rng = np.random.default_rng(seed)
        if method == "bgp_saea":
            self.model: Any = _GPEnsemble(benchmark.bounds, self.backend, members=3)
        elif method == "rddsp_saea":
            self.model = _ReducedSpaceRBFModel(
                benchmark.bounds,
                self.backend,
                latent_dim=min(8, benchmark.dimension),
            )
        else:
            self.model = _RBFModel(benchmark.bounds, self.backend, gaussian=method == "sgdlco")
        self.archive_x: list[np.ndarray] = []
        self.archive_y: list[float] = []
        self.history: list[dict[str, Any]] = []
        self.calls = 0
        self.fitted = False
        self.tightening = 1.0
        self.gray_forecast = None
        self.previous_best = None
        self.partition_count = 3 if method == "rddsp_saea" else 1

    @property
    def evaluations(self) -> int:
        return len(self.archive_y)

    def _fit(self) -> None:
        if len(self.archive_y) < 3:
            return
        x = np.asarray(self.archive_x, dtype=float)
        y = np.asarray(self.archive_y, dtype=float)
        limit = self.config.max_surrogate_samples
        if limit is not None and len(y) > limit:
            x, y = x[-limit:], y[-limit:]
        self.model.fit(x, y)
        self.fitted = True

    def _pool(self) -> np.ndarray:
        lower, upper = self.benchmark.bounds[:, 0], self.benchmark.bounds[:, 1]
        n = max(128, 4 * self.config.population_size)
        if self.method == "sade_ss" and self.archive_x:
            center = np.asarray(self.archive_x[int(np.argmin(self.archive_y))])
            span = (upper - lower) * self.tightening
            lower, upper = np.maximum(lower, center - span / 2), np.minimum(upper, center + span / 2)
            if len(self.archive_x) >= 3:
                archive = np.asarray(self.archive_x, dtype=float)
                best = archive[int(np.argmin(self.archive_y))]
                candidates = []
                for _ in range(n):
                    indices = self.rng.choice(len(archive), size=3, replace=len(archive) < 3)
                    mutant = archive[indices[0]] + 0.5 * (best - archive[indices[0]])
                    mutant += 0.7 * (archive[indices[1]] - archive[indices[2]])
                    crossover = self.rng.random(len(lower)) < 0.9
                    crossover[self.rng.integers(0, len(lower))] = True
                    parent = archive[indices[0]]
                    trial = np.where(crossover, mutant, parent)
                    candidates.append(np.clip(trial, lower, upper))
                return np.asarray(candidates, dtype=float)
        if self.method in {"sgdlco", "rddsp_saea"} and self.archive_x:
            center = np.asarray(self.archive_x[int(np.argmin(self.archive_y))])
            local = center + self.rng.normal(0.0, 0.15, size=(n // 2, len(lower))) * (upper - lower)
            global_points = self.rng.uniform(lower, upper, size=(n - len(local), len(lower)))
            return np.clip(np.vstack((global_points, local)), self.benchmark.bounds[:, 0], self.benchmark.bounds[:, 1])
        if self.method == "sagpe" and self.archive_x:
            center = np.asarray(self.archive_x[int(np.argmin(self.archive_y))])
            self.gray_forecast = _gray_forecast(self.archive_y)
            current_best = min(self.archive_y)
            radius = 0.08 if self.gray_forecast < current_best else 0.18
            return np.clip(
                center
                + self.rng.normal(0.0, radius, size=(n, len(lower))) * (upper - lower),
                self.benchmark.bounds[:, 0],
                self.benchmark.bounds[:, 1],
            )
        return self.rng.uniform(lower, upper, size=(n, len(lower)))

    def _select(self, pool: np.ndarray) -> np.ndarray:
        if self.method == "bgp_saea":
            mean, uncertainty = self.model.predict_uncertainty(pool)
            best = min(self.archive_y)
            sigma = np.maximum(uncertainty, 1e-8)
            z = (best - mean) / sigma
            acquisition = (best - mean) * _normal_cdf(z) + sigma * np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
            order = np.argsort(-acquisition)
        else:
            prediction = self.model.predict(pool)
            uncertainty = self.model.uncertainty(pool)
            weight = 0.4 if self.method == "sgdlco" else 0.2
            order = np.argsort(prediction - weight * uncertainty)
        count = min(self.batch_size, self.budget - self.evaluations)
        return pool[order[:count]]

    def evaluate(self, population: torch.Tensor) -> torch.Tensor:
        if not self.fitted:
            points = population.detach().cpu().numpy().astype(float, copy=False)
            points = points[: min(len(points), self.budget - self.evaluations)]
        else:
            points = self._select(self._pool())
        tensor = torch.as_tensor(points, dtype=population.dtype, device=population.device)
        values_tensor = self.benchmark.problem.evaluate(tensor).reshape(-1)
        values = values_tensor.detach().cpu().numpy()
        for point, value in zip(points, values):
            self.archive_x.append(np.asarray(point, dtype=float).copy())
            self.archive_y.append(float(value))
        self.calls += 1
        if not self.fitted or self.calls % self.config.surrogate_refit_interval == 0:
            self._fit()
        best = float(np.min(self.archive_y))
        if self.method == "sade_ss" and self.previous_best is not None:
            if best < self.previous_best:
                self.tightening = min(1.0, 1.05 * self.tightening)
            else:
                self.tightening = max(0.05, 0.9 * self.tightening)
        self.previous_best = best
        self.history.append({"generation": float(self.calls), "evaluations": float(self.evaluations), "best": float(np.min(self.archive_y)), "real_evaluations": float(len(points))})
        return values_tensor

    def audit(self) -> dict[str, Any]:
        if self.method == "bgp_saea":
            implementation = "BGP-SAEA_linear_weak_learner_GP_residual_boosting_EI"
            fidelity_note = "global weak trend plus GP posterior variance migrated into EI uncertainty"
        elif self.method == "rddsp_saea":
            implementation = "RDDSP-SAEA_GPU_PCA_PLS_reduced_space_RBF"
            fidelity_note = "PCA/PLS reduced-space surrogate; optional comparison"
        else:
            implementation = "legacy_generic_controller_not_used_by_default_runner"
            fidelity_note = "default runner dispatches a dedicated method controller"
        return {
            "paper": self.metadata,
            "implementation": implementation,
            "fidelity_note": fidelity_note,
            "surrogate": self.model.audit(),
            "true_evaluations": self.evaluations,
            "mechanism_state": {
                "search_space_tightening": float(self.tightening),
                "gray_forecast": self.gray_forecast,
                "decision_space_partitions": self.partition_count,
                "gpu_backend": self.backend.device_name,
                "execution_device": self.execution_device,
            },
        }


class _SADE_SSRBF:
    """Gaussian RBF from the paper's equations (6)–(7).

    The published model uses the kernel matrix ``phi`` and solves
    ``omega = phi^-1 F``.  This class deliberately avoids the intercept and
    target standardization used by the generic baselines so the SADE-SS
    reproduction has the same model structure.  Bound normalization keeps
    the Euclidean distances numerically meaningful in high dimensions.
    """

    def __init__(self, bounds: np.ndarray) -> None:
        self.bounds = np.asarray(bounds, dtype=float)
        self.centers = np.empty((0, self.bounds.shape[0]), dtype=float)
        self.weights = np.empty(0, dtype=float)
        self.sigma = 1.0
        self.fit_count = 0

    def _normalize(self, points: np.ndarray) -> np.ndarray:
        return (np.asarray(points, dtype=float) - self.bounds[:, 0]) / np.maximum(self.bounds[:, 1] - self.bounds[:, 0], 1e-12)

    def _distances(self, points: np.ndarray, centers: np.ndarray | None = None) -> np.ndarray:
        query = self._normalize(points)
        reference = self._normalize(self.centers if centers is None else centers)
        squared = np.sum(query * query, axis=1, keepdims=True) + np.sum(reference * reference, axis=1)[None, :] - 2.0 * (query @ reference.T)
        return np.sqrt(np.maximum(squared, 0.0))

    def _kernel(self, distances: np.ndarray) -> np.ndarray:
        return np.exp(-0.5 * (distances / max(self.sigma, 1e-12)) ** 2)

    def fit(self, points: np.ndarray, values: np.ndarray) -> None:
        self.centers = np.asarray(points, dtype=float).copy()
        targets = np.asarray(values, dtype=float).reshape(-1)
        distances = self._distances(self.centers)
        positive = distances[distances > 1e-12]
        self.sigma = float(np.median(positive)) if positive.size else 1.0
        phi = self._kernel(distances)
        try:
            self.weights = np.linalg.solve(phi, targets)
        except np.linalg.LinAlgError:
            # The paper specifies the inverse; pinv is only a numerical
            # fallback for duplicate samples in finite-precision runs.
            self.weights = np.linalg.pinv(phi, rcond=1e-10) @ targets
        self.fit_count += 1

    def predict(self, points: np.ndarray) -> np.ndarray:
        return self._kernel(self._distances(points)) @ self.weights


class SADE_SSController:
    """Faithful NumPy implementation of the paper's SADE-SS algorithms 1–4.

    The older ``RecentPaperController`` is retained for the other baselines.
    SADE-SS needs a separate controller because its two searches have
    different databases, DE operators, and switch state; treating it as a
    generic pool-and-acquisition method changes the algorithm materially.
    The canonical configuration key is the publication acronym ``sade_ss``;
    no legacy compatibility alias is accepted. Reports identify the method as
    ``SADE-SS``.
    """

    # Values stated in the paper's parameter section.
    GLOBAL_POPULATION_SIZE = 50
    LOCAL_POPULATION_MAX = 50
    LOCAL_POPULATION_MIN = 4
    LOCAL_SAMPLE_SIZE = 40
    MU_F_INITIAL = 0.5
    MU_CR_INITIAL = 0.8
    GROUPS = 3
    RBF_KERNEL = "gaussian"
    # nfasmax is an input to Algorithm 3 but is not assigned a numeric value
    # in the published parameter paragraph.  A bounded value keeps the
    # surrogate-only inner loop reproducible and computationally tractable.
    LOCAL_NFAS_MAX = 100
    GLOBAL_EARLY_FRACTION = 0.5

    def __init__(self, benchmark: Any, config: Any, seed: int) -> None:
        self.benchmark = benchmark
        self.config = config
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.dimension = int(benchmark.dimension)
        self.bounds = np.asarray(benchmark.bounds, dtype=float)
        self.lower = self.bounds[:, 0].copy()
        self.upper = self.bounds[:, 1].copy()
        self.span = self.upper - self.lower
        self.budget = int(config.max_evaluations_for(self.dimension))
        self.paper_nfe_max = 1000
        self.tau = 100 if self.dimension in (30, 50) else 150
        self.gs = self.tau
        self.ls = self.LOCAL_SAMPLE_SIZE
        self.mu_f = self.MU_F_INITIAL
        self.mu_cr = np.full(self.GROUPS, self.MU_CR_INITIAL, dtype=float)
        self.group_probability = np.full(self.GROUPS, 1.0 / self.GROUPS, dtype=float)
        self.archive_x: list[np.ndarray] = []
        self.archive_y: list[float] = []
        self.external_archive: list[np.ndarray] = []
        self.history: list[dict[str, Any]] = []
        self.global_switch = 1
        self.local_switch = 1
        self.action_count = 0
        self.global_model: _SADE_SSRBF | None = None
        self.local_model: _SADE_SSRBF | None = None
        self.last_local_bounds: tuple[np.ndarray, np.ndarray] | None = None
        self.last_action = "initial_lhs"
        self.radius_formula = "Algorithm 4 R_j = alpha * max(Xc-lb, ub-Xc)"
        self.backend = ComputeBackend("cpu", config.gpu_device, config.gpu_min_elements)

    @property
    def evaluations(self) -> int:
        return len(self.archive_y)

    def _latin_hypercube(self, count: int, lower: np.ndarray | None = None, upper: np.ndarray | None = None) -> np.ndarray:
        common = getattr(self.benchmark, "initial_design", None)
        if lower is None and upper is None and common is not None:
            return np.asarray(common, dtype=float)[: int(count)].copy()
        lower = self.lower if lower is None else np.asarray(lower, dtype=float)
        upper = self.upper if upper is None else np.asarray(upper, dtype=float)
        count = int(count)
        if count <= 0:
            return np.empty((0, self.dimension), dtype=float)
        result = np.empty((count, self.dimension), dtype=float)
        for j in range(self.dimension):
            order = self.rng.permutation(count)
            result[:, j] = (order + self.rng.random(count)) / count
        return lower + result * (upper - lower)

    def _evaluate_exact(self, points: np.ndarray, action: str) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=float).reshape(-1, self.dimension)
        remaining = max(0, self.budget - self.evaluations)
        points = points[:remaining]
        if not len(points):
            return points, np.empty(0, dtype=float)
        tensor = torch.as_tensor(points, dtype=torch.float64)
        values = self.benchmark.problem.evaluate(tensor).detach().cpu().numpy().reshape(-1)
        for point, value in zip(points, values):
            self.archive_x.append(point.copy())
            self.archive_y.append(float(value))
        self.last_action = action
        self.action_count += 1
        best = float(np.min(self.archive_y))
        self.history.append({
            "generation": float(self.action_count),
            "evaluations": float(self.evaluations),
            "best": best,
            "real_evaluations": float(len(points)),
            "search": action,
            "global_switch": int(self.global_switch),
            "local_switch": int(self.local_switch),
        })
        return points, values

    def _top(self, count: int) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(np.asarray(self.archive_y, dtype=float))[: min(int(count), self.evaluations)]
        return np.asarray([self.archive_x[i] for i in order], dtype=float), np.asarray([self.archive_y[i] for i in order], dtype=float)

    def _fit_model(self, count: int) -> _SADE_SSRBF:
        points, values = self._top(count)
        model = _SADE_SSRBF(self.bounds)
        model.fit(points, values)
        return model

    def _binomial_crossover(self, parent: np.ndarray, mutant: np.ndarray, cr: float) -> np.ndarray:
        mask = self.rng.random(self.dimension) < float(np.clip(cr, 0.0, 1.0))
        mask[self.rng.integers(0, self.dimension)] = True
        return np.where(mask, mutant, parent)

    def _global_trials(self, population: np.ndarray, early: bool) -> np.ndarray:
        n = len(population)
        if n < 4:
            return np.tile(population[0], (n, 1))
        best = population[int(np.argmin(self.global_model.predict(population)))] if self.global_model else population[0]
        trials = np.empty_like(population)
        for i in range(n):
            choices = [idx for idx in range(n) if idx != i]
            a, b, c = self.rng.choice(choices, size=3, replace=False)
            if early:
                mutant = population[a] + 0.5 * (population[b] - population[c])
            else:
                mutant = best + 0.5 * (population[b] - population[c])
            trials[i] = np.clip(self._binomial_crossover(population[i], mutant, 0.8), self.lower, self.upper)
        return trials

    def _global_search(self) -> bool:
        if self.evaluations < 3:
            return False
        self.global_model = self._fit_model(self.gs)
        population, _ = self._top(self.GLOBAL_POPULATION_SIZE)
        early = self.evaluations <= self.GLOBAL_EARLY_FRACTION * self.budget
        trials = self._global_trials(population, early)
        predictions = self.global_model.predict(trials)
        candidate = trials[int(np.argmin(predictions))]
        previous = min(self.archive_y)
        self._evaluate_exact(candidate.reshape(1, -1), "global_de_rand_1_bin" if early else "global_de_best_1_bin")
        improved = min(self.archive_y) <= previous
        self.global_switch = int(improved)
        return improved

    def _tightened_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        local_samples, _ = self._top(self.ls)
        lb = np.min(local_samples, axis=0)
        ub = np.max(local_samples, axis=0)
        alpha = float(self.evaluations / max(self.budget, 1))
        center = np.mean(local_samples, axis=0)
        # Algorithm 4 in the paper explicitly uses alpha here.  Eq. (12)
        # prints t=0.01 instead; retain the algorithm-box form and expose the
        # discrepancy in the audit so it cannot be mistaken for a silent edit.
        radius = alpha * np.maximum(center - self.lower, self.upper - center)
        tlb = np.maximum(self.lower, lb + radius)
        tub = np.minimum(self.upper, ub - radius)
        swap = tlb > tub
        if np.any(swap):
            midpoint = center[swap]
            tlb[swap], tub[swap] = midpoint, midpoint
        self.last_local_bounds = (tlb.copy(), tub.copy())
        return tlb, tub

    def _sample_group(self) -> int:
        return int(self.rng.choice(self.GROUPS, p=self.group_probability))

    def _local_trials(self, population: np.ndarray, model: _SADE_SSRBF) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(population)
        predictions = model.predict(population)
        order = np.argsort(predictions)
        pbest = population[order[: max(2, int(math.ceil(0.2 * n)))]]
        trials = np.empty_like(population)
        groups = np.empty(n, dtype=int)
        fs = np.empty(n, dtype=float)
        crs = np.empty(n, dtype=float)
        source_archive = self.external_archive + self.archive_x
        archive = np.asarray(source_archive, dtype=float)
        for i in range(n):
            group = self._sample_group()
            groups[i] = group
            f = float(np.clip(self.mu_f + 0.1 * self.rng.standard_cauchy(), 0.0, 1.0))
            cr = float(np.clip(self.mu_cr[group] + 0.1 * self.rng.normal(), 0.0, 1.0)) if self.mu_cr[group] > 0 else 0.0
            fs[i], crs[i] = f, cr
            r1 = self.rng.integers(0, len(archive))
            r2 = self.rng.integers(0, len(archive))
            while len(archive) > 1 and r2 == r1:
                r2 = self.rng.integers(0, len(archive))
            target = population[i]
            guide = pbest[self.rng.integers(0, len(pbest))]
            mutant = target + f * (guide - target) + f * (archive[r1] - archive[r2])
            trials[i] = np.clip(self._binomial_crossover(target, mutant, cr), self.lower, self.upper)
        return trials, predictions, groups, fs, crs

    def _update_local_parameters(self, groups: np.ndarray, fs: np.ndarray, crs: np.ndarray, old_pred: np.ndarray, new_pred: np.ndarray) -> None:
        success = new_pred < old_pred
        if not np.any(success):
            return
        delta = np.maximum(old_pred[success] - new_pred[success], 1e-12)
        weights = delta / np.sum(delta)
        sf, scr = fs[success], crs[success]
        self.mu_f = float(np.clip(np.sum(weights * sf * sf) / max(np.sum(weights * sf), 1e-12), 0.0, 1.0))
        for group in range(self.GROUPS):
            mask = success & (groups == group)
            if np.any(mask):
                w = np.maximum(old_pred[mask] - new_pred[mask], 1e-12)
                w /= np.sum(w)
                self.mu_cr[group] = float(np.clip(np.sum(w * crs[mask] * crs[mask]) / max(np.sum(w * crs[mask]), 1e-12), 0.0, 1.0))
        success_counts = np.bincount(groups[success], minlength=self.GROUPS).astype(float)
        failure_counts = np.bincount(groups[~success], minlength=self.GROUPS).astype(float)
        total_success = max(float(np.sum(success_counts)), 1.0)
        r = np.where(success_counts > 0, success_counts * success_counts / (total_success * (success_counts + failure_counts + 1e-12)), np.finfo(float).eps)
        self.group_probability = r / np.sum(r)

    def _local_search(self) -> bool:
        if self.evaluations < 3:
            return False
        tlb, tub = self._tightened_bounds()
        self.local_model = self._fit_model(self.ls)
        population = self._latin_hypercube(self.LOCAL_POPULATION_MAX, tlb, tub)
        population = np.clip(population, tlb, tub)
        best_pred = float("inf")
        best_point = population[0]
        ps = self.LOCAL_POPULATION_MAX
        for nfa in range(self.LOCAL_NFAS_MAX):
            population = population[: max(self.LOCAL_POPULATION_MIN, min(ps, len(population)))]
            trials, parent_pred, groups, fs, crs = self._local_trials(population, self.local_model)
            trial_pred = self.local_model.predict(trials)
            all_points = np.vstack((population, trials))
            all_pred = np.concatenate((parent_pred, trial_pred))
            best_index = int(np.argmin(all_pred))
            if float(all_pred[best_index]) < best_pred:
                best_pred = float(all_pred[best_index])
                best_point = all_points[best_index].copy()
            selected = np.argsort(all_pred)[: len(population)]
            old_selected = parent_pred
            new_selected = trial_pred
            for parent, is_success in zip(population, new_selected < old_selected):
                if bool(is_success):
                    self.external_archive.append(np.asarray(parent, dtype=float).copy())
            # Keep the external archive bounded, as in the DE archive used by
            # the local optimizer, so high-dimensional runs do not grow it
            # without limit.
            max_archive = max(self.LOCAL_POPULATION_MAX, 2 * self.LOCAL_SAMPLE_SIZE)
            if len(self.external_archive) > max_archive:
                keep = self.rng.choice(len(self.external_archive), size=max_archive, replace=False)
                self.external_archive = [self.external_archive[int(i)] for i in keep]
            self._update_local_parameters(groups, fs, crs, old_selected, new_selected)
            population = all_points[selected]
            # Eq. (17) as printed in the paper (the prose calls this
            # reduction, although the displayed sign increases with NFE).
            ps = int(np.clip(round((self.LOCAL_POPULATION_MAX - self.LOCAL_POPULATION_MIN) / max(self.LOCAL_NFAS_MAX, 1) * (nfa + 1) + self.LOCAL_POPULATION_MIN), self.LOCAL_POPULATION_MIN, self.LOCAL_POPULATION_MAX))
            if len(population) == 0:
                break
        previous = min(self.archive_y)
        self._evaluate_exact(best_point.reshape(1, -1), "local_de_current_pbest_1_bin")
        improved = min(self.archive_y) <= previous
        self.local_switch = int(improved)
        return improved

    def run(self) -> tuple[np.ndarray, float, int, list[dict[str, Any]], dict[str, Any]]:
        initial = self._latin_hypercube(min(self.tau, self.budget))
        self._evaluate_exact(initial, "initial_lhs")
        while self.evaluations < self.budget:
            if self.global_switch == 0 and self.local_switch == 0:
                if self.rng.random() > 0.5:
                    self.global_switch = 1
                else:
                    self.local_switch = 1
            if self.global_switch and self.evaluations < self.budget:
                self._global_search()
            if self.local_switch and self.evaluations < self.budget:
                self._local_search()
        best = int(np.argmin(self.archive_y))
        return np.asarray(self.archive_x[best], dtype=float), float(self.archive_y[best]), self.evaluations, list(self.history), self.audit()

    def audit(self) -> dict[str, Any]:
        return {
            "paper": PAPER_METADATA["sade_ss"],
            "implementation": "faithful_reimplementation_of_SADE_SS_algorithms_1_to_4",
            "true_evaluations": int(self.evaluations),
            "mechanism_state": {
                "initial_sampling": "Latin hypercube (Algorithm 1, line 2)",
                "tau": int(self.tau),
                "global_sample_size_gs": int(self.gs),
                "local_sample_size_ls": int(self.ls),
                "global_de": "DE/rand/1/bin then DE/best/1/bin",
                "global_early_fraction_p": float(self.GLOBAL_EARLY_FRACTION),
                "local_de": "DE/current-pbest/1/bin",
                "gaussian_rbf": True,
                "parameter_adaptation": "Cauchy F, Gaussian CR, weighted Lehmer mean, grouped probabilities",
                "switching": "global/local switch state from Algorithm 1",
                "search_space_tightening": self.radius_formula,
                "eq12_note": "The paper text prints t=0.01 in Eq. (12), while Algorithm 4 uses alpha; implementation follows Algorithm 4.",
                "nfasmax": int(self.LOCAL_NFAS_MAX),
                "nfasmax_note": "numeric nfasmax is not specified in the paper; fixed reproducible value used",
                "paper_nfe_max": int(self.paper_nfe_max),
                "actual_nfe_max": int(self.budget),
                "budget_policy": "shared experiment budget; set to 1000 in the paper",
                "population_size_eq17": "printed equation implemented; paper prose says reduction",
                "last_local_bounds": None if self.last_local_bounds is None else [self.last_local_bounds[0].tolist(), self.last_local_bounds[1].tolist()],
                "external_archive_size": int(len(self.external_archive)),
            },
            "surrogate": {
                "model": "gaussian_rbf_phi_inverse",
                "global_fit_count": 0 if self.global_model is None else int(self.global_model.fit_count),
                "local_fit_count": 0 if self.local_model is None else int(self.local_model.fit_count),
                "backend": self.backend.device_name,
            },
        }


class _LHS:
    """Small deterministic Latin-hypercube sampler shared by paper baselines."""

    def __init__(self, lower: np.ndarray, upper: np.ndarray, rng: np.random.Generator) -> None:
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.rng = rng

    def sample(self, count: int) -> np.ndarray:
        count = int(count)
        if count <= 0:
            return np.empty((0, len(self.lower)), dtype=float)
        result = np.empty((count, len(self.lower)), dtype=float)
        for column in range(len(self.lower)):
            order = self.rng.permutation(count)
            result[:, column] = (order + self.rng.random(count)) / count
        return self.lower + result * (self.upper - self.lower)


class SGDLCOController:
    """Unconstrained EvoX adaptation of SGDLCO's global/local workflow.

    The source paper is a constrained algorithm (CCMO, DCRLE and TLAS). EvoX
    numerical functions expose no constraints, so the feasibility layer of TLAS
    is removed explicitly while its convergence/diversity screening and the
    global/distributed-local alternation are retained.
    """

    def __init__(self, benchmark: Any, config: Any, seed: int) -> None:
        self.benchmark = benchmark
        self.config = config
        self.rng = np.random.default_rng(int(seed))
        self.dimension = int(benchmark.dimension)
        self.bounds = np.asarray(benchmark.bounds, dtype=float)
        self.lower, self.upper = self.bounds[:, 0], self.bounds[:, 1]
        self.span = np.maximum(self.upper - self.lower, 1e-12)
        self.budget = int(config.max_evaluations_for(self.dimension))
        self.batch_size = max(1, int(config.effective_real_batch_size))
        self.archive_x: list[np.ndarray] = []
        self.archive_y: list[float] = []
        self.history: list[dict[str, Any]] = []
        self.population_history: list[np.ndarray] = []
        self.calls = 0
        self.phase = "global"
        self.last_screening = ""
        # Preserve the paper's dimension-aware 2D design where the shared
        # budget permits it; reserve at least two surrogate phases when a
        # high-dimensional benchmark has a smaller FE budget.
        self.initial_size = min(max(3, 2 * self.dimension), max(3, self.budget // 3))
        self.backend = ComputeBackend("cpu", config.gpu_device, config.gpu_min_elements)
        self.model = _RBFModel(self.bounds, self.backend, gaussian=True)
        self.lhs = _LHS(self.lower, self.upper, self.rng)

    @property
    def evaluations(self) -> int:
        return len(self.archive_y)

    def _evaluate(self, points: np.ndarray, phase: str) -> None:
        remaining = self.budget - self.evaluations
        points = np.asarray(points, dtype=float).reshape(-1, self.dimension)[:remaining]
        if not len(points):
            return
        values = self.benchmark.problem.evaluate(torch.as_tensor(points, dtype=torch.float64)).detach().cpu().numpy().reshape(-1)
        self.archive_x.extend(point.copy() for point in points)
        self.archive_y.extend(float(value) for value in values)
        self.population_history.append(points.copy())
        self.calls += 1
        self.phase = phase
        self.history.append({"generation": float(self.calls), "evaluations": float(self.evaluations), "best": float(np.min(self.archive_y)), "real_evaluations": float(len(points)), "phase": phase})

    def _fit(self) -> None:
        if len(self.archive_y) < 3:
            return
        x = np.asarray(self.archive_x, dtype=float)
        y = np.asarray(self.archive_y, dtype=float)
        limit = self.config.max_surrogate_samples
        if limit is not None and len(y) > int(limit):
            x, y = x[-int(limit):], y[-int(limit):]
        self.model.fit(x, y)

    def _global_ccmo(self, count: int) -> np.ndarray:
        archive = np.asarray(self.archive_x, dtype=float)
        order = np.argsort(self.archive_y)
        best = archive[order[0]]
        candidates = np.empty((count, self.dimension), dtype=float)
        for i in range(count):
            indices = self.rng.choice(len(archive), size=4, replace=len(archive) < 4)
            # CCMO-style positive/negative classification guidance: the best
            # point supplies convergence pressure and two differential vectors
            # preserve global exploration.
            f1, f2 = self.rng.uniform(0.35, 0.85, size=2)
            mutant = best + f1 * (archive[indices[0]] - archive[indices[1]]) + f2 * (archive[indices[2]] - archive[indices[3]])
            if self.archive_y[indices[0]] > self.archive_y[indices[1]]:
                mutant += self.rng.uniform(0.0, 0.25) * (best - archive[indices[0]])
            parent = archive[indices[0]]
            mask = self.rng.random(self.dimension) < 0.85
            mask[self.rng.integers(0, self.dimension)] = True
            candidates[i] = np.where(mask, mutant, parent)
        return np.clip(candidates, self.lower, self.upper)

    def _local_dcrle(self, count: int) -> np.ndarray:
        archive = np.asarray(self.archive_x, dtype=float)
        order = np.argsort(self.archive_y)
        pool = archive[order[: min(len(order), max(3, 4 * self.batch_size))]]
        centers = [pool[0]]
        while len(centers) < min(4, len(pool)):
            distance = np.min(np.linalg.norm(pool[:, None, :] - np.asarray(centers)[None, :, :], axis=2), axis=1)
            centers.append(pool[int(np.argmax(distance))])
        result = []
        for i in range(count):
            center = centers[i % len(centers)]
            scale = self.span * (0.04 + 0.12 * (1.0 - self.evaluations / max(self.budget, 1)))
            result.append(center + self.rng.normal(size=self.dimension) * scale)
        return np.clip(np.asarray(result), self.lower, self.upper)

    def _tlas_screen(self, candidates: np.ndarray) -> np.ndarray:
        prediction = self.model.predict(candidates)
        normalized = (prediction - np.min(prediction)) / max(float(np.ptp(prediction)), 1e-12)
        archive = np.asarray(self.archive_x, dtype=float)
        diversity = np.min(np.linalg.norm(candidates[:, None, :] - archive[None, :, :], axis=2), axis=1)
        diversity = (diversity - np.min(diversity)) / max(float(np.ptp(diversity)), 1e-12)
        # TLAS has feasibility, diversity and convergence layers. With an
        # empty constraint set, retain the latter two layers explicitly.
        score = 0.7 * normalized - 0.3 * diversity
        self.last_screening = "TLAS(convergence+diversity; feasibility layer empty)"
        return candidates[np.argsort(score)[: min(self.batch_size, self.budget - self.evaluations)]]

    def run(self) -> tuple[np.ndarray, float, int, list[dict[str, Any]], dict[str, Any]]:
        common = getattr(self.benchmark, "initial_design", None)
        initial = (
            np.asarray(common, dtype=float)[: min(self.initial_size, self.budget)].copy()
            if common is not None
            else self.lhs.sample(min(self.initial_size, self.budget))
        )
        self._evaluate(initial, "initial_lhs")
        while self.evaluations < self.budget:
            self._fit()
            count = min(max(32, 4 * self.batch_size), 128)
            if self.phase in {"initial_lhs", "local"}:
                candidates = self._global_ccmo(count)
                phase = "global"
            else:
                candidates = self._local_dcrle(count)
                phase = "local"
            self._evaluate(self._tlas_screen(candidates), phase)
        best = int(np.argmin(self.archive_y))
        return self.archive_x[best].copy(), float(self.archive_y[best]), self.evaluations, list(self.history), self.audit()

    def audit(self) -> dict[str, Any]:
        return {"paper": PAPER_METADATA["sgdlco"], "implementation": "SGDLCO_global_CCMO_local_DCRLE_TLAS_unconstrained_adaptation", "true_evaluations": self.evaluations, "surrogate": self.model.audit(), "mechanism_state": {"initial_sampling": "Latin hypercube", "initial_population_size": int(self.initial_size), "global_operator": "CCMO-style classified differential mutation", "local_operator": "DCRLE-style distributed centers with local exploration", "screening": self.last_screening, "constraint_adaptation": "empty constraint set for EvoX numerical benchmarks", "global_local_alternation": True, "budget_adaptation": "2D initial design capped at one third of the shared FE budget"}}


class _SAGPE_CubicRBF:
    """Cubic RBF plus linear polynomial term from SAGPE's model equations."""

    def __init__(self, bounds: np.ndarray) -> None:
        self.bounds = np.asarray(bounds, dtype=float)
        self.centers = np.empty((0, len(bounds)), dtype=float)
        self.weights: np.ndarray | None = None
        self.linear: np.ndarray | None = None
        self.fit_count = 0

    def _normalize(self, points: np.ndarray) -> np.ndarray:
        return 2.0 * (np.asarray(points, dtype=float) - self.bounds[:, 0]) / np.maximum(self.bounds[:, 1] - self.bounds[:, 0], 1e-12) - 1.0

    def fit(self, points: np.ndarray, values: np.ndarray) -> None:
        self.centers = np.asarray(points, dtype=float).copy()
        x = self._normalize(self.centers)
        y = np.asarray(values, dtype=float).reshape(-1)
        distances = np.sqrt(np.maximum(np.sum(x * x, axis=1, keepdims=True) + np.sum(x * x, axis=1)[None, :] - 2.0 * (x @ x.T), 0.0))
        phi = distances ** 3
        p = np.column_stack((np.ones(len(x)), x))
        system = np.block([[phi, p], [p.T, np.zeros((p.shape[1], p.shape[1]))]])
        rhs = np.concatenate((y, np.zeros(p.shape[1])))
        try:
            solution = np.linalg.solve(system, rhs)
        except np.linalg.LinAlgError:
            solution = np.linalg.pinv(system, rcond=1e-10) @ rhs
        self.weights, self.linear = solution[: len(x)], solution[len(x):]
        self.fit_count += 1

    def predict(self, points: np.ndarray) -> np.ndarray:
        x = self._normalize(points)
        c = self._normalize(self.centers)
        distances = np.sqrt(np.maximum(np.sum(x * x, axis=1, keepdims=True) + np.sum(c * c, axis=1)[None, :] - 2.0 * (x @ c.T), 0.0))
        p = np.column_stack((np.ones(len(x)), x))
        return distances ** 3 @ self.weights + p @ self.linear

    def audit(self) -> dict[str, Any]:
        return {"model": "cubic_rbf_with_linear_polynomial", "kernel": "phi(r)=r^3", "fit_count": self.fit_count, "center_count": len(self.centers)}


class SAGPEController:
    """SAGPE workflow: EGM(1,1)-style generations, RBF screening and learning."""

    ETA = 1e-7

    def __init__(self, benchmark: Any, config: Any, seed: int) -> None:
        self.benchmark = benchmark
        self.config = config
        self.rng = np.random.default_rng(int(seed))
        self.dimension = int(benchmark.dimension)
        self.bounds = np.asarray(benchmark.bounds, dtype=float)
        self.lower, self.upper = self.bounds[:, 0], self.bounds[:, 1]
        self.span = np.maximum(self.upper - self.lower, 1e-12)
        self.budget = int(config.max_evaluations_for(self.dimension))
        self.batch_size = max(1, int(config.effective_real_batch_size))
        self.population_size = min(max(4, 2 * self.dimension), max(4, self.budget // 3))
        self.tau = 2 * self.dimension
        self.archive_x: list[np.ndarray] = []
        self.archive_y: list[float] = []
        self.populations: list[np.ndarray] = []
        self.history: list[dict[str, Any]] = []
        self.calls = 0
        self.phase = "global"
        self.gray_fallbacks = 0
        self.backend = ComputeBackend("cpu", config.gpu_device, config.gpu_min_elements)
        self.initial_size = min(max(3, self.tau), max(3, self.budget // 3), self.budget)
        self.model = _SAGPE_CubicRBF(self.bounds)
        self.lhs = _LHS(self.lower, self.upper, self.rng)

    @property
    def evaluations(self) -> int:
        return len(self.archive_y)

    def _evaluate(self, points: np.ndarray, phase: str) -> None:
        points = np.asarray(points, dtype=float).reshape(-1, self.dimension)[: self.budget - self.evaluations]
        if not len(points):
            return
        values = self.benchmark.problem.evaluate(torch.as_tensor(points, dtype=torch.float64)).detach().cpu().numpy().reshape(-1)
        self.archive_x.extend(point.copy() for point in points)
        self.archive_y.extend(float(value) for value in values)
        self.populations.append(points.copy())
        self.populations = self.populations[-3:]
        self.calls += 1
        self.phase = phase
        self.history.append({"generation": float(self.calls), "evaluations": float(self.evaluations), "best": float(np.min(self.archive_y)), "real_evaluations": float(len(points)), "phase": phase})

    def _fit(self) -> None:
        x = np.asarray(self.archive_x, dtype=float)
        y = np.asarray(self.archive_y, dtype=float)
        limit = self.config.max_surrogate_samples
        if limit is not None and len(y) > int(limit):
            x, y = x[-int(limit):], y[-int(limit):]
        if len(y) >= 3:
            self.model.fit(x, y)

    def _gray_population(self) -> np.ndarray:
        if len(self.populations) < 3:
            self.gray_fallbacks += 1
            center = np.asarray(self.archive_x[int(np.argmin(self.archive_y))])
            return np.clip(center + self.rng.normal(scale=0.12, size=(self.population_size, self.dimension)) * self.span, self.lower, self.upper)
        p0, p1, p2 = self.populations[-3:]
        m = min(len(p0), len(p1), len(p2), self.population_size)
        p0, p1, p2 = p0[:m], p1[:m], p2[:m]
        step = p2 - p1
        if float(np.linalg.norm(step)) < self.ETA:
            self.gray_fallbacks += 1
            return np.clip(p2 + self.rng.normal(scale=0.08, size=p2.shape) * self.span, self.lower, self.upper)
        # EGM(1,1) fallback/forecast applied coordinate-wise to the three
        # population generations; finite differences keep it stable in high D.
        forecast = p2 + step + 0.5 * (p2 - 2.0 * p1 + p0)
        return np.clip(forecast, self.lower, self.upper)

    def _inferior_learning(self, candidates: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        incumbent = np.asarray(self.archive_x[int(np.argmin(self.archive_y))])
        worse = predicted > float(np.min(predicted))
        learned = candidates.copy()
        learned[worse] += self.rng.random((int(np.sum(worse)), self.dimension)) * (incumbent - learned[worse])
        return np.clip(learned, self.lower, self.upper)

    def run(self) -> tuple[np.ndarray, float, int, list[dict[str, Any]], dict[str, Any]]:
        initial = self.lhs.sample(self.initial_size)
        self._evaluate(initial, "initial_lhs")
        while self.evaluations < self.budget:
            self._fit()
            candidates = self._gray_population()
            if len(self.archive_y) >= 3:
                predicted = self.model.predict(candidates)
                candidates = self._inferior_learning(candidates, predicted)
                predicted = self.model.predict(candidates)
                order = np.argsort(predicted)
                selected = candidates[order[: min(self.batch_size, self.budget - self.evaluations)]]
            else:
                selected = candidates[: min(self.batch_size, self.budget - self.evaluations)]
            self._evaluate(selected, self.phase)
            self.phase = "local" if self.phase == "global" else "global"
        best = int(np.argmin(self.archive_y))
        return self.archive_x[best].copy(), float(self.archive_y[best]), self.evaluations, list(self.history), self.audit()

    def audit(self) -> dict[str, Any]:
        return {"paper": PAPER_METADATA["sagpe"], "implementation": "SAGPE_EGM_population_RBF_global_local_switch", "true_evaluations": self.evaluations, "surrogate": self.model.audit(), "mechanism_state": {"initial_sampling": "Latin hypercube", "initial_population_size": int(self.initial_size), "population_size": int(self.population_size), "tau": int(self.tau), "eta": self.ETA, "gray_prediction": "three-generation coordinate-wise EGM forecast with random perturbation fallback", "inferior_offspring_learning": True, "global_local_switch": "alternating on surrogate search phases", "gray_fallbacks": int(self.gray_fallbacks), "budget_adaptation": "2D initial design capped at one third of the shared FE budget"}}


class RecentPaperProblem(Problem):
    def __init__(self, controller: RecentPaperController) -> None:
        super().__init__()
        self.controller = controller
        self.evaluations = 0

    def evaluate(self, population: torch.Tensor) -> torch.Tensor:
        values = self.controller.evaluate(population)
        self.evaluations = self.controller.evaluations
        return values


class RecentPaperAlgorithm(Algorithm):
    def __init__(self, controller: RecentPaperController) -> None:
        super().__init__()
        self.controller = controller

    def _random_population(self) -> torch.Tensor:
        device = torch.device(
            "cpu"
            if self.controller.execution_device == "cpu"
            else f"cuda:{self.controller.config.gpu_device}"
        )
        lower = torch.as_tensor(self.controller.benchmark.bounds[:, 0], dtype=torch.float64, device=device)
        upper = torch.as_tensor(self.controller.benchmark.bounds[:, 1], dtype=torch.float64, device=device)
        return lower + torch.rand(
            (self.controller.config.population_size, self.controller.benchmark.dimension),
            device=device,
        ) * (upper - lower)

    def init_step(self) -> None:
        self.evaluate(self._random_population())

    def step(self) -> None:
        if self.controller.evaluations < self.controller.budget:
            self.evaluate(self._random_population())

    def final_step(self) -> None:
        self.step()

    def record_step(self) -> dict[str, Any]:
        return {
            "evaluations": torch.tensor(
                self.controller.evaluations,
                device=(
                    "cpu"
                    if self.controller.execution_device == "cpu"
                    else f"cuda:{self.controller.config.gpu_device}"
                ),
            )
        }


class _LocalPaperConfig:
    """Minimal configuration view shared by the local comparison ports."""

    def __init__(self, suite_config: Any) -> None:
        self.population_size = int(suite_config.population_size)
        self.total_evals = int(suite_config.total_evals)
        self.initial_evals = int(suite_config.initial_evals)
        self.gpu_device = 0
        self.gpu_min_elements = 1
        self.max_surrogate_samples = min(128, self.total_evals)
        self.surrogate_refit_interval = 5

    def max_evaluations_for(self, dimension: int) -> int:
        del dimension
        return self.total_evals

    @property
    def effective_real_batch_size(self) -> int:
        return max(1, self.population_size // 4)


def _history_rows(
    history: list[dict[str, Any]],
    evaluations: int,
    best_f: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in history:
        value = item.get("evaluations", item.get("real_evaluations"))
        if value is None:
            continue
        rows.append(
            {
                "evaluations": int(value),
                "best_f": float(item.get("best", best_f)),
            }
        )
    if not rows or rows[-1]["evaluations"] < int(evaluations):
        rows.append({"evaluations": int(evaluations), "best_f": float(best_f)})
    return rows


def run_paper_algorithm(
    method: str,
    problem: str,
    dimension: int,
    seed: int,
    config: Any,
) -> dict[str, Any]:
    """Run one of the three local paper-baseline implementations.

    This entry point intentionally receives the suite benchmark through the
    local package.  It never imports the user's separate ``SAEA(2)`` checkout,
    so the comparison remains runnable after that checkout is moved or absent.
    """

    from ..expensive_suite import _shared_initial_design, get_suite_benchmark

    benchmark = get_suite_benchmark(problem, int(dimension))
    benchmark.initial_design = _shared_initial_design(config.initial_evals, benchmark.bounds, seed)
    local_config = _LocalPaperConfig(config)
    method = str(method)
    if method == "SADE-SS":
        controller: Any = SADE_SSController(benchmark, local_config, int(seed))
        controller.tau = min(local_config.initial_evals, local_config.total_evals)
        controller.gs = controller.tau
        best_x, best_f, evaluations, history, audit = controller.run()
    elif method == "SGDLCO":
        controller = SGDLCOController(benchmark, local_config, int(seed))
        controller.initial_size = min(
            max(3, local_config.initial_evals),
            local_config.total_evals,
        )
        best_x, best_f, evaluations, history, audit = controller.run()
    elif method == "BGP-SAEA":
        controller = RecentPaperController(
            benchmark,
            local_config,
            "bgp_saea",
            int(seed),
        )
        rng = np.random.default_rng(int(seed))
        lower, upper = benchmark.bounds[:, 0], benchmark.bounds[:, 1]
        while controller.evaluations < local_config.total_evals:
            remaining = local_config.total_evals - controller.evaluations
            count = min(
                local_config.initial_evals
                if controller.evaluations == 0
                else local_config.population_size,
                remaining,
            )
            if controller.evaluations == 0 and getattr(benchmark, "initial_design", None) is not None:
                points = np.asarray(benchmark.initial_design, dtype=float)[:count].copy()
            else:
                points = rng.uniform(lower, upper, size=(count, int(dimension)))
            controller.evaluate(torch.as_tensor(points, dtype=torch.float64))
        best_index = int(np.argmin(controller.archive_y))
        best_x = np.asarray(controller.archive_x[best_index], dtype=float)
        best_f = float(controller.archive_y[best_index])
        evaluations = int(controller.evaluations)
        history = list(controller.history)
        audit = controller.audit()
    else:
        raise ValueError(f"unsupported local paper baseline: {method}")

    del best_x
    evaluations = int(evaluations)
    best_f = float(best_f)
    audit = dict(audit or {})
    audit.update(
        {
            "implementation": "local_expensive_algorithms",
            "runtime_external_dependency": False,
            "true_evaluations": evaluations,
        }
    )
    return {
        "algorithm": method,
        "problem": problem,
        "dimension": int(dimension),
        "seed": int(seed),
        "best_f": best_f,
        "initial_best_f": float(_history_rows(history, evaluations, best_f)[0]["best_f"]),
        "real_evals": evaluations,
        "initial_evals": int(config.initial_evals),
        "best_f_history": _history_rows(history, evaluations, best_f),
        "audit": audit,
    }


__all__ = [
    "PAPER_METADATA",
    "RecentPaperAlgorithm",
    "RecentPaperController",
    "RecentPaperProblem",
    "SAGPEController",
    "SADE_SSController",
    "SGDLCOController",
    "run_paper_algorithm",
]
