import time

import numpy as np
from scipy import linalg
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from pathlib import Path

from scripts.metrics import normalized_frame_error

pymanopt = None
manifolds = None
optimizers = None

ro = None
numpy2ri = None
localconverter = None
_r_varimax_loaded = False

# Helper functions for R dependencies. 
def ensure_r_varimax_deps():
    global ro, numpy2ri, localconverter, _r_varimax_loaded

    if _r_varimax_loaded:
        return

    try:
        import rpy2.robjects as ro_module
        from rpy2.robjects import numpy2ri as numpy2ri_module
        from rpy2.robjects.conversion import localconverter as localconverter_fn
    except ImportError as exc:
        raise ImportError(
            "Running pca_varimax with R requires rpy2 and a working R installation. "
            "Install R, then run `pip install rpy2`."
        ) from exc

    ro = ro_module
    numpy2ri = numpy2ri_module
    localconverter = localconverter_fn

    r_script = Path(__file__).resolve().parent / "r_varimax.R"
    ro.r.source(str(r_script))

    _r_varimax_loaded = True


def np_to_r(x):
    ensure_r_varimax_deps()
    with localconverter(ro.default_converter + numpy2ri.converter):
        return ro.conversion.py2rpy(np.asarray(x, dtype=float))


def r_to_np(x):
    ensure_r_varimax_deps()
    with localconverter(ro.default_converter + numpy2ri.converter):
        return np.asarray(ro.conversion.rpy2py(x), dtype=float)

def ensure_optimization_deps():
    global pymanopt, manifolds, optimizers
    if pymanopt is not None:
        return
    try:
        import autograd.numpy as autograd_np
        import pymanopt as pymanopt_module
        import pymanopt.manifolds as pymanopt_manifolds
        import pymanopt.optimizers as pymanopt_optimizers
    except ImportError as exc:
        raise ImportError(
            'Regenerating experiment CSVs requires pymanopt and autograd. '
            'Create the conda environment with `conda env create -f environment.yml` '
            'and activate it with `conda activate wamfigs`.'
        ) from exc
    globals()['np'] = autograd_np
    pymanopt = pymanopt_module
    manifolds = pymanopt_manifolds
    optimizers = pymanopt_optimizers


def get_adaptive_iterations(dim):
    if dim <= 10:
        return 1000
    if dim <= 25:
        return 3000
    return 5000


def wfa1_cost_factory(manifold, data, use_mean=True):
    ensure_optimization_deps()

    @pymanopt.function.autograd(manifold)
    def cost(basis_matrix):
        dist = []
        for point in data:
            closest_axis_index = np.argmax(np.abs(np.dot(point, basis_matrix)))
            closest_axis = basis_matrix[:, closest_axis_index]
            projected_point = np.dot(point, closest_axis) * closest_axis
            squared_distance = np.linalg.norm((point - projected_point) ** 2)
            dist.append(squared_distance)
        if use_mean:
            return (1 / data.shape[0]) * np.sum(dist)
        return (1 / data.size) * np.trapz(dist)

    return cost


def run_wfa1(data, dim, initial_point=None, log_verbosity=0, max_iterations=None):
    ensure_optimization_deps()
    start_time = time.time()
    manifold = manifolds.Stiefel(dim, dim)
    if max_iterations is None:
        max_iterations = get_adaptive_iterations(dim)
    solver = optimizers.SteepestDescent(
        verbosity=0,
        log_verbosity=log_verbosity,
        max_iterations=max_iterations,
        min_gradient_norm=1e-10,
        min_step_size=1e-12,
    )
    problem = pymanopt.Problem(manifold=manifold, cost=wfa1_cost_factory(manifold, data))
    if initial_point is None:
        initial_point = manifold.random_point()
    result = solver.run(problem, initial_point=initial_point)
    runtime = time.time() - start_time
    details = {
        'gradient_norm': getattr(result, 'gradient_norm', None),
        'iterations': getattr(result, 'iterations', None),
        'step_size': getattr(result, 'step_size', None),
        'stopping_criterion': getattr(result, 'stopping_criterion', None),
        'log': getattr(result, 'log', None),
    }
    return result.point, runtime, True, details


class DeflationVarimax:
    def __init__(self, r, s=None, method='random'):
        self.r = r
        self.s = s if s is not None else r
        self.method = method
        self.Q_hat = None

    def initialize_next_row(self, U_r=None, N=50):
        r = self.Q_hat.shape[0]
        non_zero_rows = 0
        for i in range(r):
            if np.any(self.Q_hat[i] != 0):
                non_zero_rows += 1
            else:
                break

        P_perp = np.eye(r)
        for i in range(non_zero_rows):
            P_perp -= np.outer(self.Q_hat[i], self.Q_hat[i])

        if self.method == 'random':
            if non_zero_rows == 0:
                g = np.random.normal(size=r)
                return g / np.linalg.norm(g)
            eigvals, eigvecs = np.linalg.eigh(P_perp)
            leading_count = r - non_zero_rows
            V_perp = eigvecs[:, r - leading_count:]
            g = np.random.normal(size=leading_count)
            q_init = V_perp @ g
            return q_init / np.linalg.norm(q_init)

        if self.method == 'moments':
            if U_r is None:
                raise ValueError('U_r is required for moments initialization.')
            max_gap = -1
            best_v1 = None
            for _ in range(N):
                G = np.random.normal(size=(r, r))
                M_hat = np.zeros((r, r))
                for t in range(U_r.shape[1]):
                    u_t = U_r[:, t:t + 1]
                    M_hat += (u_t @ u_t.T) * (u_t.T @ G @ u_t)
                M_hat = M_hat / (3 * U_r.shape[1]) - G - G.T
                if non_zero_rows > 0:
                    M_hat = P_perp @ M_hat @ P_perp
                try:
                    U, s, _ = linalg.svd(M_hat, full_matrices=False)
                except Exception:
                    continue
                if len(s) >= 2 and s[0] - s[1] > max_gap:
                    max_gap = s[0] - s[1]
                    best_v1 = U[:, 0]
            if best_v1 is not None:
                return best_v1
            self.method = 'random'
            return self.initialize_next_row(U_r=U_r)

        raise ValueError(f'Unknown initialization method: {self.method}')

    def fit(self, U):
        ensure_optimization_deps()
        r, n = U.shape
        s = min(self.s, r)
        self.Q_hat = np.zeros((s, r))
        for k in range(s):
            manifold = manifolds.Sphere(r)

            @pymanopt.function.autograd(manifold)
            def cost(q):
                Uq = U.T @ q
                return -np.sum(Uq ** 4) / (12 * n)

            problem = pymanopt.Problem(manifold=manifold, cost=cost)
            optimizer = optimizers.SteepestDescent(verbosity=0, max_iterations=get_adaptive_iterations(r))
            result = optimizer.run(problem, initial_point=self.initialize_next_row(U_r=U))
            self.Q_hat[k] = result.point

        U_svd, _, V_svd = np.linalg.svd(self.Q_hat, full_matrices=False)
        return U_svd @ V_svd


class PCAWithDeflationVarimax:
    def __init__(self, r, initialization='random'):
        self.r = r
        self.initialization = initialization

    def fit(self, X):
        pca = PCA(n_components=self.r)
        pca.fit(X)
        V_r = pca.components_.T
        D_r = np.diag(np.sqrt(pca.explained_variance_))
        X_centered = X - pca.mean_
        U_r = D_r @ V_r.T @ X_centered.T
        varimax = DeflationVarimax(self.r, method=self.initialization)
        Q = varimax.fit(U_r)
        Lambda_hat = V_r @ D_r @ Q
        _, singular_values, _ = np.linalg.svd(Lambda_hat)
        Lambda_hat = Lambda_hat / singular_values[0]
        self.components_ = Q
        self.loadings_ = Lambda_hat
        self.mean_ = pca.mean_
        return self


def run_deflation_varimax(data, dim, initialization='random'):
    start_time = time.time()
    model = PCAWithDeflationVarimax(r=dim, initialization=initialization)
    model.fit(data)
    return model.components_.T, time.time() - start_time, True, {
        'stopping_criterion': 'Deflation algorithm (non-pymanopt)'
    }


def varimax_loss_factory(manifold, pca_components):
    """Kaiser-style Varimax loss from the original comparison experiment."""
    ensure_optimization_deps()

    @pymanopt.function.autograd(manifold)
    def cost(Rot):
        UR = pca_components @ Rot
        _, k = UR.shape
        varimax_sum = 0.0
        for col in range(k):
            column = UR[:, col]
            fourth_moment = np.mean(column ** 4)
            second_moment_squared = np.mean(column ** 2) ** 2
            varimax_sum += fourth_moment - second_moment_squared
        return varimax_sum

    return cost


def run_pca_varimax(data, dim):
    ensure_r_varimax_deps()

    start_time = time.time()

    pca = PCA(n_components=dim)
    pca.fit(data)

    components = pca.components_  # dim x p

    res = ro.globalenv["pca_varimax_from_components"](
        np_to_r(components),
        normalize=False,
        eps=1e-5,
    )

    rotated_components_t = r_to_np(res.rx2("rotated"))  # p x dim
    rotmat = r_to_np(res.rx2("rotmat"))                 # dim x dim

    return rotated_components_t, time.time() - start_time, True, {
        "stopping_criterion": "R stats::varimax",
        "normalize": False,
        "rotmat_shape": rotmat.shape,
    }

def solver_rsc(Y, lambda_):
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise ImportError(
            'Regenerating RSC comparison CSVs requires a working cvxpy installation. '
            'Create the conda environment with `conda env create -f environment.yml` '
            'and activate it with `conda activate wamfigs`; if cvxpy is already installed, '
            'check that its version is compatible with the installed NumPy version.'
        ) from exc

    N = Y.shape[1]
    Z = cp.Variable((N, N))
    objective = cp.Minimize(0.5 * cp.norm(Y - Y @ Z, 'fro') ** 2 + lambda_ * cp.norm(Z, 1))
    constraints = [cp.diag(Z) == 0]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    if Z.value is None:
        raise RuntimeError('CVXPY did not return an RSC solution.')
    Z_result = np.zeros((N, N))
    for i in range(N):
        Z_result[i, :i] = Z.value[i, :i]
        Z_result[i, i + 1:] = Z.value[i, i + 1:]
    return Z_result


def build_adjacency(CMat, K=0):
    N = CMat.shape[0]
    CAbs = np.abs(CMat)
    Ind = np.argsort(-CAbs, axis=0)
    if K == 0:
        for i in range(N):
            CAbs[:, i] = CAbs[:, i] / (CAbs[Ind[0, i], i] + np.finfo(float).eps)
    else:
        for i in range(N):
            for j in range(K):
                CAbs[Ind[j, i], i] = CAbs[Ind[j, i], i] / (CAbs[Ind[0, i], i] + np.finfo(float).eps)
    return CAbs + CAbs.T, CAbs


def compute_pca_per_cluster(samples, labels, n_components=1):
    pca_components = []
    for label in np.unique(labels):
        cluster_samples = samples[labels == label]
        if len(cluster_samples) >= n_components:
            pca = PCA(n_components=n_components)
            pca.fit(cluster_samples)
            pca_components.append(pca.components_[0])
        else:
            random_dir = np.random.randn(samples.shape[1])
            pca_components.append(random_dir / np.linalg.norm(random_dir))
    return np.array(pca_components)


def run_robust_subspace_clustering(data, dim):
    start_time = time.time()
    Y = data.T
    Z = solver_rsc(Y, lambda_=0.01)
    adj, _ = build_adjacency(Z)
    spectral = SpectralClustering(n_clusters=dim, affinity='precomputed')
    labels = spectral.fit_predict(adj)
    factors = compute_pca_per_cluster(data, labels, n_components=1)
    if factors.shape[0] == dim and factors.shape[1] == dim:
        estimated = factors.T
    else:
        estimated = np.eye(dim)
        min_dim = min(factors.shape[0], dim)
        estimated[:min_dim, :min_dim] = factors[:min_dim, :].T[:min_dim, :]
    return estimated, time.time() - start_time, True, {
        'stopping_criterion': 'LASSO + Spectral clustering (non-pymanopt)'
    }


def run_method(method, data, dim):
    if method == 'deflation_varimax_random':
        return run_deflation_varimax(data, dim, initialization='random')
    if method == 'robust_subspace_clustering':
        return run_robust_subspace_clustering(data, dim)
    if method == 'pca_varimax':
        return run_pca_varimax(data, dim)
    if method == 'wfa1_manifold_optimization':
        return run_wfa1(data, dim)
    raise ValueError(f'Unknown method: {method}')


def run_trial(method, data, dim, true_frames):
    estimated, runtime, success, details = run_method(method, data, dim)
    assignment_error, frame_error = normalized_frame_error(estimated, true_frames)
    return assignment_error, frame_error, runtime, success, details
