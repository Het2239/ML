"""Batched Lasso on the GPU (PyTorch) via FISTA with adaptive restart.

Solves, for every fold f and every alpha a simultaneously,

    min_w  1/(2n) ||y_f - A_f w - b||^2 + a ||w||_1

which is exactly sklearn.linear_model.Lasso's objective (intercept b is not
penalised; with column-centred A it equals mean(y_f)).

Working in Gram form (G = A^T A / n, c = A^T y / n) makes each iteration a
single batched matmul of shape (F, p, p) x (F, p, n_alphas), so all
(fold, alpha) problems for a given polynomial degree are solved in one go.
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:  # pragma: no cover
    torch = None
    HAS_CUDA = False


def default_device():
    return "cuda" if HAS_CUDA else "cpu"


def _soft(x, t):
    return torch.sign(x) * torch.clamp(x.abs() - t, min=0.0)


def fista_lasso_gram(G, c, alphas, max_iter=20000, tol=1e-6, check_every=50):
    """Batched FISTA.

    G      : (F, p, p) Gram matrices (already divided by n)
    c      : (F, p)    A^T y / n  (y centred)
    alphas : (K,)      penalties
    returns W : (F, p, K), n_iter
    """
    F, p, _ = G.shape
    K = alphas.numel()
    # Lipschitz constant per fold = largest eigenvalue of G
    L = torch.linalg.eigvalsh(G)[:, -1].clamp(min=1e-12)          # (F,)
    step = (1.0 / L).view(F, 1, 1)
    thr = step * alphas.view(1, 1, K)                               # (F,1,K)
    cK = c.unsqueeze(-1)                                            # (F,p,1)

    W = torch.zeros(F, p, K, dtype=G.dtype, device=G.device)
    Z = W.clone()
    t = torch.ones(1, 1, K, dtype=G.dtype, device=G.device).expand(F, 1, K).clone()
    it = 0
    for it in range(1, max_iter + 1):
        grad = torch.bmm(G, Z) - cK
        W_new = _soft(Z - step * grad, thr)
        # gradient-based adaptive restart (O'Donoghue & Candes)
        restart = ((Z - W_new) * (W_new - W)).sum(dim=1, keepdim=True) > 0
        t_new = 0.5 * (1 + torch.sqrt(1 + 4 * t * t))
        t_new = torch.where(restart, torch.ones_like(t_new), t_new)
        mom = torch.where(restart, torch.zeros_like(t), (t - 1) / t_new)
        Z = W_new + mom * (W_new - W)
        if it % check_every == 0:
            diff = (W_new - W).abs().amax(dim=1)
            scale = W_new.abs().amax(dim=1).clamp(min=1e-12)
            if (diff / scale).max().item() < tol:
                W = W_new
                break
        W, t = W_new, t_new
    return W, it


def _prep_fold(P, y, tr, va, dtype, device):
    Pt = torch.as_tensor(P[tr], dtype=dtype, device=device)
    Pv = torch.as_tensor(P[va], dtype=dtype, device=device)
    mu = Pt.mean(0)
    sd = Pt.std(0, unbiased=False)
    sd = torch.where(sd == 0, torch.ones_like(sd), sd)
    A = (Pt - mu) / sd
    B = (Pv - mu) / sd
    yt = torch.as_tensor(y[tr], dtype=dtype, device=device)
    return A, B, yt, torch.as_tensor(y[va], dtype=dtype, device=device)


def lasso_cv_gpu(P, y, splits, alphas, dtype="float64", device=None,
                 max_iter=20000, tol=1e-6):
    """CV for all alphas on expanded features P (n, p).

    Returns dict of numpy arrays with shape (n_folds, n_alphas):
    val_mse, val_r2, train_mse, nonzero; plus 'n_iter'.
    """
    device = device or default_device()
    dt = getattr(torch, dtype)
    folds = [_prep_fold(P, y, tr, va, dt, device) for tr, va in splits]
    # all folds have equal train size with KFold(5) on 1000 rows; handle generic case
    G = torch.stack([A.T @ A / A.shape[0] for A, _, _, _ in folds])
    ym = torch.stack([yt.mean() for _, _, yt, _ in folds])
    c = torch.stack([A.T @ (yt - yt.mean()) / A.shape[0] for A, _, yt, _ in folds])
    a = torch.as_tensor(np.asarray(alphas), dtype=dt, device=device)

    W, n_iter = fista_lasso_gram(G, c, a, max_iter=max_iter, tol=tol)

    vm, vr, tm, nz = [], [], [], []
    for f, (A, B, yt, yv) in enumerate(folds):
        pv = B @ W[f] + ym[f]
        pt = A @ W[f] + ym[f]
        mse_v = ((pv - yv[:, None]) ** 2).mean(0)
        vm.append(mse_v)
        vr.append(1 - mse_v / yv.var(unbiased=False))
        tm.append(((pt - yt[:, None]) ** 2).mean(0))
        nz.append((W[f] != 0).sum(0).to(dt))
    out = {k: torch.stack(v).cpu().numpy() for k, v in
           dict(val_mse=vm, val_r2=vr, train_mse=tm, nonzero=nz).items()}
    out["n_iter"] = n_iter
    del G, W, folds
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


class GPULasso(RegressorMixin, BaseEstimator):
    """Minimal sklearn-style Lasso estimator backed by the FISTA solver.

    Same objective as sklearn.linear_model.Lasso. Intended to sit after a
    StandardScaler in a Pipeline.
    """

    def __init__(self, alpha=1.0, max_iter=50000, tol=1e-7, dtype="float64", device=None):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.dtype = dtype
        self.device = device

    def fit(self, X, y, sample_weight=None):
        dev = self.device or default_device()
        dt = getattr(torch, self.dtype)
        Xt = torch.as_tensor(np.asarray(X), dtype=dt, device=dev)
        yt = torch.as_tensor(np.asarray(y), dtype=dt, device=dev)
        n = Xt.shape[0]
        if sample_weight is None:
            sw = torch.ones(n, dtype=dt, device=dev)
        else:
            sw = torch.as_tensor(np.asarray(sample_weight), dtype=dt, device=dev)
            sw = sw * (n / sw.sum())  # sklearn rescales weights to sum to n
        xm = (sw[:, None] * Xt).sum(0) / n
        ym = (sw * yt).sum() / n
        A = Xt - xm
        Aw = A * sw[:, None]
        G = (Aw.T @ A / n).unsqueeze(0)
        c = (Aw.T @ (yt - ym) / n).unsqueeze(0)
        a = torch.tensor([self.alpha], dtype=dt, device=dev)
        W, self.n_iter_ = fista_lasso_gram(G, c, a, self.max_iter, self.tol)
        w = W[0, :, 0]
        self.coef_ = w.cpu().numpy()
        self.intercept_ = float((ym - xm @ w).item())
        return self

    def predict(self, X):
        return np.asarray(X) @ self.coef_ + self.intercept_
