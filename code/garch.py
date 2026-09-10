# -*- coding: utf-8 -*-
"""GARCH(1,1) 条件异方差模型：极大似然估计 + 多步波动率预测

模型:  r_t = eps_t,  eps_t ~ N(0, sigma_t^2)
      sigma_t^2 = omega + alpha * eps_{t-1}^2 + beta * sigma_{t-1}^2
约束:  omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1 (协方差平稳)
"""
import numpy as np
from scipy.optimize import minimize

_EPS = 1e-10


def _nll(theta, r):
    omega, alpha, beta = theta
    if omega <= _EPS or alpha < 0 or beta < 0 or alpha + beta >= 0.9999:
        return 1e12
    T = len(r)
    s2 = np.empty(T, dtype=float)
    s2[0] = r.var() + _EPS
    for t in range(1, T):
        s2[t] = omega + alpha * r[t - 1] ** 2 + beta * s2[t - 1]
    if np.any(s2 <= 0) or not np.isfinite(s2).all():
        return 1e12
    return float(0.5 * np.sum(np.log(2.0 * np.pi * s2) + r ** 2 / s2))


def fit(resid):
    """对残差序列拟合 GARCH(1,1)，返回参数字典"""
    r = np.asarray(resid, dtype=float)
    r = r - r.mean()                      # 零均值化
    v0 = float(r.var())
    best, bx = None, np.inf
    for a0, b0 in [(0.08, 0.88), (0.15, 0.75), (0.05, 0.9), (0.3, 0.6)]:
        x0 = np.array([v0 * (1 - a0 - b0), a0, b0])
        res = minimize(_nll, x0, args=(r,), method='L-BFGS-B',
                       bounds=[(1e-10, None), (1e-8, 0.9999), (1e-8, 0.9999)],
                       options={'maxiter': 400, 'ftol': 1e-12})
        if res.fun < bx:
            bx, best = res.fun, res.x
    omega, alpha, beta = best
    omega = max(omega, 1e-12)
    # 末端条件方差
    T = len(r)
    s2 = np.empty(T, dtype=float)
    s2[0] = r.var() + _EPS
    for t in range(1, T):
        s2[t] = omega + alpha * r[t - 1] ** 2 + beta * s2[t - 1]
    long_run = omega / max(1 - alpha - beta, 1e-8)
    return {
        'omega': float(omega), 'alpha': float(alpha), 'beta': float(beta),
        'persist': float(alpha + beta),
        'long_run_vol': float(np.sqrt(long_run)),
        'last_var': float(s2[-1]), 'last_eps2': float(r[-1] ** 2),
        'loglik': float(-bx), 'sigma': np.sqrt(s2),
    }


def forecast_var(g, horizon):
    """向前 h 步条件方差预测（逐步递推，含长程均值回复）"""
    omega, alpha, beta = g['omega'], g['alpha'], g['beta']
    s2 = omega + alpha * g['last_eps2'] + beta * g['last_var']
    out = np.empty(horizon, dtype=float)
    for h in range(horizon):
        out[h] = s2
        s2 = omega + (alpha + beta) * s2
    return out


def ci_band(g, horizon, z=1.96):
    """累积方差 -> 对数价格的 95% 置信半宽（元/吨空间按指数还原）"""
    cum = np.cumsum(forecast_var(g, horizon))
    return z * np.sqrt(np.maximum(cum, 0.0))
