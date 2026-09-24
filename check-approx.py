import math
from scipy.optimize import brentq
import sys

def compute_P_star_lower(
    m: float, 
    gamma: float, 
    tol: float = 1e-20
) -> float:
    """Computes P_*(m) = 
    e^(-m) * sum_{n=0}^inf (m^n / n!) * (1 / (1 + gamma * n))
    using dynamic series truncation.
    """
    if gamma < 0:
        raise ValueError("Gamma should be non-negative.")

    total_sum = 0.0
    # Initial term for n = 0: (e^-m * m^0 / 0!) * (1 / (1 + 0))
    term = math.exp(-m)  

    n = 0
    # Ensure loop runs past the peak at n \approx m
    while term > tol or n < m:  
        weight = 1.0 / (1.0 + gamma * n)
        total_sum += term * weight

        n += 1
        # Recurrence relation: term(n) = term(n-1) * (m / n)
        term *= m / n

    return total_sum

def compute_P_star_upper(
    m: float, 
    gamma: float, 
    tol: float = 1e-20
) -> float:
    """Computes P^*(m) = 
    e^(-m) * sum_{n=0}^inf (m^n / n!) * (gamma / (gamma + n))
    using dynamic series truncation.
    """
    if gamma <= 0:
        raise ValueError("Gamma must be greater than 0.")
    if m < 0:
        raise ValueError("m must be non-negative.")
    if m == 0:
        return 1.0

    total_sum = 0.0
    # Initial Poisson PMF term for n = 0: P(X = 0) = e^(-m)
    poisson_term = math.exp(-m)

    n = 0
    # Compute until terms become negligible past the mean (n > m)
    while poisson_term > tol or n < m:
        weight = gamma / (gamma + n)
        total_sum += poisson_term * weight

        n += 1
        # Recurrence relation for Poisson PMF
        poisson_term *= m / n  

    return total_sum

def compute_Q_tau(
    d: float, 
    tau: float, 
    beta: float, 
    k: float
) -> float:
    """Computes Q_tau(d) defined as:
        (beta + (1 - beta) * tau) * e^(-k / (2d)) 
        + (1 - beta) * (1 - tau) * e^(-k / d)        if d > 0
        0                                            if d <= 0
    """
    if d <= 0:
        return 0.0

    term1 = (beta + (1 - beta) * tau) * math.exp(-k / (2 * d))
    term2 = (1 - beta) * (1 - tau) * math.exp(-k / d)

    return term1 + term2

def compute_R_C(
    v: float, 
    lmbda: float, 
    tau: float, 
    beta: float, 
    k: float, 
    gamma: float
) -> float:
    """Computes R_C(v; lambda) = 
    lambda * Q_tau( P_*(v) - P^*(lambda - v) )"""
    m1 = max(0.0, v)
    m2 = max(0.0, lmbda - v)

    p_lower = compute_P_star_lower(m1, gamma)
    p_upper = compute_P_star_upper(m2, gamma)

    d = p_lower - p_upper
    return lmbda * compute_Q_tau(d, tau, beta, k)


def compute_R_O(
    v: float, 
    lmbda: float, 
    tau: float, 
    beta: float, 
    k: float, 
    gamma: float
) -> float:
    """Computes R_O(v; lambda) = 
    lambda * (1 - Q_tau( P^*(lambda - v) - P_*(v) ))"""
    m1 = max(0.0, lmbda - v)
    m2 = max(0.0, v)

    p_upper = compute_P_star_upper(m1, gamma)
    p_lower = compute_P_star_lower(m2, gamma)

    d = p_upper - p_lower
    return lmbda * (1.0 - compute_Q_tau(d, tau, beta, k))

def compute_x( # x(lambda)
    lmbda: float,
    tau: float,
    beta: float,
    k: float,
    gamma: float,
    xtol: float = 1e-12,
) -> float:
    """Solves for v such that R_C(v; lambda) = v."""
    # Objective function: f(v) = R_C(v; lambda) - v
    def objective(v: float) -> float:
        return compute_R_C(v, lmbda, tau, beta, k, gamma) - v

    # Check edge cases
    f_0 = objective(0.0)
    if abs(f_0) < xtol:
        return 0.0

    f_lmbda = objective(lmbda)
    if abs(f_lmbda) < xtol:
        return lmbda

    # Use Brent's method to find root on [0, lambda]
    v_sol = brentq(objective, a=0.0, b=lmbda, xtol=xtol)
    return float(v_sol)

def compute_y( # y(lambda)
    lmbda: float,
    tau: float,
    beta: float,
    k: float,
    gamma: float,
    xtol: float = 1e-12,
) -> float:
    """Solves for v such that R_O(v; lambda) = v."""
    # Objective function: f(v) = R_O(v; lambda) - v
    def objective(v: float) -> float:
        return compute_R_O(v, lmbda, tau, beta, k, gamma) - v

    # Check boundaries
    f_0 = objective(0.0)
    if abs(f_0) < xtol:
        return 0.0

    f_lmbda = objective(lmbda)
    if abs(f_lmbda) < xtol:
        return lmbda

    # Use Brent's method to find root on [0, lambda]
    v_sol = brentq(objective, a=0.0, b=lmbda, xtol=xtol)
    return float(v_sol)

def compute_z_bar(
    lmbda: float,
    tau: float,
    beta: float,
    k: float,
    gamma: float,
    xtol: float = 1e-20,
) -> float:
    """Computes z_underline = k / P^*(x(lambda))"""
    # 1. Solve x(lambda) = v*
    x_val = compute_x(lmbda, tau, beta, k, gamma, xtol=xtol)

    # 2. Compute P^*(x(lambda))
    p_star = compute_P_star_upper(x_val, gamma)

    if p_star == 0:
        raise ZeroDivisionError(f"P^*({x_val}) evaluated to 0.")

    # 3. Compute z_underline
    return k / p_star

def compute_xi(
    lmbda: float,
    tau: float,
    beta: float,
    k: float,
    gamma: float,
    xtol: float = 1e-12,
) -> float:
    """Computes xi(lambda) = beta / 
    (beta + (1 - beta) * (tau + (1 - tau) * e^(-z_underline / 2)))
    """
    z_under = compute_z_bar(lmbda, tau, beta, k, gamma, xtol=xtol)

    denom = beta + (1.0 - beta) 
    denom *= (tau + (1.0 - tau) * math.exp(-z_under / 2.0))

    if denom == 0:
        raise ZeroDivisionError("Denominator in xi(lambda) is 0.")

    return beta / denom


def compute_Delta_B(
    lmbda: float,
    tau: float,
    beta: float,
    k: float,
    gamma: float,
    xtol: float = 1e-12,
) -> float:
    """Computes Delta_B(lambda) = 
    xi(lambda) * (1 - e^(-x(lambda))) - beta."""
    # 1. Solve x(lambda) = v*
    x_val = compute_x(lmbda, tau, beta, k, gamma, xtol=xtol)

    # 2. Compute xi(lambda) using x_val
    xi_val = compute_xi(lmbda, tau, beta, k, gamma)

    # 3. Compute Delta_B(lambda)
    return xi_val * (1.0 - math.exp(-x_val)) - beta

def compute_Delta_G(
    lmbda: float,
    tau: float,
    beta: float,
    k: float,
    gamma: float,
    xtol: float = 1e-20,
) -> float:
    """Computes Delta_G(lambda) = 
    (1 - beta) * (1 - e^(-y(lambda))) - 1 + xi(lambda)"""
    # 1. Solve y(lambda) directly using R_O fixed point
    y_val = compute_y(lmbda, tau, beta, k, gamma, xtol=xtol)

    # 2. Compute x(lambda) = lambda - y(lambda)
    x_val = lmbda - y_val

    # 3. Compute xi(lambda) using x(lambda)
    xi_val = compute_xi(lmbda, tau, beta, k, gamma)

    # 4. Compute Delta_G(lambda)
    return (1.0 - beta) * (1.0 - math.exp(-y_val)) - 1.0 + xi_val
