import numpy as np

def heston_jump_price_MC(S0, K, T, r, v0, kappa, theta, sigma_v, rho,
                         lambda_j, mu_j, sigma_j, n_steps=252, n_sim=3000, seed=42):
    dt = T / n_steps
    discount_factor = np.exp(-r * T)
    chol = np.linalg.cholesky(np.array([[1, rho], [rho, 1]]))
    S_paths = np.full((n_sim, n_steps + 1), S0)
    v_paths = np.full((n_sim, n_steps + 1), v0)
    np.random.seed(seed)

    for t in range(1, n_steps + 1):
        Z = np.random.normal(size=(n_sim, 2))
        dW = Z @ chol.T * np.sqrt(dt)

        v_prev = v_paths[:, t - 1]
        v_next = np.abs(v_prev + kappa * (theta - v_prev) * dt + sigma_v * np.sqrt(v_prev) * dW[:, 1])
        v_paths[:, t] = v_next

        jumps = np.random.poisson(lambda_j * dt, n_sim)
        jump_sizes = np.exp(mu_j + sigma_j * np.random.randn(n_sim)) * jumps

        S_prev = S_paths[:, t - 1]
        S_next = S_prev * np.exp((r - 0.5 * v_prev) * dt + np.sqrt(v_prev) * dW[:, 0]) * (1 + jump_sizes)
        S_paths[:, t] = S_next

    payoffs = np.maximum(S_paths[:, -1] - K, 0)
    return discount_factor * np.mean(payoffs)
