import torch
from torch.distributions.normal import Normal

def rbf_kernel_torch(x1, x2, length_scale=1.0, sigma_f=1.0):
    """
    Computes the RBF (squared exponential) kernel between two sets of points.
    Args:
        x1: (N, D) tensor
        x2: (M, D) tensor
        length_scale: kernel length scale
        sigma_f: kernel signal variance
    Returns:
        (N, M) kernel matrix
    """
    sqdist = torch.sum(x1**2, dim=1).view(-1, 1) + torch.sum(x2**2, dim=1) - 2 * torch.mm(x1, x2.T)
    return sigma_f**2 * torch.exp(-0.5 / length_scale**2 * sqdist)

def gp_posterior_torch(x_test, X_train, y_train, length_scale=1.0, sigma_f=1.0, sigma_y=1e-8):
    """
    Computes the posterior mean and covariance for test points in Gaussian Process regression.
    Args:
        x_test: (N_test, D) test inputs
        X_train: (N_train, D) training inputs
        y_train: (N_train,) or (N_train, 1) training targets
    Returns:
        mu_n: (N_test,) posterior mean
        sigma_n_squared: (N_test,) posterior variance
    """
    K = rbf_kernel_torch(X_train, X_train, length_scale, sigma_f) + sigma_y**2 * torch.eye(X_train.size(0), device=X_train.device)
    K_s = rbf_kernel_torch(x_test, X_train, length_scale, sigma_f)
    K_ss = rbf_kernel_torch(x_test, x_test, length_scale, sigma_f)

    K_inv = torch.linalg.inv(K)
    y_train = y_train.view(-1, 1)

    mu_n = torch.mm(torch.mm(K_s, K_inv), y_train)
    sigma_n_squared = K_ss - torch.mm(torch.mm(K_s, K_inv), K_s.T)

    return mu_n.view(-1), torch.diagonal(sigma_n_squared)

def expected_improvement_general_torch(Delta_n, sigma_n):
    """
    Calculates the Expected Improvement (EI) acquisition function.
    Args:
        Delta_n: (N,) tensor, f_max - mu_n for each candidate
        sigma_n: (N,) tensor, standard deviation for each candidate
    Returns:
        EI: (N,) tensor, expected improvement for each candidate
    """
    sigma_n = torch.clamp(sigma_n, min=1e-10)  # Prevent divide by zero
    Z = Delta_n / sigma_n
    normal = Normal(0, 1)
    # EI formula
    EI = torch.clamp(Delta_n, min=0) + sigma_n * normal.log_prob(Z).exp() - torch.abs(Delta_n) * normal.cdf(Z)
    return EI

def upper_confidence_bound_torch(mu_n, sigma_n_squared, beta):
    """
    Calculates the Upper Confidence Bound (UCB) acquisition function.
    Args:
        mu_n: (N,) tensor, posterior mean
        sigma_n_squared: (N,) tensor, posterior variance
        beta: float, exploration-exploitation trade-off parameter
    Returns:
        (N,) tensor, UCB values for each candidate
    """
    sigma_n = torch.sqrt(torch.clamp(sigma_n_squared, min=1e-10))
    return mu_n + beta * sigma_n

def probability_of_improvement_torch(mu_n, sigma_n_squared, f_max):
    """
    Calculates the Probability of Improvement (PI) acquisition function.
    Args:
        mu_n: (N,) tensor, posterior mean
        sigma_n_squared: (N,) tensor, posterior variance
        f_max: scalar, current best observation
    Returns:
        (N,) tensor, probability of improvement for each candidate
    """
    sigma_n = torch.sqrt(torch.clamp(sigma_n_squared, min=1e-10))
    normal = Normal(0,1)
    Z = (mu_n - f_max) / sigma_n
    return normal.cdf(Z)

def bayessian_optimisation_torch(X_train, y_train, dimention, number=5, method='ei', beta=2.0):
    """
    Runs Bayesian Optimization using GP surrogate and given acquisition function.
    Args:
        X_train: (N_train, D) training points
        y_train: (N_train,) training values
        dimention: int, search space dimension
        number: int, number of top candidates to return
        method: str, acquisition function ('ei', 'pi', 'ucb')
        beta: float, UCB beta parameter
    Returns:
        (number, dimention) tensor, the best candidate points selected by acquisition function
    """
    device = X_train.device
    # Generate candidate points uniformly at random
    X_test = torch.rand(5000, dimention, device=device)
    # Compute GP posterior mean and variance at candidate points
    mu_x, sigma_x = gp_posterior_torch(X_test, X_train, y_train, length_scale=1.0, sigma_f=1.0, sigma_y=1e-8)
    f_max = torch.max(y_train)

    if method == 'ei':
        Delta_n = f_max - mu_x
        acq_value = expected_improvement_general_torch(Delta_n, sigma_x)
    elif method == 'pi':
        acq_value = probability_of_improvement_torch(mu_x, sigma_x, f_max)
    elif method == 'ucb':
        acq_value = upper_confidence_bound_torch(mu_x, sigma_x, beta)
    else:
        raise ValueError("Unsupported acquisition function method: choose 'ei', 'pi', or 'ucb'.")

    # Select top candidate points according to acquisition value
    top_5_indices = torch.argsort(acq_value, descending=True)[:number]
    top_5_points = X_test[top_5_indices]

    return top_5_points

if __name__ == "__main__":
    # Example Usage
    A = torch.normal(0, 1, size=(50, 4096)).to(torch.float32)
    X_train = torch.randn(5, 50)
    print(X_train)
    # Transform initial points
    X_start = torch.mm(X_train, A)
    print(X_start)
