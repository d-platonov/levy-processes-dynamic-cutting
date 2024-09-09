from alpha_stable_process import AlphaStableProcess
from approximation_type import ApproximationType
from monte_carlo import MonteCarlo


if __name__ == '__main__':
    # Plot some trajectories
    process = AlphaStableProcess(
        t=1, h=0.05, eps=0.01, n_grid=100, z_0=0, seed=2024, alpha=1.5, approximation_type=ApproximationType.CUT_2
    )
    process.plot_trajectories(10)

    # Monte-Carlo
    monte_carlo = MonteCarlo()
    n_simulations = 100
    n_trajectories = 1000
    base_params = {
        't': 1, 'h': 0.05, 'eps': 0.01, 'n_grid': 100, 'z_0': 0, 'seed': 12
    }

    params_cut_1 = {**base_params, 'approximation_type': ApproximationType.CUT_1}
    params_cut_2 = {**base_params, 'approximation_type': ApproximationType.CUT_2}

    # Moments of the alpha-stable process
    p = 0.3
    alpha_values = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]

    alpha_stable_results_cut_1 = monte_carlo.estimate_alpha_stable_moments(
        params_cut_1, n_simulations, n_trajectories, p, alpha_values
    )
    print(f"Alpha-stable moments for CUT_1: \n{alpha_stable_results_cut_1}")

    alpha_stable_results_cut_2 = monte_carlo.estimate_alpha_stable_moments(
        params_cut_2, n_simulations, n_trajectories, p, alpha_values
    )
    print(f"Alpha-stable moments for CUT_2: \n{alpha_stable_results_cut_2}")

    # Tempered-stable process: Example 1
    alpha_plus = alpha_minus = 0.2
    beta_plus = beta_minus = 1.0
    tempered_stable_results_1 = monte_carlo.estimate_tempered_stable_cumulants(
        params_cut_2, alpha_plus, beta_plus, alpha_minus, beta_minus, n_simulations, n_trajectories, seed=12
    )
    print(f"Tempered-stable results for alpha_plus = {alpha_plus}, alpha_minus = {alpha_minus}, "
          f"beta_plus = {beta_plus}, beta_minus = {beta_minus}: \n{tempered_stable_results_1}")

    # Tempered-stable process: Example 2
    alpha_plus, alpha_minus = 0.8, 0.2
    beta_plus = beta_minus = 1.0
    tempered_stable_results_2 = monte_carlo.estimate_tempered_stable_cumulants(
        params_cut_2, alpha_plus, beta_plus, alpha_minus, beta_minus, n_simulations, n_trajectories, seed=12
    )
    print(f"Tempered-stable results for alpha_plus = {alpha_plus}, alpha_minus = {alpha_minus}, "
          f"beta_plus = {beta_plus}, beta_minus = {beta_minus}: \n{tempered_stable_results_2}")

    # Tempered-stable process: Example 3
    alpha_plus = alpha_minus = 0.8
    beta_plus = beta_minus = 1.0
    tempered_stable_results_3 = monte_carlo.estimate_tempered_stable_cumulants(
        params_cut_2, alpha_plus, beta_plus, alpha_minus, beta_minus, n_simulations, n_trajectories, seed=12
    )
    print(f"Tempered-stable results for alpha_plus = {alpha_plus}, alpha_minus = {alpha_minus}, "
          f"beta_plus = {beta_plus}, beta_minus = {beta_minus}: \n{tempered_stable_results_3}")
