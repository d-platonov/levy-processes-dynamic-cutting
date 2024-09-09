from typing import List

import numpy as np
import pandas as pd
from scipy.special import gamma
from scipy.stats import kurtosis, skew

from alpha_stable_process import AlphaStableProcess


class MonteCarlo:
    """Class to run Monte-Carlo simulations for alpha-stable and tempered stable processes."""

    @staticmethod
    def get_true_alpha_stable_moment(p: float, alpha: float) -> float:
        """Calculate true p-th moment of a symmetric alpha-stable distribution."""
        return (2 ** p * gamma((1 + p) / 2) * gamma(1 - p / alpha)) / (gamma(1 - p / 2) * gamma(1 / 2))

    @staticmethod
    def get_true_tempered_stable_cumulant(
            n: int,
            alpha_plus: float,
            gamma_plus: float,
            beta_plus: float,
            alpha_minus: float,
            gamma_minus: float,
            beta_minus: float,
    ) -> float:
        """Calculate true cumulants for tempered-stable process."""
        return (gamma(n - alpha_plus) * gamma_plus / (beta_plus ** (n - alpha_plus)) +
                (-1) ** n * gamma(n - alpha_minus) * gamma_minus / (beta_minus ** (n - alpha_minus)))

    def estimate_alpha_stable_moments(
            self,
            alpha_stable_params_except_alpha: dict,
            n_simulations: int,
            n_trajectories: int,
            p: float,
            alpha_values: List[float]
    ) -> pd.DataFrame:
        """Estimate p-th moment of a symmetric alpha-stable distribution for different alpha values."""
        print("Running Monte-Carlo simulations for Alpha-Stable process...")
        params = alpha_stable_params_except_alpha.copy()
        estimates, true_values, stds = [], [], []
        for alpha in alpha_values:
            params['alpha'] = alpha
            process = AlphaStableProcess(**params)
            moments = [
                np.mean(
                    [np.abs(process.simulate_path()[1][-1]) ** p for _ in range(n_trajectories)]
                ) for _ in range(n_simulations)
            ]
            estimates.append(np.mean(moments))
            stds.append(np.std(moments))
            true_values.append(self.get_true_alpha_stable_moment(p=p, alpha=alpha))

        df_result = pd.DataFrame({
            'alpha': alpha_values,
            'true_value': true_values,
            'estimate': estimates,
            'std': stds
        })
        return df_result

    @staticmethod
    def simulate_tempered_stable_process_half(
            alpha_stable_params_except_alpha: dict,
            alpha: float,
            b: float,
            n_simulations: int,
            n_trajectories: int,
            rng: np.random.RandomState,
            sign: int = 1
    ) -> np.ndarray:
        """Simulate half of the tempered-stable process using accept-rejection method."""
        params = alpha_stable_params_except_alpha.copy()
        params['alpha'] = alpha

        process = AlphaStableProcess(**params)

        z_2d = np.zeros((n_simulations, n_trajectories))
        for i in range(n_simulations):
            j = 0
            while j < n_trajectories:
                _, values = process.simulate_path()

                increments = np.diff(values)
                v = increments[increments > 0].sum()

                u = rng.rand()
                if u <= np.exp(-b * v):
                    z_2d[i][j] = sign * v
                    j += 1

        return z_2d

    def estimate_tempered_stable_cumulants(
            self,
            alpha_stable_params_except_alpha: dict,
            alpha_plus: float,
            beta_plus: float,
            alpha_minus: float,
            beta_minus: float,
            n_simulations: int,
            n_trajectories: int,
            seed: int = 12
    ) -> pd.DataFrame:
        """Estimate mean, variance, skewness and kurtosis of the tampered-stable process."""
        print("Running Monte-Carlo simulations for Tempered-Stable process...")
        rng = np.random.RandomState(seed)
        params = alpha_stable_params_except_alpha.copy()

        z_2d_plus = self.simulate_tempered_stable_process_half(
            params, alpha_plus, beta_plus, n_simulations, n_trajectories, rng, sign=1
        )

        z_2d_minus = self.simulate_tempered_stable_process_half(
            params, alpha_minus, beta_minus, n_simulations, n_trajectories, rng, sign=-1
        )

        z_2d = z_2d_plus + z_2d_minus

        gamma_plus = alpha_plus * AlphaStableProcess.calculate_m_constant(alpha_plus)
        gamma_minus = alpha_minus * AlphaStableProcess.calculate_m_constant(alpha_minus)

        cumulant_2 = self.get_true_tempered_stable_cumulant(
            2, alpha_plus, gamma_plus, beta_plus, alpha_minus, gamma_minus, beta_minus
        )

        true_mean = self.get_true_tempered_stable_cumulant(
            1, alpha_plus, gamma_plus, beta_plus, alpha_minus, gamma_minus, beta_minus
        )

        true_variance = cumulant_2

        true_skewness = self.get_true_tempered_stable_cumulant(
            3, alpha_plus, gamma_plus, beta_plus, alpha_minus, gamma_minus, beta_minus
        ) / cumulant_2 ** (3 / 2)

        true_kurtosis = (self.get_true_tempered_stable_cumulant(
            4, alpha_plus, gamma_plus, beta_plus, alpha_minus, gamma_minus, beta_minus
        ) / cumulant_2 ** 2) - 3

        data = {
            "True Values": {
                "Mean": true_mean,
                "Variance": true_variance,
                "Skewness": true_skewness,
                "Kurtosis": true_kurtosis
            },
            "Estimates Mean": {
                "Mean": z_2d.mean(axis=1).mean(),
                "Variance": z_2d.var(axis=1).mean(),
                "Skewness": skew(z_2d, axis=1).mean(),
                "Kurtosis": kurtosis(z_2d, axis=1).mean()
            },
            "Estimates Std": {
                "Mean": z_2d.mean(axis=1).std(),
                "Variance": z_2d.var(axis=1).std(),
                "Skewness": skew(z_2d, axis=1).std(),
                "Kurtosis": kurtosis(z_2d, axis=1).std()
            }
        }

        return pd.DataFrame(data)
