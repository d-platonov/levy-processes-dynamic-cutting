from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import quad

from approximation_type import ApproximationType

import warnings

warnings.filterwarnings("ignore")


@dataclass
class AlphaStableProcess:
    t: float
    h: float
    eps: float
    n_grid: int
    z_0: float
    seed: int
    alpha: float
    approximation_type: ApproximationType
    rng: np.random.RandomState = field(init=False)

    def __post_init__(self):
        self.rng = np.random.RandomState(self.seed)
        self.m_const = self.calculate_m_constant(self.alpha)

    @staticmethod
    def calculate_m_constant(alpha) -> float:
        """Calculate the M constant used in the definition of tau."""
        result = quad(lambda u, a: (1 - np.cos(u)) / u**(1 + a), 0, np.inf, args=(alpha,))[0]
        return 1 / (2 * alpha * result)

    def _tau(self, t: float) -> float:
        """Calculate threshold value for jump sizes at time t."""
        return (t * self.m_const) ** (1 / self.alpha)

    def _sigma(self, t: np.ndarray) -> np.ndarray:
        """Calculate standard deviations of normal increments (substitution of small jumps)."""
        scale_factor = self.m_const ** (2 / self.alpha)
        numerator = 2 * self.alpha ** 2 * scale_factor * t
        denominator = ((2 - self.alpha) * self.eps + self.alpha) * (2 - self.alpha)
        time_scaling = (t * self.h) ** ((2 - self.alpha) * self.eps / self.alpha)
        return np.sqrt(numerator / denominator * time_scaling)

    def _inverse_intensity(self, t: float) -> float:
        """Calculate the inverse of the lambda function for Poisson process jump times."""
        return (t * (1 - self.eps) * (self.h ** self.eps)) ** (1 / (1 - self.eps))

    def _inverse_jump_distribution(self, u: float, t: float) -> float:
        """Inverse transform method to generate jump sizes."""
        tau_threshold = self._tau((t * self.h) ** self.eps)
        if u <= tau_threshold:
            return self._tau((self.eps / (u * (t * self.h) ** (1 - self.eps))) ** (1 / (self.eps - 1)))
        return self._tau((1 - self.eps) * ((t * self.h) ** self.eps) / (1 - u))

    def _generate_jump_times(self) -> np.ndarray:
        """Inverse time transformation method."""
        jump_times = []
        accumulated_exponential = self.rng.exponential()
        current_time = self._inverse_intensity(accumulated_exponential)
        while 0 < current_time < self.t:
            jump_times.append(current_time)
            accumulated_exponential += self.rng.exponential()
            current_time = self._inverse_intensity(accumulated_exponential)
        return np.array(jump_times)

    def simulate_path(self) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate the entire path of the alpha-stable Lévy process."""
        grid_times = np.linspace(0, self.t, self.n_grid)
        pos_jump_times = self._generate_jump_times()
        neg_jump_times = self._generate_jump_times()
        all_times = np.unique(np.concatenate([grid_times, pos_jump_times, neg_jump_times]))

        # Process initialization
        z = np.zeros_like(all_times)
        z[0] = self.z_0

        # Simulate normal increments if required
        if self.approximation_type == ApproximationType.CUT_2:
            normal_increments = self._sigma(np.diff(all_times, prepend=all_times[0])) * self.rng.normal(size=len(all_times))
            z += normal_increments

        # Apply jumps
        for jump_times, sign in [(pos_jump_times, 1), (neg_jump_times, -1)]:
            if jump_times.size > 0:
                inv_jump_dist_vect = np.vectorize(self._inverse_jump_distribution, otypes=[float])
                jump_sizes = inv_jump_dist_vect(self.rng.uniform(size=jump_times.size), jump_times)
                indices = np.searchsorted(all_times, jump_times)
                np.add.at(z, indices, sign * jump_sizes)

        return all_times, np.cumsum(z)

    def plot_trajectories(self, n_simulations: int = 10, figsize: Tuple[int, int] = (12, 6)):
        """Plot multiple trajectories of the simulated Lévy process."""
        plt.figure(figsize=figsize)
        for _ in range(n_simulations):
            times, values = self.simulate_path()
            plt.step(times, values, where='post', alpha=0.5)
        plt.title('Simulated Lévy Process Trajectories')
        plt.xlabel('Time')
        plt.ylabel('Process Value')
        plt.grid(True)
        plt.show()
