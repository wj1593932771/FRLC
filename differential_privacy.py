import numpy as np
import torch
from typing import Dict, List, Tuple, Optional


class DifferentialPrivacy:

    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5,
                 clip_norm: float = 1.0, noise_multiplier: float = 0.1):
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier

        self.epsilon_spent = 0.0
        self.epsilon_released = 0.0
        self.num_steps = 0
        self.last_release_step = 0

        self.release_rate = 0.03
        self.min_release_interval = 10

        self.privacy_history = []
        self.noise_history = []
        self.budget_history = []

    def add_noise_to_gradients(self, gradients: torch.Tensor,
                               sensitivity: float = None) -> Tuple[torch.Tensor, float]:
        if sensitivity is None:
            sensitivity = self.clip_norm

        remaining_budget_ratio = max(0.1, 1.0 - self.get_net_spent_ratio())
        adjusted_noise_multiplier = self.noise_multiplier * (2.0 - remaining_budget_ratio)

        noise_std = adjusted_noise_multiplier * sensitivity * 0.01

        noise = torch.randn_like(gradients) * noise_std

        noisy_gradients = gradients + noise

        if torch.norm(gradients) > 0:
            actual_noise_level = torch.norm(noise).item() / torch.norm(gradients).item()
        else:
            actual_noise_level = 0.01

        self.noise_history.append(actual_noise_level)

        return noisy_gradients, actual_noise_level

    def clip_gradients(self, gradients: torch.Tensor) -> torch.Tensor:
        grad_norm = torch.norm(gradients, p=2)

        if grad_norm > self.clip_norm:
            gradients = gradients * (self.clip_norm / grad_norm)

        return gradients

    def apply_differential_privacy(self, model_params: Dict[str, torch.Tensor],
                                   learning_rate: float = 0.01) -> Dict[str, torch.Tensor]:
        self._try_release_budget()

        dp_params = {}
        actual_noise_added = []

        for param_name, param_value in model_params.items():
            if 'weight' in param_name and param_value.requires_grad:
                remaining_budget_ratio = max(0.1, 1.0 - self.get_net_spent_ratio())
                noise_scale = self.noise_multiplier * self.clip_norm * 0.01 * (2.0 - remaining_budget_ratio)

                noise = torch.randn_like(param_value) * noise_scale

                noisy_param = param_value + noise
                dp_params[param_name] = noisy_param

                if torch.norm(param_value) > 0:
                    actual_noise = torch.norm(noise).item() / torch.norm(param_value).item()
                else:
                    actual_noise = 0.01
                actual_noise_added.append(actual_noise)
            else:
                dp_params[param_name] = param_value

        if actual_noise_added:
            avg_noise = np.mean(actual_noise_added)
            self.noise_history.append(avg_noise)

            net_spent = self.epsilon_spent - self.epsilon_released
            if net_spent < self.epsilon * 0.3:
                step_epsilon = 0.08
            elif net_spent < self.epsilon * 0.7:
                step_epsilon = 0.05
            else:
                step_epsilon = 0.02

            self.epsilon_spent += step_epsilon
            self.num_steps += 1

            self.privacy_history.append({
                'step': self.num_steps,
                'step_epsilon': step_epsilon,
                'total_spent': self.epsilon_spent,
                'total_released': self.epsilon_released,
                'net_spent': self.epsilon_spent - self.epsilon_released,
                'noise_level': avg_noise
            })

            self.budget_history.append({
                'step': self.num_steps,
                'spent': self.epsilon_spent,
                'released': self.epsilon_released,
                'net': self.epsilon_spent - self.epsilon_released
            })

        return dp_params

    def _try_release_budget(self):
        if self.num_steps - self.last_release_step >= self.min_release_interval:
            net_spent = self.epsilon_spent - self.epsilon_released

            if net_spent > 0:
                release_amount = min(net_spent * self.release_rate, net_spent * 0.3)
                self.epsilon_released += release_amount
                self.last_release_step = self.num_steps

                if self.epsilon_released > self.epsilon_spent:
                    self.epsilon_released = self.epsilon_spent

    def get_net_spent_ratio(self) -> float:
        net_spent = self.epsilon_spent - self.epsilon_released
        return min(1.0, net_spent / self.epsilon) if self.epsilon > 0 else 0

    def get_statistics(self) -> Dict:
        net_spent = self.epsilon_spent - self.epsilon_released

        stats = {
            'total_privacy_spent': self.epsilon_spent,
            'total_privacy_released': self.epsilon_released,
            'net_privacy_spent': net_spent,
            'privacy_budget_usage': self.get_net_spent_ratio(),
            'num_updates': self.num_steps,
            'average_noise_level': np.mean(self.noise_history) if self.noise_history else 0,
            'max_noise_level': np.max(self.noise_history) if self.noise_history else 0,
            'min_noise_level': np.min(self.noise_history) if self.noise_history else 0,
            'current_noise_multiplier': self.noise_multiplier,
            'clip_norm': self.clip_norm,
            'remaining_budget': max(0, self.epsilon - net_spent),
            'release_rate': self.release_rate
        }

        return stats

    def reset(self):
        self.epsilon_spent = 0.0
        self.epsilon_released = 0.0
        self.num_steps = 0
        self.last_release_step = 0
        self.privacy_history = []
        self.noise_history = []
        self.budget_history = []


class LocalDifferentialPrivacy:

    def __init__(self, epsilon: float = 1.0, mechanism: str = 'laplace'):
        self.epsilon = epsilon
        self.mechanism = mechanism

    def randomized_response(self, data: np.ndarray, domain_size: int = 2) -> np.ndarray:
        p = np.exp(self.epsilon) / (np.exp(self.epsilon) + domain_size - 1)
        mask = np.random.random(data.shape) < p
        random_values = np.random.randint(0, domain_size, data.shape)
        perturbed_data = np.where(mask, data, random_values)
        return perturbed_data

    def add_laplace_noise(self, data: np.ndarray, sensitivity: float = 1.0) -> np.ndarray:
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale, data.shape)
        return data + noise

    def add_gaussian_noise(self, data: np.ndarray, sensitivity: float = 1.0, delta: float = 1e-5) -> np.ndarray:
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / self.epsilon
        noise = np.random.normal(0, sigma, data.shape)
        return data + noise

    def perturb_data(self, data: np.ndarray, sensitivity: float = 1.0) -> np.ndarray:
        if self.mechanism == 'laplace':
            return self.add_laplace_noise(data, sensitivity)
        elif self.mechanism == 'gaussian':
            return self.add_gaussian_noise(data, sensitivity)
        else:
            raise ValueError(f"Unknown mechanism: {self.mechanism}")
