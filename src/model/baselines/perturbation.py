import torch
from torch import nn
import pytorch_lightning as pl

class PerturbationModel(pl.LightningModule):
    def __init__(self, noise_ratio=0.01, variance=0.01):
        super().__init__()
        self.noise_ratio = noise_ratio
        self.variance = variance

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Applies random perturbation to the trajectory.
        
        Args:
            trajectory: Tensor of shape (..., 2) containing lat/lon coordinates
        
        Returns:
            Perturbed trajectory.
        """
        return self.random_perturbation(trajectory)

    def random_perturbation(self, trajectory: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """
        Add uniform random noise to the trajectory to generate n_samples.

        Args:
            trajectory: Tensor of shape (..., 2) containing lat/lon coordinates.
            n_samples: Number of perturbed samples to generate.

        Returns:
            Tensor of shape (n_samples, ..., 2) containing perturbed trajectories.
        """
        # Generate noise for n_samples
        noise = (torch.rand((n_samples, *trajectory.shape)) * 2 * self.noise_ratio) - self.noise_ratio
        # Expand trajectory and add noise
        trajectory_expanded = trajectory.unsqueeze(0).expand(n_samples, *trajectory.shape)
        return trajectory_expanded + noise

    def gaussian_perturbation(self, trajectory: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """
        Add Gaussian noise to the trajectory to generate n_samples.

        Args:
            trajectory: Tensor of shape (..., 2) containing lat/lon coordinates.
            n_samples: Number of perturbed samples to generate.

        Returns:
            Tensor of shape (n_samples, ..., 2) containing perturbed trajectories.
        """
        # Generate Gaussian noise for n_samples
        noise = torch.randn((n_samples, *trajectory.shape)) * torch.sqrt(torch.tensor(self.variance))
        # Expand trajectory and add noise
        trajectory_expanded = trajectory.unsqueeze(0).expand(n_samples, *trajectory.shape)
        return trajectory_expanded + noise

    def sample(self, trajectory: torch.Tensor, method="gaussian", n_samples: int = 1) -> torch.Tensor:
        """
        Sample perturbed trajectories using the specified method.

        Args:
            trajectory: Tensor of shape (..., 2) containing lat/lon coordinates.
            method: Perturbation method - "random" or "gaussian".
            n_samples: Number of perturbed samples to generate.

        Returns:
            Tensor of shape (n_samples, ..., 2) containing sampled perturbed trajectories.
        """
        if method == "random":
            return self.random_perturbation(trajectory, n_samples)
        elif method == "gaussian":
            return self.gaussian_perturbation(trajectory, n_samples)
        else:
            raise ValueError(f"Unknown sampling method: {method}")    

if __name__ == "__main__":
    # Initialize model
    model = PerturbationModel()
    example_trajectory = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])

    # Forward pass
    perturbed = model(example_trajectory)
    print("Forward (Random Perturbation):\n", perturbed)

    # Sampling
    sampled_random = model.sample(example_trajectory, method="random")
    print("Sampled (Random):\n", sampled_random)

    sampled_gaussian = model.sample(example_trajectory, method="gaussian")
    print("Sampled (Gaussian):\n", sampled_gaussian)

    # Reconstruction
    reconstructed = model.reconstruct(example_trajectory)
    print("Reconstructed:\n", reconstructed)
