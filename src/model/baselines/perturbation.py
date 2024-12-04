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

    def random_perturbation(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Add uniform random noise to the trajectory.
        
        Args:
            trajectory: Tensor of shape (..., 2) containing lat/lon coordinates
        
        Returns:
            Perturbed trajectory
        """
        noise = (torch.rand_like(trajectory) * 2 * self.noise_ratio) - self.noise_ratio
        return trajectory + noise

    def gaussian_perturbation(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise to the trajectory.
        
        Args:
            trajectory: Tensor of shape (..., 2) containing lat/lon coordinates
        
        Returns:
            Perturbed trajectory
        """
        noise = torch.randn_like(trajectory) * torch.sqrt(torch.tensor(self.variance))
        return trajectory + noise

    def sample(self, trajectory: torch.Tensor, method="random") -> torch.Tensor:
        """
        Sample perturbed trajectories using the specified method.
        
        Args:
            trajectory: Tensor of shape (..., 2) containing lat/lon coordinates
            method: Perturbation method - "random" or "gaussian"
        
        Returns:
            Sampled perturbed trajectory
        """
        if method == "random":
            return self.random_perturbation(trajectory)
        elif method == "gaussian":
            return self.gaussian_perturbation(trajectory)
        else:
            raise ValueError(f"Unknown sampling method: {method}")

    def reconstruct(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Simulate a "reconstruction" process by removing the mean noise (for demo purposes).
        
        Args:
            trajectory: Tensor of shape (..., 2) containing lat/lon coordinates
        
        Returns:
            Reconstructed trajectory
        """
        mean_noise = torch.mean(trajectory, dim=0, keepdim=True)
        return trajectory - mean_noise

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
