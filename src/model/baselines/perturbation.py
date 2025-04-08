import torch
from torch import nn
import pytorch_lightning as pl

class PerturbationModel(pl.LightningModule):
    def __init__(self, config ):
        super().__init__()
        self.config = config
        self.noise_ratio = config['noise_ratio']
        self.variance = config['variance']

    def to(self, device):
        return self

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
        noise = (torch.rand((trajectory.shape)) * 2 * self.noise_ratio) - self.noise_ratio
        noise = noise.to(trajectory.device)
        # Expand trajectory and add noise
        #trajectory_expanded = trajectory.unsqueeze(0).expand(n_samples, *trajectory.shape)
        return trajectory + noise

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
        noise = torch.randn((trajectory.shape)) * torch.sqrt(torch.tensor(self.variance))

        noise = noise.to(trajectory.device)
        # Expand trajectory and add noise
        #trajectory_expanded = trajectory.unsqueeze(0).expand(n_samples, *trajectory.shape)
        return trajectory + noise

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
            return self.random_perturbation(trajectory, n_samples), []
        elif method == "gaussian":
            return self.gaussian_perturbation(trajectory, n_samples), []
        else:
            raise ValueError(f"Unknown sampling method: {method}")    

    def reconstruct(self, x, con, cat, grid):
        return self.sample(x, n_samples = x.shape[0])

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
