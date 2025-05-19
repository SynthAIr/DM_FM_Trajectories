import torch
import pytorch_lightning as pl

class PerturbationModel(pl.LightningModule):
    """
    Perturbation model that applies random or Gaussian noise to trajectories.
    """
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
            trajectory:
        
        Returns:
            Perturbed trajectory.
        """
        return self.random_perturbation(trajectory)

    def random_perturbation(self, trajectory: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """
        Add uniform random noise to the trajectory to generate n_samples.

        Args:
            trajectory:
            n_samples:

        Returns:

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
            trajectory:
            n_samples:

        Returns:

        """
        noise = torch.randn((trajectory.shape)) * torch.sqrt(torch.tensor(self.variance))

        noise = noise.to(trajectory.device)
        return trajectory + noise

    def sample(self, trajectory: torch.Tensor, method="random", n_samples: int = 1) -> torch.Tensor:
        """
        Sample perturbed trajectories using the specified method.

        Args:
            trajectory:
            method:
            n_samples:

        Returns:
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
