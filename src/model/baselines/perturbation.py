import torch
import pandas as pd

def random_perturbation(trajectory: torch.Tensor, noise_ratio = 0.01) -> torch.Tensor:
    """
    Add uniform random noise between -0.01 and 0.01 to trajectory.
    
    Args:
        trajectory: Tensor of shape (..., 2) containing lat/lon coordinates
    Returns:
        Perturbed trajectory
    """
    noise = (torch.rand_like(trajectory) * 2 * noise_ratio) - noise_ratio # Generate noise between -0.01 and 0.01
    return trajectory + noise

def gaussian_perturbation(trajectory: torch.Tensor, variance = 0.01) -> torch.Tensor:
    """
    Add Gaussian noise with mean 0 and variance 0.01 to trajectory.
    
    Args:
        trajectory: Tensor of shape (..., 2) containing lat/lon coordinates
    Returns:
        Perturbed trajectory
    """
    noise = torch.randn_like(trajectory) * torch.sqrt(torch.tensor(variance))  # std = sqrt(variance)
    return trajectory + noise

# Example usage
if __name__ == "__main__":
    # Create example trajectory
    trajectory = torch.tensor([
        [40.7128, -74.0060],  # New York
        [34.0522, -118.2437], # Los Angeles
        [41.8781, -87.6298],  # Chicago
    ])
    
    # Random perturbation
    rp_trajectory = random_perturbation(trajectory)
    print("\nRandom Perturbation:")
    print("Original:", trajectory[0])
    print("Perturbed:", rp_trajectory[0])
    
    # Gaussian perturbation
    gp_trajectory = gaussian_perturbation(trajectory)
    print("\nGaussian Perturbation:")
    print("Original:", trajectory[0])
    print("Perturbed:", gp_trajectory[0])
