"""
Taken from Murad TODO
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyproj import Geod
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from matplotlib.patches import Patch
from traffic.core import Traffic


def get_flight_durations(traffic):
    durations = []
    for flight in traffic:
        duration = (flight.data['timestamp'].max() - flight.data['timestamp'].min()).total_seconds() / 60  # in minutes
        durations.append(duration)
    return np.array(durations)

def get_flight_speeds(traffic, method='calculate', remove_outliers=True, lower_quantile=0.01, upper_quantile=0.99):
    all_speeds = []
    geod = Geod(ellps="WGS84")
    
    for flight in traffic:
        if method == 'groundspeed' and 'groundspeed' in flight.data.columns:
            speeds = flight.data['groundspeed'].values
        elif method == 'calculate':
            coords = np.column_stack((flight.data['longitude'], flight.data['latitude']))
            times = flight.data['timestamp'].values
            
            distances = geod.inv(coords[:-1, 0], coords[:-1, 1], coords[1:, 0], coords[1:, 1])[2]
            time_diffs = np.diff(times).astype('timedelta64[s]').astype(float)
            
            # Filter out invalid or negative time differences
            valid_mask = (time_diffs > 0) & (distances >= 0)
            valid_distances = distances[valid_mask]
            valid_time_diffs = time_diffs[valid_mask]
            
            # Calculate speeds and convert to km/h
            speeds = valid_distances / valid_time_diffs * 3.6  
        else:
            raise ValueError("Method must be either 'groundspeed' or 'calculate'")
        
        # Remove zero speeds
        speeds = speeds[speeds > 0]
        
        all_speeds.extend(speeds)
    
    all_speeds = np.array(all_speeds)
    if remove_outliers:
        lower_bound = np.quantile(all_speeds, lower_quantile)
        upper_bound = np.quantile(all_speeds, upper_quantile)
        all_speeds = all_speeds[(all_speeds >= lower_bound) & (all_speeds <= upper_bound)]
    
    return all_speeds



def duration_and_speed(training_trajectories, synthetic_trajectories, model_name = "model"):
    # Suppress specific UserWarnings related to set_xticklabels
    warnings.filterwarnings("ignore", message=".*set_ticklabels.*")

    # Set the style for a more professional look
    sns.set_style("whitegrid")
    sns.set_context("paper")
    training_durations = get_flight_durations(training_trajectories)
    synthetic_durations = get_flight_durations(synthetic_trajectories)

    training_speeds = get_flight_speeds(training_trajectories, method='groundspeed')
    synthetic_speeds = get_flight_speeds(synthetic_trajectories, method='groundspeed')

    # Create a single figure with two rows and two columns
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Row 1: Flight Durations
    sns.histplot(training_durations, kde=True, element="step", label='Real', color='#1f77b4', linewidth=1.5, ax=axes[0, 0])
    sns.histplot(synthetic_durations, kde=True, element="step", label='Synthetic', color='#ff7f0e', linewidth=1.5, ax=axes[0, 0])
    axes[0, 0].set_title('Flight Durations', fontsize=10)
    axes[0, 0].set_xlabel('Duration (minutes)', fontsize=8)
    axes[0, 0].set_ylabel('Density', fontsize=8)
    axes[0, 0].legend(fontsize=6)
    axes[0, 0].tick_params(labelsize=6)

    sns.boxplot(data=[training_durations, synthetic_durations], palette=['#1f77b4', '#ff7f0e'], ax=axes[0, 1])
    axes[0, 1].set_xticklabels(['Real', 'Synthetic'], fontsize=8)
    axes[0, 1].set_title('Flight Durations', fontsize=10)
    axes[0, 1].set_ylabel('Duration (minutes)', fontsize=8)
    axes[0, 1].tick_params(labelsize=6)

    # Row 2: Flight Speeds
    sns.histplot(training_speeds, kde=True, element="step", label='Real', color='#1f77b4', linewidth=1.5, ax=axes[1, 0])
    sns.histplot(synthetic_speeds, kde=True, element="step", label='Synthetic', color='#ff7f0e', linewidth=1.5, ax=axes[1, 0])
    axes[1, 0].set_title('Flight Speeds', fontsize=10)
    axes[1, 0].set_xlabel('Speed (km/h)', fontsize=8)
    axes[1, 0].set_ylabel('Density', fontsize=8)
    axes[1, 0].legend(fontsize=6)
    axes[1, 0].tick_params(labelsize=6)

    sns.boxplot(data=[training_speeds, synthetic_speeds], palette=['#1f77b4', '#ff7f0e'], ax=axes[1, 1])
    axes[1, 1].set_xticklabels(['Real', 'Synthetic'], fontsize=8)
    axes[1, 1].set_title('Flight Speeds', fontsize=10)
    axes[1, 1].set_ylabel('Speed (km/h)', fontsize=8)
    axes[1, 1].tick_params(labelsize=6)

    # Adjust layout and remove top and right spines
    plt.tight_layout()
    for ax in axes.flatten():
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Add a main title for the entire figure
    # fig.suptitle('Comparison of Real and Synthetic Flight Data', fontsize=12, y=1.02)

    # Save the figure
    plt.savefig(f"./figures/{model_name}_distribution_plots.png", bbox_inches='tight')


def timeseries_plot(
    real_traffic,
    synthetic_traffic,
    features: list,
    units: dict,
    n_plot_samples: int = 1000,
    alpha: float = 0.3,
    model_name = "model"
):
    # Set the style for a more professional look
    sns.set(style="whitegrid")
    sns.set_context("paper")

    # Create subplots
    fig, axes = plt.subplots(2, len(features), figsize=(4 * len(features), 6), sharex=True)

    # Prepare data
    datasets = [real_traffic, synthetic_traffic]
    dataset_names = ['Real', 'Synthetic']
    colors = ['#1f77b4', '#ff7f0e']  # Specified colors

    for feature_idx, feature in enumerate(features):
        # Top row: individual trajectories
        for dataset_idx, (dataset, color) in enumerate(zip(datasets, colors)):
            feature_data = np.array([flight.data[feature].values for flight in dataset])
            sample_ind = np.random.randint(0, len(feature_data), min(n_plot_samples, len(feature_data)))
            for idx in sample_ind:
                axes[0, feature_idx].plot(feature_data[idx], alpha=alpha, color=color)

        axes[0, feature_idx].set_title(f"{feature.capitalize()}", fontsize=12)
        axes[0, feature_idx].tick_params(labelsize=10)
        axes[0, feature_idx].set_ylabel(f"{feature.capitalize()} ({units[feature]})", fontsize=12)

        # Bottom row: mean and confidence intervals
        for dataset_idx, (dataset, color) in enumerate(zip(datasets, colors)):
            feature_data = np.array([flight.data[feature].values for flight in dataset])
            mean_data = np.mean(feature_data, axis=0)
            std_data = np.std(feature_data, axis=0)

            axes[1, feature_idx].plot(mean_data, color=color, linewidth=2)

            t_value = stats.t.ppf(0.975, df=len(feature_data)-1)
            ci = t_value * std_data * np.sqrt(1 + 1/len(feature_data))
            axes[1, feature_idx].fill_between(
                range(len(mean_data)),
                mean_data - ci,
                mean_data + ci,
                color=color,
                alpha=0.2
            )

        axes[1, feature_idx].set_xlabel("Time Steps", fontsize=12)
        axes[1, feature_idx].set_ylabel(f"{feature.capitalize()} ({units[feature]})", fontsize=12)
        axes[1, feature_idx].tick_params(labelsize=10)

        # Remove top and right spines for both rows
        for row in range(2):
            axes[row, feature_idx].spines['top'].set_visible(False)
            axes[row, feature_idx].spines['right'].set_visible(False)

    # Create custom legend elements
    legend_elements = []
    for color, name in zip(colors, dataset_names):
        legend_elements.append(plt.Line2D([0], [0], color=color, lw=3, label=f'{name}'))
        legend_elements.append(Patch(facecolor=color, edgecolor=color, alpha=alpha, label=f'{name} 95% CI'))

    # Add a single legend for the entire figure
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, fontsize=12, bbox_to_anchor=(0.5, 1.05))

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"./figures/{model_name}_timeseries_ci.png", bbox_inches='tight')

# Usage


