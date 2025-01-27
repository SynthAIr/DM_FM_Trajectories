"""
From SINTEF SynthAIr
https://github.com/SynthAIr/T-VQ-VAE-TrajGen/blob/main/timevqvae/evaluation/flyability_eval.py
"""
import argparse
import glob
import os
import pickle
import re
from typing import Any, Dict, List, Tuple
import scipy.stats as stats

from matplotlib.figure import Figure

import cartopy.crs as ccrs
import cartopy.feature
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.basemap import Basemap
from traffic.core import Traffic

from evaluation.flyability_utils import (clean, discret_frechet, e_dtw, e_edr, e_erp,
                          e_hausdorff, e_lcss, e_sspd, frechet, s_dtw, s_edr,
                          s_erp, s_hausdorff, s_lcss, s_sspd, simulate)
from utils import extract_airport_coordinates, extract_geographic_info


def get_longest_non_outlier_flight_duration(
    training_data_path: str, outlier_threshold: float = 5.0
) -> str:
    # Load the data
    training_data = Traffic.from_file(training_data_path)

    # Group data by flight_id and calculate the maximum duration for each flight
    flight_durations = training_data.groupby(["flight_id"]).agg(
        duration=("timedelta", "max")
    )
    flight_durations = flight_durations.sort_values(by="duration", ascending=False)

    # Calculate IQR and determine outliers
    Q1 = flight_durations["duration"].quantile(0.25)
    Q3 = flight_durations["duration"].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + outlier_threshold * IQR

    # Filter out outliers
    non_outliers = flight_durations[flight_durations["duration"] <= upper_bound]

    # Select the longest duration that is not an outlier
    if not non_outliers.empty:
        longest_non_outlier_duration = non_outliers.iloc[0]["duration"]
    else:
        raise ValueError("All flights are outliers based on the current threshold.")

    # Convert the duration to a more readable format
    longest_duration_timedelta = pd.to_timedelta(longest_non_outlier_duration, unit="s")
    hours, remainder = divmod(longest_duration_timedelta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    formatted_duration = f"{hours:02}:{minutes:02}:{seconds:02}"

    # Print and return the formatted duration
    print(f"Longest non-outlier flight duration: {formatted_duration}")
    return formatted_duration


def get_most_common_ac_type(training_data_path: str) -> str:
    training_data = Traffic.from_file(training_data_path)

    # value counts of AC types
    v_c = training_data.data["AC Type"].value_counts()

    ac_types = training_data.groupby(["AC Type"]).agg(count=("AC Type", "count"))
    ac_types = ac_types.sort_values(by="count", ascending=False)

    most_common_ac_type = ac_types.index[0]
    print(f"Most common AC type: {most_common_ac_type}")
    return most_common_ac_type


def load_and_edit_generated_trajectories(
    directory: str, model_type: str, AC_type: str
) -> List[Traffic]:
    # Construct the file pattern to match generated traffic files
    pattern: str = os.path.join(directory, f"{model_type}_traf_gen*.pkl")

    # Find all matching traffic generation files
    traf_files: List[str] = sorted(glob.glob(pattern))
    print(f"Found {len(traf_files)} generated traffic files:\n     {traf_files}")

    # Initialize the last flight ID from the last available ID or start anew
    last_flight_id = 0

    # Load each file as a Traffic object and edit the data: change the flight_id to int64 and add the AC Type
    traffic_list = []
    for i, f in enumerate(traf_files):
        traj = Traffic.from_file(f)
        current_ids = traj.data["flight_id"].str.replace("TRAJ_", "").astype("int64")
        min_current_id = current_ids.min()
        max_current_id = current_ids.max()

        # Offset new IDs to continue from the last ID used
        offset = last_flight_id - min_current_id + 1
        traj.data["flight_id"] = current_ids + offset

        # Update the last_flight_id for the next iteration
        last_flight_id = max_current_id + offset

        traj.data["AC Type"] = AC_type
        # traj.to_pickle(f"{simulation_output_dir}{model_type}_traf_gen_{i}.pkl")
        traffic_list.append(traj)

    return traffic_list


def filter_simulated_traffic(
    SimTrajs_list: List[Traffic], training_data_path: str
) -> List[Traffic]:
    """
    filter simulated traffic data based on their proximity to the destination airport,
    i.e., remove redundant simulation points after the aircraft has landed at its destination.

    Args:
    SimTrajs_list (List[Traffic]): A list of Traffic instances containing simulated data.
    training_data_path (str): Path to training data to extract airport coordinates.

    Returns:
    List[Traffic]: A list of Traffic instances with filtered data.
    """
    # Extract target airport coordinates
    ADES_lat, ADES_lon = extract_airport_coordinates(training_data_path)

    # Filter traffic based on minimum distance to the airport and remove all subsequent points
    filtered_traffics = []
    for sim_traffic in SimTrajs_list:
        sim_traffic.data["distance"] = np.sqrt(
            (sim_traffic.data["latitude"] - ADES_lat) ** 2
            + (sim_traffic.data["longitude"] - ADES_lon) ** 2
        )

        filtered_results = []
        for flight_id, group in sim_traffic.data.groupby("flight_id"):
            min_index = group["distance"].idxmin()
            filtered_results.append(group.loc[:min_index])

        filtered_df = pd.concat(filtered_results).reset_index(drop=True)
        filtered_traffics.append(Traffic(filtered_df))

    return filtered_traffics


def simulate_traffic(
    traffic_list: List[Traffic], config: Dict[str, Any]
) -> List[Traffic]:

    SimTrajs_list = []
    for i, traffic in enumerate(traffic_list):
        print(f"Simulating traffic {i}")
        simulated_trajectories = simulate(generated_trajectories, simulation_config)
        #SimTrajs_list.append(simulate(traffic, config))
        clean()

    return SimTrajs_list


def plot_simulation_results(
    GenTrajs_list: List[Traffic], SimuTrajs_list: List[Traffic], training_data_path: str
) -> Figure:

    ADEP_code, ADES_code, geographic_extent = extract_geographic_info(
        training_data_path
    )

    plt.style.use("ggplot")

    fig = plt.figure(figsize=(20, 20))
    gs = gridspec.GridSpec(2, 2, wspace=0.001)
    ax0 = fig.add_subplot(gs[0], projection=ccrs.EuroPP())
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2], projection=ccrs.EuroPP())
    ax3 = fig.add_subplot(gs[3])

    # Setup color normalization
    max_trajectories = max(len(GenTrajs_list), len(SimuTrajs_list))
    color_norm = Normalize(vmin=0, vmax=max_trajectories)
    colormap = plt.cm.inferno

    # Function to plot trajectories
    def plot_trajectories(trajectories, ax, title):
        for i, t in enumerate(trajectories):
            color = colormap(color_norm(i))
            t.plot(ax, alpha=0.1, color=color, linewidth=1)
            #t["TRAJ_0"].plot(ax, color=color, linewidth=2)
        ax.set_title(title)
        ax.coastlines()
        ax.add_feature(cartopy.feature.BORDERS, linestyle=":", alpha=1.0)
        ax.set_extent(geographic_extent)
        ax.gridlines(draw_labels=True, color="gray", alpha=0.5, linestyle="--")

    # Plotting trajectories
    plot_trajectories(
        GenTrajs_list, ax0, f"Generated Trajectories: {ADEP_code} -> {ADES_code}"
    )
    plot_trajectories(
        SimuTrajs_list, ax2, f"Simulated Trajectories: {ADEP_code} -> {ADES_code}"
    )

    # Function to plot trajectories with altitude on a Basemap
    def plot_altitude_trajectories(traj_list, ax, title):
        for i, t in enumerate(traj_list):
            if i == 0:
                trajectories = t
            else:
                trajectories += t

        df = trajectories.data
        m = Basemap(
            projection="merc",
            llcrnrlat=geographic_extent[2],
            urcrnrlat=geographic_extent[3],
            llcrnrlon=geographic_extent[0],
            urcrnrlon=geographic_extent[1],
            lat_ts=20,
            resolution="i",
            ax=ax,
        )
        m.drawcoastlines()
        m.drawcountries()
        m.fillcontinents(color="lightgray", lake_color="aqua")
        m.drawmapboundary(fill_color="aqua")

        # Plotting and labeling specifics
        x, y = m(df["longitude"].values, df["latitude"].values)
        ax.plot(x, y, color="black", alpha=0.2, zorder=1)
        is_simulated = "simulated" in title.lower()
        sns.scatterplot(
            x=x,
            y=y,
            hue=df["altitude"],
            palette="viridis",
            size=df["altitude"],
            sizes=(20, 200),
            legend="brief",
            ax=ax,
            edgecolor="black",
            alpha=(
                0.1 if is_simulated else 1
            ),  # Adjust alpha based on whether it's simulated
        )
        norm = Normalize(vmin=df["altitude"].min(), vmax=df["altitude"].max())
        sm = ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, aspect=40)
        cbar.set_label("Altitude (feet)")
        ax.legend(loc="upper right")
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    # Plotting trajectories with altitude
    plot_altitude_trajectories(
        GenTrajs_list, ax1, f"Generated Trajectories: {ADEP_code} -> {ADES_code}"
    )
    plot_altitude_trajectories(
        SimuTrajs_list, ax3, f"Simulated Trajectories: {ADEP_code} -> {ADES_code}"
    )
    # plt.tight_layout()

    os.makedirs("./results/evaluation/figures", exist_ok=True)
    plt.savefig(
        "./results/evaluation/figures/generated_vs_simulated_trajectories.png", bbox_inches="tight"
    )
    return fig


def calculate_trajectory_distances(
    gen_traj: Traffic, simulated_traj: Traffic, ADEP_lat: float, ADEP_lon: float
) -> Dict[str, List[float]]:
    """
    Calculate various distance metrics between generated and simulated trajectories.

    Args:
        gen_traj (Traffic): Traffic object containing generated trajectories.
        simulated_traj (Traffic): Traffic object containing simulated trajectories.

    Returns:
        dict: Dictionary containing lists of distance metrics.
    """
    results = {
        "SSPD Euclidean": [],
        "SSPD Spherical": [],
        "DTW Euclidean": [],
        "DTW Spherical": [],
        "Hausdorff Euclidean": [],
        "Hausdorff Spherical": [],
        "LCSS Euclidean": [],
        "LCSS Spherical": [],
        "ERP Euclidean": [],
        "ERP Spherical": [],
        "EDR Euclidean": [],
        "EDR Spherical": [],
        "Discrete Frechet": [],
        "Frechet": [],
    }

    # ADEP_lat, ADEP_lon = extract_airport_coordinates(training_data_path)  # For ERP
    # Suppose you decide that a significant deviation for your analysis is around 1 kilometer.
    # To convert 1 kilometer to a degree measure for latitude: 1 km ≈ 1/111 ≈ 0.009 degrees.
    eps = 0.009  # For LCSS and EDR

    for i in range(len(gen_traj)):
    #for flight, simulated_flight in zip(gen_traj, simulated_traj):
        flight = gen_traj[i]
        simulated_flight = simulated_traj[i]
        trajectory_gen = flight.data[["latitude", "longitude"]].values
        trajectory_sim = simulated_flight.data[["latitude", "longitude"]].values

        # Compute SSPD distances
        results["SSPD Euclidean"].append(e_sspd(trajectory_gen, trajectory_sim))
        results["SSPD Spherical"].append(s_sspd(trajectory_gen, trajectory_sim))

        # Compute DTW distances
        results["DTW Euclidean"].append(e_dtw(trajectory_gen, trajectory_sim))
        results["DTW Spherical"].append(s_dtw(trajectory_gen, trajectory_sim))

        # Compute Hausdorff distances
        results["Hausdorff Euclidean"].append(
            e_hausdorff(trajectory_gen, trajectory_sim)
        )
        results["Hausdorff Spherical"].append(
            s_hausdorff(trajectory_gen, trajectory_sim)
        )

        # Compute LCSS distances
        results["LCSS Euclidean"].append(e_lcss(trajectory_gen, trajectory_sim, eps))
        results["LCSS Spherical"].append(
            s_lcss(trajectory_gen, trajectory_sim, eps * 1e6)
        )  # If I set eps to small value, then almost all points will be considered similar

        # Compute ERP distances
        # g represents a gap value or a reference point. This point is used to calculate the penalty when a point in one trajectory does not have a corresponding match in the other trajectory during the alignment process. Essentially, g acts as a virtual "base" point against which distances are measured when no real corresponding point exists in the other trajectory.
        # Boundary Values:If trajectories are fairly localized or if you want to ensure g is somewhat removed from actual data points (to avoid skewing results if g coincides too closely with common trajectory points), you might choose g as a point on the boundary of the geographical extent of your trajectories.
        g = (ADEP_lat, ADEP_lon)  # Reference point
        results["ERP Euclidean"].append(e_erp(trajectory_gen, trajectory_sim, g))
        results["ERP Spherical"].append(s_erp(trajectory_gen, trajectory_sim, g))

        # Compute EDR distances
        results["EDR Euclidean"].append(e_edr(trajectory_gen, trajectory_sim, eps))
        results["EDR Spherical"].append(s_edr(trajectory_gen, trajectory_sim, eps))

        # Compute Discrete Frechet distance
        results["Discrete Frechet"].append(
            discret_frechet(trajectory_gen, trajectory_sim)
        )

        # Compute Frechet distance
        results["Frechet"].append(frechet(trajectory_gen, trajectory_sim))

    return results


def plot_distances_cumulative_distributions(
    all_distances_results: Dict[str, List[float]]
) -> Figure:
    n_metrics = len(all_distances_results)
    ncols = 2
    nrows = (n_metrics + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows * 5))
    axes = axes.flatten()

    # # set ligth sns bakcground
    sns.set_style("white")

    for i, (key, values) in enumerate(all_distances_results.items()):
        if values:
            sns.kdeplot(
                values,
                fill=True,
                ax=axes[i],
                cumulative=True,
                label=f"{key}",
                common_norm=False,
                color="darkblue",
            )
            axes[i].set_title(f"{key}", fontweight="bold")

            if i >= len(all_distances_results) - ncols:
                axes[i].set_xlabel("Distance")
            else:
                axes[i].set_xlabel("")
            if i % ncols == 0:
                axes[i].set_ylabel("Cumulative Probability")
            else:
                axes[i].set_ylabel("")
            axes[i].legend()
            axes[i].grid(True, linestyle="--", alpha=0.6)
        else:
            axes[i].set_title(f"No Data for {key}", fontweight="bold")
            axes[i].set_visible(False)

    for j in range(i + 1, nrows * ncols):
        axes[j].set_visible(False)

    fig.suptitle(
        "Cumulative Distributions of distances between generated and simulated trajectories",
        fontsize=16,
        fontweight="bold",
        y=1.001,
    )
    for ax in axes:
        ax.set_facecolor("white")  # Sets the individual subplot background color
    fig.set_facecolor("white")  # Sets the overall figure background color

    plt.tight_layout()
    os.makedirs("./results/evaluation/figures", exist_ok=True)
    plt.savefig(
        "./results/evaluation/figures/distances_cumulative_distributions.png", bbox_inches="tight"
    )
    return fig


def create_percentile_plot(data, output_filename):
    """
    Creates a percentile plot from the given dataframe and saves it to the specified file.
    
    Parameters:
        data (pd.DataFrame): The input dataframe containing data to plot.
        output_filename (str): Path to save the resulting plot.
    """
    # Set up seaborn style
    sns.set(style="whitegrid")
    sns.set_style("whitegrid", {'axes.grid': True, 'grid.color': '.8', 'grid.linestyle': ':'})
    
    # Define a custom color palette (colorblind-friendly)
    custom_colors = [
        "#0077BB",  # Blue
        "#33BBEE",  # Cyan
        "#009988",  # Teal
        "#EE7733",  # Orange
        "#CC3311",  # Red
        "#EE3377",  # Magenta
        "#BBBBBB",  # Grey
    ]
    
    # Define line styles
    line_styles = ['-', '--', '-.', ':']

    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=(12, 6))

    # Prepare the plot
    num_columns = len(data.columns)
    colors = custom_colors * (num_columns // len(custom_colors) + 1)
    styles = line_styles * (num_columns // len(line_styles) + 1)

    for idx, column in enumerate(data.columns):
        column_data = data[column]
        percentiles = [stats.percentileofscore(column_data, value) for value in column_data]
        ax.plot(sorted(column_data), sorted(percentiles),
                label=column,
                color=colors[idx],
                linestyle=styles[idx],
                linewidth=3)

    # Customize the plot
    ax.set_xlabel('Distance (log scale)', fontsize=14)
    ax.set_ylabel('Percentile', fontsize=14)
    ax.set_xscale('log')
    ax.grid(True, linestyle='--', alpha=0.7, color='#CCCCCC')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=10)

    # Add a box around the plot
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)

    # Customize legend
    legend = ax.legend(fontsize=12,
                       loc='center left',
                       bbox_to_anchor=(0.86, 0.5),
                       frameon=True,
                       fancybox=True,
                       shadow=True,
                       borderpad=1)
    legend.set_title("Metrics", prop={'size': 14})

    # Adjust the layout and save
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight')
    return fig


def create_correlation_heatmap(data, output_filename, title=None):
    """
    Creates a correlation heatmap from the given dataframe and saves it to the specified file.
    
    Parameters:
        data (pd.DataFrame): The input dataframe containing the data for correlation computation.
        output_filename (str): Path to save the resulting heatmap.
        title (str, optional): Title of the heatmap. Defaults to None.
    """
    sns.set(style="whitegrid")
    
    # Compute the correlation matrix
    correlation_matrix = data.corr()
    
    # Create a mask to hide the upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    # Set up the figure size
    fig = plt.figure(figsize=(12, 8))
    
    # Create the heatmap with additional styling
    heatmap = sns.heatmap(correlation_matrix,
                          annot=True,
                          fmt='.2f',
                          cmap='RdBu_r',  # A more color-blind friendly palette
                          vmin=-1, vmax=1,
                          center=0,
                          linewidths=0.5,
                          linecolor='white',
                          cbar_kws={'shrink': 0.8, 'aspect': 30, 'label': 'Correlation Coefficient'},
                          square=True,
                          mask=mask)  # Apply the mask to hide upper triangle
    
    # Add title if provided
    if title:
        plt.title(title, fontsize=20, weight='bold', pad=20)
    
    # Customize tick labels
    plt.xticks(fontsize=12, rotation=45, ha='right')
    plt.yticks(fontsize=12, rotation=0)
    
    # Adjust colorbar label
    cbar = heatmap.collections[0].colorbar
    cbar.ax.set_ylabel('Correlation Coefficient', fontsize=14, labelpad=15)
    cbar.ax.tick_params(labelsize=12)
    
    # Add a subtle border around the heatmap
    heatmap.figure.axes[0].add_artist(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='gray', lw=1))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight')
    return fig



def run(training_data_path: str, synthetic_data_path: str, logger = None) -> None:

    simulation_time = get_longest_non_outlier_flight_duration(training_data_path)
    #ADEP_code, ADES_code, geographic_extent = extract_geographic_info(training_data_path)
    most_common_ac_type = "A319"
    generated_trajectories = Traffic.from_file(synthetic_data_path)
    generated_trajectories.data["AC Type"] = most_common_ac_type
    #generated_trajectories = generated_trajectories[generated_trajectories['ADES'] != 'EHAM']

    simulation_config = {
        "delta": 2000,
        "batch_size": 256,
        "early_batch_stop": False,
        "logs_directory": os.path.expanduser("~/bluesky/output/"),
        "simulation_time": simulation_time,
    }

    simulated_trajectories = simulate(generated_trajectories, simulation_config)
    clean()

    # SimuTrajs_list = filter_simulated_traffic(SimuTrajs_list, args.data_path)
    simulated_trajectories = filter_simulated_traffic(simulated_trajectories, training_data_path)


    #SimuTrajs_list = simulate_traffic(GenTrajs_list, simulation_config)
    #SimuTrajs_list = filter_simulated_traffic(SimuTrajs_list, args.data_path)
    fig_simulation = plot_simulation_results(generated_trajectories,simulated_trajectories , training_data_path)

    ADEP_lat, ADEP_lon = extract_airport_coordinates(training_data_path)

    all_distances_results = {
        "DTW Euclidean": [],
        "DTW Spherical": [],
        "SSPD Euclidean": [],
        "SSPD Spherical": [],
        "LCSS Euclidean": [],
        "LCSS Spherical": [],
        "Hausdorff Euclidean": [],
        "Hausdorff Spherical": [],
        "ERP Euclidean": [],
        "ERP Spherical": [],
        "EDR Euclidean": [],
        "EDR Spherical": [],
        "Discrete Frechet": [],
        "Frechet": [],
    }

    print("Calculating distances between generated and simulated trajectories...")
    #for gen_traj, simulated_traj in zip(generated_trajectories,simulated_trajectories):
    distances = calculate_trajectory_distances(
        generated_trajectories, simulated_trajectories, ADEP_lat, ADEP_lon
    )
    for key, values in distances.items():
        all_distances_results[key].extend(values)


    # Convert the dictionary to a DataFrame

    df = pd.DataFrame(all_distances_results)
    # Save the DataFrame to a CSV file
    file_path = synthetic_data_path.replace(".pkl", "_distances.csv")
    print("Saving to distances to CSV")
    df.to_csv(file_path, index=False)

    fig_distance = plot_distances_cumulative_distributions(all_distances_results)

    if logger:
        logger.experiment.log_figure(logger.run_id,fig_simulation, f"figures/flyability_synthetic_simulation.png")
        logger.experiment.log_figure(logger.run_id,fig_distance, f"figures/flyability_distances.png")

    def get_euclidean(df: pd.DataFrame) -> pd.DataFrame:
        df_euclidean = df.filter(like="Euclidean").copy()  # Filter columns with Euclidean metrics and create a copy
        df_euclidean.loc[:, 'Frechet'] = df['Frechet']
        df_euclidean.loc[:, 'Discrete Frechet'] = df['Discrete Frechet']
        df_euclidean.columns = df_euclidean.columns.str.replace('Euclidean', '').str.strip()
        df_euclidean.columns = df_euclidean.columns.str.replace('Discrete', 'Dis.').str.strip()
        return df_euclidean

    def get_sphereical(df: pd.DataFrame) -> pd.DataFrame:
        df_spherical = df.filter(like="Spherical")  # Filter columns with Spherical metrics
        # Add Frechet distance to Euclidean metrics using .loc to avoid SettingWithCopyWarning
        df_spherical.columns = df_spherical.columns.str.replace('Spherical', '').str.strip()
        return df_spherical

    df_euclidean = get_euclidean(df)
    df_spherical = get_sphereical(df)

    fig_euc = create_percentile_plot(df_euclidean, "results/evaluation/figures/euclidean_percentile_plot.png")
    fig_sph = create_percentile_plot(df_spherical, "results/evaluation/figures/spherical_percentile_plot.png")
    corr_euc = create_correlation_heatmap(df_euclidean, 'figures/correlation_heatmap_euclidean.png', title='Correlation Heatmap (Euclidean)')
    corr_sph = create_correlation_heatmap(df_spherical, 'figures/correlation_heatmap_spherical.png', title='Correlation Heatmap (Spherical)')

    if logger:
        logger.experiment.log_figure(logger.run_id,fig_euc, f"figures/euclidean_percentile.png")
        logger.experiment.log_figure(logger.run_id,fig_sph, f"figures/spherical_percentile.png")
        logger.experiment.log_figure(logger.run_id,corr_euc, f"figures/correlation_heatmap_euclidean.png")
        logger.experiment.log_figure(logger.run_id,corr_sph, f"figures/correlation_heatmap_spherical.png")
        logger.finalize()
    
