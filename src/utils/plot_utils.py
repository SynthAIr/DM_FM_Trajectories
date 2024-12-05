import glob
import os
import pickle
from typing import Any, Dict, List, Tuple

import altair as alt
import cartopy.crs as ccrs
import cartopy.feature
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from cartopy.crs import EuroPP, PlateCarree
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from mpl_toolkits.basemap import Basemap
from traffic.core import Traffic
from traffic.data import airports


def extract_geographic_info(
    training_data_path: str,
) -> Tuple[str, str, float, float, float, float, float, float]:

    training_data = Traffic.from_file(training_data_path)
    training_data = Traffic(training_data.data[training_data.data['ADEP'] == 'EHAM'])

    # raise an error if there exists more than one destination airport or if there are more than one origin airport
    # But why?
    if len(training_data.data["ADES"].unique()) > 1:
        raise ValueError("There are multiple destination airports in the training data")
    if len(training_data.data["ADES"].unique()) == 0:
        raise ValueError("There are no destination airports in the training data")
    if len(training_data.data["ADEP"].unique()) > 1:
        raise ValueError("There are multiple origin airports in the training data")
    if len(training_data.data["ADEP"].unique()) == 0:
        raise ValueError("There are no origin airports in the training data")

    ADEP_code = training_data.data["ADEP"].value_counts().idxmax()
    ADES_code = training_data.data["ADES"].value_counts().idxmax()

    # Determine the geographic bounds for plotting
    lon_min = training_data.data["longitude"].min()
    lon_max = training_data.data["longitude"].max()
    lat_min = training_data.data["latitude"].min()
    lat_max = training_data.data["latitude"].max()

    # Padding to apply around the bounds
    lon_padding = 1
    lat_padding = 1

    geographic_extent = [
        lon_min - lon_padding,
        lon_max + lon_padding,
        lat_min - lat_padding,
        lat_max + lat_padding,
    ]

    return ADEP_code, ADES_code, geographic_extent


def extract_airport_coordinates(
    training_data_path: str,
) -> Tuple[float, float]:

    training_data = Traffic.from_file(training_data_path)

    if len(training_data.data["ADES"].unique()) > 1:
        raise ValueError("There are multiple destination airports in the training data")
    if len(training_data.data["ADES"].unique()) == 0:
        raise ValueError("There are no destination airports in the training data")

    ADES_code = training_data.data["ADES"].value_counts().idxmax()

    ADES_code = training_data.data["ADES"].value_counts().idxmax()
    ADES_lat = airports[ADES_code].latitude
    ADES_lon = airports[ADES_code].longitude
    return ADES_lat, ADES_lon


def plot_generated_trajectories_and_latent_spaces(
    wind_dict: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    save_separately: bool = True,
    output_dir: str = "output_plots",
    training_data_path = "",
    name = "standard",
) -> None:
    """
    Plot generated trajectories and their corresponding latent space projections.
    """
    ADEP_code, ADES_code, geographic_extent = extract_geographic_info(
        training_data_path
    )

    plt.style.use("ggplot")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir+"/trajectories", exist_ok=True)
    os.makedirs(output_dir+"/lat_space", exist_ok=True)

    if save_separately:
        fig1 = plt.figure(figsize=(12, 12))
        ax1 = fig1.add_subplot(1, 1, 1, projection=ccrs.EuroPP())
        fig2 = plt.figure(figsize=(12, 12))
        ax2_1 = fig2.add_subplot(2, 1, 1)
        ax2_2 = fig2.add_subplot(2, 1, 2)
    else:
        fig = plt.figure(figsize=(36, 12))
        gs = gridspec.GridSpec(1, 3, width_ratios=[2, 2, 1])
        ax1 = fig.add_subplot(gs[1], projection=ccrs.EuroPP())
        ax2 = fig.add_subplot(gs[2])

    ax1.coastlines()
    ax1.add_feature(cartopy.feature.BORDERS, linestyle=":", alpha=1.0)
    ax1.set_extent(geographic_extent)
    
    for index, (wind_dir, (traffic_list, Z_gen)) in enumerate(wind_dict.items()):
                # Plot trajectories
        colormap = plt.cm.tab10
        #print(Z_gen['type'].unique())
        color_idx = index / len(wind_dict)
        color = colormap(color_idx + 0.3)
        ax2_1.scatter(
            Z_gen.query("type.isnull()").X1,
            Z_gen.query("type.isnull()").X2,
            color="black",
            s=2,
            alpha=0.1,
        )
        
        lat_space_points_x1 = []
        lat_space_points_x2 = []

        for i, t in enumerate(traffic_list):
            if i == 0:
                traj = t
            else:
                traj += t

            lat_space_points_x1.append(Z_gen.query(f"type == 'GEN{i+1}'").X1)
            lat_space_points_x2.append(Z_gen.query(f"type == 'GEN{i+1}'").X2)

        X1 = np.concatenate(lat_space_points_x1)
        X2 = np.concatenate(lat_space_points_x2)

        ax2_1.scatter(
        X1,
        X2,
        color=color,
        s=20,
        alpha=0.3,
        label=f"GT {wind_dir}",
        )
        ax2_2.scatter(
            X1,
            X2,
            color=color,
            s=20,
            alpha=0.3,
            label=f"GT {wind_dir}",
        )


        t = traj
        t.plot(ax1, alpha=0.1, color=color, linewidth=0.2)
        t["TRAJ_0"].plot(
            ax1,
            color=color,
            linewidth=0.2,
            label=f"GT {wind_dir}",
        )
        ax1.legend(loc="upper right")
    ax1.gridlines(draw_labels=True, color="gray", alpha=0.5, linestyle="--")

    ax2_1.set_title("Latent Space projection")
    ax2_1.legend(loc="upper right", fontsize=12)
    # remove gridlines
    ax2_1.grid(False)
    # remove axis labels and ticks
    ax2_1.set_xticks([])
    ax2_1.set_yticks([])

    ax2_2.set_title("Latent Space projection")
    ax2_2.legend(loc="upper right", fontsize=12)
    # remove gridlines
    ax2_2.grid(False)
    # remove axis labels and ticks
    ax2_2.set_xticks([])
    ax2_2.set_yticks([])
    
    for index, (wind_dir, (traffic_list, Z_gen)) in enumerate(wind_dict.items()):
        hell = False
        if index == 0:
            hell = True
        for i, t in enumerate(traffic_list):
            if i == 0 and hell:
                trajectories = t
            else:
                trajectories += t

    df = trajectories.data
    
    plt.tight_layout()
    
    # Save the figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_separately:
        fig1.savefig(os.path.join(output_dir+"/trajectories", f'{name}_trajectories.png'))
        fig2.savefig(os.path.join(output_dir+"lat_space", f'{name}_embeddings.png'))
        plt.close(fig1)
        plt.close(fig2)
    else:
        plt.savefig(os.path.join(output_dir, f'wind_trajectories_embeddings.png'))
        plt.close(fig)
    
    print(f"Saved plots for {name} in {output_dir}")


def plot_latent_space(
    traffic_list: List[Traffic],
    Z_gen: pd.DataFrame,
    colors: List[Any],
) -> Figure:
    """
    Plot the latent space of generated data.
    """

    plt.style.use("ggplot")
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(
        Z_gen.query("type.isnull()").X1,
        Z_gen.query("type.isnull()").X2,
        color="black",
        s=4,
        alpha=0.1,
        label="Training Data Latent Space",
    )
    
    for i, t in enumerate(traffic_list):
        id_ = t.data["gen_number"].values[0]
        ax.scatter(
            Z_gen.query(f"type == 'GEN{id_}'").X1,
            Z_gen.query(f"type == 'GEN{id_}'").X2,
            color=colors[i],
            s=50,
            alpha=1.0,
            label=f"Generated Trajectories (Pseudo-input {i})",
        )

    ax.set_title("Latent Space of Generated Trajectories")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.legend(loc="upper right", fontsize=12)
    # remove gridlines
    ax.grid(False)
    # remove axis labels and ticks
    ax.set_xticks([])
    ax.set_yticks([])
    return fig

def plot_generated_trajectories(
    training_data_path: str,
    generated_data_dir: str,
    save_path: str,
    geographic_extent,
    traffic_list: list
) -> None:

    ADEP_code, ADES_code, geographic_extent = extract_geographic_info(
        training_data_path
    )

    traffic_list: List[Traffic] = load_generated_trajectories(
        generated_data_dir, "Diffusion"
    )
    """
    Plot generated trajectories.
    """
    plt.style.use("ggplot")
    fig3 = plt.figure(figsize=(12, 12))
    ax3 = fig3.add_subplot(1, 1, 1)

    for i, t in enumerate(traffic_list):
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
        ax=ax3,
    )
    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents(color="lightgray", lake_color="aqua")
    m.drawmapboundary(fill_color="aqua")

    x, y = m(df["longitude"].values, df["latitude"].values)
    ax3.plot(x, y, color="black", alpha=0.2, zorder=1)
    sns.scatterplot(
        x=x,
        y=y,
        hue=df["altitude"],
        palette="viridis",
        size=df["altitude"],
        sizes=(20, 200),
        legend="brief",
        ax=ax3,
        edgecolor="black",
    )
    norm = Normalize(vmin=df["altitude"].min(), vmax=df["altitude"].max())
    sm = ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax3, aspect=30)
    cbar.set_label("Altitude (feet)")
    ax3.legend(loc="upper right")
    ax3.set_title(f"Flight Path from {ADEP_code} to {ADES_code}")
    ax3.set_xlabel("Longitude")
    ax3.set_ylabel("Latitude")

    plt.tight_layout()

    fig3.savefig(
        f"{save_path}/generated_trajectories_with_altitude_{ADEP_code}_to_{ADES_code}.png",
        bbox_inches="tight",
    )
    print(
        f"Saved figure to {save_path}/generated_trajectories_with_altitude_{ADEP_code}_to_{ADES_code}.png"
    )

def plot_generated_trajectories_and_latent_space(
    training_data_path: str,
    generated_data_dir: str,
    save_separately: bool = True,
) -> None:
    """
    Plot generated trajectories and their corresponding latent space projections.
    """

    ADEP_code, ADES_code, geographic_extent = extract_geographic_info(
        training_data_path
    )
    traffic: Traffic = load_generated_trajectories(
        generated_data_dir, "TCVAE"
    )

    traffic_list: List[Traffic] = [Traffic(t) for _, t in traffic.groupby("gen_number")]

    Z_gen: pd.DataFrame = pd.read_pickle(
        generated_data_dir + "/latent_space_vampprior_TCVAE.pkl"
    )

    plt.style.use("ggplot")
    # Create figures and axes based on the save_separately option
    if save_separately:
        fig1 = plt.figure(figsize=(12, 12))
        ax1 = fig1.add_subplot(1, 1, 1, projection=ccrs.EuroPP())
        fig2 = plt.figure(figsize=(6, 12))
        ax2 = fig2.add_subplot(1, 1, 1)
        fig3 = plt.figure(figsize=(12, 12))
        ax3 = fig3.add_subplot(1, 1, 1)
    else:
        fig = plt.figure(figsize=(36, 12))
        gs = gridspec.GridSpec(1, 3, width_ratios=[2, 2, 1])
        ax1 = fig.add_subplot(gs[1], projection=ccrs.EuroPP())
        ax2 = fig.add_subplot(gs[2])
        ax3 = fig.add_subplot(gs[0])

    # Configure the map and scatter plots
    ax1.coastlines()
    ax1.add_feature(cartopy.feature.BORDERS, linestyle=":", alpha=1.0)
    ax1.set_extent(geographic_extent)
    ax2.scatter(
        Z_gen.query("type.isnull()").X1,
        Z_gen.query("type.isnull()").X2,
        color="black",
        s=4,
        alpha=0.1,
        label="Training Data Latent Space",
    )

    # Process each trajectory for plotting
    colormap = plt.cm.inferno
    # colormap = plt.cm.tab20
    for i, t in enumerate(traffic_list):
        color_idx = i / len(traffic_list)
        color = colormap(color_idx + 0.3)
        t.plot(ax1, alpha=0.1, color=color, linewidth=1)
        t["TRAJ_0"].plot(
            ax1,
            color=color,
            linewidth=2,
            label=f"Generated Trajectories (Pseudo-input {i+1})",
        )
        ax2.scatter(
            Z_gen.query(f"type == 'GEN{i+1}'").X1,
            Z_gen.query(f"type == 'GEN{i+1}'").X2,
            color=color,
            s=50,
            alpha=1.0,
            label=f"Generated Trajectories (Pseudo-input {i+1})",
        )
        # ax2.scatter(Z_gen.query(f"type == 'PI{i+1}'").X1, Z_gen.query(f"type == 'PI{i+1}'").X2, c=color, s=50, label=f"Pseudo-input {i+1}")

    # Add additional map features and legends
    ax1.legend(loc="upper right")
    ax1.gridlines(draw_labels=True, color="gray", alpha=0.5, linestyle="--")
    ax2.set_title("Latent Space projection")
    ax2.legend(loc="upper right", fontsize=12)
    # remove gridlines
    ax2.grid(False)
    # remove axis labels and ticks
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Combine trajectories for a comprehensive path
    for i, t in enumerate(traffic_list):
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
        ax=ax3,
    )
    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents(color="lightgray", lake_color="aqua")
    m.drawmapboundary(fill_color="aqua")

    # Plotting and labeling specifics
    x, y = m(df["longitude"].values, df["latitude"].values)
    ax3.plot(x, y, color="black", alpha=0.2, zorder=1)
    sns.scatterplot(
        x=x,
        y=y,
        hue=df["altitude"],
        palette="viridis",
        size=df["altitude"],
        sizes=(20, 200),
        legend="brief",
        ax=ax3,
        edgecolor="black",
    )
    norm = Normalize(vmin=df["altitude"].min(), vmax=df["altitude"].max())
    sm = ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax3, aspect=30)
    cbar.set_label("Altitude (feet)")
    ax3.legend(loc="upper right")
    ax3.set_title(f"Flight Path from {ADEP_code} to {ADES_code}")
    ax3.set_xlabel("Longitude")
    ax3.set_ylabel("Latitude")

    plt.tight_layout()

    # results_directory = os.path.dirname(generated_data_path)
    save_path = os.path.join(generated_data_dir, "figures")
    os.makedirs(save_path, exist_ok=True)

    # Save figures based on configuration
    if save_separately:
        # fig1.savefig(f"./figures/generated_trajectories_{ADEP_code}_to_{ADES_code}.png", bbox_inches="tight")
        # fig2.savefig(f"./figures/latent_space_projection_{ADEP_code}_to_{ADES_code}.png", bbox_inches="tight")
        # fig3.savefig(
        #     f"./figures/generated_trajectories_with_altitude_{ADEP_code}_to_{ADES_code}.png", bbox_inches="tight"
        # )
        fig1.savefig(
            f"{save_path}/generated_trajectories_{ADEP_code}_to_{ADES_code}.png",
            bbox_inches="tight",
        )
        print(
            f"Saved figure to {save_path}/generated_trajectories_{ADEP_code}_to_{ADES_code}.png"
        )
        fig2.savefig(
            f"{save_path}/latent_space_projection_{ADEP_code}_to_{ADES_code}.png",
            bbox_inches="tight",
        )
        print(
            f"Saved figure to {save_path}/latent_space_projection_{ADEP_code}_to_{ADES_code}.png"
        )
        fig3.savefig(
            f"{save_path}/generated_trajectories_with_altitude_{ADEP_code}_to_{ADES_code}.png",
            bbox_inches="tight",
        )
        print(
            f"Saved figure to {save_path}/generated_trajectories_with_altitude_{ADEP_code}_to_{ADES_code}.png"
        )
    else:
        # plt.savefig(
        #     f"./figures/generated_trajectories_with_altitude_and_latent_space_{ADEP_code}_to_{ADES_code}.png",
        #     bbox_inches="tight",
        # )
        plt.savefig(
            f"{save_path}/generated_trajectories_with_altitude_and_latent_space_{ADEP_code}_to_{ADES_code}.png",
            bbox_inches="tight",
        )
        print(
            f"Saved figure to {save_path}/generated_trajectories_with_altitude_and_latent_space_{ADEP_code}_to_{ADES_code}.png"
        )


    # plt.show()


def load_generated_trajectories(directory: str, model_type: str) -> List[Traffic]:
    # Construct the file pattern to match generated traffic files
    traf_save_name = os.path.join(directory, f"{model_type}_traf_gen.pkl")
    return Traffic.from_file(traf_save_name)

def plot_tcvae_reconstruction(
    training_data_path: str,
    traffics_path: str,
    z_tcvae_path: str,
    save_separately: bool = False,
) -> None:

    ADEP_code, ADES_code, geographic_extent = extract_geographic_info(
        training_data_path
    )

    with open(traffics_path, "rb") as f:
        traffics = pickle.load(f)
    z_tcvae = pd.read_pickle(z_tcvae_path)

    # Define color cycle for plotting
    color_cycle = plt.cm.inferno(np.linspace(0, 1, len(z_tcvae.label.unique())))
    # different color cycle
    # color_cycle = plt.cm.tab20(np.linspace(0, 1, len(z_tcvae.label.unique())))

    np.random.seed(49)
    color_cycle = np.random.permutation(color_cycle)
    # Apply colors based on labels
    colors = [color_cycle[int(i) % len(color_cycle)] for i in z_tcvae.label]

    plt.style.use("ggplot")

    if save_separately:

        fig1 = plt.figure(figsize=(12, 12))
        ax0 = fig1.add_subplot(1, 1, 1, projection=ccrs.EuroPP())
        fig2 = plt.figure(figsize=(9, 12))
        ax1 = fig2.add_subplot(1, 1, 1)
    else:
        fig = plt.figure(figsize=(24, 12))
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
        ax0 = fig.add_subplot(gs[0], projection=ccrs.EuroPP())
        ax1 = fig.add_subplot(gs[1])

    ax0.coastlines()
    ax0.add_feature(cartopy.feature.BORDERS, linestyle=":", alpha=1.0)
    ax0.set_extent(geographic_extent)
    airports[ADEP_code].point.plot(
        ax0, color="red", label=f"Origin: {ADEP_code}", s=500, zorder=5
    )
    airports[ADES_code].point.plot(
        ax0, color="green", label=f"Destination: {ADES_code}", s=500, zorder=5
    )

    # Plot reconstructed trajectories
    ax0.set_title(
        f"Reconstructed Trajectories from {ADEP_code} to {ADES_code} using TCVAE"
    )
    for i, traf in enumerate(traffics):
        traf.plot(ax=ax0, alpha=0.2, color=color_cycle[i % len(color_cycle)])

    ax0.legend(loc="upper right")
    gridlines = ax0.gridlines(draw_labels=True, color="gray", alpha=0.5, linestyle="--")
    gridlines.top_labels = False
    gridlines.right_labels = False

    ax1.scatter(z_tcvae.X1, z_tcvae.X2, s=4, c=colors)
    ax1.set_title(
        f"Latent space of trajectories from {ADEP_code} to {ADES_code} using TCVAE"
    )
    ax1.grid(False)
    ax1.set_xticks([])
    ax1.set_yticks([])

    results_directory = os.path.dirname(z_tcvae_path)
    save_path = os.path.join(results_directory, "figures")
    os.makedirs(save_path, exist_ok=True)

    # plt.tight_layout()
    if save_separately:
        # fig1.savefig(f"./figures/reconstructed_trajectories_{ADEP_code}_to_{ADES_code}.png", bbox_inches="tight")
        # fig2.savefig(f"./figures/latent_space_projection_{ADEP_code}_to_{ADES_code}.png", bbox_inches="tight")
        fig1.savefig(
            f"{save_path}/reconstructed_trajectories_{ADEP_code}_to_{ADES_code}_TCVAE.png",
            bbox_inches="tight",
        )
        print(
            f"Saved figure to {save_path}/reconstructed_trajectories_{ADEP_code}_to_{ADES_code}_TCVAE.png"
        )
        fig2.savefig(
            f"{save_path}/latent_space_projection_{ADEP_code}_to_{ADES_code}_TCVAE.png",
            bbox_inches="tight",
        )
        print(
            f"Saved figure to {save_path}/latent_space_projection_{ADEP_code}_to_{ADES_code}_TCVAE.png"
        )
    else:
        # plt.savefig(
        #     f"./figures/reconstructed_trajectories_and_latent_space_{ADEP_code}_to_{ADES_code}.png", bbox_inches="tight"
        # )
        plt.savefig(
            f"{save_path}/reconstructed_trajectories_and_latent_space_{ADEP_code}_to_{ADES_code}_TCVAE.png"
        )
        print(
            f"Saved figure to {save_path}/reconstructed_trajectories_and_latent_space_{ADEP_code}_to_{ADES_code}_TCVAE.png"
        )

    # plt.show()


def plot_fcvae_reconstruction(
    training_data_path: str,
    traffics_path: str,
    z_fcvae_path: str,
    save_separately: bool = False,
) -> None:

    ADEP_code, ADES_code, geographic_extent = extract_geographic_info(
        training_data_path
    )

    with open(traffics_path, "rb") as f:
        traffics = pickle.load(f)
    z_fcvae = pd.read_pickle(z_fcvae_path)

    # Define color cycle for plotting
    color_cycle = plt.cm.inferno(np.linspace(0, 1, len(z_fcvae.label.unique())))
    np.random.seed(49)
    color_cycle = np.random.permutation(color_cycle)
    # Apply colors based on labels
    colors = [color_cycle[int(i) % len(color_cycle)] for i in z_fcvae.label]

    plt.style.use("ggplot")

    if save_separately:

        fig1 = plt.figure(figsize=(12, 12))
        ax0 = fig1.add_subplot(1, 1, 1, projection=ccrs.EuroPP())
        fig2 = plt.figure(figsize=(9, 12))
        ax1 = fig2.add_subplot(1, 1, 1)
    else:
        fig = plt.figure(figsize=(24, 12))
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
        ax0 = fig.add_subplot(gs[0], projection=ccrs.EuroPP())
        ax1 = fig.add_subplot(gs[1])

    ax0.coastlines()
    ax0.add_feature(cartopy.feature.BORDERS, linestyle=":", alpha=1.0)
    ax0.set_extent(geographic_extent)
    airports[ADEP_code].point.plot(
        ax0, color="red", label=f"Origin: {ADEP_code}", s=500, zorder=5
    )
    airports[ADES_code].point.plot(
        ax0, color="green", label=f"Destination: {ADES_code}", s=500, zorder=5
    )

    # Plot reconstructed trajectories
    ax0.set_title(
        f"Reconstructed Trajectories from {ADEP_code} to {ADES_code} using FCVAE"
    )
    for i, traf in enumerate(traffics):
        traf.plot(ax=ax0, alpha=0.2, color=color_cycle[i % len(color_cycle)])

    ax0.legend(loc="upper right")
    ax0.gridlines(draw_labels=True, color="gray", alpha=0.5, linestyle="--")

    ax1.scatter(z_fcvae.X1, z_fcvae.X2, s=4, c=colors)
    ax1.set_title(
        f"Latent space of trajectories from {ADEP_code} to {ADES_code} using FCVAE"
    )
    ax1.grid(False)
    ax1.set_xticks([])
    ax1.set_yticks([])
    plt.tight_layout()

    results_directory = os.path.dirname(z_fcvae_path)
    save_path = os.path.join(results_directory, "figures")
    os.makedirs(save_path, exist_ok=True)
    if save_separately:
        fig1.savefig(
            f"{save_path}/reconstructed_trajectories_{ADEP_code}_to_{ADES_code}_FCVAE.png",
            bbox_inches="tight",
        )
        print(
            f"Saved figure to {save_path}/reconstructed_trajectories_{ADEP_code}_to_{ADES_code}_FCVAE.png"
        )
        fig2.savefig(
            f"{save_path}/latent_space_projection_{ADEP_code}_to_{ADES_code}_FCVAE.png",
            bbox_inches="tight",
        )
        print(
            f"Saved figure to {save_path}/latent_space_projection_{ADEP_code}_to_{ADES_code}_FCVAE.png"
        )
        # fig1.savefig(f"./figures/reconstructed_trajectories_{ADEP_code}_to_{ADES_code}.png", bbox_inches="tight")
        # fig2.savefig(f"./figures/latent_space_projection_{ADEP_code}_to_{ADES_code}.png", bbox_inches="tight")
    else:
        plt.savefig(
            f"{save_path}/reconstructed_trajectories_and_latent_space_{ADEP_code}_to_{ADES_code}_FCVAE.png"
        )
        print(
            f"Saved figure to {save_path}/reconstructed_trajectories_and_latent_space_{ADEP_code}_to_{ADES_code}_FCVAE.png"
        )
        # plt.savefig(
        # f"./figures/reconstructed_trajectories_and_latent_space_{ADEP_code}_to_{ADES_code}.png", bbox_inches="tight"
        # )


def plot_sample_reconstruction_fcvae(
    training_data_path: str,
    reconstruction_fcvae_path: str,
) -> None:

    # Load reconstructed data from files
    reconstruction_fcvae = Traffic.from_file(reconstruction_fcvae_path)

    plt.style.use("ggplot")
    ADEP_code, ADES_code, geographic_extent = extract_geographic_info(
        training_data_path
    )

    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(projection=EuroPP()))

    # Configure the FCVAE plot
    ax.set_title(f" Sample reconstruction\n{ADEP_code} to {ADES_code} using FCVAE")
    reconstruction_fcvae[0].plot(ax=ax, lw=2, label="original")
    reconstruction_fcvae[1].plot(ax=ax, lw=2, label="reconstructed")
    ax.coastlines()
    ax.set_extent(geographic_extent)

    # Add legends and format plots
    ax.legend(loc="upper right")

    gridlines = ax.gridlines(draw_labels=True, color="gray", alpha=0.5, linestyle="--")
    gridlines.top_labels = False
    gridlines.right_labels = False

    plt.tight_layout()

    results_directory = os.path.dirname(reconstruction_fcvae_path)
    save_path = os.path.join(results_directory, "figures")
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(
        f"{save_path}/sample_reconstruction_fcvae_{ADEP_code}_to_{ADES_code}.png",
        bbox_inches="tight",
    )
    print(
        f"Saved figure to {save_path}/sample_reconstruction_fcvae_{ADEP_code}_to_{ADES_code}.png"
    )
    # plt.show()
def plot_sample_reconstruction_tcvae_conditioning(
    reconstructed_tcvae,
    training_data_path: str,
    reconstruction_tcvae_path: str,
) -> None:

    # Load reconstructed data from files

    plt.style.use("ggplot")
    ADEP_code, ADES_code, geographic_extent = extract_geographic_info(
        training_data_path
    )

    for i, k in enumerate(reconstructed_tcvae.keys()):
        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(projection=EuroPP()))

        # Configure the TCVAE plot
        ax.set_title(f"Sample reconstruction\n{ADEP_code} to {ADES_code} using TCVAE")
        reconstructed_tcvae['original'].plot(ax=ax, lw=2, label=f"original")
        if k == "original":
            continue

        reconstructed_tcvae[k].plot(ax=ax, lw=2, label=f"{k}")
        #reconstruction_tcvae[0].plot(ax=ax, lw=2, label="original")
        #reconstruction_tcvae[1].plot(ax=ax, lw=2, label="reconstructed")
        ax.coastlines()
        ax.set_extent(geographic_extent)
        ax.gridlines(draw_labels=True, color="gray", alpha=0.5, linestyle="--")

        # Add legends and format plots
        ax.legend(loc="upper right")

        plt.tight_layout()

        results_directory = os.path.dirname(reconstruction_tcvae_path)
        save_path = os.path.join(results_directory, "figures")
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(
            f"{save_path}/sample_reconstruction_tcvae_{ADEP_code}_to_{ADES_code}_{k}.png",
            bbox_inches="tight",
        )
        print(
            f"Saved figure to {save_path}/sample_reconstruction_tcvae_{ADEP_code}_to_{ADES_code}_{k}.png"
        )
    # plt.show()


### Suggestion
def plot_sample_reconstruction_tcvae(
    training_data_path: str,
    reconstruction_tcvae_path: str,
) -> None:

    # Load reconstructed data from files
    reconstruction_tcvae = Traffic.from_file(reconstruction_tcvae_path)

    plt.style.use("ggplot")
    ADEP_code, ADES_code, geographic_extent = extract_geographic_info(
        training_data_path
    )

    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(projection=EuroPP()))

    # Configure the TCVAE plot
    ax.set_title(f"Sample reconstruction\n{ADEP_code} to {ADES_code} using TCVAE")
    reconstruction_tcvae[0].plot(ax=ax, lw=2, label="original")
    reconstruction_tcvae[1].plot(ax=ax, lw=2, label="reconstructed")
    ax.coastlines()
    ax.add_feature(cartopy.feature.BORDERS, linestyle=":", alpha=1.0)
    ax.set_extent(geographic_extent)
    # ax.gridlines(draw_labels=True, color="gray", alpha=0.5, linestyle="--")

    gridlines = ax.gridlines(draw_labels=True, color="gray", alpha=0.5, linestyle="--")
    gridlines.top_labels = False
    gridlines.right_labels = False

    # Add legends and format plots
    ax.legend(loc="upper right")

    plt.tight_layout()

    results_directory = os.path.dirname(reconstruction_tcvae_path)
    save_path = os.path.join(results_directory, "figures")
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(
        f"{save_path}/sample_reconstruction_tcvae_{ADEP_code}_to_{ADES_code}.png",
        bbox_inches="tight",
    )
    print(
        f"Saved figure to {save_path}/sample_reconstruction_tcvae_{ADEP_code}_to_{ADES_code}.png"
    )
    # plt.show()


def plot_training_data(training_data_path: str) -> None:

    plt.style.use("ggplot")

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={"projection": EuroPP()})
    ADEP_code, ADES_code, geographic_extent = extract_geographic_info(
        training_data_path
    )
    training_data = Traffic.from_file(training_data_path)

    training_data.plot(ax, alpha=0.2, color="darkblue", linewidth=1)

    ax.coastlines()
    ax.add_feature(cartopy.feature.BORDERS, linestyle=":", alpha=1.0)
    ax.set_extent(geographic_extent)

    # Plot the origin and destination airports
    airports[ADEP_code].point.plot(
        ax, color="red", label=f"Origin: {ADEP_code}", s=500, zorder=5
    )
    airports[ADES_code].point.plot(
        ax, color="green", label=f"Destination: {ADES_code}", s=500, zorder=5
    )

    plt.title(f"Training data of flight trajectories from {ADEP_code} to {ADES_code}")
    plt.legend(loc="upper right")

    # Add gridlines
    gridlines = ax.gridlines(draw_labels=True, color="gray", alpha=0.5, linestyle="--")
    gridlines.top_labels = False
    gridlines.right_labels = False

    # tight layout
    plt.tight_layout()

    # save the figure to the figures directory in the data directory
    data_directory = os.path.dirname(training_data_path)
    save_path = os.path.join(data_directory, "figures")
    os.makedirs(save_path, exist_ok=True)
    save_name = os.path.basename(training_data_path).replace(".pkl", ".png")
    plt.savefig(f"{save_path}/{save_name}", bbox_inches="tight")
    print(f"Saved figure to {save_path}/{save_name}")

def plot_training_data_with_altitude(training_data_path: str) -> None:
    # Set up the map
    ADEP_code, ADES_code, geographic_extent = extract_geographic_info(
        training_data_path
    )
    training_data = Traffic.from_file(training_data_path)
    training_data = (
        training_data.sample(1000) if len(training_data) > 1000 else training_data
    )
    df = training_data.data
    fig, ax = plt.subplots(figsize=(13, 12))
    m = Basemap(
        projection="merc",
        llcrnrlat=geographic_extent[2],
        urcrnrlat=geographic_extent[3],
        llcrnrlon=geographic_extent[0],
        urcrnrlon=geographic_extent[1],
        lat_ts=20,
        resolution="i",
    )

    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents(color="lightgray", lake_color="aqua")
    m.drawmapboundary(fill_color="aqua")

    # Convert latitude and longitude to x and y coordinates
    x, y = m(df["longitude"].values, df["latitude"].values)

    # Connect points with a line
    plt.plot(x, y, color="black", alpha=0.2, zorder=1)

    # Plot the points with altitude as hue and size
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
    )

    # Add color bar for altitude
    norm = plt.Normalize(vmin=df["altitude"].min(), vmax=df["altitude"].max())
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array(
        []
    )  # This is necessary because of a Matplotlib bug when using scatter with norm.
    cbar = plt.colorbar(sm, ax=ax, aspect=30)
    cbar.set_label("Altitude (feet)")
    # set legend upper right
    plt.legend(loc="upper right")

    # Add title and labels
    plt.title(f"Flight Path from {ADEP_code} to {ADES_code}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    # Adjust layout and display plot
    plt.tight_layout()

    data_directory = os.path.dirname(training_data_path)
    save_path = os.path.join(data_directory, "figures")
    os.makedirs(save_path, exist_ok=True)
    save_name = os.path.basename(training_data_path).replace(
        ".pkl", "_with_altitude.png"
    )
    plt.savefig(f"{save_path}/{save_name}", bbox_inches="tight")
    print(f"Saved figure to {save_path}/{save_name}")


def plot_traffic(traffic: Traffic) -> Figure:
    with plt.style.context("traffic"):
        fig, ax = plt.subplots(1, figsize=(5, 5), subplot_kw=dict(projection=EuroPP()))
        traffic[1].plot(ax, c="orange", label="reconstructed")
        traffic[0].plot(ax, c="purple", label="original")
        ax.legend()

    return fig


def plot_clusters(traffic: Traffic, cluster_label: str = "cluster") -> Figure:
    assert (
        cluster_label in traffic.data.columns
    ), f"Underlying dataframe should have a {cluster_label} column"
    clusters = sorted(list(traffic.data[cluster_label].value_counts().keys()))
    n_clusters = len(clusters)
    if n_clusters > 3:
        nb_cols = 3
        nb_lines = n_clusters // nb_cols + ((n_clusters % nb_cols) > 0)

        with plt.style.context("traffic"):
            fig, axs = plt.subplots(
                nb_lines,
                nb_cols,
                figsize=(10, 15),
                subplot_kw=dict(projection=EuroPP()),
            )

            for n, cluster in enumerate(clusters):
                ax = axs[n // nb_cols][n % nb_cols]
                ax.set_title(f"cluster {cluster}")
                t_cluster = traffic.query(f"{cluster_label} == {cluster}")
                t_cluster.plot(ax, alpha=0.5)
                t_cluster.centroid(nb_samples=None, projection=EuroPP()).plot(
                    ax, color="red", alpha=1
                )
    else:
        with plt.style.context("traffic"):
            fig, axs = plt.subplots(
                n_clusters,
                figsize=(10, 15),
                subplot_kw=dict(projection=EuroPP()),
            )

            for n, cluster in enumerate(clusters):
                ax = axs[n]
                ax.set_title(f"cluster {cluster}")
                t_cluster = traffic.query(f"{cluster_label} == {cluster}")
                t_cluster.plot(ax, alpha=0.5)
                t_cluster.centroid(nb_samples=None, projection=EuroPP()).plot(
                    ax, color="red", alpha=1
                )
    return fig


def unpad_sequence(padded: torch.Tensor, lengths: torch.Tensor) -> List:
    return [padded[i][: lengths[i]] for i in range(len(padded))]


def cumul_dist_plot(
    df: pd.DataFrame, scales: Dict[Any, Tuple[float, float]], domain: List[str]
):
    alt.data_transformers.disable_max_rows()

    base = alt.Chart(df)
    legend_config = dict(
        labelFontSize=12,
        titleFontSize=13,
        labelFont="Ubuntu",
        titleFont="Ubuntu",
        orient="none",
        legendY=430,
    )

    chart = (
        alt.vconcat(
            *[
                base.transform_window(
                    cumulative_count="count()",
                    sort=[{"field": col}],
                    groupby=["generation", "reconstruction"],
                )
                .transform_joinaggregate(
                    total="count()", groupby=["generation", "reconstruction"]
                )
                .transform_calculate(
                    normalized=alt.datum.cumulative_count / alt.datum.total
                )
                .mark_line(clip=True)
                .encode(
                    alt.X(
                        col,
                        title="Distance",
                        scale=alt.Scale(domain=scales[col]),
                    ),
                    alt.Y("normalized:Q", title="Cumulative ratio"),
                    alt.Color(
                        "generation",
                        legend=alt.Legend(title="Generation method", **legend_config),
                        scale=alt.Scale(domain=domain),
                    ),
                    alt.StrokeDash(
                        "reconstruction",
                        legend=alt.Legend(
                            title="Reconstruction method",
                            legendX=200,
                            **legend_config,
                        ),
                        scale=alt.Scale(
                            domain=["Navigational points", "Douglas-Peucker"]
                        ),
                    ),
                )
                .properties(title=col.upper(), height=150)
                for col in scales
            ]
        )
        .configure_view(stroke=None)
        .configure_title(font="Fira Sans", fontSize=16, anchor="start")
        .configure_axis(
            labelFont="Fira Sans",
            labelFontSize=14,
            titleFont="Ubuntu",
            titleFontSize=12,
        )
    )

    return chart
