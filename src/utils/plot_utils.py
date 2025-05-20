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

def plot_traffic(traffic: Traffic) -> Figure:
    with plt.style.context("traffic"):
        fig, ax = plt.subplots(1, figsize=(5, 5), subplot_kw=dict(projection=EuroPP()))
        traffic[1].plot(ax, c="orange", label="reconstructed")
        traffic[0].plot(ax, c="purple", label="original")
        ax.legend()

    return fig
