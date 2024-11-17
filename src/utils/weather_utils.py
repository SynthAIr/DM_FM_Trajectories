import xarray as xr
import pickle
import os
from tqdm import tqdm
import torch
import numpy as np


def load_weather_data(file_paths, traffic, preprocess, save_path):
    # Load dataset using xarray's open_mfdataset with the preprocess function
    ds = xr.open_mfdataset(file_paths, combine='by_coords', preprocess=preprocess, chunks={'time': 100})
    print("Data loaded")

    grid_conditions = []

    name = f"flight_processed_{len(traffic)}.pkl"
    if not os.path.isfile(save_path + name):
        print("ERA5 file not found - creating new")

        for flight in tqdm(traffic):
            # Extracting the average time for the flight and rounding it to the nearest hour
            t = flight.mean("timestamp").round('h')
            formatted_timestamp = t.strftime('%Y-%m-%d %H:00:00')
            
            # Select sub-dataset for the timestamp
            sub = ds.sel(time=formatted_timestamp)
            
            # Convert the selected subset to a PyTorch FloatTensor, filling NaN values if necessary
            data_array = sub.to_array().fillna(0).values  # Filling NaNs with 0 or another appropriate value
            grid_conditions.append(torch.FloatTensor(data_array))
        
        # Save the processed grid_conditions as a pickle file
        with open(save_path + name, "wb") as fp:
            pickle.dump(grid_conditions, fp)
    else:
        # Load existing pickle file if available
        print("File found - Loading from pickle")
        with open(save_path + name, 'rb') as f:
            grid_conditions = pickle.load(f)

    return grid_conditions

def pad_or_crop_grid(grid, target_shape):
    """Pads or crops the grid to match the target shape (num_levels, grid_size, grid_size)."""
    target_channels, target_rows, target_cols = target_shape
    current_channels, current_rows, current_cols = grid.shape

    # Pad or crop channels
    if current_channels < target_channels:
        pad_channels = target_channels - current_channels
        grid = np.pad(grid, ((0, pad_channels), (0, 0), (0, 0)), mode='constant')
    elif current_channels > target_channels:
        grid = grid[:target_channels]

    # Pad or crop rows
    if current_rows < target_rows:
        pad_rows = target_rows - current_rows
        grid = np.pad(grid, ((0, 0), (0, pad_rows), (0, 0)), mode='constant')
    elif current_rows > target_rows:
        grid = grid[:, :target_rows]

    # Pad or crop columns
    if current_cols < target_cols:
        pad_cols = target_cols - current_cols
        grid = np.pad(grid, ((0, 0), (0, 0), (0, pad_cols)), mode='constant')
    elif current_cols > target_cols:
        grid = grid[:, :, :target_cols]

    return grid

def load_weather_data_function(file_paths, traffic, preprocess, save_path, grid_size=5, num_levels=3):
    """
    Load and process weather data with adjustable grid size and levels.

    Parameters:
    - file_paths: list of file paths to the dataset
    - traffic: list of flight trajectories
    - preprocess: preprocessing function for the dataset
    - save_path: path to save processed data
    - grid_size: size of the grid (default 5x5)
    - num_levels: number of vertical levels to extract (default 3)
    """
    # Load dataset using xarray's open_mfdataset with the preprocess function
    ds = xr.open_mfdataset(file_paths, combine='by_coords', preprocess=preprocess, chunks={'time': 100})
    print("Data loaded")

    grid_conditions = []

    name = f"flight_processed_{len(traffic)}_{grid_size}x{grid_size}_levels_{num_levels}.pkl"
    if not os.path.isfile(save_path + name):
        print("ERA5 file not found - creating new")

        for flight in tqdm(traffic):
            # Extracting the average time for the flight and rounding it to the nearest hour
            t = flight.mean("timestamp").round('h')
            formatted_timestamp = t.strftime('%Y-%m-%d %H:00:00')
            
            # Select sub-dataset for the timestamp
            sub = ds.sel(time=formatted_timestamp)
            
            flight_grids = []
            for i in range(len(flight)):
                point = flight.data.loc[i]
                lon, lat, alt = point['longitude'], point['latitude'], point['altitude']
                
                # Extract a grid_size x grid_size grid around the point for num_levels
                half_grid = grid_size // 2
                grid = sub.sel(
                    longitude=slice(lon - half_grid, lon + half_grid), 
                    latitude=slice(lat - half_grid, lat + half_grid), 
                    level=slice(alt - (num_levels // 2), alt + (num_levels // 2))
                ).to_array().fillna(0).values  # Filling NaNs with 0

                # Ensure the grid shape is (num_levels, grid_size, grid_size)
                if grid.shape[1:3] != (grid_size, grid_size) or grid.shape[0] != num_levels:
                    # Handle cases where grid extraction may be smaller due to boundaries
                    grid = pad_or_crop_grid(grid, target_shape=(num_levels, grid_size, grid_size))

                flight_grids.append(torch.FloatTensor(grid))
            
            # Collect grids for all trajectory points in this flight
            grid_conditions.append(torch.stack(flight_grids))  # Shape: (200, grid_size, grid_size, num_levels)
        
        # Save the processed grid_conditions as a pickle file
        with open(save_path + name, "wb") as fp:
            pickle.dump(grid_conditions, fp)
    else:
        # Load existing pickle file if available
        print("File found - Loading from pickle")
        with open(save_path + name, 'rb') as f:
            grid_conditions = pickle.load(f)

    return grid_conditions

