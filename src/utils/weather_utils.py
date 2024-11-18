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

def retrieve_closest_pressure(pressure_hPa, pressure_levels = np.array([ 100,  150,  200,  250,  300,  400,  500,  600,  700,  850,  925, 1000] )):
    """ Retrieve the closest pressure level in the ERA5 dataset to the given pressure
    Args:
        pressure_hPa: float
    """
    return min(pressure_levels, key=lambda x:abs(x-pressure_hPa))

def hPa_to_m(pressure_hPa):
    """ Convert pressure in hPa to meters
    Args:
        pressure_hPa: float
    """
    return 145366.45 * (1 - (pressure_hPa / 1013.25)**0.190284)

def m_to_hPa(altitude_m):
    """ Convert altitude in meters to hPa
    Args:
        altitude_m: float
    """
    return 1013.25 * (1 - 2.2577e-5 * altitude_m )**5.25588

def find_nearest_pressure_levels(altitude, num_levels, pressure_levels=np.array([100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000])):
    """ 
    Map an altitude to the nearest available pressure level and return surrounding levels.
    
    Args:
        altitude: float, altitude in meters
        num_levels: int, number of levels to retrieve (total number, so num_levels // 2 above and below)
        pressure_levels: numpy array of available pressure levels in hPa
    
    Returns:
        list: Sorted list of pressure levels including the nearest and surrounding levels.
    """
    # Convert altitude to pressure (if needed), here using a placeholder conversion for example
    pressure_hPa = m_to_hPa(altitude)  # Convert altitude (meters) to pressure in hPa
    
    # Find the index of the nearest pressure level
    closest_index = np.abs(pressure_levels - pressure_hPa).argmin()
    
    # Calculate the range of indices to select surrounding levels
    half_num_levels = num_levels // 2
    start_index = max(0, closest_index - half_num_levels)
    end_index = min(len(pressure_levels), closest_index + half_num_levels + 1)  # +1 to include the end index
    
    # Select surrounding pressure levels
    selected_levels = pressure_levels[start_index:end_index]
    
    return selected_levels

def load_weather_data_function(file_paths, traffic, preprocess, save_path, grid_size=5, num_levels=3, 
                               pressure_levels = np.array([ 100,  150,  200,  250,  300,  400,  500,  600,  700,  850,  925, 1000])):
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

            for i in tqdm(range(len(flight))):
                point = flight.data.loc[i]
                lon, lat, alt = point['longitude'], point['latitude'], point['altitude']
                
                nearest_levels = find_nearest_pressure_levels(alt, num_levels)
                half_grid = grid_size // 2
                grid = sub.sel(
                    longitude=slice(lon - half_grid, lon + half_grid), 
                    latitude=slice(lat - half_grid, lat + half_grid), 
                    level=nearest_levels  # This will select the appropriate levels directly
                ).to_array().fillna(0).values  # Filling NaNs with 0

                print("GRID SHAPE:", grid.shape)
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

