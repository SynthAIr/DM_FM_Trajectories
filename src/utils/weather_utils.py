import xarray as xr
import pickle
import os
from tqdm import tqdm
import torch


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
