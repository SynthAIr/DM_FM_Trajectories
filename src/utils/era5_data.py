import xarray as xr
import pandas as pd
import argparse
import os

class ERA5Dataset:
    """
    A class to handle the ERA5 dataset.
    """
    def __init__(self, data_path):
        self.data_path = data_path
        self.dataset = xr.open_dataset(data_path)

    def get_data(self, start_time, end_time, lat_min, lat_max, lon_min, lon_max):
        """
        Get a subset of the dataset based on time and spatial region.
        Parameters
        ----------
        start_time
        end_time
        lat_min
        lat_max
        lon_min
        lon_max

        Returns
        -------

        """
        latitude_range = slice(lat_max, lat_min)  # lat_max first since latitude decreases from north to south
        longitude_range = slice(lon_min, lon_max)

        subset = self.dataset.sel(
            time=slice(start_time, end_time),
            latitude=latitude_range,
            longitude=longitude_range
        )

        return subset

def download_data(start_time, end_time, lat_min, lat_max, lon_min, lon_max, save_path):
    """
    Download ERA5 data for a specified time period and spatial region, and save it to NetCDF files.
    Uses Google Cloud Storage (GCS) to access the dataset.
    Parameters
    ----------
    start_time
    end_time
    lat_min
    lat_max
    lon_min
    lon_max
    save_path

    Returns
    -------

    """
    era5 = xr.open_zarr(
        "gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2",
        chunks={'time': 48},
        consolidated=True,
    )

    variables_to_save = [
        '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature',
        'temperature', 'vertical_velocity', 'v_component_of_wind', 'u_component_of_wind',
        'total_precipitation', 'total_cloud_cover'
    ]

    latitude_range = slice(lat_max, lat_min)  # lat_max first since latitude decreases from north to south
    longitude_range = slice(lon_min, lon_max)

    dates = pd.date_range(start=start_time, end=end_time, freq='MS')

    for i in range(len(dates) - 1):
        current_start = dates[i]
        current_end = dates[i + 1] - pd.Timedelta(seconds=1)  # End of the month

        subset = era5.sel(
            time=slice(current_start, current_end),
            latitude=latitude_range,
            longitude=longitude_range
        )[variables_to_save]

        month_str = current_start.strftime('%Y-%m')
        filename = f"era5_subset_{month_str}.nc"

        subset.to_netcdf(os.path.join(save_path, filename))
        print(f"Saved {filename}")

def concat_data(save_path):
    """
    Concatenate all NetCDF files in the specified directory into a single xarray dataset.
    Parameters
    ----------
    save_path

    Returns
    -------

    """
    files = os.path.join(save_path, "era5_subset_*.nc")

    combined_dataset = xr.open_mfdataset(files, combine='by_coords')

    print(combined_dataset)
    return combined_dataset

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Download or concat ERA5 data with specified parameters.")
    
    parser.add_argument('--mode', choices=['download', 'concat'], required=True,
                        help="Specify whether to 'download' data or 'concat' files.")
    parser.add_argument('--start_time', type=str, default='2020-01-01',
                        help="Start time for downloading data (format YYYY-MM-DD).")
    parser.add_argument('--end_time', type=str, default='2020-12-31',
                        help="End time for downloading data (format YYYY-MM-DD).")
    parser.add_argument('--lat_min', type=float, default=38.0, help="Minimum latitude for the region.")
    parser.add_argument('--lat_max', type=float, default=64.0, help="Maximum latitude for the region.")
    parser.add_argument('--lon_min', type=float, default=0.0, help="Minimum longitude for the region.")
    parser.add_argument('--lon_max', type=float, default=20.0, help="Maximum longitude for the region.")
    parser.add_argument('--save_path', type=str, default='/mnt/data/synthair/synthair_diffusion/data/era5/',
                        help="Path to save the NetCDF files or concat files.")

    args = parser.parse_args()

    # Execute based on the mode
    if args.mode == 'download':
        # Download data for the specified period and region
        download_data(args.start_time, args.end_time, args.lat_min, args.lat_max, args.lon_min, args.lon_max, args.save_path)
    elif args.mode == 'concat':
        # Concatenate all NetCDF files in the save path
        concat_data(args.save_path)

#python era5_script.py --mode download --start_time 2020-01-01 --end_time 2020-12-31 --lat_min 38 --lat_max 64 --lon_min 0 --lon_max 20 --save_path /your/save/path/
