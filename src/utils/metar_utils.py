import os
import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import argparse
import re
import pandas as pd
import metpy.io as metpy
import torch
import pickle
from tqdm import tqdm

def extract_metar_data(lines):
    """
    Extract METAR data from lines of text.
    Parameters
    ----------
    lines

    Returns
    -------

    """
    df_all = None
    first = True
    for line in lines:
        match = re.match(r"(\d{4})(\d{2})\d{6}\s(METAR.*)", line)
        if match:
            year, month, metar = match.groups()
            df =  metpy.parse_metar_to_dataframe(metar, year=int(year), month=int(month))

        df_all = pd.concat([df_all, df], ignore_index=True) if not first else df
        first = False
    return df_all

def read_metar_file(filename):
    """
    Read METAR data from a file and extract the relevant information.
    Parameters
    ----------
    filename

    Returns
    -------

    """
    with open(filename, 'r') as file:
        lines = file.readlines()
    return extract_metar_data(lines)

def preprocess_metar(df):
    """
    Preprocess METAR data:
    - Fill NaNs in categorical columns with "unknown".
    - Fill NaNs in numerical columns with 0.
    - Encode categorical columns as integers for embedding.
    
    Args:
        df (pd.DataFrame): Raw METAR DataFrame
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame
    """
    categorical_cols = [
        "current_wx1", "current_wx2", "current_wx3",
        "low_cloud_type", "medium_cloud_type", "high_cloud_type",
        "highest_cloud_type", "remarks"
    ]
    
    numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()
    
    categorical_cols = [col for col in categorical_cols if col in df.columns]
    numerical_cols = [col for col in numerical_cols if col in df.columns]

    df[categorical_cols] = df[categorical_cols].fillna("unknown")
    df[numerical_cols] = df[numerical_cols].fillna(0)

    for col in categorical_cols:
        df[col] = df[col].astype("category").cat.codes

    return df

def load_metar_data(file_path, traffic, save_path):
    """
    Load METAR data from a file, preprocess it, and save the processed data as a pickle file.
    Parameters
    ----------
    file_path
    traffic
    save_path

    Returns
    -------

    """

    important_features = [
        'wind_speed', 'wind_gust', 'wind_direction', 'eastward_wind', 'northward_wind',
        'visibility', 'cloud_coverage', 'low_cloud_level',
        'air_temperature', 'dew_point_temperature', 'air_pressure_at_sea_level',
        'altimeter', 'elevation', "low_cloud_type",
    ]
    name = f"flight_processed_{len(traffic)}_METAR_ADES.pkl"
    if not os.path.isfile(save_path + name):
        print("METAR file not found - creating new")
        df = read_metar_file(file_path)
        df = df.sort_values(by=["date_time"]).reset_index()
        df = df.set_index('date_time')
        df.index = df.index.tz_localize("UTC")
        df = preprocess_metar(df)[important_features]
        
        print("Data loaded")
        closest_rows = []
        for f in tqdm(traffic):
            time = f.max("timestamp").round("min")
            closest_index = df.index.get_indexer([time], method="nearest")[0]
            closest_rows.append(df.iloc[closest_index])

        t_df = pd.DataFrame(closest_rows)
        grid_conditions = torch.FloatTensor(t_df.values)
        
        with open(save_path + name, "wb") as fp:
            pickle.dump(grid_conditions, fp)
    else:
        print("File found - Loading from pickle")
        with open(save_path + name, 'rb') as f:
            grid_conditions = pickle.load(f)

    return grid_conditions














def fetch_metar_data(base_url: str, icao24: str, start_date: datetime, end_date: datetime, save_folder: str) -> None:
    """
    Fetch METAR data from the API, remove HTML tags, and save the clean METAR data into files with dates in a folder.
    The data is split by month into separate text files.

    :param base_url: The base URL for the API query.
    :param start_date: The starting datetime for the query.
    :param end_date: The ending datetime for the query.
    :param save_folder: The folder where the files will be saved.
    """
    # Ensure the folder exists
    os.makedirs(save_folder, exist_ok=True)

    # Loop through the date range month by month
    current_start = start_date
    while current_start <= end_date:
        # Calculate the last date of the current month
        next_month_start = (current_start.replace(day=28) + timedelta(days=4)).replace(day=1)
        current_end = min(next_month_start - timedelta(days=1), end_date)

        # Generate the API URL with parameters
        query_params = {
            'lang': 'en',
            'lugar': icao24,
            'tipo': 'ALL',
            'ord': 'REV',
            'nil': 'NO',
            'fmt': 'txt',
            'ano': current_start.year,
            'mes': current_start.month,
            'day': current_start.day,
            'hora': current_start.hour,
            'anof': current_end.year,
            'mesf': current_end.month,
            'dayf': current_end.day,
            'horaf': current_end.hour,
            'minf': current_end.minute,
            'send': 'send'
        }

        # Fetch data
        response = requests.get(base_url, params=query_params)
        if response.status_code != 200:
            print(f"Failed to fetch METAR data for {current_start.strftime('%Y-%m-%d')}: {response.status_code}")
            current_start = next_month_start
            continue

        # Use BeautifulSoup to clean HTML tags
        soup = BeautifulSoup(response.text, 'html.parser')
        metar_data = soup.get_text()  # Extracts the text content from the HTML

        # Remove leading/trailing whitespace
        metar_data = metar_data.strip()

        # Save to file with date
        file_name = f"METAR_{icao24}_{current_start.strftime('%Y%m')}.txt"
        file_path = os.path.join(save_folder, file_name)
        
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(metar_data)

        print(f"METAR data saved to {file_path}")

        # Move to the next month
        current_start = next_month_start


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Download METAR data for an airport over a specified date range.")
    parser.add_argument('start', type=str, help='Start date (YYYY-MM-DD HH:MM)')
    parser.add_argument('end', type=str, help='End date (YYYY-MM-DD HH:MM)')
    parser.add_argument('icao24', type=str, help='ICAO code for the airport (e.g., LIRF for Roma Fiumicino)')
    parser.add_argument('--output_dir', type=str, default="./metar_data", help='Output directory for METAR files')
    args = parser.parse_args()

    # Convert the input date strings to datetime objects (now with time support)
    start_date = datetime.strptime(args.start, '%Y-%m-%d %H:%M')
    end_date = datetime.strptime(args.end, '%Y-%m-%d %H:%M')


    # Call the METAR fetch function with provided arguments
    fetch_metar_data(
        base_url="https://www.ogimet.com/display_metars2.php",
        icao24=args.icao24,
        start_date=start_date,
        end_date=end_date,
        save_folder=args.output_dir
    )

    #python metar_utils.py "2018-01-01 08:00" "2018-01-03 10:00" LIRF --output_dir ./my_metar_data
