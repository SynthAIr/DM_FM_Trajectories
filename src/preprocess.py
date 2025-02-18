import glob
import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple
from cartopy.crs import EuroPP
from scipy.stats import zscore
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from traffic.core import Flight, Traffic
from traffic.data import airports
from utils import (calculate_consecutive_distances,
                           calculate_final_distance,
                           calculate_initial_distance, plot_training_data,
                           plot_training_data_with_altitude)
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import scipy.stats as stats

print(os.getcwd())


import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Create a StreamHandler that sends output to the standard output
info_handler = logging.StreamHandler()
info_handler.setLevel(level=logging.DEBUG)
logger.addHandler(info_handler)
logger.info("Imports completed")
logger.debug("Debugging")

def enforce_increasing(points):
    """
    Adjust points to ensure an increasing sequence from the first to the last,
    with a constraint on maximum growth between consecutive points.
    
    Parameters:
    points (array-like): A sequence of n altitude points.
    max_growth (float): Maximum allowable difference between consecutive points.
    
    Returns:
    np.ndarray: Adjusted sequence with no downward trends and controlled growth.
    """
    points = np.array(points)  # Ensure input is a NumPy array for easy manipulation
    
    # Initialize an array to hold adjusted points
    adjusted_points = points.copy()
    
    # Traverse the points from start to end
    for i in range(1, len(points)):
        # Enforce non-decreasing trend
        if adjusted_points[i] < adjusted_points[i - 1]:
            adjusted_points[i] = adjusted_points[i - 1]
    
    return adjusted_points

def enforce_non_increasing(points):
    """
    Adjust points to ensure a non-increasing sequence from the first to the last.
    
    Parameters:
    points (array-like): A sequence of n altitude points.
    
    Returns:
    np.ndarray: Adjusted sequence with no upward trends from the first to the last point.
    """
    points = np.array(points)  # Ensure input is a NumPy array for easy manipulation
    
    # Initialize an array to hold adjusted points
    adjusted_points = points.copy()
    
    # Traverse the points from start to end
    for i in range(1, len(points)):
        # Ensure the current point is not greater than the previous point
        if adjusted_points[i] > adjusted_points[i - 1]:
            adjusted_points[i] = adjusted_points[i - 1]
    
    return adjusted_points

def enforce_increasing_with_limit(points, max_growth=1000):
    """
    Adjust points to ensure an increasing sequence from the first to the last,
    with a constraint on maximum growth between consecutive points.
    
    Parameters:
    points (array-like): A sequence of n altitude points.
    max_growth (float): Maximum allowable difference between consecutive points.
    
    Returns:
    np.ndarray: Adjusted sequence with no downward trends and controlled growth.
    """
    points = np.array(points)  # Ensure input is a NumPy array for easy manipulation
    
    # Initialize an array to hold adjusted points
    adjusted_points = points.copy()
    
    # Traverse the points from start to end
    for i in range(1, len(points)):
        # Enforce non-decreasing trend
        if adjusted_points[i] < adjusted_points[i - 1]:
            adjusted_points[i] = adjusted_points[i - 1]
        # Enforce maximum growth limit
        elif adjusted_points[i] > adjusted_points[i - 1] + max_growth:
            adjusted_points[i] = adjusted_points[i - 1]
    
    return adjusted_points
    
def clean_and_smooth_flight_with_tight_threshold(flight, target_length, column):
    """
    Removes outliers and smooths the altitude for a single flight with a tighter threshold.
    """
    df = flight.data.copy()

    if df.loc[0, 'altitude'] > 500:
        df.loc[0, 'altitude'] = 0

    #if df.loc[1, 'altitude'] > 2000:
        #df.loc[1, 'altitude'] = 600

    #if df.loc[2, 'altitude'] > 4000:
        #df.loc[2, 'altitude'] = 1200

    df.loc[int(target_length*0.935):, column] = enforce_non_increasing(df.loc[int(target_length*0.935):, column])
    df.loc[:int(target_length*0.30), column] = enforce_increasing(df.loc[:int(target_length*0.30),column])
    
    if 'altitude' not in df.columns:
        return flight  # Skip if no altitude data available

    # First Pass: Initial cleaning using a rolling median and standard deviation
    rolling_median = df[column].rolling(window=5, center=True).median()
    rolling_std = df[column].rolling(window=5, center=True).std()
    threshold = 2 * rolling_std
    df['is_outlier'] = np.abs(df[column] - rolling_median) > threshold
    df['altitude_cleaned'] = np.where(df['is_outlier'], rolling_median, df[column])

    # Second Pass: Tighter threshold for persistent outliers
    rolling_median_2 = df['altitude_cleaned'].rolling(window=5, center=True).median()
    rolling_std_2 = df['altitude_cleaned'].rolling(window=5, center=True).std()
    tighter_threshold = 1.5 * rolling_std_2
    df['is_outlier_2'] = np.abs(df['altitude_cleaned'] - rolling_median_2) > tighter_threshold
    df['altitude_cleaned'] = np.where(df['is_outlier_2'], rolling_median_2, df['altitude_cleaned'])

    # Smooth the final cleaned altitude data using a Savitzky-Golay filter
    df['altitude_smoothed'] = savgol_filter(df['altitude_cleaned'].bfill().ffill(),
                                            window_length=11, polyorder=2)

    # Replace the original flight's altitude data with cleaned and smoothed data
    flight.data['altitude_cleaned'] = df['altitude_cleaned']
    flight.data['altitude_smoothed'] = df['altitude_smoothed']
    flight.data[column] = df['altitude_smoothed']
    
    return flight

def clean_trajectory_data(df, column, n, threshold=3):
    """
    Clean trajectory data by:
    1. Identifying and replacing outliers with NaN.
    2. Capping the last n values to a maximum height if needed.
    3. Interpolating missing values at the end.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing trajectory data.
    column (str): Column name to clean.
    n (int): Number of points at the end to be capped.
    threshold (float): Number of standard deviations for outlier detection.
    
    Returns:
    pd.Series: Cleaned data series.
    """
    # Calculate z-scores to identify outliers
    z_scores = np.abs(stats.zscore(df[column]))
    
    # Create a copy of the data to modify
    cleaned_data = df[column].copy()
    
    # Replace outliers with NaN
    cleaned_data[z_scores > threshold] = np.nan
    
    # Perform interpolation across the entire trajectory
    cleaned_data = cleaned_data.interpolate(method='linear')

    return cleaned_data

def add_time_based_features(df: pd.DataFrame, time_col: str = 'Time Over') -> pd.DataFrame:
    """
    Add time-based features to the dataframe
    """
    # Add time-based features
    # assert coulmn timestamp exists

    # assert 'Time Over' in df.columns, "timestamp column must exist in the dataframe"
    # timestamps = df['Time Over']
    assert time_col in df.columns, "timestamp column must exist in the dataframe"
    timestamps = df[time_col]
    if not type(timestamps.iloc[0]) == pd.Timestamp: 
        timestamps = pd.to_datetime(timestamps, dayfirst=True)
        
    months = timestamps.dt.month
    hours = timestamps.dt.hour + timestamps.dt.minute / 60.0
    day_of_week = timestamps.dt.dayofweek

    # Calculate sine and cosine for months (12 months in a year)
    #month_sin = np.sin(2 * np.pi * months / 12)
    #month_cos = np.cos(2 * np.pi * months / 12)

    # Calculate sine and cosine for hours (24 hours in a day)
    #hour_sin = np.sin(2 * np.pi * hours / 24)
    #hour_cos = np.cos(2 * np.pi * hours / 24)

    #day_of_week_sin = np.sin(2 * np.pi * day_of_week / 7)
    #day_of_week_cos = np.cos(2 * np.pi * day_of_week / 7)

    df['hour'] =  hours
    df['month'] = months
    df['day_of_week'] = day_of_week
    #df['hour_cos'] = hour_cos
    #df['month_sin'] = month_sin
    #df['month_cos'] = month_cos
    #df['day_of_week_sin'] = day_of_week_sin
    #df['day_of_week_cos'] = day_of_week_cos
    if 'track' in df.columns:
        print("Adding Track sine and cosine")
        df['track_cos'] = np.cos(2 * np.pi * df['track'] / 360)
        df['track_sin'] = np.sin(2 * np.pi * df['track'] / 360)
    return df


def add_embedding_encoding_index(df: pd.DataFrame, column_name: str, new_column_name: str) -> pd.DataFrame:
    """
    Add an index encoding for a categorical column
    Sorts values to ensure consistent encoding
    """
    unique_values = df[column_name].unique()
    unique_values.sort()
    encoding = {value: i for i, value in enumerate(unique_values)}
    df[f"{new_column_name}"] = df[column_name].map(encoding)
    return df

def month_flight_filter(df:pd.DataFrame, month: int, timestamp_col:str='Time Over') -> pd.DataFrame:
    """
    Filter flights that both took off and landed within the specified month.
    Overlapping flights are disregarded for simplicity.
    The timestamp_col must be of type timestamp

    """
    assert type(df[timestamp_col].iloc[0]) == pd.Timestamp, "timestamp_col must be of type timestamp"
    assert month in range(1, 13), "Month must be between 1 and 12"
    ext_df = df[df[timestamp_col].dt.month != month] # flights that did not start or finish in the specified month

    ids = np.unique(ext_df['ECTRL ID']) # unique flight ids of outside boundaries flights
    in_df = df[~df['ECTRL ID'].isin(ids)] # filtering out according to ids

    return in_df


def load_flights_points(file_flights: str, flight_points_file_path: str, ADEP_code: str, ADES_code: str) -> pd.DataFrame:
    # Load flights data from a CSV file
    flights_df = pd.read_csv(file_flights)

    # Filter flights originating from ADEP_code and destined for ADES_code
    flights = flights_df[
        (flights_df["ADEP"] == ADEP_code) & (flights_df["ADES"] == ADES_code)
    ]
    print(f"Number of flights from {ADEP_code} to {ADES_code}: {len(flights)}")

    # Load flight points data from another CSV file
    flight_points = pd.read_csv(flight_points_file_path)

    # Merge data to only include flight points for the filtered flights
    flights_points = flight_points[flight_points["ECTRL ID"].isin(flights["ECTRL ID"])]

    flights_points = flights_points.merge(
        flights[["ECTRL ID", "ADEP", "ADES", "AC Type", 
                 "AC Operator", "FILED OFF BLOCK TIME",
                 "ACTUAL OFF BLOCK TIME", "ACTUAL ARRIVAL TIME",
                 "Actual Distance Flown (nm)"]], on="ECTRL ID"
    )
    # Add relevant flight information to the flight points
    #flights_points = flights_points.merge(
    #    flights[["ECTRL ID", "ADEP", "ADES", "AC Type"]], on="ECTRL ID"
    #)

    # Calculate the average sequence length of the flights
    sequence_lengths = flights_points.groupby("ECTRL ID").size()
    avg_sequence_length = sequence_lengths.mean()
    print(f"Average sequence length: {avg_sequence_length}")

    # # add weather data
    # flight_file_folder = os.path.dirname(file_flights)
    # flights_points = add_weather_data(flights_points, folder = flight_file_folder)

    # Add information about the flight duration

    # Add seasonal and other cyclic features
    # TODO: Temp solution, move function to utils
    flights_points = add_time_based_features(flights_points)

    return flights_points


def assign_flight_ids(opensky_data: pd.DataFrame, window: int = 6) -> pd.DataFrame:

    # Initialize the flight_id column and a dictionary to track the last time per (icao24, callsign) df['flight_id'] = None
    opensky_data["flight_id"] = None
    last_flight_times = {}

    # Function to determine flight id based on past flight times and a 6-hour window
    def assign_flight_id_fn(row, window=window):
        key = (row["icao24"], row["callsign"])
        current_time = row["timestamp"]
        if (
            key in last_flight_times
            and (current_time - last_flight_times[key]["time"]).total_seconds() / 3600
            <= window
        ):
            # If within 6 hours of the last flight with the same icao24 and callsign, use the same flight id
            return last_flight_times[key]["flight_id"]
        else:
            # Otherwise, create a new flight id
            formatted_time = current_time.strftime("%Y%m%d_%H%M%S")
            # new_flight_id = f"{row['icao24']}_{row['callsign']}_{current_time.isoformat()}"
            new_flight_id = f"{row['icao24']}_{row['callsign']}_{formatted_time}"
            last_flight_times[key] = {"time": current_time, "flight_id": new_flight_id}
            return new_flight_id

    # Apply the function to each row df['flight_id'] = df.apply(assign_flight_id, axis=1)
    opensky_data["flight_id"] = opensky_data.apply(assign_flight_id_fn, axis=1)

    return opensky_data


def remove_outliers(
    opensky_data: pd.DataFrame, thresholds: List[float]
) -> Tuple[pd.DataFrame, float]:

    # print the number of unique flight ids
    num_flights = opensky_data["flight_id"].nunique()
    print(f"Number of unique flight ids before removing outliers: {num_flights}")

    def find_outliers_zscore(df, column, threshold=2.5):
        # Calculate z-scores
        df["z_score"] = zscore(df[column])

        # Filter and return outlier rows
        outliers = df[df["z_score"].abs() > threshold]
        return outliers.drop(columns="z_score")

    (
        consecutive_distance_threshold,
        altitude_threshold,
        lowest_sequence_length_threshold,
    ) = thresholds

    consecutive_distance_outliers = calculate_consecutive_distances(
        opensky_data, distance_threshold=consecutive_distance_threshold
    )
    print(
        f"Found {len(consecutive_distance_outliers)} flights with excessive consecutive distances."
    )

    ADEP_code = opensky_data["ADEP"].value_counts().idxmax()
    ADES_code = opensky_data["ADES"].value_counts().idxmax()
    ADEP_lat_lon = airports[ADEP_code].latlon
    ADES_lat_lon = airports[ADES_code].latlon
    # find outliers where the distance between the first point in the flight and the origin airport is greater than 100 km
    initial_distance_outliers = calculate_initial_distance(
        opensky_data, ADEP_lat_lon, distance_threshold=100
    )
    print(
        f"Found {len(initial_distance_outliers)} flights with excessive initial distances."
    )
    print(
        f"Number of unique flight ids in initial distance outliers that are in consecutive distance outliers: {len(set(initial_distance_outliers).intersection(set(consecutive_distance_outliers)))}"
    )

    # find outliers where the distance between the last point in the flight and the destination airport is greater than 100 km
    final_distance_outliers = calculate_final_distance(
        opensky_data, ADES_lat_lon, distance_threshold=100
    )
    print(
        f"Found {len(final_distance_outliers)} flights with excessive final distances."
    )
    print(
        f"Number of unique flight ids in final distance outliers that are in consecutive distance outliers: {len(set(final_distance_outliers).intersection(set(consecutive_distance_outliers)))}"
    )
    print(
        f"Number of unique flight ids in final distance outliers that are in initial distance outliers: {len(set(final_distance_outliers).intersection(set(initial_distance_outliers)))}"
    )

    altitude_outliers = find_outliers_zscore(
        opensky_data, "altitude", threshold=altitude_threshold
    )
    print(
        f"Found {len(altitude_outliers)} outliers in column 'altitude', with threshold {altitude_threshold}"
    )
    print(altitude_outliers[["flight_id", "altitude"]])
    # print(altitude_outliers['flight_id'].unique())
    print(
        f"Number of unique flight ids in altitude outliers: {altitude_outliers['flight_id'].nunique()}\n"
    )

    # drop rows with altitude outliers
    print("Dropping rows with altitude outliers...")
    opensky_data = opensky_data.drop(altitude_outliers.index).reset_index(drop=True)

    # drop flights with consecutive distance outliers
    print("Dropping flights with consecutive distance outliers...")
    opensky_data = opensky_data[
        ~opensky_data["flight_id"].isin(consecutive_distance_outliers)
    ]

    # drop flights with initial distance outliers that are not dropped by consecutive distance outliers
    initial_distance_outliers = [
        flight_id
        for flight_id in initial_distance_outliers
        if flight_id not in consecutive_distance_outliers
    ]
    print("Dropping flights with initial distance outliers...")
    opensky_data = opensky_data[
        ~opensky_data["flight_id"].isin(initial_distance_outliers)
    ]

    # drop flights with final distance outliers that are not dropped by consecutive distance outliers or initial distance outliers
    final_distance_outliers = [
        flight_id
        for flight_id in final_distance_outliers
        if flight_id not in consecutive_distance_outliers
        and flight_id not in initial_distance_outliers
    ]
    print("Dropping flights with final distance outliers...")
    opensky_data = opensky_data[
        ~opensky_data["flight_id"].isin(final_distance_outliers)
    ]

    # reset the index
    opensky_data = opensky_data.reset_index(drop=True)

    # find the average number of rows in each flight with unique flight_id
    avg_sequence_length = opensky_data.groupby("flight_id").size().mean()

    # count the number of rows in each flight with unique flight_id, and make it a dataframe
    size = opensky_data.groupby("flight_id").size().reset_index(name="counts")

    # calculate z-scores for the counts
    size["z_score"] = zscore(size["counts"])

    # drop flights with lowest sequence length
    low_counts_outliers = size[size["z_score"] < lowest_sequence_length_threshold]
    print(
        f"Found {len(low_counts_outliers)} outliers in column 'counts', with threshold {lowest_sequence_length_threshold}"
    )
    # print(low_counts_outliers)

    # drop the low counts outliers
    opensky_data = opensky_data[
        ~opensky_data["flight_id"].isin(low_counts_outliers["flight_id"])
    ]
    # reset the index
    opensky_data = opensky_data.reset_index(drop=True)

    # remove flights with duplicate rows of the same timestamp. To address: ValueError: cannot reindex on an axis with duplicate labels
    duplicate_rows = opensky_data[
        opensky_data.duplicated(subset=["flight_id", "timestamp"], keep=False)
    ]
    duplicate_flights = duplicate_rows["flight_id"].unique()
    print(f"Found {len(duplicate_flights)} flights with duplicate rows")
    opensky_data = opensky_data[~opensky_data["flight_id"].isin(duplicate_flights)]

    return opensky_data, avg_sequence_length


def load_OpenSky_flights_points(
    base_path: str, ADEP_code: str, ADES_code: str
) -> pd.DataFrame:

    # Look for any .csv files in the base_path
    files = glob.glob(os.path.join(base_path, "*.csv"))
    print(f"Found {len(files)} files in the directory: {base_path}")

    # Select only the files that contain the ADEP and ADES codes
    files = [file for file in files if ADEP_code in file and ADES_code in file]
    print(f"Found {len(files)} files with ADEP and ADES codes in the directory: {base_path}")

    all_data = []  # List to collect DataFrames for each file
    avg_seq_len = 0
    # Process each file
    for file in files:
        print(f"Processing file: {file}")
        opensky_data = pd.read_csv(file)
        print(opensky_data.columns)

        # Drop the 'Unnamed: 0' column if it exists
        if 'Unnamed: 0' in opensky_data.columns:
            opensky_data = opensky_data.drop(columns=["Unnamed: 0"])

        # Drop rows with NaN values and reset the index
        opensky_data = opensky_data.dropna().reset_index(drop=True)

        # Drop rows with negative values in the 'altitude' column
        opensky_data = opensky_data[opensky_data["altitude"] >= 0]

        # Rename columns to match the expected format: ADEP and ADES
        opensky_data = opensky_data.rename(
            columns={"estdepartureairport": "ADEP", "estarrivalairport": "ADES"}
        )

        # Convert the 'timestamp' column to datetime and sort by 'timestamp'
        opensky_data["timestamp"] = pd.to_datetime(opensky_data["timestamp"])
        opensky_data.sort_values("timestamp", inplace=True)

        # Assign flight IDs
        opensky_data = assign_flight_ids(opensky_data, window=6)

        # Remove outliers
        opensky_data, avg_sequence_length = remove_outliers(
            opensky_data, thresholds=[50, 2.2, -1.4]
        )
        avg_seq_len += avg_sequence_length
        print("Removed outliers, now getting trajectories...")

        # Drop the 'z_score' column if it exists
        if 'z_score' in opensky_data.columns:
            opensky_data.drop(columns=["z_score"], inplace=True)

        # Add time-based features
        opensky_data = add_time_based_features(opensky_data, time_col='timestamp')

        # Append the cleaned DataFrame to the list
        all_data.append(opensky_data)

    # Concatenate all DataFrames in the list into a single DataFrame
    combined_df = pd.concat(all_data, ignore_index=True)

    return combined_df, int(avg_seq_len/len(files))


def get_trajectories(flights_points: pd.DataFrame) -> Traffic:

    # Convert timestamp to datetime object
    flights_points["timestamp"] = pd.to_datetime(flights_points["timestamp"], format="%d-%m-%Y %H:%M:%S", utc=True)

    # Create Flight objects for each unique flight ID
    grouped_flights = flights_points.groupby("flight_id")
    flights_list = [Flight(group) for _, group in grouped_flights]

    # Create a Traffic object containing all the flights
    trajectories = Traffic.from_flights(flights_list)
    return trajectories

def prepare_trajectories(
    trajectories: Traffic, n_samples: int, n_jobs: int, douglas_peucker_coeff: float
) -> Traffic:

    # trajectories = trajectories.compute_xy(projection=EuroPP())

    # Simplify trajectories with Douglas-Peucker algorithm if a coefficient is provided
    if douglas_peucker_coeff is not None:
        print("Simplification...")
        trajectories = trajectories.simplify(tolerance=1e3).eval(desc="")

    # Add elapsed time since start for each flight
    trajectories = Traffic.from_flights(
        flight.assign(
            timedelta=lambda r: (r.timestamp - flight.start).apply(
                lambda t: t.total_seconds()
            )
        )
        for flight in trajectories
    )

    # clustering
    print("Clustering...")
    np.random.seed(
        199
    )  # random seed for reproducibility (has big impact on the clustering shape)
    trajectories = trajectories.clustering(
        nb_samples=n_samples,
        projection=EuroPP(),
        features=["latitude", "longitude"],
        clustering=GaussianMixture(n_components=5),
        transform=StandardScaler(),
    ).fit_predict()

    # Resample trajectories for uniformity
    print("Resampling...")
    trajectories = (
        trajectories.resample(n_samples).unwrap().eval(max_workers=n_jobs, desc="resampling")
    )
    return trajectories

# def prepare_trajectories(trajectories: Traffic, n_samples: int, n_jobs: int, douglas_peucker_coeff: float) -> Traffic:
#     # Resample trajectories for uniformity
#     trajectories = trajectories.resample(n_samples).unwrap().eval(max_workers=n_jobs, desc="resampling")
#     trajectories = trajectories.compute_xy(projection=EuroPP())

#     # simplify trjectories with Douglas-Peucker algorithm if a coefficient is provided
#     if douglas_peucker_coeff is not None:
#         print("Simplification...")
#         trajectories = trajectories.simplify(tolerance=1e3).eval(desc="")

#     # Add elapsed time since start for each flight
#     trajectories = Traffic.from_flights(
#         flight.assign(timedelta=lambda r: (r.timestamp - flight.start).apply(lambda t: t.total_seconds()))
#         for flight in trajectories
#     )

#     return trajectories

def main(base_path: str, ADEP: str, ADES: str, data_source: str) -> None:

    flights_points, avg_sequence_length = load_OpenSky_flights_points(
        base_path, ADEP, ADES
    )


    print("Adding weather data")
    #flights_points = add_weather_data_gcsfs(flights_points, "./data/ecmwf_gcsfs/")

    # Create Traffic object from flight points
    trajectories = get_trajectories(flights_points)
    del flights_points

    # Prepare trajectories for training
    trajectories = prepare_trajectories(
        trajectories, int(avg_sequence_length), n_jobs=7, douglas_peucker_coeff=None
    )

    # Save the prepared trajectories to a pickle file in the parent directory of the base_path
    save_path = (
        Path(base_path).parent
        / f"{data_source}_{ADEP}_{ADES}_trajectories.pkl"
    )

    trajectories.to_pickle(Path(save_path))
    print(f"Saved trajectories to {save_path}")

    del trajectories

    # Plot the training data
    #plot_training_data(training_data_path=save_path)

    # Plot the training data with altitude
    #plot_training_data_with_altitude(training_data_path=save_path)

from traffic.data.datasets import (
    landing_amsterdam_2019,
    landing_cdg_2019,
    landing_dublin_2019,
    landing_heathrow_2019,
    landing_londoncity_2019,
    landing_toulouse_2017,
    landing_zurich_2019
)

def get_airport_data(icao_code: str):
    # Total 7 airports
    airport_mapping = {
        "EHAM": landing_amsterdam_2019,  # Amsterdam Schiphol
        "LFPG": landing_cdg_2019,        # Paris Charles de Gaulle
        "EIDW": landing_dublin_2019,     # Dublin Airport
        "EGLL": landing_heathrow_2019,   # London Heathrow
        "EGLC": landing_londoncity_2019, # London City
        "LFBO": landing_toulouse_2017,   # Toulouse Blagnac
        "LSZH": landing_zurich_2019      # Zurich Airport
    }
    return airport_mapping.get(icao_code.upper(), None)  # Return None if not found

def main_landing(base_path: str, data_source: str, ADES: str) -> None:
    print("Preprocessing Landing")

    flight_points = (
    #landing_zurich_2019
    get_airport_data(ADES)
    #.query("runway == '14' and initial_flow == '162-216'")
    .assign_id()
    .unwrap()
    .resample(200)
    #.drop_duplicates()
    .eval()
    )
    print("Finished Preprocessing Landing")

    flight_points.data = add_time_based_features(flight_points.data, "timestamp")
    
    if "origin" in flight_points.data.columns:
        flight_points.data['ADEP'] = flight_points.data['origin']
    else:
        flight_points.data['ADEP'] = "ZZZZ"
    #flight_points.data['ADES'] = 'LSZH'
    flight_points.data['ADES'] = ADES
    flight_points.data['runway'] = ADES


    print("Adding weather data")
    #flights_points = add_weather_data_gcsfs(flights_points, "./data/ecmwf_gcsfs/")

    # Create Traffic object from flight points
    flight_points.data["timestamp"] = pd.to_datetime(flight_points.data["timestamp"], format="%d-%m-%Y %H:%M:%S", utc=True)
    #trajectories = Traffic.from_flights(flight_points)
    #trajectories = get_trajectories(flight_points)
    #del flight_points

    # Prepare trajectories for training
    #trajectories = prepare_trajectories(
        #trajectories, int(avg_sequence_length), n_jobs=7, douglas_peucker_coeff=None
    #)
    trajectories = Traffic.from_flights(
        flight.assign(
            timedelta=lambda r: (r.timestamp - flight.start).apply(
                lambda t: t.total_seconds()
            )
        )
        for flight in flight_points
    )
    trajectories.data["ADEP"] = trajectories.data["ADEP"].fillna("ZZZZ")

    avg_sequence_length = 200
    # Save the prepared trajectories to a pickle file in the parent directory of the base_path
    os.makedirs(Path(base_path) / f"landing_{ADES}", exist_ok=True)
    save_path = (
        Path(base_path) / f"landing_{ADES}"
        / f"{data_source}_trajectories_{ADES}_{avg_sequence_length}.pkl"
    )

    trajectories.to_pickle(Path(save_path))
    print(f"Saved trajectories to {save_path}")

    del trajectories


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ADEP", dest="ADEP", type=str, default="EHAM")
    parser.add_argument("--ADES", dest="ADES", type=str, default="LIMC")
    parser.add_argument(
        "--data_dir", dest="base_path", type=str, default="../data/Opensky"
    )
    # source of data: Either Eurocontrol or OpenSky
    parser.add_argument(
        "--data_source", dest="data_source", type=str, default="OpenSky"
    )
    parser.add_argument(
        "--landing", dest="landing", action='store_true'
    )

    args = parser.parse_args()
    if args.landing:
        args.data_source = "landing"
        main_landing(args.base_path, args.data_source, args.ADES)
    else:
        main(args.base_path, args.ADEP, args.ADES, args.data_source)
