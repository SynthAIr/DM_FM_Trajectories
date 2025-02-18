import os
import pickle
from traffic.core import Traffic
from argparse import ArgumentParser
from preprocess import clean_trajectory_data, clean_and_smooth_flight_with_tight_threshold
import pandas as pd


def load_traffic_object(filepath):
    """Loads a Traffic object from a .pkl file."""
    traffic_obj = Traffic.from_file(filepath)
    return traffic_obj

def interpolate_trajectory(traffic_obj, target_length):
    """Interpolates the trajectory of a Traffic object to a target length."""
    return traffic_obj.resample(target_length)



def main(directory, target_length, output_filepath, filter_alt = False):
    """Loads all .pkl files in the directory, interpolates each Traffic object, combines them, and saves as one."""
    big_traffic = None

    # Iterate over all .pkl files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.pkl'):
            filepath = os.path.join(directory, filename)
            
            print(f"Processing {filename}")
            
            # Load the Traffic object
            traffic_obj = load_traffic_object(filepath)
            print("object loaded")
            #traffic_obj = traffic_obj.drop_duplicates()
            print("duplicates dropped")
            #traffic_obj = traffic_obj.filter(altitude=(17, 53))
            print("Filtering Altitude")
            #traffic_obj = traffic_obj.drop_duplicates()
            print("duplicates dropped")
            #traffic_obj = traffic_obj.resample(target_length).eval()
            print("resampled")

            #feet to m
            traffic_obj.data['altitude'] = traffic_obj.data['altitude'] * 0.3048
            # Knots to m/s
            traffic_obj.data['groundspeed'] = traffic_obj.data['groundspeed'] * 0.514444
            # f/min to m/s 
            traffic_obj.data['vertical_rate'] = traffic_obj.data['vertical_rate'] * 0.00508
            c2 = 0
            for f in traffic_obj:
                if len(f) == 201:
                    c2 += 1

            if c2 != 0:
                print("Found invalid flights", c2)
                df = traffic_obj.data
                # Count the number of samples per flight
                flight_counts = df.groupby("flight_id").size()

                # Identify flights with 201 samples
                flights_to_trim = flight_counts[flight_counts == 201].index

                # Remove one sample from each of these flights
                trimmed_dfs = []
                for flight_id in flights_to_trim:
                    flight_df = df[df["flight_id"] == flight_id]
                    trimmed_dfs.append(flight_df.iloc[1:])  # Drop the first sample (or change logic as needed)

                # Combine modified and unmodified flights
                new_df = pd.concat([df[~df["flight_id"].isin(flights_to_trim)]] + trimmed_dfs)
                #new_df['runway'] = "toulouse"
                # Create a new Traffic object
                df = new_df

                # Identify flight_ids that contain any NaN values
                flights_with_nan = df[df.isna().any(axis=1)]["flight_id"].unique()

                # Remove all rows associated with these flight_ids
                df_clean = df[~df["flight_id"].isin(flights_with_nan)]

                # Create a new Traffic object with the cleaned DataFrame
                traffic_obj = Traffic(df_clean)

            
            if filter_alt:
                print("Filtering Altitude")
                for n in range(target_length):
                    traffic_obj.data.loc[n,'altitude'] = clean_trajectory_data(traffic_obj.data.loc[n], 'altitude',n, 2.5)
                    traffic_obj.data.loc[n,'groundspeed'] = clean_trajectory_data(traffic_obj.data.loc[n], 'groundspeed',n, 2.5)
                    traffic_obj.data.loc[n,'vertical_rate'] = clean_trajectory_data(traffic_obj.data.loc[n], 'vertical_rate',n, 2.5)

                print("Cleaning Altitudes")
                cleaned_flights = [clean_and_smooth_flight_with_tight_threshold(flight, target_length, 'altitude') for flight in traffic_obj]
                traffic_obj = Traffic.from_flights(cleaned_flights)
                cleaned_flights = [clean_and_smooth_flight_with_tight_threshold(flight, target_length, 'groundspeed') for flight in traffic_obj]
                traffic_obj = Traffic.from_flights(cleaned_flights)
                cleaned_flights = [clean_and_smooth_flight_with_tight_threshold(flight, target_length, 'vertical_rate') for flight in traffic_obj]
                
                traffic_obj = Traffic.from_flights(cleaned_flights)
                print("Columns:", traffic_obj.data.columns)
                    
                    
            # Combine into a single Traffic object
            if big_traffic is None:
                big_traffic = traffic_obj
            else:
                big_traffic = big_traffic + traffic_obj  # Combine Traffic objects

            print(f"Processed {filename}")

    # Feet to meters

    big_traffic = big_traffic.cumulative_distance().eval()
    big_traffic = big_traffic = big_traffic.query('flight_id != "SWR983_17905"')

    

    # Save the combined Traffic object
    if not filter_alt:
        ades_names = "_".join(sorted(big_traffic.data['ADES'].unique()))
        output_filepath = f"/mnt/data/synthair/synthair_diffusion/data/resampled/combined_traffic_resampled_landing_{ades_names}_{target_length}.pkl"
    if big_traffic is not None:
        big_traffic.to_pickle(output_filepath)
        print(f"Saved combined Traffic object to {output_filepath}")
    else:
        print("No Traffic objects found to combine.")

if __name__ == "__main__":
    # Directory containing .pkl files
    #directory = "./data"
    
    # Desired target length for all trajectories
    target_length = 200
    filter_alt = False
    # Change as per your requirements
    
    # Output file for the combined Traffic object
    parser = ArgumentParser()
    parser.add_argument("--length", type=int, default=target_length)
    parser.add_argument(
        "--data_dir", dest="base_path", type=str, default="./data"
    )
    output_filepath = f"./data/resampled/combined_traffic_resampled_{target_length}.pkl" if filter_alt else f"./data/resampled/combined_traffic_resampled_landing_EHAM_{target_length}.pkl" 
    # source of data: Either Eurocontrol or OpenSky
    parser.add_argument(
        "--data_source", dest="data_source", type=str, default=output_filepath
    )

    args = parser.parse_args()
    output_filepath = f"/mnt/data/synthair/synthair_diffusion/data/resampled/combined_traffic_resampled_{target_length}.pkl" if filter_alt else f"/mnt/data/synthair/synthair_diffusion/data/resampled/combined_traffic_resampled_landing_{target_length}.pkl" 
    args.data_source = output_filepath
    
    # Process, interpolate, and combine all Traffic objects
    main(args.base_path, args.length, args.data_source, filter_alt)

