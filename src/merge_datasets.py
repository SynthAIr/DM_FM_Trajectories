import os
import pickle
from traffic.core import Traffic
from argparse import ArgumentParser
from preprocess import clean_trajectory_data, clean_and_smooth_flight_with_tight_threshold


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
            
            # Load the Traffic object
            
            traffic_obj = load_traffic_object(filepath)
            print("object loaded")
            traffic_obj = traffic_obj.drop_duplicates()
            print("duplicates dropped")
            traffic_obj = traffic_obj.filter(altitude=(17, 53))
            print("Filtering Altitude")
            traffic_obj = traffic_obj.drop_duplicates()
            print("duplicates dropped")
            traffic_obj = traffic_obj.resample(target_length).phases().eval()
            print("resampled")
            traffic_obj.data['altitude'] = traffic_obj.data['altitude'] * 0.3048
            
            if filter_alt:
                print("Filtering Altitude")
                for n in range(target_length):
                    traffic_obj.data.loc[n,'altitude'] = clean_trajectory_data(traffic_obj.data.loc[n], 'altitude',n, 2.5)
                    traffic_obj.data.loc[n,'groundspeed'] = clean_trajectory_data(traffic_obj.data.loc[n], 'groundspeed',n, 2.5)


                cleaned_flights = [clean_and_smooth_flight_with_tight_threshold(flight, target_length, 'altitude') for flight in traffic_obj]
                cleaned_flights = [clean_and_smooth_flight_with_tight_threshold(flight, target_length, 'groundspeed') for flight in traffic_obj]
                
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

    # Save the combined Traffic object
    if big_traffic is not None:
        big_traffic.to_pickle(output_filepath)
        print(f"Saved combined Traffic object to {output_filepath}")
    else:
        print("No Traffic objects found to combine.")

if __name__ == "__main__":
    # Directory containing .pkl files
    #directory = "./data"
    
    # Desired target length for all trajectories
    target_length = 600
    filter_alt = True
    # Change as per your requirements
    
    # Output file for the combined Traffic object
    parser = ArgumentParser()
    parser.add_argument("--length", type=int, default=target_length)
    parser.add_argument(
        "--data_dir", dest="base_path", type=str, default="./data"
    )
    output_filepath = f"./data/resampled/combined_traffic_resampled_{target_length}.pkl" if filter_alt else f"./data/resampled/combined_traffic_resampled_no_filter_{target_length}.pkl" 
    # source of data: Either Eurocontrol or OpenSky
    parser.add_argument(
        "--data_source", dest="data_source", type=str, default=output_filepath
    )

    args = parser.parse_args()
    
    # Process, interpolate, and combine all Traffic objects
    main(args.base_path, args.length, args.data_source, filter_alt)

