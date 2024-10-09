import os
import pickle
from traffic.core import Traffic
from argparse import ArgumentParser


def load_traffic_object(filepath):
    """Loads a Traffic object from a .pkl file."""
    traffic_obj = Traffic.from_file(filepath)
    return traffic_obj

def interpolate_trajectory(traffic_obj, target_length):
    """Interpolates the trajectory of a Traffic object to a target length."""
    return traffic_obj.resample(target_length)

def main(directory, target_length, output_filepath):
    """Loads all .pkl files in the directory, interpolates each Traffic object, combines them, and saves as one."""
    big_traffic = None

    # Iterate over all .pkl files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.pkl'):
            filepath = os.path.join(directory, filename)
            
            # Load the Traffic object
            traffic_obj = load_traffic_object(filepath)
            traffic_obj = traffic_obj.drop_duplicates()
            traffic_obj = traffic_obj.resample(target_length).eval()
            
            
            # Combine into a single Traffic object
            if big_traffic is None:
                big_traffic = traffic_obj
            else:
                big_traffic = big_traffic + traffic_obj  # Combine Traffic objects

            print(f"Processed {filename}")

    
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
    target_length = 200
    # Change as per your requirements
    
    # Output file for the combined Traffic object
    parser = ArgumentParser()
    parser.add_argument("--length", type=int, default=target_length)
    parser.add_argument(
        "--data_dir", dest="base_path", type=str, default="./data"
    )
    output_filepath = f"./data/resampled/combined_traffic_resampled_{target_length}.pkl"
    # source of data: Either Eurocontrol or OpenSky
    parser.add_argument(
        "--data_source", dest="data_source", type=str, default=output_filepath
    )

    args = parser.parse_args()
    
    # Process, interpolate, and combine all Traffic objects
    main(args.base_path, args.length, args.data_source)

