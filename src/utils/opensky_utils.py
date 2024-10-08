import traffic
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


from traffic.data import opensky

import argparse



def main(args):

    logging.info(f"Downloading data from OpenSky for {args.departure_airport} to {args.arrival_airport} from {args.start} to {args.end}")
    downloaded_traffic= opensky.history(
        start=args.start,
        stop=args.end,
        departure_airport=args.departure_airport,
        arrival_airport=args.arrival_airport,
        selected_columns=args.selected_columns,
    )

    logging.info(f"Data downloaded. Saving to {args.output_dir}")


    save_path = args.output_dir + "/opensky_" + args.departure_airport + "_" + args.arrival_airport + "_" + args.start + "_" + args.end + ".csv"
    downloaded_traffic.to_csv(save_path)
    logging.info(f"Data saved to {save_path}")


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Download OpenSky data')
    parser.add_argument('start', type=str, help='Start date')
    parser.add_argument('end', type=str, help='End date')
    parser.add_argument('departure_airport', type=str, help='Departure airport')
    parser.add_argument('arrival_airport', type=str, help='Arrival airport')
    parser.add_argument('--output_dir', type=str, default="./", help='Output directory')


    parser.add_argument('--selected_columns', nargs='+', default=["StateVectorsData4.time", "icao24", "callsign", "lat", "lon", "baroaltitude", "FlightsData4.estdepartureairport", "FlightsData4.estarrivalairport"], help='Selected columns')

    args = parser.parse_args()
    main(args)

    # python get_data.py 2019-11-01 2019-11-02 EHAM LIMC 
