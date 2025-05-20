import logging
from traffic.data import opensky
import argparse
from datetime import datetime, timedelta
import os

def daterange(start_date, end_date):
    """
    Generate a range of dates, one month at a time.
    Parameters
    ----------
    start_date
    end_date

    Returns
    -------

    """
    current_start = start_date
    while current_start < end_date:
        # Move to the next month
        next_month = current_start.replace(day=28) + timedelta(days=4)
        next_month_start = next_month.replace(day=1)
        yield (current_start, min(next_month_start - timedelta(days=1), end_date))
        current_start = next_month_start

def download_data_for_month(start, end, departure_airport, arrival_airport, selected_columns, output_dir):
    """
    Download OpenSky data for a specific month and save it to a CSV file.
    Parameters
    ----------
    start
    end
    departure_airport
    arrival_airport
    selected_columns
    output_dir

    Returns
    -------

    """
    logging.info(f"Downloading data from OpenSky for {departure_airport} to {arrival_airport} from {start} to {end}")
    start_date = start.strftime('%Y-%m-%d')
    end_date = end.strftime('%Y-%m-%d')
    #print(start_date, end_date)

    downloaded_traffic = opensky.history(
        start=start_date,
        stop=end_date,
        departure_airport=departure_airport,
        arrival_airport=arrival_airport,
        selected_columns=selected_columns,
    )
    
    save_path = os.path.join(output_dir, f"opensky_{departure_airport}_{arrival_airport}_{start_date}_{end_date}.csv")
    
    if downloaded_traffic:
        downloaded_traffic.to_csv(save_path)
        logging.info(f"Data saved to {save_path}")
    else:
        logging.info(f"No data found for {start} to {end}")

def main(args):
    start_date = datetime.strptime(args.start, '%Y-%m-%d')
    end_date = datetime.strptime(args.end, '%Y-%m-%d')

    logging.info(f"Downloading data for each month from {start_date} to {end_date}")
    
    for start, end in daterange(start_date, end_date):
        download_data_for_month(
            start=start,
            end=end,
            departure_airport=args.departure_airport,
            arrival_airport=args.arrival_airport,
            selected_columns=args.selected_columns,
            output_dir=args.output_dir
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download OpenSky data')
    parser.add_argument('start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('departure_airport', type=str, help='Departure airport')
    parser.add_argument('arrival_airport', type=str, help='Arrival airport')
    parser.add_argument('--output_dir', type=str, default="./", help='Output directory')
    parser.add_argument('--selected_columns', nargs='+', default=["StateVectorsData4.time", "icao24", "callsign", 
                                                                  "lat", "lon", "baroaltitude", 
                                                                  "FlightsData4.estdepartureairport", "FlightsData4.estarrivalairport",
                                                                  "StateVectorsData4.velocity", "StateVectorsData4.heading", "StateVectorsData4.vertrate",
                                                                  #"VelocityData4.ewvelocity", "VelocityData4.nsvelocity"
                                                                  ], help='Selected columns')

    args = parser.parse_args()
    main(args)

