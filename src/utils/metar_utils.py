from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
import os
import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import argparse
import re
from typing import List, Optional

@dataclass
class METAR:
    raw: str  # The raw METAR string for reference
    airport_code: str  # ICAO airport code
    observation_time: str  # Observation time in Zulu format
    wind_direction: Optional[int] = None  # Wind direction in degrees
    wind_speed: Optional[float] = None  # Wind speed in meters per second or knots
    wind_gusts: Optional[float] = None  # Gust speed if present
    variable_wind: Optional[List[int]] = field(default_factory=list)  # Wind variability [min, max]
    visibility: Optional[int] = None  # Visibility in meters
    runway_visual_range: List[str] = field(default_factory=list)  # RVR information
    weather_phenomena: List[str] = field(default_factory=list)  # Significant weather phenomena
    cloud_layers: List[str] = field(default_factory=list)  # Cloud layers and coverage
    temperature: Optional[int] = None  # Temperature in Celsius
    dew_point: Optional[int] = None  # Dew point in Celsius
    pressure: Optional[float] = None  # Altimeter pressure in hPa
    trend: Optional[str] = None  # TREND forecast (e.g., NOSIG)
    runway_condition: Optional[str] = None  # Runway condition information
    remarks: Optional[str] = None  # Any additional remarks

    @staticmethod
    def from_string(metar_string: str) -> 'METAR':
        # Initialize fields with default values
        airport_code = None
        observation_time = None
        wind_direction = None
        wind_speed = None
        wind_gusts = None
        variable_wind = []
        visibility = None
        runway_visual_range = []
        weather_phenomena = []
        cloud_layers = []
        temperature = None
        dew_point = None
        pressure = None
        trend = None
        runway_condition = None
        remarks = None

        # Use regex to parse METAR components
        parts = metar_string.split()
        if parts:
            # Airport code
            airport_code = parts[1] if len(parts) > 1 else None

            # Observation time
            observation_match = re.search(r"\b(\d{6}Z)\b", metar_string)
            if observation_match:
                observation_time = observation_match.group(1)

            # Wind info
            wind_match = re.search(r"(\d{3})(\d{2})(G\d{2})?(KT|MPS)", metar_string)
            if wind_match:
                wind_direction = int(wind_match.group(1))
                wind_speed = int(wind_match.group(2))
                if wind_match.group(3):
                    wind_gusts = int(wind_match.group(3)[1:])  # Remove 'G'

            # Variable wind
            variable_wind_match = re.search(r"(\d{3})V(\d{3})", metar_string)
            if variable_wind_match:
                variable_wind = [int(variable_wind_match.group(1)), int(variable_wind_match.group(2))]

            # Visibility
            visibility_match = re.search(r"\b(\d{4})\b", metar_string)
            if visibility_match:
                visibility = int(visibility_match.group(1))

            # Runway visual range
            runway_matches = re.findall(r"R\d{2}/[PNU]?\d{4}[UDN]?", metar_string)
            runway_visual_range = runway_matches

            # Weather phenomena
            weather_phenomena_match = re.findall(r"[\+\-]?[A-Z]{2,}", metar_string)
            weather_phenomena = weather_phenomena_match

            # Cloud layers
            cloud_layer_matches = re.findall(r"(FEW|SCT|BKN|OVC)\d{3}", metar_string)
            cloud_layers = cloud_layer_matches

            # Temperature and dew point
            temp_match = re.search(r"M?\d{2}/M?\d{2}", metar_string)
            if temp_match:
                temps = temp_match.group(0).split("/")
                temperature = int(temps[0].replace("M", "-"))
                dew_point = int(temps[1].replace("M", "-"))

            # Pressure
            pressure_match = re.search(r"Q(\d{4})", metar_string)
            if pressure_match:
                pressure = int(pressure_match.group(1)) / 10.0

            # TREND forecast
            trend_match = re.search(r"(NOSIG|BECMG|TEMPO|INTER)", metar_string)
            if trend_match:
                trend = trend_match.group(1)

            # Runway condition
            runway_condition_match = re.search(r"88\d{2}//\d{2}", metar_string)
            if runway_condition_match:
                runway_condition = runway_condition_match.group(0)

            # Remarks
            remarks_index = metar_string.find("RMK")
            if remarks_index != -1:
                remarks = metar_string[remarks_index + 4:]

        return METAR(
            raw=metar_string,
            airport_code=airport_code,
            observation_time=observation_time,
            wind_direction=wind_direction,
            wind_speed=wind_speed,
            wind_gusts=wind_gusts,
            variable_wind=variable_wind,
            visibility=visibility,
            runway_visual_range=runway_visual_range,
            weather_phenomena=weather_phenomena,
            cloud_layers=cloud_layers,
            temperature=temperature,
            dew_point=dew_point,
            pressure=pressure,
            trend=trend,
            runway_condition=runway_condition,
            remarks=remarks
        )

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
            'nil': 'SI',
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
    """
    base_url = "https://www.ogimet.com/display_metars2.php"
    save_folder = "./metar_data/"
    icao24 = "LIRF"
    start_date = datetime(2018, 1, 1, 0, 0)  # Example start date
    end_date = datetime(2018, 3, 2, 0, 0)   # Example end date

    # Fetch and save METAR data
    fetch_metar_data(base_url, icao24, start_date, end_date, save_folder)
    """
