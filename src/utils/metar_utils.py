from dataclasses import dataclass, field
from typing import Optional, List

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

if __name__ == "__main__":
    metar_string = "METAR LIMC 151850Z VRB02KT 3000 MIFG NSC 04/03 Q1024 NOSIG="
    metar = METAR.from_string(metar_string)
    print(metar)
