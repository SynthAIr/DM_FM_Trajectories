import abc
from tqdm import tqdm
import xarray as xr
import logging
import os
from argparse import ArgumentParser
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path
from typing import Any, List, Optional, Protocol, Tuple, TypedDict, Union
import joblib
import utm

import numpy as np
import pandas as pd
import torch
import yaml
from minio import Minio
from minio.error import S3Error
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from torch.utils.data import Dataset
from traffic.core import Traffic
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from .condition_utils import Condition
from .weather_utils import load_weather_data, load_weather_data_function, load_weather_data_arrival_airport
from .metar_utils import load_metar_data


logger = logging.getLogger(__name__)


def parse_runtime_env(filename):
    with open(file=filename, mode="r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def export_environment_variables(runtime_env_filename):
    runtime_env = parse_runtime_env(runtime_env_filename)
    for key, value in runtime_env["env_vars"].items():
        os.environ[key] = value


def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance


def calculate_consecutive_distances(df, distance_threshold):
    """Calculates distances between consecutive points and flags flights with any excessive distance."""
    # Calculate distances for each point to the next within each flight
    df = df.sort_values(["flight_id", "timestamp"])
    df["next_latitude"] = df.groupby("flight_id")["latitude"].shift(-1)
    df["next_longitude"] = df.groupby("flight_id")["longitude"].shift(-1)

    # Apply the Haversine formula
    df["segment_distance"] = df.apply(
        lambda row: (
            haversine(
                row["latitude"],
                row["longitude"],
                row["next_latitude"],
                row["next_longitude"],
            )
            if not pd.isna(row["next_latitude"])
            else 0
        ),
        axis=1,
    )

    # Find flights with any segment exceeding the threshold
    outlier_flights = df[df["segment_distance"] > distance_threshold][
        "flight_id"
    ].unique()
    return outlier_flights


def calculate_initial_distance(df, origin_lat_lon, distance_threshold):
    """Calculates distances between the first point in each flight and the origin airport."""
    # Calculate distances from the origin airport to the first point of each flight

    # first point of each flight
    first_points = df.groupby("flight_id").first()
    # Calculate distances from the origin airport to the first point of each flight
    first_points["initial_distance"] = [
        haversine(lat, lon, origin_lat_lon[0], origin_lat_lon[1])
        for lat, lon in zip(first_points["latitude"], first_points["longitude"])
    ]

    # Find flights with the first point exceeding the threshold
    outlier_flights = first_points[
        first_points["initial_distance"] > distance_threshold
    ].index
    return outlier_flights


def calculate_final_distance(df, destination_lat_lon, distance_threshold):
    """Calculates distances between the last point in each flight and the destination airport."""
    # Calculate distances from the destination airport to the last point of each flight

    # last point of each flight
    last_points = df.groupby("flight_id").last()
    # Calculate distances from the destination airport to the last point of each flight
    last_points["final_distance"] = [
        haversine(lat, lon, destination_lat_lon[0], destination_lat_lon[1])
        for lat, lon in zip(last_points["latitude"], last_points["longitude"])
    ]

    # Find flights with the last point exceeding the threshold
    outlier_flights = last_points[
        last_points["final_distance"] > distance_threshold
    ].index
    return outlier_flights


class Downloader(abc.ABC):
    @abc.abstractmethod
    def download(self):
        pass


class MinioInterface:

    def __init__(
        self,
        bucket_name: str,
        object_name: str,
        local_path: str,
        endpoint_url: Optional[str] = None,
        **kwargs,
    ) -> None:
        env_endpoint_url = os.environ.get("MINIO_ENDPOINT_URL", None)
        if endpoint_url is None:
            endpoint_url = env_endpoint_url
        self.client = Minio(
            endpoint_url,
            access_key=os.environ.get("MINIO_ACCESS_KEY_ID"),
            secret_key=os.environ.get("MINIO_SECRET_ACCESS_KEY"),
            secure=False,
        )
        self.bucket_name = bucket_name
        self.object_name = object_name
        self.local_path = local_path
        self.kwargs = kwargs


class MinioDirectoryDownloader(MinioInterface, Downloader):

    def _download(self):

        objects = self.client.list_objects(
            self.bucket_name, prefix=self.object_name, recursive=True
        )
        for obj in objects:
            object_name = obj.object_name
            local_file_path = os.path.join(self.local_path, object_name)
            try:
                self.client.fget_object(self.bucket_name, object_name, local_file_path)
                if self.kwargs.get("verbose", True):
                    print(
                        f"File '{object_name}' downloaded as '{local_file_path}' from bucket '{self.bucket_name}'."
                    )
            except S3Error as err:
                logger.error(err)

    def download(self):
        self._download()


class BuilderProtocol(Protocol):
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame: ...


class TransformerProtocol(Protocol):
    def fit(self, X: np.ndarray) -> "TransformerProtocol": ...

    def fit_transform(self, X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]: ...

    def transform(self, X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]: ...

    def inverse_transform(self, X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]: ...


class Infos(TypedDict):
    features: List[str]
    index: Optional[int]


class DatasetParams(TypedDict):
    features: List[str]
    file_path: Optional[Path]
    input_dim: int
    scaler: Optional[TransformerProtocol]
    seq_len: int
    shape: str
    conditional_features: List[Condition]
    variables: List[str]
    metar: bool


class TrafficDataset(Dataset):
    """Traffic Dataset

    Args:
        traffic: Traffic object to extract data from.
        features: features to extract from traffic.
        shape (optional): shape of datapoints when:

            - ``'image'``: tensor of shape
              :math:`(\\text{feature}, \\text{seq})`.
            - ``'linear'``: tensor of shape
              :math:`(\\text{feature} \\times \\text{seq})`.
            - ``'sequence'``: tensor of shape
              :math:`(\\text{seq}, \\text{feature})`. Defaults to
              ``'sequence'``.
        scaler (optional): scaler to apply to the data. You may want to
            consider `StandardScaler()
            <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html>`_.
            Defaults to None.
    """

    _repr_indent = 4
    _available_shapes = ["linear", "sequence", "image"]


    def __init__(
        self,
        traffic: Traffic,
        features: List[str],
        shape: str = "linear",
        scaler: Optional[TransformerProtocol] = None,
        conditional_features = [],
        variables = ['v_component_of_wind', 'u_component_of_wind', 'temperature', 'vertical_velocity'],
        metar = False,
    ) -> None:

        assert shape in self._available_shapes, (
            f"{shape} shape is not available. " + f"Available shapes are: {self._available_shapes}"
        )

        self.file_path: Optional[Path] = None
        self.features = features
        self.conditional_features = conditional_features 
        self.shape = shape
        self.scaler = scaler
        self.data: torch.Tensor
        self.continuous_conditions: torch.Tensor
        self.categorical_conditions: torch.Tensor
        self.variables = variables
        self.metar = metar
        
        if "lat_ref" in traffic.data.columns:
            self.lat_refs = np.stack(list(f.data['lat_ref'].values[0] for f in traffic)).reshape(-1,1)
            self.lon_refs = np.stack(list(f.data['lon_ref'].values[0] for f in traffic)).reshape(-1,1)
        else:
            self.lat_refs = np.zeros((len(traffic), 1))
            self.lon_refs = np.zeros((len(traffic), 1))

        print(self.lat_refs.shape, self.lon_refs.shape)
        assert self.lon_refs.shape == self.lat_refs.shape

        data = np.stack(list(f.data[self.features].values.ravel() for f in traffic))
        data = data.reshape(data.shape[0], -1, len(self.features))
        assert self.lon_refs.shape[0] == data.shape[0]


        data = data.reshape(data.shape[0], -1)
        print(data.shape)

        weather_variable_names = "".join(sorted(word for word in self.variables if word))
        self.con_cond_scaler = None
        self.cat_cond_scaler = None
        self.grid_cond_scaler = None

        self.con_conditions = torch.empty(len(data))
        self.cat_conditions = torch.empty(len(data))
        self.grid_conditions = torch.empty(len(data))

        #pressure_levels = np.array([ 100,  150,  200,  250,  300,  400,  500,  600,  700,  850,  925, 1000])
        #pressure_levels = np.array([ 925,  950,  975, 1000])[::-1]
        pressure_levels = np.array([1000])[::-1]
        print(data.shape)
        assert not np.isnan(data).any(), "Tensor contains NaN values"

        if metar:
            print("Initing with metar")
            file_path = "/mnt/data/synthair/synthair_diffusion/data/metar/metar_landing.txt"  # Change this to the actual file path
            save_path = "/mnt/data/synthair/synthair_diffusion/data/metar/"
            self.grid_conditions = load_metar_data(file_path, traffic, save_path)

        elif self.variables is not None and len(self.variables) > 0:
            print("Initing with era5")
            save_path = "/mnt/data/synthair/synthair_diffusion/data/era5/"
            # List all .nc files in the directory
            nc_files = [save_path + f for f in os.listdir(save_path) if f.endswith('.nc')]
            #self.grid_conditions = load_weather_data(nc_files, traffic, preprocess, save_path)
            grid_size = 5
            num_levels = 1
            #self.grid_conditions = load_weather_data_function(nc_files, traffic, preprocess, save_path, grid_size = grid_size, num_levels=num_levels, pressure_levels = pressure_levels)
            print("No NaN values in data")
            self.grid_conditions = load_weather_data_arrival_airport(nc_files, traffic, variables, save_path, grid_size = grid_size, 
                                                                 num_levels=num_levels, pressure_levels = pressure_levels, variable_names = weather_variable_names)
        
        assert len(traffic) == len(self.grid_conditions)
        print(len(self.grid_conditions))
        print(self.grid_conditions[0].shape)

        def save_scaler_if_not_exists(scaler, path):
            if not os.path.exists(path):
                joblib.dump(scaler, path)
                print(f"Scaler saved at: {path}")
            else:
                print(f"Scaler already exists at: {path}")


        if self.conditional_features is not None and len(self.conditional_features) > 0:
            
            def load_scaler_if_exists(path):
                """Load and return the scaler if it exists, otherwise return None."""
                if os.path.exists(path):
                    print(f"Loading scaler from: {path}")
                    return joblib.load(path)
                else:
                    print(f"Scaler not found at: {path}")
                    return None

            def scale_conditions(conditions_fs: List[torch.Tensor], axis=1):

                if len(conditions_fs) == 0:
                    return torch.empty(len(data)), None

                
                if len(conditions_fs) == 0:
                    conditions = conditions_fs[0]
                else:
                    conditions = np.concatenate(conditions_fs, axis=axis)

                if conditions.shape[-1] == 0:
                    print("No values in dataset")
                    return torch.empty(len(data)), None

                original_shape = conditions.shape
                print(original_shape)
                if len(conditions.shape) != 2:
                    # Reshape to 2D (preserving the first dimension, flatten the rest)
                    conditions = conditions.reshape(conditions.shape[0], -1)
                    print(conditions.shape)
                
                s = load_scaler_if_exists("/mnt/data/synthair/synthair_diffusion/data/resampled/scalers/7_datasets_con_utm_standard.gz")
                if s is None:
                    #s = MinMaxScaler(feature_range=(-1, 1))
                    s = StandardScaler()
                    s.fit(conditions)

                conditions = s.transform(conditions)
                conditions = torch.FloatTensor(conditions)

                    # Reshape back to the original shape if necessary (e.g., return to 3D)
                if len(original_shape) != 2:
                    conditions = conditions.reshape(*original_shape)

                return conditions, s
            
            if self.metar:
                preprocessed , self.gid_cond_scaler = scale_conditions([self.grid_conditions[:,:-1]], 0)
                self.grid_conditions = torch.cat((preprocessed, self.grid_conditions[:,-1].unsqueeze(1)), dim=1)

            elif self.variables is not None and len(self.variables) > 0:
                self.grid_conditions, self.gid_cond_scaler = scale_conditions(self.grid_conditions, 1)
                print(self.grid_conditions.shape)
                print(self.grid_conditions.ndim)
                if self.grid_conditions.ndim != 1:
                    self.grid_conditions = self.grid_conditions.reshape(-1, len(variables), num_levels, grid_size, grid_size)

            con_condition_fs, cat_condition_fs = self._get_conditions(traffic)

            self.con_conditions, self.con_cond_scaler = scale_conditions(con_condition_fs)
            save_scaler_if_not_exists(self.con_cond_scaler,"/mnt/data/synthair/synthair_diffusion/data/resampled/scalers/7_datasets_con_utm_standard.gz") 

            self.cat_conditions = np.concatenate(cat_condition_fs, axis=1)
            self.cat_conditions = torch.IntTensor(self.cat_conditions)

            print(self.con_conditions.shape, self.cat_conditions.shape)

        if self.scaler is not None:
            try:
                # If scaler already fitted, only transform
                check_is_fitted(self.scaler)
                data = self.scaler.transform(data)
            except NotFittedError:
                # If not: fit and transform
                self.scaler = self.scaler.fit(data)
                data = self.scaler.transform(data)
        

        save_scaler_if_not_exists(self.scaler, f"/mnt/data/synthair/synthair_diffusion/data/resampled/scalers/7_datasets_utm_standard.gz")
        exit()

        data = torch.FloatTensor(data)
        self.data = data
        if self.shape in ["sequence", "image"]:
            self.data = self.data.view(self.data.size(0), -1, len(self.features))
            #self.conditions = self.conditions.view(self.conditions.size(0), -1, len(condition_feature_names))
            if self.shape == "image":
                self.data = torch.transpose(self.data, 1, 2)
                #self.conditions = torch.transpose(self.conditions, 1, 2)


    def _get_conditions(self, traffic: Traffic) -> List[torch.Tensor]: 
        condition_continuous = []
        condition_categorical = []
        for feature in self.conditional_features:
            feature_type = feature.get_type()
            feature_names = feature.get_feature_names()
            feature_data = np.array([f.data[feature_names].values.ravel() for f in traffic])

            if feature_type == "continuous":
                #feature_data = np.array([f.data[feature_names].values.ravel() for f in traffic])
                feature_data = feature.to_tensor(feature_data).squeeze()
                if len(feature_data.shape) == 1:
                    feature_data = feature_data.reshape(-1, 1)
                print(feature_names)
                print(feature_data.shape)
                condition_continuous.append(feature_data)

            elif feature_type == "cyclic":
                #feature_data = np.array([f.data[feature.label].values.ravel() for f in traffic])
                feature_data = feature.to_tensor(feature_data)
                print(feature_names)
                print(feature_data.shape)
                condition_continuous.append(feature_data)

            elif feature_type == "categorical":
                #feature_data = np.array([f.data[feature_names].values.ravel() for f in traffic])
                print(feature_names)
                feature_data = feature.to_tensor(feature_data)
                condition_categorical.append(feature_data)


        print("Continuous conditions: ", len(condition_continuous))
        print("Categorical conditions: ", len(condition_categorical))
        return condition_continuous, condition_categorical
    
    def inverse_airport_coordinates(self, data, idx) -> torch.Tensor:
        shape = data.shape
        data = data.reshape(-1, 200, len(self.features))
        easting_ref, northing_ref, zone_number, zone_letter = utm.from_latlon(self.lat_refs[idx], self.lon_refs[idx])
        # Compute absolute easting/northing
        easting = easting_ref + data[:, :, 0]
        northing = northing_ref + data[:, :, 1]

        # Convert back to latitude/longitude
        lat, lon = utm.to_latlon(easting, northing, zone_number, zone_letter)

        data[:, :, 0] = lat
        data[:, :, 1] = lon

        # Restore latitude
        #data[:, :, 0] += self.lat_refs[idx]

        # Restore longitude with scaling correction
        #data[:, :, 1] = (data[:, :, 1] / np.cos(np.deg2rad(self.lat_refs[idx]))) + self.lon_refs[idx]
        #data[:, :, 1] = data[:, :, 1] + self.lon_refs[idx]

        return data.reshape(shape)

    @classmethod
    def from_file(
        cls,
        file_path: Union[str, Path],
        features: List[str],
        shape: str = "linear",
        scaler: Optional[TransformerProtocol] = None,
        conditional_features = [],
        variables = ['v_component_of_wind', 'u_component_of_wind', 'temperature', 'vertical_velocity'],
        metar = False,
    ) -> "TrafficDataset":
        file_path = file_path if isinstance(file_path, Path) else Path(file_path)
        traffic = Traffic.from_file(file_path)

        dataset = cls(traffic, features, shape, scaler, conditional_features, variables, metar)
        dataset.file_path = file_path
        return dataset

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, List[Any]]:
        """Get item method, returns datapoint at some index.

        Args:
            index (int): An index. Should be :math:`<len(self)`.

        Returns:
            torch.Tensor: The trajectory data shaped accordingly to self.shape.
            int: The length of the trajectory.
            list: List of informations that could be needed like, labels or
                original latitude and longitude values.
        """
        return self.data[index], self.con_conditions[index], self.cat_conditions[index], self.grid_conditions[index]

    @property
    def input_dim(self) -> int:
        """Returns the size of datapoint's features.

        .. warning::
            If the `self.shape` is ``'linear'``, the returned size will be
            :math:`\\text{feature_n} \\times \\text{sequence_len}`
            since the temporal dimension is not taken into account with this
            shape.
        """
        if self.shape in ["linear", "sequence"]:
            return self.data.shape[-1]
        elif self.shape == "image":
            return self.data.shape[1]
        else:
            raise ValueError(f"Invalid shape value: {self.shape}.")

    @property
    def seq_len(self) -> int:
        """Returns sequence length (i.e. maximum sequence length)."""
        if self.shape == "linear":
            return int(self.input_dim / len(self.features))
        elif self.shape == "sequence":
            return self.data.shape[1]
        elif self.shape == "image":
            return self.data.shape[2]
        else:
            raise ValueError(f"Invalid shape value: {self.shape}.")

    @property
    def parameters(self) -> DatasetParams:
        """Returns parameters of the TrafficDataset object in a TypedDict.

        * features (List[str])
        * file_path (Path, optional)
        * input_dim (int)
        * scaler (Any object that matches TransformerProtocol, see TODO)
        * seq_len (int)
        * shape (str): either ``'image'``, ``'linear'`` or ```'sequence'``.
        """
        return DatasetParams(
            features=self.features,
            file_path=self.file_path,
            input_dim=self.input_dim,
            scaler=self.scaler,
            seq_len=self.seq_len,
            shape=self.shape,
            conditional_features = self.conditional_features,
            variables = self.variables,
            metar = self.metar
        )

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        # if self.file_path is not None:
        #     body.append(f"File location: {self.file_path}")
        if self.scaler is not None:
            body += [repr(self.scaler)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    @classmethod
    def add_argparse_args(cls, parser: ArgumentParser) -> ArgumentParser:
        """Adds TrafficDataset arguments to ArgumentParser.

        List of arguments:

            * ``--data_path``: The path to the traffic data file. Default to
              None.
            * ``--features``: The features to keep for training. Default to
              ``['latitude','longitude','altitude','timedelta']``.

              Usage:

              .. code-block:: console

                python main.py --features track groundspeed altitude timedelta

            * ``--info_features``: Features not passed through the model but
              useful to keep. For example, if you chose as main features:
              track, groundspeed, altitude and timedelta ; it might help to
              keep the latitude and longitude values of the first or last
              coordinates to reconstruct the trajectory. The values are picked
              up at the index specified at ``--info_index``. You can also
              get some labels.

              Usage:

              .. code-block:: console

                python main.py --info_features latitude longitude

                python main.py --info_features label

            * ``--info_index``: Index of information features. Default to None.

        Args:
            parser (ArgumentParser): ArgumentParser to update.

        Returns:
            ArgumentParser: updated ArgumentParser with TrafficDataset
            arguments.
        """
        p = parser.add_argument_group("TrafficDataset")
        p.add_argument(
            "--data_path",
            dest="data_path",
            type=Path,
            default=None,
        )
        p.add_argument(
            "--features",
            dest="features",
            nargs="+",
            default=["latitude", "longitude", "altitude", "timedelta"],
        )
        p.add_argument(
            "--info_features",
            dest="info_features",
            nargs="+",
            default=[],
        )
        p.add_argument(
            "--info_index",
            dest="info_index",
            type=int,
            default=None,
        )
        return parser


def extract_features(
    traffic: Traffic,
    features: List[str],
    init_features: List[str] = [],
) -> np.ndarray:
    """Extract features from Traffic data according to the feature list.

    Parameters
    ----------
    traffic: Traffic
    features: List[str]
        Labels of the columns to extract from the underlying dataframe of
        Traffic object.
    init_features: List[str]
        Labels of the features to extract from the first row of each Flight
        underlying dataframe.
    Returns
    -------
    np.ndarray
        Feature vector `(N, HxL)` with `N` number of flights, `H` the number
        of features and `L` the sequence length.
    """
    X = np.stack(list(f.data[features].values.ravel() for f in traffic))

    if len(init_features) > 0:
        init_ = np.stack(list(f.data[init_features].iloc[0].values.ravel() for f in traffic))
        X = np.concatenate((init_, X), axis=1)

    return X


def init_dataframe(
    data: np.ndarray,
    features: List[str],
    init_features: List[str] = [],
) -> pd.DataFrame:
    """TODO:"""
    # handle dense features (features)
    dense: np.ndarray = data[:, len(init_features) :]
    nb_samples = data.shape[0]
    dense = dense.reshape(nb_samples, -1, len(features))
    nb_obs = dense.shape[1]
    # handle sparce features (init_features)
    if len(init_features) > 0:
        sparce = data[:, : len(init_features)]
        sparce = sparce[:, np.newaxis]
        sparce = np.insert(sparce, [1] * (nb_obs - 1), [np.nan] * len(init_features), axis=1)
        dense = np.concatenate((dense, sparce), axis=2)
        features = features + init_features

    # generate dataframe
    df = pd.DataFrame({feature: dense[:, :, i].ravel() for i, feature in enumerate(features)})
    return df


def traffic_from_data(
    data: np.ndarray,
    features: List[str],
    init_features: List[str] = [],
    builder: Optional[BuilderProtocol] = None,
) -> Traffic:

    df = init_dataframe(data, features, init_features)

    if builder is not None:
        df = builder(df)

    return Traffic(df)
