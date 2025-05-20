"""
Code is adapted from https://github.com/SynthAIr/SynTraj, who adapted it from https://github.com/kruuZHAW/deep-traffic-generation-paper
"""
import logging
import os
from pathlib import Path
from typing import Any, List, Optional, Protocol, Tuple, TypedDict, Union
import joblib
import utm
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from torch.utils.data import Dataset
from traffic.core import Traffic
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from .condition_utils import Condition
from .weather_utils import load_weather_data, load_weather_data_function, load_weather_data_arrival_airport
from .metar_utils import load_metar_data


logger = logging.getLogger(__name__)


class TransformerProtocol(Protocol):
    """
    Protocol for a transformer that can be fitted and applied to data.
    """
    def fit(self, X: np.ndarray) -> "TransformerProtocol": ...

    def fit_transform(self, X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]: ...

    def transform(self, X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]: ...

    def inverse_transform(self, X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]: ...


class DatasetParams(TypedDict):
    """
    Dataset parameters
    """
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
    """
    Class is adapted from https://github.com/SynthAIr/SynTraj, who adapted it from https://github.com/kruuZHAW/deep-traffic-generation-paper
    Traffic Dataset
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
         
        if "lat_original" in traffic.data.columns:
            print("Initing refs as original")
            self.lat_original = np.stack(list(f.data['lat_original'].values for f in traffic)).reshape(-1, 200)
            self.lon_original = np.stack(list(f.data['lon_original'].values for f in traffic)).reshape(-1, 200)
        if "lat_ref" in traffic.data.columns:
            print("Initing refs as refs")
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
            """
            Save the scaler to the specified path if it doesn't already exist.
            Parameters
            ----------
            scaler
            path

            Returns
            -------

            """
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
                """
                Scale the conditions using StandardScaler or MinMaxScaler.
                Parameters
                ----------
                conditions_fs
                axis

                Returns
                -------

                """

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
                
                #s = load_scaler_if_exists("data/scaler.gz")
                s = None
                if s is None:
                    #s = MinMaxScaler(feature_range=(-1, 1))
                    s = StandardScaler()
                    s.fit(conditions)
                
                print("Nan in conditions pre transform", np.isnan(conditions).any())
                conditions = s.transform(conditions)
                conditions = torch.FloatTensor(conditions)
                print("Nan in conditions post transform", torch.isnan(conditions).any())

                if torch.isnan(conditions).any().item():
                    print("aborting, found nan in cond")
                    exit()

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
            #save_scaler_if_not_exists(self.con_cond_scaler,"data/scaler.gz")

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
        
        data = torch.FloatTensor(data)
        self.data = data
        if self.shape in ["sequence", "image"]:
            self.data = self.data.view(self.data.size(0), -1, len(self.features))
            #self.conditions = self.conditions.view(self.conditions.size(0), -1, len(condition_feature_names))
            if self.shape == "image":
                self.data = torch.transpose(self.data, 1, 2)
                #self.conditions = torch.transpose(self.conditions, 1, 2)


    def _get_conditions(self, traffic: Traffic) -> List[torch.Tensor]:
        """
        Private method for getting conditions from traffic data to pytorch tensors
        Parameters
        ----------
        traffic

        Returns
        -------

        """
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
                print("Data contains nan? ",torch.isnan(feature_data).any())
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
        """
        Inverse the airport coordinates to lat/lon coordinates
        Parameters
        ----------
        data
        idx

        Returns
        -------

        """
        return data
        shape = data.shape
        data = data.reshape(-1, 200, len(self.features))
        easting_ref, northing_ref, _, _= utm.from_latlon(self.lat_refs[idx], self.lon_refs[idx])
        # Compute absolute easting/northing
        easting = data[:, :, 0]
        northing = data[:, :, 1]

        _, _, zone_number, zone_letter = utm.from_latlon(0, 0)

        lat, lon = utm.to_latlon(easting, northing, zone_number, zone_letter, strict = False)

        data[:, :, 0] = lat
        data[:, :, 1] = lon

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
        """
        Create a TrafficDataset from a file.
        Parameters
        ----------
        file_path
        features
        shape
        scaler
        conditional_features
        variables
        metar

        Returns
        -------

        """
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
        if self.scaler is not None:
            body += [repr(self.scaler)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

