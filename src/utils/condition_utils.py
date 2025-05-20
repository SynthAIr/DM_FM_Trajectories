"""
Code is adapted from https://github.com/SynthAIr/SynTraj
"""
import torch
import numpy as np
import abc
from typing import Dict, Any, List
import pandas as pd

class Condition(abc.ABC):
    """
    Abstract class for conditions
    """
    def __init__(self, label):
        self.label = label

    @abc.abstractmethod
    def to_tensor(self, data) -> torch.Tensor:
        """
        Convert the data to a tensor
        """
        pass

    @abc.abstractmethod
    def get_type(self) -> str:
        """
        Get the type of the condition
        """
        return "condition"

    @abc.abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Get the feature names
        """
        return [self.label]

class ContinuousCondition(Condition):
    """
    Class for continuous conditions (single value)
    """
    def __init__(self, label, first = False, last = False):
        super().__init__(label)
        self.first = first
        self.last = last
    
    def to_tensor(self, data) -> torch.Tensor:
        if self.first:
            data = data[:, 0]
            data = data.reshape(-1, 1)
        if self.last:
            data = data[:, -1]
            data = data.reshape(-1, 1)
        if self.first and self.last:
            data = data[:, [0, -1]]

        data = torch.tensor(data, dtype=torch.float)

        if len(data.shape) == 1:
            return data.reshape(-1, 1)

        return data

    def get_type(self) -> str:
        return "continuous"

    def get_feature_names(self) -> List[str]:
        return [self.label]


class CyclicCondition(Condition):
    """
    Class for cyclical conditions represented as sin and cos
    """

    def __init__(self, label, max_val, bins=None, labels = None):
        super().__init__(label)
        self.max_val = max_val
        self.bins = bins
        self.labels = labels
        self.single = True


    def _cyclic_to_tensor(self, data) -> torch.Tensor:
        if self.labels is not None:
            index = self.labels.index(data)
            data = index
        
        if self.bins is not None:
            sin = np.sin(2*np.pi * self.bins[data] / self.max_val)
            cos = np.cos(2*np.pi * self.bins[data] / self.max_val)

        if self.single:
            data = data[:, 0]

        sin = np.sin(2*np.pi * data / self.max_val)
        cos = np.cos(2*np.pi * data / self.max_val)
        return torch.tensor(np.array([sin, cos]), dtype=torch.float).T
    
    def to_tensor(self, data) -> torch.Tensor:
        return self._cyclic_to_tensor(data)

    def get_type(self) -> str:
        return "cyclic"

    def get_feature_names(self) -> List[str]:
        return [self.label]
        #return [self.label + "_sin", self.label + "_cos"]


class CategoricalCondition(Condition):
    """
    Class for embedding conditions
    """
    def __init__(self, label, categories = []):
        super().__init__(label) 
        self.categories = categories
        self.to_index = {c: i for i, c in enumerate(categories)}
        #self.to_index = {"EHAM" : 0, "LIMC" : 1, "LFPG":2, "EGKK":3, "LIRF": 4, "LOWW":5, "EGLL":6, "ESSA": 7, "EDDF": 8, "EDDT": 9}

    def to_tensor(self, data) -> torch.Tensor:
        data = data[:,0]
        data = [self.to_index[d] for d in data]
        data = torch.tensor([data], dtype=torch.int)
        return data.reshape(-1, 1)

    def get_type(self) -> str:
        return "categorical"

    def get_feature_names(self) -> List[str]:
        return [self.label]

class WeatherGridCondition(Condition):
    """
    Class for weather grid conditions
    """
    def __init__(self, label, levels = 12):
        super().__init__(label)
        self.levels = levels
    
    def to_tensor(self, data) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float)

    def get_type(self) -> str:
        return "weather_grid"

    def get_feature_names(self) -> List[str]:
        return [self.label]


class ConditionHandler:
    """
    Class for handling conditions and converting them to tensors
    """
    def __init__(self, conditions = None, scaler = None):
        self.condition_map = {}
        self.scaler = scaler
        if conditions is not None:
            self.condition_map = {c.label: c for c in conditions}
    
    def add_condition(self, label, type, max_val=0, bins=None, labels=None, n_categories=None, embedding_dim=None):
        """
        Add a condition to the handler
        """
        if type == "cyclic":
            self.condition_map[label] = CyclicCondition(label, max_val, bins, labels)
        elif type == "continuous":
            self.condition_map[label] = ContinuousCondition(label)
        elif type == "categorical":
            self.condition_map[label] = CategoricalCondition(label)

    def process(self, name, data) -> torch.Tensor:
        """
        Process a single condition
        """
        if name in self.condition_map:
            return self.condition_map[name].to_tensor(data)
        else:
            raise ValueError(f"Condition {name} not found")
    
    def process_all(self, data: Dict[str, Any], feature_order: List[Condition] = None) -> torch.Tensor:
        """
        Process all conditions
        """
        all_conditions = []
        if not self.condition_map:
            return torch.empty(0)
        
        feature_order = [f.label for f in feature_order] if feature_order is not None else list(data.keys())

        for label in feature_order:
            all_conditions.append(self.process(label, data[label]))

        all_conditions = torch.cat(all_conditions, dim=0).reshape(1, -1)
        if self.scaler is not None:
            all_conditions = self.scaler.transform(all_conditions)

        return torch.FloatTensor(all_conditions)


def load_conditions(config, dataset: pd.DataFrame = None) -> List[Condition]:
    """
    Load conditions from a config file
    """
    condtions = []

    for name in config["conditional_features"].keys():

        condition = config["conditional_features"][name]
        match condition["type"]:
            case "continuous":
                first = condition.get("first", False)
                last = condition.get("last", False)
                condtions.append(ContinuousCondition(name, first, last))
            case "cyclic":
                bins = condition.get("bins", None)
                labels = condition.get("labels", None)
                condtions.append(CyclicCondition(name, condition["max_value"], bins, labels))
            case "categorical":
                condtions.append(CategoricalCondition(name, condition["categories"]))
            case _:
                NotImplementedError("Condition type not implemented")
        


    return condtions


