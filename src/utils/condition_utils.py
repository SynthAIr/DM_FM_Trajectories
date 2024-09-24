import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import abc
from typing import Dict, Any, List
from enum import Enum
import pandas as pd

#WIND_DIRECTIONS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
WIND_DIRECTIONS = ['N', 'E', 'S',  'W']

class Condition(abc.ABC):
    """
    Abstract class for conditions
    """
    def __init__(self, label):
        self.label = label

    @abc.abstractmethod
    def to_tensor(self, data, n_repeat = 1) -> torch.Tensor:
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

class ContinuousConditions(Condition):
    """
    Class for continuous conditions (single value)
    """
    def __init__(self, label):
        super().__init__(label)
    
    def to_tensor(self, data, n_repeat = 31) -> torch.Tensor:
        return torch.tensor([data]*n_repeat, dtype=torch.float)

    def get_type(self) -> str:
        return "continuous"

    def get_feature_names(self) -> List[str]:
        return [self.label]


class CyclicalConditions(Condition):
    """
    Class for cyclical conditions represented as sin and cos
    """

    def __init__(self, label, max_val, bins=None, labels = None):
        super().__init__(label)
        self.max_val = max_val
        self.bins = bins
        self.labels = labels


    def _cyclic_to_tensor(self, data, n_repeat = 31) -> torch.Tensor:
        if self.labels is not None:
            index = self.labels.index(data)
            data = index
        
        if self.bins is not None:
            sin = np.sin(2*np.pi * self.bins[data] / self.max_val)
            cos = np.cos(2*np.pi * self.bins[data] / self.max_val)

        sin = np.sin(2*np.pi * data / self.max_val)
        cos = np.cos(2*np.pi * data / self.max_val)
        return torch.tensor([sin, cos]*n_repeat, dtype=torch.float)
    
    def to_tensor(self, data, n_repeat = 31) -> torch.Tensor:
        return self._cyclic_to_tensor(data, n_repeat)

    def get_type(self) -> str:
        return "cyclical"

    def get_feature_names(self) -> List[str]:
        return [self.label + "_sin", self.label + "_cos"]

class OneHotConditions(Condition):
    """
    Class for one hot conditions
    Main idea being the label is the int value of the category from preprocessing

    Category col is the column name of the category as string values
    Values are are the unique values of the category that are sorted which is used to get the index of the category

    """
    def __init__(self, label: str, category_col: str, values = None):
        super().__init__(label)
        self.category_col = category_col
        self.values = values
        if values is not None:
            self.values = sorted(values)
            self.n_categories = len(self.values)

    def init_categories(self, values: List[str]):
        self.values = sorted(values)
        self.n_categories = len(self.values)
    
    def to_tensor(self, data, n_repeat = 1) -> torch.Tensor:
        """
        Convert the data to a tensor
        Supports both using the index of the category directly or the string value
        """
        if self.values is None:
            raise ValueError("Values not set for one hot condition")
        
        if type(data) == str or issubclass(type(data), Enum):
            data = self.values.index(data)

        return F.one_hot(torch.tensor(data, dtype=torch.int64), num_classes=self.n_categories)

    def get_type(self) -> str:
        return "one_hot"

    def get_feature_names(self) -> List[str]:
        return [self.label]

class EmbeddingConditions(Condition):
    """
    Class for embedding conditions
    """
    def __init__(self, label, n_categories, embedding_dim):
        super().__init__(label)
        self.n_categories = n_categories
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(n_categories, embedding_dim)
    
    def to_tensor(self, data, n_repeat = 1) -> torch.Tensor:
        return self.embedding(data)

    def get_type(self) -> str:
        return "embedding"

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
        if type == "cyclical":
            self.condition_map[label] = CyclicalConditions(label, max_val, bins, labels)
        elif type == "continuous":
            self.condition_map[label] = ContinuousConditions(label)
        elif type == "embedding":
            self.condition_map[label] = EmbeddingConditions(label, n_categories, embedding_dim)
        #elif type == "one_hot":
            #self.condition_map[label] = OneHotConditions(label, )

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

def get_cond_len(conditional_features: List[Condition], seq_len = 31) -> int:
    """
    Get the amount single features from the conditional features
    """
    cond_len = 0
    for cf in conditional_features:
        match cf.get_type():
            case "continuous":
                cond_len += 1 * seq_len
            case "cyclical":
                cond_len += 2 * seq_len
            case "embedding":
                cond_len += cf.embedding_dim
            case "one_hot":
                cond_len += cf.n_categories
            case _:
                NotImplementedError("Condition type not implemented")
    return cond_len

def load_conditions(config, dataset: pd.DataFrame = None) -> List[Condition]:
    """
    Load conditions from a config file
    """
    condtions = []

    for name in config["conditional_features"].keys():

        condition = config["conditional_features"][name]
        match condition["type"]:
            case "continuous":
                condtions.append(ContinuousConditions(name))
            case "cyclical":
                bins = condition.get("bins", None)
                labels = condition.get("labels", None)
                condtions.append(CyclicalConditions(name, condition["max_value"], bins, labels))
            case "embedding":
                condtions.append(EmbeddingConditions(name, condition["n_categories"], condition["embedding_dim"]))
            case "one_hot":
                categories_col = condition.get("category_col", None)
                if categories_col is not None:
                    categories = list(set([f.data[categories_col].values[0] for f in dataset.traffic])) if dataset is not None else None
                    condtions.append(OneHotConditions(name, categories_col, categories))
                else:
                    condtions.append(OneHotConditions(name, None))
            case _:
                NotImplementedError("Condition type not implemented")
        


    return condtions


