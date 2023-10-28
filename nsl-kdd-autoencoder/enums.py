from enum import Enum


class ClassifierMode(Enum):
    BINARY = 1
    MULTI_CLASS = 2


class ModelMode(Enum):
    AUTOENCODER = 1
    CLASSIFIER = 2
