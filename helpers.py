from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TrainTestSplit:
    X_train: Any
    X_test: Any
    y_train: Any
    y_test: Any