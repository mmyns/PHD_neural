
from datetime import datetime
from typing import List
import json
from dataclasses import dataclass, InitVar, field, asdict


@dataclass
class PytorchSettings:

    filename: str =  "/esat/biomeddata/mmaeyens/logs/Experiment:" + datetime.now().strftime("%Y%m%d-%H%M%S")
    selected_labels: List[str] = field(default_factory=list("All"))
    frames: List[int] = None

    batch_size: int = 16
    label_smoothing: float = 0
    epochs: int = 10
    split_number: int = 0
    dropout_rate: float = 0
    lr: float = 0.001
    images: str = "normal"

    model: str = "visiontransformer"
    activation: str = "Binary"
    input_shape: tuple = (1012,1356,3)

    augment_N: int = 2
    augment_M: int = 3


    samples: int = 0


    @classmethod
    def load(cls,filename):
        with open(filename) as json_data_file:
            values = json.load(json_data_file)
        return PytorchSettings(**values)

    def save(self,filename):
        with open(filename + "_Settings", 'w') as fp:
            json.dump(asdict(self), fp)


