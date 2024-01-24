from dataclasses import dataclass

@dataclass
class MainConfig:
    experiment_name: str
    


@dataclass
class DataConfig:
    dataset: str
    

class DataModule:
    pass