import enum

class Event(enum.Enum):
    INFERENCE_START = 0
    INFERENCE_END = 1
    TRAINING_START = 2
    TRAINING_END = 3
    BACKWARD_START = 4
    BACKWARD_END = 5