
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict


class PredictionRequest(BaseModel):

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False
    )

    Type: Literal["L", "M", "H"]

    air_temperature: float = Field(gt=0)

    process_temperature: float = Field(gt=0)

    rotational_speed: int = Field(ge=0)

    torque: float = Field(ge=0)

    tool_wear: int = Field(ge=0)

    def to_features(self):

        return {
            "Type": self.Type,
            "Air temperature [K]": self.air_temperature,
            "Process temperature [K]": self.process_temperature,
            "Rotational speed [rpm]": self.rotational_speed,
            "Torque [Nm]": self.torque,
            "Tool wear [min]": self.tool_wear,
        }


class PredictionResponse(BaseModel):

    prediction_id: int

    prediction: Literal[0, 1]

    failure_probability: float

    model_name: str

    model_version: str


class FeedbackRequest(BaseModel):

    prediction_id: int = Field(gt=0)

    actual_label: Literal[0, 1]
