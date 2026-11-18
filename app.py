
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Milk Batch Prediction")

model = joblib.load("models/milk_model.pkl")

class MilkData(BaseModel):
    milk_quantity: float = Field(gt=0)
    collection_temperature: float = Field(ge=-1, le=40)
    transport_duration: float = Field(gt=0, le=24)
    cooling_delay: float = Field(ge=0, le=24)
    milk_fat: float = Field(gt=0, le=15)
    acidity: float = Field(gt=0, le=1)
    container_type: str
    ambient_temperature: float = Field(ge=-10, le=60)

@app.post("/predict")
def predict(data: MilkData):
    values = data.model_dump()

    values["temperature_exposure_index"] = (
        values["collection_temperature"] * values["cooling_delay"]
    )

    values["transport_delay_risk"] = (
        values["transport_duration"] * values["ambient_temperature"]
    )

    prediction = model.predict(pd.DataFrame([values]))[0]

    return {
        "prediction": "Rejection Risk" if prediction == "Yes"
        else "Likely Accepted"
    }
