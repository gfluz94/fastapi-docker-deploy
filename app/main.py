import dill
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.requests import Request

app = FastAPI(title="Predicting Wine Class")

class Wine(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float

@app.on_event("startup")
def load_clf():
    with open("/app/models/wine.pkl", "rb") as file:
        global clf
        clf = dill.load(file)

@app.get("/")
def home():
    return "Home"

@app.post("/predict")
def predict(request: Request, wine: Wine):
    if request.method == "POST":
        data_point = np.array(
            [
                [
                    wine.alcohol,
                    wine.malic_acid,
                    wine.ash,
                    wine.alcalinity_of_ash,
                    wine.magnesium,
                    wine.total_phenols,
                    wine.flavanoids,
                    wine.nonflavanoid_phenols,
                    wine.proanthocyanins,
                    wine.color_intensity,
                    wine.hue,
                    wine.od280_od315_of_diluted_wines,
                    wine.proline,
                ]
            ]
        )

        pred = clf.predict(data_point).tolist()
        pred = pred[0]
        score = clf.predict_proba(data_point).max()
        return {"Prediction": pred, "Confidence": score}
    return "No post request found"