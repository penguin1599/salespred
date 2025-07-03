import pickle
import numpy as np
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load model on startup
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# ---------- HTML form routes ----------
@app.get("/", response_class=HTMLResponse)
async def show_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict_form(
    request: Request,
    TV: float = Form(...),
    Radio: float = Form(...),
    Newspaper: float = Form(...),
):
    features = np.array([[TV, Radio, Newspaper]])
    prediction = model.predict(features)[0]
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": f"Predicted Sales: {prediction:.2f}"},
    )

# ---------- JSON API route ----------
class InputData(BaseModel):
    TV: float
    Radio: float
    Newspaper: float

@app.post("/predict/")
async def predict_json(data: InputData):
    features = np.array([[data.TV, data.Radio, data.Newspaper]])
    pred = model.predict(features)[0]
    return {"predicted_sales": float(pred)}
