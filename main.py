import os
import json
import joblib
import numpy as np
import requests
import sqlite3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler

model = joblib.load('weather_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')

app = FastAPI()

API_KEY = "0a3a0055b9c2edebded1d21e92397c87"  

class PredictionRequest(BaseModel):
    temperature: float
    humidity: float

class LocationRequest(BaseModel):
    location: str  

def get_weather_data(location: str):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={API_KEY}"
        response = requests.get(url)
        response.raise_for_status()  

        data = response.json()

        temperature = data["main"]["temp"]
        humidity = data["main"]["humidity"]

        return temperature, humidity
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error fetching weather data: {str(e)}")

def get_last_10_records():
    try:
        conn = sqlite3.connect('dataweather.db')
        cursor = conn.cursor()

        cursor.execute("SELECT temp, humidity, timestamp FROM weather_data ORDER BY timestamp DESC LIMIT 10")
        records = cursor.fetchall()

        conn.close()

        return records
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Error accessing database: {str(e)}")

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        temperature = request.temperature
        humidity = request.humidity

        new_data = np.array([[temperature, humidity, 0, 0]])  
        new_data_scaled = scaler.transform(new_data)

        prediction = model.predict(new_data_scaled)

        return {"predicted_temperature": prediction[0], "predicted_humidity": humidity}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during prediction: {str(e)}")

@app.post("/predict/location")
async def predict_location(request: LocationRequest):
    try:
        location = request.location
        
        temperature, humidity = get_weather_data(location)

        new_data = np.array([[temperature, humidity, 0, 0]])  
        new_data_scaled = scaler.transform(new_data)

        prediction = model.predict(new_data_scaled)

        return {"predicted_temperature": prediction[0], "predicted_humidity": humidity}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during prediction: {str(e)}")

@app.get("/get_data")
async def get_data():
    try:
        records = get_last_10_records()
        result = [{"temperature": record[0], "humidity": record[1], "timestamp": record[2]} for record in records]
        return {"data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data from database: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, debug=True)

