import os
import json
import joblib
import numpy as np
import requests
import sqlite3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
from datetime import datetime

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
        print("Giá trị data", data)

        temperature_kelvin = data["main"]["temp"]
        humidity = data["main"]["humidity"]

        wind = data.get("wind", {})
        rain = data.get("rain", {})

        result = {
            "temperature": temperature_kelvin,
            "humidity": humidity
        }
        if wind:
            result["wind"] = wind
        if rain:
            result["rain"] = rain
        return result
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error fetching weather data: {str(e)}")

def kelvin_to_celsius(kelvin):
    return kelvin - 273.15

def format_two_decimals(value):
    return round(value, 2)

def insert_weather_data(temp, humidity):
    try:
        conn = sqlite3.connect('dataweather.db')
        cursor = conn.cursor()

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        cursor.execute("""
            INSERT INTO weather_data (temp, humidity, timestamp)
            VALUES (?, ?, ?)
        """, (temp, humidity, timestamp))

        conn.commit()
        conn.close()

    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Error inserting data into database: {str(e)}")

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
        temperature = format_two_decimals(request.temperature)
        humidity = format_two_decimals(request.humidity)

        new_data = np.array([[temperature, humidity, 0, 0]])  
        new_data_scaled = scaler.transform(new_data)

        prediction = model.predict(new_data_scaled)

        predicted_temperature = format_two_decimals(prediction[0][0])
        predicted_humidity = format_two_decimals(prediction[0][1])

        return {
            "predicted_temperature": predicted_temperature,
            "predicted_humidity": predicted_humidity
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during prediction: {str(e)}")
@app.post("/predict/location")
async def predict_location(request: LocationRequest):
    try:
        location = request.location
        
        weather_data = get_weather_data(location)
        temperature_kelvin = weather_data.get("temperature")
        humidity = weather_data.get("humidity")
        wind = weather_data.get("wind", "Không có dữ liệu")  
        rain = weather_data.get("rain", "Không có dữ liệu")  

        temperature_celsius = kelvin_to_celsius(temperature_kelvin)
        temperature_celsius = format_two_decimals(temperature_celsius)
        humidity = format_two_decimals(humidity)

        insert_weather_data(temperature_celsius, humidity)

        new_data = np.array([[temperature_celsius, humidity, 0, 0]])  
        new_data_scaled = scaler.transform(new_data)

        prediction = model.predict(new_data_scaled)
        print("Giá trị predict", prediction)

        predicted_temperature = format_two_decimals(prediction[0][0])
        predicted_humidity = format_two_decimals(prediction[0][1])

        response = {
            "predicted_temperature": predicted_temperature,
            "predicted_humidity": predicted_humidity,
            "actual_temperature": temperature_celsius,
            "actual_humidity": humidity,
            "wind": wind,  
            "rain": rain   
        }

        return response

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during prediction: {str(e)}")


@app.get("/get_data")
async def get_data():
    try:
        records = get_last_10_records()
        result = [{"temperature": format_two_decimals(record[0]), "humidity": format_two_decimals(record[1]), "timestamp": record[2]} for record in records]
        return {"data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data from database: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port="8000", debug=True)
