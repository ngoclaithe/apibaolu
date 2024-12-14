import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

df = pd.read_csv('weather_data.csv')
print(df.head())

X = df[['Outdoor Drybulb Temperature [C]', 'Outdoor Relative Humidity [%]', 
        'Diffuse Solar Radiation [W/m2]', 'Direct Solar Radiation [W/m2]']]
y = df[['6h Prediction Outdoor Drybulb Temperature [C]', '6h Prediction Outdoor Relative Humidity [%]']]  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = MultiOutputRegressor(LinearRegression())
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
print("Mean Squared Error (MSE):", mse)

def save_model_and_scaler(model, scaler, model_filename='weather_prediction_model.pkl', scaler_filename='scaler.pkl'):
    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)
    print(f"Model đã được lưu vào file '{model_filename}'")
    print(f"Scaler đã được lưu vào file '{scaler_filename}'")

save_model_and_scaler(model, scaler)

new_data = [[17.81, 68.12, 0, 0]] 
new_data = scaler.transform(new_data)

prediction = model.predict(new_data)
print("Dự đoán sau 6 giờ (Nhiệt độ, Độ ẩm):", prediction)
