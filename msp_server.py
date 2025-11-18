from flask import Flask, jsonify, request
import requests
import os
from dotenv import load_dotenv

# Load the environment variables from .env file (API key should be in this file)
load_dotenv()

app = Flask(__name__)

# Simple landing route so root URL does not 404
@app.route("/", methods=["GET"])
def index():
    return jsonify(
        {
            "status": "ok",
            "message": "Use /weather?location=City to fetch current weather.",
        }
    )

# Silence favicon requests (common from browsers)
@app.route("/favicon.ico")
def favicon():
    return ("", 204)

# Define a route to query external weather data
@app.route('/weather', methods=['GET'])
def get_weather():
    location = request.args.get('location', 'Abu Dhabi')  # Default to Abu Dhabi if no location is provided
    if not location:
        return jsonify({"error": "Location parameter is required"}), 400
    
    # Call external weather API
    weather_data = get_weather_data_from_api(location)
    return jsonify(weather_data)

# Helper function to fetch weather data from the API
def get_weather_data_from_api(location):
    API_KEY = os.getenv("OPENWEATHER_API_KEY")  # Load API Key from the .env file
    if not API_KEY:
        return {"error": "OPENWEATHER_API_KEY is not set in the environment"}

    url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()  # Return the weather data
        return {"error": f"Failed to fetch data: {response.status_code}"}
    except requests.RequestException as exc:
        return {"error": f"Request failed: {exc}"}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Start the server on port 5000
