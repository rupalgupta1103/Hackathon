import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./mess_optimization.db")
    
    # Model paths
    MODEL_PATH = os.getenv("MODEL_PATH", "models/demand_model.pkl")
    SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.pkl")
    
    # Hostel configuration
    TOTAL_STUDENTS = int(os.getenv("TOTAL_STUDENTS", 2000))
    MEAL_TYPES = ["breakfast", "lunch", "snacks", "dinner"]
    
    # Cost parameters (INR)
    AVG_MEAL_COST = float(os.getenv("AVG_MEAL_COST", 50))
    WASTE_COST_PER_KG = float(os.getenv("WASTE_COST_PER_KG", 30))
    
    # Environmental factors
    CO2_PER_KG_FOOD = float(os.getenv("CO2_PER_KG_FOOD", 2.5))  # kg CO2 per kg food
    WATER_PER_KG_FOOD = float(os.getenv("WATER_PER_KG_FOOD", 1000))  # liters per kg food
    
    # API Keys
    WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")