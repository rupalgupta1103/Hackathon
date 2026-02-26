from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class MealConsumption(Base):
    __tablename__ = "meal_consumption"
    
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, default=datetime.utcnow)
    meal_type = Column(String)  # breakfast, lunch, dinner
    dish_name = Column(String)
    quantity_prepared = Column(Float)  # kg
    quantity_consumed = Column(Float)  # kg
    quantity_wasted = Column(Float)  # kg
    student_count = Column(Integer)
    day_of_week = Column(Integer)
    is_weekend = Column(Boolean)
    is_holiday = Column(Boolean)
    temperature = Column(Float)
    rainfall = Column(Float)
    special_event = Column(String, nullable=True)

class MenuItem(Base):
    __tablename__ = "menu_items"
    
    id = Column(Integer, primary_key=True)
    dish_name = Column(String)
    category = Column(String)  # main_course, bread, dessert, etc.
    ingredients = Column(String)  # JSON string of ingredients
    base_price = Column(Float)
    preparation_time = Column(Integer)  # minutes
    popularity_score = Column(Float, default=0.0)
    sustainability_score = Column(Float, default=0.0)  # eco-friendly rating

class Procurement(Base):
    __tablename__ = "procurement"
    
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, default=datetime.utcnow)
    ingredient_name = Column(String)
    quantity = Column(Float)  # kg
    unit_price = Column(Float)
    total_cost = Column(Float)
    vendor = Column(String)
    predicted_quantity = Column(Float)
    actual_used = Column(Float, nullable=True)

class WasteLog(Base):
    __tablename__ = "waste_log"
    
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, default=datetime.utcnow)
    meal_type = Column(String)
    dish_name = Column(String)
    waste_quantity = Column(Float)  # kg
    waste_reason = Column(String)  # overproduction, quality, etc.
    co2_footprint = Column(Float)
    water_footprint = Column(Float)
    cost_loss = Column(Float)