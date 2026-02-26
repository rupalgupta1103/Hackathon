from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List
import uvicorn
from datetime import datetime

from app.database.database import get_db, init_db
from app.database import models
from app.api import schemas, endpoints
from app.ml_models.demand_predictor import DemandPredictor
from app.optimizer.procurement_optimizer import ProcurementOptimizer
from app.sustainability.metrics_calculator import SustainabilityMetrics
from app.config import Config

# Initialize FastAPI app
app = FastAPI(
    title="Smart Mess Optimization System",
    description="AI-Driven Food Demand Prediction & Dynamic Meal Planning Engine",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database on startup
@app.on_event("startup")
def startup_event():
    init_db()

# Initialize ML models
predictor = DemandPredictor(model_type='xgboost')
optimizer = ProcurementOptimizer()
sustainability = SustainabilityMetrics(Config)

# Health check endpoint
@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "message": "Smart Mess Optimization System API",
        "version": "1.0.0"
    }

# Demand Prediction Endpoints
@app.post("/api/predict/demand", response_model=schemas.DemandPredictionResponse)
async def predict_demand(
    request: schemas.DemandPredictionRequest,
    db: Session = Depends(get_db)
):
    """Predict demand for next day"""
    try:
        # Get historical data
        historical_data = db.query(models.MealConsumption).all()
        
        # Prepare features
        features = {
            'date': request.date,
            'day_of_week': request.date.weekday(),
            'temperature': request.temperature,
            'rainfall': request.rainfall,
            'is_weekend': request.date.weekday() >= 5,
            'is_holiday': request.is_holiday,
            'is_exam': request.is_exam,
            'special_event': request.special_event
        }
        
        # Get meal types and dishes
        meal_types = ["breakfast", "lunch", "dinner"]
        dishes = ["rice", "chapati", "dal", "sabzi", "curry"]
        
        # Make predictions
        predictions = predictor.predict_next_day(features, meal_types, dishes)
        
        return schemas.DemandPredictionResponse(
            date=request.date,
            predictions=predictions,
            confidence=0.85,  # Could calculate from model
            model_used="XGBoost"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Procurement Optimization Endpoint
@app.post("/api/optimize/procurement")
async def optimize_procurement(
    request: schemas.ProcurementRequest
):
    """Optimize ingredient procurement"""
    try:
        # Recipe database (simplified for demo)
        recipe_db = {
            "rice": {"rice": 0.1, "water": 0.2},
            "dal": {"dal": 0.05, "water": 0.1, "spices": 0.01},
            "chapati": {"wheat": 0.05, "water": 0.02, "oil": 0.005},
            "sabzi": {"vegetables": 0.1, "oil": 0.01, "spices": 0.005},
            "curry": {"vegetables": 0.08, "spices": 0.01, "oil": 0.01}
        }
        
        # Ingredient prices
        prices = {
            "rice": 40, "wheat": 30, "dal": 80, "vegetables": 30,
            "oil": 100, "spices": 200, "water": 0.1
        }
        
        # Run optimization
        result = optimizer.optimize_procurement(
            predicted_demand=request.predictions,
            ingredient_prices=prices,
            recipe_database=recipe_db,
            budget_constraint=request.budget_constraint
        )
        
        # Get recommendations
        recommendations = optimizer.get_procurement_recommendations(
            result['optimal_quantities'],
            request.current_stock
        )
        
        return {
            **result,
            'recommendations': recommendations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Waste Monitoring Endpoint
@app.post("/api/waste/log")
async def log_waste(
    waste_data: schemas.WasteLogRequest,
    db: Session = Depends(get_db)
):
    """Log waste data"""
    try:
        # Calculate environmental impact
        co2_footprint = waste_data.waste_quantity * Config.CO2_PER_KG_FOOD
        water_footprint = waste_data.waste_quantity * Config.WATER_PER_KG_FOOD
        cost_loss = waste_data.waste_quantity * Config.WASTE_COST_PER_KG
        
        # Create waste log
        waste_log = models.WasteLog(
            date=datetime.utcnow(),
            meal_type=waste_data.meal_type,
            dish_name=waste_data.dish_name,
            waste_quantity=waste_data.waste_quantity,
            waste_reason=waste_data.waste_reason,
            co2_footprint=co2_footprint,
            water_footprint=water_footprint,
            cost_loss=cost_loss
        )
        
        db.add(waste_log)
        db.commit()
        
        return {
            "message": "Waste logged successfully",
            "id": waste_log.id,
            "co2_footprint": co2_footprint,
            "water_footprint": water_footprint,
            "cost_loss": cost_loss
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Sustainability Metrics Endpoint
@app.get("/api/sustainability/metrics")
async def get_sustainability_metrics(
    days: int = 30,
    db: Session = Depends(get_db)
):
    """Get sustainability metrics"""
    try:
        # Get waste logs from last N days
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        waste_logs = db.query(models.WasteLog).filter(
            models.WasteLog.date >= cutoff_date
        ).all()
        
        # Convert to dict for metrics calculator
        waste_logs_dict = [
            {
                'waste_quantity': log.waste_quantity,
                'meal_type': log.meal_type,
                'waste_reason': log.waste_reason,
                'cost_loss': log.cost_loss
            }
            for log in waste_logs
        ]
        
        # Calculate metrics
        metrics = sustainability.calculate_waste_metrics(waste_logs_dict, days)
        
        # Calculate sustainability score
        score = sustainability.calculate_sustainability_score(
            metrics,
            procurement_efficiency=0.75,  # Example value
            renewable_percentage=0.3
        )
        
        # Generate report
        report = sustainability.generate_impact_report(metrics)
        
        return {
            'metrics': metrics,
            'sustainability_score': score,
            'report': report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Menu Optimization Endpoint
@app.post("/api/menu/optimize")
async def optimize_menu(
    request: schemas.MenuOptimizationRequest
):
    """Optimize menu based on popularity and waste trends"""
    try:
        # Calculate popularity scores (simplified)
        dish_popularity = {}
        for dish, data in request.historical_consumption.items():
            consumption_rate = data['consumed'] / data['prepared'] if data['prepared'] > 0 else 0
            dish_popularity[dish] = consumption_rate
        
        # Sort dishes by popularity
        sorted_dishes = sorted(
            dish_popularity.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Generate optimized menu
        optimized_menu = {
            'breakfast': [],
            'lunch': [],
            'dinner': []
        }
        
        # Assign top dishes to meals (simplified logic)
        for i, (dish, score) in enumerate(sorted_dishes[:9]):
            if i < 3:
                optimized_menu['breakfast'].append({
                    'dish': dish,
                    'popularity_score': score,
                    'suggested_portion': 'normal'
                })
            elif i < 6:
                optimized_menu['lunch'].append({
                    'dish': dish,
                    'popularity_score': score,
                    'suggested_portion': 'normal'
                })
           