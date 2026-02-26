"""
Complete API endpoints for Smart Mess Optimization System
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Path, status
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta, date
import logging

from app.database.database import get_db
from app.database.crud import CRUDOperations
from app.database.queries import QueryBuilder
from app.database import models
from app.api import schemas
from app.api.dependencies import (
    required_auth, optional_auth, rate_limiter,
    request_logger, PaginationParams, DateFilter
)
from app.api.errors import NotFoundException, ValidationException
from app.ml_models.demand_predictor import DemandPredictor
from app.optimizer.procurement_optimizer import ProcurementOptimizer
from app.sustainability.metrics_calculator import SustainabilityMetrics
from app.config import config

# Initialize router
router = APIRouter(tags=["API"])

# Initialize ML components
predictor = DemandPredictor(model_type='xgboost')
optimizer = ProcurementOptimizer()
sustainability = SustainabilityMetrics(config)

logger = logging.getLogger(__name__)

# ==================== Health Check ====================

@router.get(
    "/health",
    response_model=schemas.HealthCheckResponse,
    summary="Health Check",
    description="Check if the API is running"
)
async def health_check(
    request_log = Depends(request_logger)
):
    """
    Health check endpoint to verify API status
    """
    return schemas.HealthCheckResponse(
        status="healthy",
        message="Smart Mess Optimization System API",
        version="1.0.0"
    )

# ==================== Demand Prediction Endpoints ====================

@router.post(
    "/predict/demand",
    response_model=schemas.DemandPredictionResponse,
    summary="Predict Food Demand",
    description="Predict next-day food demand based on historical data and factors"
)
async def predict_demand(
    request: schemas.DemandPredictionRequest,
    db: Session = Depends(get_db),
    auth = Depends(optional_auth),
    _ = Depends(rate_limiter)
):
    """
    Predict demand for next day for all meals and dishes
    
    - **date**: Date to predict for
    - **temperature**: Expected temperature in Celsius
    - **rainfall**: Expected rainfall in mm
    - **is_holiday**: Whether it's a holiday
    - **is_exam**: Whether there are exams
    - **special_event**: Any special event name
    """
    try:
        logger.info(f"Predicting demand for {request.date}")
        
        # Get historical data for context
        crud = CRUDOperations(db)
        historical_data = crud.get_all_meals(limit=500)
        
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
        
        # Get meal types and dishes from database
        menu_items = crud.get_all_menu_items()
        dishes = [item.dish_name for item in menu_items]
        
        if not dishes:
            # Default dishes if none in database
            dishes = ["rice", "chapati", "dal", "sabzi", "curry"]
        
        meal_types = config.MEAL_TYPES
        
        # Make predictions
        predictions = predictor.predict_next_day(features, meal_types, dishes)
        
        # Calculate confidence based on data availability
        confidence = min(0.95, 0.5 + len(historical_data) / 1000)
        
        return schemas.DemandPredictionResponse(
            date=request.date,
            predictions=predictions,
            confidence=confidence,
            model_used="XGBoost with Time Series"
        )
        
    except Exception as e:
        logger.error(f"Demand prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Demand prediction failed: {str(e)}"
        )

# ==================== Procurement Optimization Endpoints ====================

@router.post(
    "/optimize/procurement",
    response_model=schemas.ProcurementResponse,
    summary="Optimize Procurement",
    description="Optimize ingredient procurement using linear programming"
)
async def optimize_procurement(
    request: schemas.ProcurementRequest,
    db: Session = Depends(get_db),
    auth = Depends(required_auth)
):
    """
    Optimize ingredient procurement based on predicted demand
    
    - **predictions**: Predicted demand by meal and dish
    - **budget_constraint**: Maximum budget (optional)
    - **current_stock**: Current stock levels (optional)
    """
    try:
        logger.info("Optimizing procurement")
        
        # Get recipe database from menu items
        crud = CRUDOperations(db)
        menu_items = crud.get_all_menu_items()
        
        recipe_db = {}
        for item in menu_items:
            recipe_db[item.dish_name] = item.get_ingredients_dict()
        
        # Get ingredient prices from config
        prices = config.INGREDIENT_PRICES
        
        # Run optimization
        result = optimizer.optimize_procurement(
            predicted_demand=request.predictions,
            ingredient_prices=prices,
            recipe_database=recipe_db,
            budget_constraint=request.budget_constraint
        )
        
        # Get recommendations with current stock
        recommendations = optimizer.get_procurement_recommendations(
            result['optimal_quantities'],
            request.current_stock
        )
        
        return schemas.ProcurementResponse(
            optimal_quantities=result['optimal_quantities'],
            total_cost=result['total_cost'],
            traditional_cost=result['traditional_cost'],
            savings=result['savings'],
            savings_percentage=result['savings_percentage'],
            status=result['status'],
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Procurement optimization failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Procurement optimization failed: {str(e)}"
        )

# ==================== Waste Monitoring Endpoints ====================

@router.post(
    "/waste/log",
    response_model=schemas.WasteLogResponse,
    summary="Log Waste",
    description="Log food waste and calculate environmental impact"
)
async def log_waste(
    waste_data: schemas.WasteLogRequest,
    db: Session = Depends(get_db),
    auth = Depends(required_auth)
):
    """
    Log waste data and calculate environmental impact
    
    - **meal_type**: Type of meal (breakfast, lunch, etc.)
    -