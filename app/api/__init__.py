"""
API module for Smart Mess Optimization System
Exposes REST endpoints for all functionality
"""

from app.api.endpoints import router
from app.api.schemas import (
    # Request schemas
    DemandPredictionRequest,
    ProcurementRequest,
    WasteLogRequest,
    MenuOptimizationRequest,
    MealConsumptionCreate,
    MenuItemCreate,
    ProcurementCreate,
    WasteLogCreate,
    
    # Response schemas
    DemandPredictionResponse,
    ProcurementResponse,
    SustainabilityMetricsResponse,
    WasteLogResponse,
    HealthCheckResponse,
    MealConsumptionResponse,
    MenuItemResponse,
    ProcurementResponse as ProcurementDataResponse,
    WasteLogResponse as WasteLogDataResponse,
    
    # Common schemas
    ErrorResponse,
    PaginatedResponse
)

__all__ = [
    'router',
    
    # Request schemas
    'DemandPredictionRequest',
    'ProcurementRequest',
    'WasteLogRequest',
    'MenuOptimizationRequest',
    'MealConsumptionCreate',
    'MenuItemCreate',
    'ProcurementCreate',
    'WasteLogCreate',
    
    # Response schemas
    'DemandPredictionResponse',
    'ProcurementResponse',
    'SustainabilityMetricsResponse',
    'WasteLogResponse',
    'HealthCheckResponse',
    'MealConsumptionResponse',
    'MenuItemResponse',
    'ProcurementDataResponse',
    'WasteLogDataResponse',
    
    # Common schemas
    'ErrorResponse',
    'PaginatedResponse'
]