"""
Optimization module for Smart Mess Optimization System
Provides procurement optimization, menu planning, and resource allocation
"""

from app.optimizer.procurement_optimizer import ProcurementOptimizer
from app.optimizer.menu_optimizer import MenuOptimizer
from app.optimizer.inventory_optimizer import InventoryOptimizer
from app.optimizer.cost_optimizer import CostOptimizer
from app.optimizer.constraints import ConstraintsManager
from app.optimizer.solver import SolverEngine

__version__ = '1.0.0'
__all__ = [
    # Main optimizers
    'ProcurementOptimizer',
    'MenuOptimizer',
    'InventoryOptimizer',
    'CostOptimizer',
    
    # Supporting classes
    'ConstraintsManager',
    'SolverEngine'
]

# Package metadata
__author__ = 'Smart Mess Optimization Team'
__description__ = 'Optimization algorithms for food procurement and menu planning'
__license__ = 'MIT'

# ==================== Optimization Configuration ====================

# Default optimization parameters
DEFAULT_OPTIMIZATION_CONFIG = {
    'procurement': {
        'solver': 'pulp',  # pulp, scipy, ortools
        'objective': 'minimize_cost',  # minimize_cost, maximize_efficiency
        'buffer_percentage': 0.1,  # 10% safety buffer
        'min_order_quantity': 1.0,  # Minimum order in kg
        'max_storage_days': 7,  # Maximum days to store fresh items
        'reorder_point_multiplier': 1.5,  # Reorder when stock < 1.5x daily usage
        'bulk_discount_threshold': 100,  # kg threshold for bulk discount
        'bulk_discount_rate': 0.05  # 5% discount for bulk orders
    },
    'menu': {
        'objective': 'maximize_satisfaction',  # maximize_satisfaction, minimize_waste, balanced
        'rotation_days': 7,  # Days before menu repeats
        'min_variety': 3,  # Minimum dishes per meal
        'max_variety': 5,  # Maximum dishes per meal
        'nutrition_weight': 0.3,  # Weight for nutrition in scoring
        'popularity_weight': 0.4,  # Weight for popularity in scoring
        'sustainability_weight': 0.3,  # Weight for sustainability in scoring
        'seasonal_preference': 0.2,  # Boost for seasonal items
        'local_preference': 0.15  # Boost for locally sourced items
    },
    'inventory': {
        'min_stock_days': 2,  # Minimum stock in days
        'max_stock_days': 7,  # Maximum stock in days
        'perishable_threshold': 3,  # Days considered perishable
        'abc_analysis': True,  # Perform ABC analysis
        'fifo_priority': True  # Prioritize FIFO in allocation
    },
    'cost': {
        'labor_cost_per_hour': 150,  # INR per hour
        'storage_cost_per_kg': 2,  # INR per kg per day
        'transport_cost_per_kg': 5,  # INR per kg
        'waste_disposal_cost': 10,  # INR per kg
        'budget_soft_constraint': 0.1  # 10% flexibility in budget
    }
}

# Solver options
SOLVER_OPTIONS = {
    'pulp': {
        'name': 'PuLP Solver',
        'description': 'Linear programming solver for Python',
        'available_solvers': ['PULP_CBC_CMD', 'GLPK_CMD', 'CPLEX_CMD'],
        'default_solver': 'PULP_CBC_CMD',
        'time_limit': 60,  # seconds
        'gap_tolerance': 0.01  # 1% optimality gap
    },
    'scipy': {
        'name': 'SciPy Optimize',
        'description': 'Scientific computing optimization',
        'methods': ['linprog', 'minimize'],
        'default_method': 'linprog',
        'max_iterations': 1000
    },
    'ortools': {
        'name': 'Google OR-Tools',
        'description': 'Google optimization tools',
        'available_solvers': ['GLOP', 'CBC', 'CP-SAT'],
        'default_solver': 'CBC',
        'time_limit': 30,
        'enable_logging': False
    }
}

# ==================== Constraint Definitions ====================

class ConstraintTypes:
    """Types of constraints used in optimization"""
    
    # Procurement constraints
    MIN_ORDER = 'min_order'
    MAX_ORDER = 'max_order'
    BUDGET_LIMIT = 'budget_limit'
    STORAGE_CAPACITY = 'storage_capacity'
    VENDOR_CAPACITY = 'vendor_capacity'
    
    # Nutritional constraints
    CALORIE_MIN = 'calorie_min'
    CALORIE_MAX = 'calorie_max'
    PROTEIN_MIN = 'protein_min'
    CARBS_MIN = 'carbs_min'
    FAT_MIN = 'fat_min'
    
    # Menu constraints
    DIETARY_RESTRICTION = 'dietary_restriction'
    MENU_VARIETY = 'menu_variety'
    REPETITION_LIMIT = 'repetition_limit'
    
    # Inventory constraints
    SAFETY_STOCK = 'safety_stock'
    SHELF_LIFE = 'shelf_life'
    ROTATION_RULE = 'rotation_rule'

# ==================== Nutritional Guidelines ====================

NUTRITIONAL_GUIDELINES = {
    'adult_male': {
        'calories': 2500,  # kcal per day
        'protein': 60,  # grams
        'carbs': 300,  # grams
        'fat': 70,  # grams
        'fiber': 30,  # grams
        'iron': 10,  # mg
        'calcium': 800  # mg
    },
    'adult_female': {
        'calories': 2000,
        'protein': 50,
        'carbs': 250,
        'fat': 60,
        'fiber': 25,
        'iron': 15,
        'calcium': 800
    },
    'student': {
        'calories': 2200,  # Average student requirement
        'protein': 55,
        'carbs': 275,
        'fat': 65,
        'fiber': 28,
        'iron': 12,
        'calcium': 800
    }
}

# ==================== Helper Functions ====================

def get_optimizer_info(optimizer_type: str = None) -> dict:
    """
    Get information about available optimizers
    
    Args:
        optimizer_type: Specific optimizer type to get info for
        
    Returns:
        Dictionary with optimizer information
    """
    optimizer_info = {
        'procurement': {
            'name': 'Procurement Optimizer',
            'description': 'Optimizes ingredient purchasing using linear programming',
            'algorithms': ['Linear Programming', 'Mixed Integer Programming'],
            'constraints': ['Budget', 'Storage', 'Minimum Order', 'Vendor Capacity'],
            'outputs': ['Optimal Quantities', 'Cost Savings', 'Procurement Schedule']
        },
        'menu': {
            'name': 'Menu Optimizer',
            'description': 'Creates optimal menu rotations based on popularity and nutrition',
            'algorithms': ['Multi-objective Optimization', 'Genetic Algorithms'],
            'constraints': ['Nutritional', 'Variety', 'Budget', 'Seasonal'],
            'outputs': ['Weekly Menu', 'Dish Scores', 'Rotation Schedule']
        },
        'inventory': {
            'name': 'Inventory Optimizer',
            'description': 'Manages inventory levels and reorder points',
            'algorithms': ['ABC Analysis', 'EOQ Model', 'Safety Stock Calculation'],
            'constraints': ['Storage', 'Shelf Life', 'Service Level'],
            'outputs': ['Reorder Points', 'Stock Levels', 'Expiry Alerts']
        },
        'cost': {
            'name': 'Cost Optimizer',
            'description': 'Minimizes total cost including procurement, storage, and waste',
            'algorithms': ['Cost-Benefit Analysis', 'Trade-off Optimization'],
            'constraints': ['Budget', 'Quality', 'Service Level'],
            'outputs': ['Cost Breakdown', 'Savings Opportunities', 'ROI Analysis']
        }
    }
    
    if optimizer_type:
        return optimizer_info.get(optimizer_type, {})
    return optimizer_info

def validate_optimization_config(config: dict, optimizer_type: str = 'procurement') -> bool:
    """
    Validate optimization configuration
    
    Args:
        config: Configuration dictionary
        optimizer_type: Type of optimizer
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    required_keys = {
        'procurement': ['solver', 'objective', 'buffer_percentage'],
        'menu': ['objective', 'rotation_days', 'min_variety'],
        'inventory': ['min_stock_days', 'max_stock_days'],
        'cost': ['labor_cost_per_hour', 'storage_cost_per_kg']
    }
    
    required = required_keys.get(optimizer_type, [])
    
    for key in required:
        if key not in config:
            raise ValueError(f"Missing required configuration key for {optimizer_type}: {key}")
    
    # Validate specific values
    if optimizer_type == 'procurement':
        if config.get('buffer_percentage', 0) < 0 or config.get('buffer_percentage', 0) > 0.5:
            raise ValueError("buffer_percentage must be between 0 and 0.5")
        
        if config.get('min_order_quantity', 0) < 0:
            raise ValueError("min_order_quantity must be non-negative")
    
    elif optimizer_type == 'menu':
        if config.get('rotation_days', 0) < 1:
            raise ValueError("rotation_days must be at least 1")
        
        total_weight = (
            config.get('nutrition_weight', 0) +
            config.get('popularity_weight', 0) +
            config.get('sustainability_weight', 0)
        )
        if abs(total_weight - 1.0) > 0.01:  # Allow small floating point误差
            raise ValueError("Weights must sum to 1.0")
    
    return True

def create_optimizer(optimizer_type: str = 'procurement', **kwargs):
    """
    Factory function to create optimizer instances
    
    Args:
        optimizer_type: Type of optimizer to create
        **kwargs: Additional arguments for the optimizer
        
    Returns:
        Optimizer instance
    """
    config = DEFAULT_OPTIMIZATION_CONFIG.get(optimizer_type, {}).copy()
    config.update(kwargs)
    
    # Validate config
    validate_optimization_config(config, optimizer_type)
    
    if optimizer_type == 'procurement':
        from app.optimizer.procurement_optimizer import ProcurementOptimizer
        return ProcurementOptimizer(**config)
    
    elif optimizer_type == 'menu':
        from app.optimizer.menu_optimizer import MenuOptimizer
        return MenuOptimizer(**config)
    
    elif optimizer_type == 'inventory':
        from app.optimizer.inventory_optimizer import InventoryOptimizer
        return InventoryOptimizer(**config)
    
    elif optimizer_type == 'cost':
        from app.optimizer.cost_optimizer import CostOptimizer
        return CostOptimizer(**config)
    
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

def get_available_solvers() -> dict:
    """
    Get list of available solvers and their status
    
    Returns:
        Dictionary of solvers with availability status
    """
    available_solvers = {}
    
    # Check PuLP solvers
    try:
        import pulp
        available_solvers['pulp'] = {
            'available': True,
            'solvers': pulp.listSolvers(onlyAvailable=True)
        }
    except ImportError:
        available_solvers['pulp'] = {'available': False, 'solvers': []}
    
    # Check SciPy
    try:
        import scipy
        available_solvers['scipy'] = {
            'available': True,
            'version': scipy.__version__
        }
    except ImportError:
        available_solvers['scipy'] = {'available': False}
    
    # Check OR-Tools
    try:
        from ortools.linear_solver import pywraplp
        available_solvers['ortools'] = {
            'available': True,
            'solvers': ['GLOP', 'CBC', 'CP-SAT']
        }
    except ImportError:
        available_solvers['ortools'] = {'available': False}
    
    return available_solvers

# ==================== Initialize Module ====================

import logging

# Set up module logger
optimizer_logger = logging.getLogger(__name__)
optimizer_logger.info(f"Optimizer module initialized (version {__version__})")
optimizer_logger.info(f"Available optimizers: {', '.join(get_optimizer_info().keys())}")

# Check solver availability
solvers = get_available_solvers()
available_solvers = [name for name, info in solvers.items() if info.get('available')]
optimizer_logger.info(f"Available solvers: {', '.join(available_solvers)}")

if not available_solvers:
    optimizer_logger.warning("No optimization solvers found! Install PuLP, SciPy, or OR-Tools.")

# Export main classes and functions
__all__.extend([
    'ConstraintTypes',
    'NUTRITIONAL_GUIDELINES',
    'DEFAULT_OPTIMIZATION_CONFIG',
    'SOLVER_OPTIONS',
    'get_optimizer_info',
    'validate_optimization_config',
    'create_optimizer',
    'get_available_solvers'
])