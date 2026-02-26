"""
Menu Optimizer Module
Handles dynamic menu planning, dish selection, and rotation optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from collections import defaultdict
import random
from dataclasses import dataclass, field

from app.optimizer.constraints import ConstraintsManager, ConstraintTypes
from app.config import config

logger = logging.getLogger(__name__)


@dataclass
class Dish:
    """Represents a dish with its properties"""
    id: int
    name: str
    category: str
    popularity_score: float
    sustainability_score: float
    nutritional_score: float
    cost_per_serving: float
    preparation_time: int  # minutes
    ingredients: Dict[str, float]
    seasonal_months: List[int]
    is_vegetarian: bool = True
    is_vegan: bool = False
    is_gluten_free: bool = False
    avg_waste_percentage: float = 0.0
    times_served_last_month: int = 0
    
    @property
    def composite_score(self) -> float:
        """Calculate composite score based on multiple factors"""
        return (
            self.popularity_score * 0.4 +
            self.sustainability_score * 0.3 +
            self.nutritional_score * 0.3
        )


@dataclass
class MealPlan:
    """Represents a meal plan for a specific day and meal type"""
    date: datetime
    meal_type: str  # breakfast, lunch, dinner
    main_dish: Optional[Dish] = None
    side_dish1: Optional[Dish] = None
    side_dish2: Optional[Dish] = None
    bread: Optional[Dish] = None
    dessert: Optional[Dish] = None
    beverage: Optional[Dish] = None
    total_cost: float = 0.0
    total_preparation_time: int = 0
    estimated_students: int = 0
    sustainability_score: float = 0.0
    
    def get_all_dishes(self) -> List[Dish]:
        """Get all dishes in this meal"""
        dishes = []
        for dish in [self.main_dish, self.side_dish1, self.side_dish2, 
                     self.bread, self.dessert, self.beverage]:
            if dish:
                dishes.append(dish)
        return dishes
    
    def calculate_total_cost(self) -> float:
        """Calculate total cost of the meal"""
        return sum(dish.cost_per_serving for dish in self.get_all_dishes())
    
    def calculate_sustainability_score(self) -> float:
        """Calculate average sustainability score"""
        dishes = self.get_all_dishes()
        if not dishes:
            return 0.0
        return sum(d.sustainability_score for d in dishes) / len(dishes)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'date': self.date.strftime('%Y-%m-%d'),
            'meal_type': self.meal_type,
            'dishes': [d.name for d in self.get_all_dishes()],
            'total_cost': round(self.total_cost, 2),
            'estimated_students': self.estimated_students,
            'sustainability_score': round(self.sustainability_score, 2)
        }


class MenuOptimizer:
    """
    Advanced menu optimizer for creating optimal meal plans
    """
    
    def __init__(self, **kwargs):
        # Load configuration
        self.config = config.OPTIMIZATION_CONFIG.get('menu', {})
        self.config.update(kwargs)
        
        # Optimization parameters
        self.objective = self.config.get('objective', 'maximize_satisfaction')
        self.rotation_days = self.config.get('rotation_days', 7)
        self.min_variety = self.config.get('min_variety', 3)
        self.max_variety = self.config.get('max_variety', 5)
        
        # Weights for scoring
 self.nutrition_weight = self.config.get('nutrition_weight', 0.3)
        self.popularity_weight = self.config.get('popularity_weight', 0.4)
        self.sustainability_weight = self.config.get('sustainability_weight', 0.3)
        
        # Preference boosts
        self.seasonal_preference = self.config.get('seasonal_preference', 0.2)
        self.local_preference = self.config.get('local_preference', 0.15)
        
        # Constraints
        self.constraints_manager = ConstraintsManager()
        
        # Dish database
        self.dish_database: Dict[str, List[Dish]] = {
            'breakfast': [],
            'lunch': [],
            'dinner': []
        }
        
        # Menu history
        self.menu_history: List[MealPlan] = []
        
        logger.info(f"MenuOptimizer initialized with objective: {self.objective}")
    
    def load_dishes_from_db(self, db_session) -> None:
        """
        Load dishes from database
        """
        try:
            from app.database.models import MenuItem
            
            items = db_session.query(MenuItem).all()
            
            for item in items:
                dish = Dish(
                    id=item.id,
                    name=item.dish_name,
                    category=item.category,
                    popularity_score=item.popularity_score,
                    sustainability_score=item.sustainability_score,
                    nutritional_score=getattr(item, 'nutritional_score', 0.5),
                    cost_per_serving=item.base_price,
                    preparation_time=item.preparation_time,
                    ingredients=item.get_ingredients_dict(),
                    seasonal_months=self._parse_seasonal_months(item.seasonal_months),
                    is_vegetarian=item.is_vegetarian,
                    is_vegan=item.is_vegan,
                    is_gluten_free=item.is_gluten_free
                )
                
                # Categorize by meal type (simplified - you might want more sophisticated logic)
                if 'breakfast' in item.category.lower():
                    self.dish_database['breakfast'].append(dish)
                elif 'lunch' in item.category.lower():
                    self.dish_database['lunch'].append(dish)
                elif 'dinner' in item.category.lower():
                    self.dish_database['dinner'].append(dish)
                else:
                    # Default assignment
                    self.dish_database['lunch'].append(dish)
            
            logger.info(f"Loaded {len(items)} dishes into database")
            
        except Exception as e:
            logger.error(f"Error loading dishes from database: {e}")
            # Load default dishes
            self._load_default_dishes()
    
    def _load_default_dishes(self) -> None:
        """Load default dishes for testing"""
        default_dishes = {
            'breakfast': [
                Dish(1, "Poha", "breakfast", 0.9, 0.8, 0.7, 15, 15, {"poha": 0.1}, [1,2,3,4,5,6,7,8,9,10,11,12]),
                Dish(2, "Upma", "breakfast", 0.8, 0.8, 0.6, 12, 10, {"sooji": 0.1}, [1,2,3,4,5,6,7,8,9,10,11,12]),
                Dish(3, "Idli", "breakfast", 0.95, 0.9, 0.8, 10, 20, {"rice": 0.05, "urad dal": 0.02}, [1,2,3,4,5,6,7,8,9,10,11,12]),
                Dish(4, "Dosa", "breakfast", 0.92, 0.85, 0.75, 12, 15, {"rice": 0.06, "urad dal": 0.02}, [1,2,3,4,5,6,7,8,9,10,11,12]),
                Dish(5, "Sandwich", "breakfast", 0.85, 0.7, 0.6, 20, 10, {"bread": 0.1, "vegetables": 0.05}, [1,2,3,4,5,6,7,8,9,10,11,12]),
            ],
            'lunch': [
                Dish(6, "Rice", "main_course", 0.9, 0.8, 0.7, 10, 20, {"rice": 0.15}, [1,2,3,4,5,6,7,8,9,10,11,12]),
                Dish(7, "Chapati", "bread", 0.95, 0.9, 0.8, 5, 30, {"wheat": 0.05}, [1,2,3,4,5,6,7,8,9,10,11,12]),
                Dish(8, "Dal", "main_course", 0.88, 0.85, 0.9, 8, 25, {"dal": 0.08}, [1,2,3,4,5,6,7,8,9,10,11,12]),
                Dish(9, "Mixed Vegetables", "side_dish", 0.85, 0.9, 0.85, 12, 20, {"vegetables": 0.15}, [1,2,3,4,5,6,7,8,9,10,11,12]),
                Dish(10, "Curd", "side_dish", 0.9, 0.7, 0.8, 5, 5, {"milk": 0.1}, [1,2,3,4,5,6,7,8,9,10,11,12]),
                Dish(11, "Salad", "side_dish", 0.7, 0.95, 0.95, 8, 10, {"vegetables": 0.1}, [1,2,3,4,5,6,7,8,9,10,11,12]),
                Dish(12, "Papad", "side_dish", 0.8, 0.7, 0.5, 2, 2, {"papad": 0.02}, [1,2,3,4,5,6,7,8,9,10,11,12]),
            ],
            'dinner': [
                Dish(13, "Rice", "main_course", 0.9, 0.8, 0.7, 10, 20, {"rice": 0.15}, [1,2,3,4,5,6,7,8,9,10,11,12]),
                Dish(14, "Roti", "bread", 0.95, 0.9, 0.8, 5, 30, {"wheat": 0.05}, [1,2,3,4,5,6,7,8,9,10,11,12]),
                Dish(15, "Paneer Curry", "main_course", 0.9, 0.75, 0.85, 25, 25, {"paneer": 0.1, "spices": 0.01}, [1,2,3,4,5,6,7,8,9,10,11,12]),
                Dish(16, "Dal", "main_course", 0.88, 0.85, 0.9, 8, 25, {"dal": 0.08}, [1,2,3,4,5,6,7,8,9,10,11,12]),
                Dish(17, "Vegetable Curry", "main_course", 0.85, 0.9, 0.85, 15, 25, {"vegetables": 0.2}, [1,2,3,4,5,6,7,8,9,10,11,12]),
                Dish(18, "Khichdi", "main_course", 0.8, 0.9, 0.9, 12, 25, {"rice": 0.08, "dal": 0.05}, [1,2,3,4,5,6,7,8,9,10,11,12]),
            ]
        }
        
        self.dish_database = default_dishes
        logger.info("Loaded default dishes")
    
    def _parse_seasonal_months(self, months_str: Optional[str]) -> List[int]:
        """Parse seasonal months string to list of integers"""
        if not months_str:
            return list(range(1, 13))
        try:
            return [int(m.strip()) for m in months_str.split(',')]
        except:
            return list(range(1, 13))
    
    def optimize_weekly_menu(self, 
                            start_date: datetime,
                            predicted_attendance: Dict[str, Dict[str, int]],
                            budget_per_day: Optional[float] = None,
                            dietary_restrictions: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate optimized menu for a week
        
        Args:
            start_date: Starting date of the week
            predicted_attendance: Predicted student counts per meal
            budget_per_day: Maximum budget per day
            dietary_restrictions: List of dietary restrictions to consider
            
        Returns:
            Weekly menu plan with optimization results
        """
        logger.info(f"Generating weekly menu starting from {start_date.date()}")
        
        weekly_menu = []
        total_cost = 0
        total_sustainability_score = 0
        
        current_date = start_date
        for day in range(7):
            day_menu = {}
            day_cost = 0
            
            for meal_type in ['breakfast', 'lunch', 'dinner']:
                # Get predicted attendance for this meal
                attendance = predicted_attendance.get(meal_type, {}).get(str(current_date.date()), 200)
                
                # Optimize single meal
                meal_plan = self.optimize_single_meal(
                    date=current_date,
                    meal_type=meal_type,
                    estimated_students=attendance,
                    previous_meals=[m for m in weekly_menu[-3:] if m] if weekly_menu else [],
                    budget_remaining=budget_per_day - day_cost if budget_per_day else None
                )
                
                if meal_plan:
                    day_menu[meal_type] = meal_plan
                    day_cost += meal_plan.total_cost
                    total_cost += meal_plan.total_cost
                    total_sustainability_score += meal_plan.sustainability_score
            
            weekly_menu.append({
                'date': current_date.date(),
                'day_of_week': current_date.strftime('%A'),
                'meals': day_menu,
                'total_cost': round(day_cost, 2)
            })
            
            current_date += timedelta(days=1)
        
        # Calculate metrics
        avg_daily_cost = total_cost / 7
        avg_sustainability = total_sustainability_score / 21  # 3 meals * 7 days
        
        # Generate recommendations
        recommendations = self._generate_menu_recommendations(weekly_menu)
        
        return {
            'weekly_menu': weekly_menu,
            'summary': {
                'total_cost': round(total_cost, 2),
                'avg_daily_cost': round(avg_daily_cost, 2),
                'avg_sustainability_score': round(avg_sustainability, 2),
                'total_dishes_used': len(set(
                    dish.name for day in weekly_menu 
                    for meal in day['meals'].values() 
                    for dish in meal.get_all_dishes()
                ))
            },
            'recommendations': recommendations,
            'optimization_status': 'success'
        }
    
    def optimize_single_meal(self,
                            date: datetime,
                            meal_type: str,
                            estimated_students: int,
                            previous_meals: List[MealPlan] = None,
                            budget_remaining: Optional[float] = None) -> Optional[MealPlan]:
        """
        Optimize a single meal
        
        Args:
            date: Date of the meal
            meal_type: Type of meal
            estimated_students: Expected number of students
            previous_meals: List of recent meals to avoid repetition
            budget_remaining: Remaining budget for the day
            
        Returns:
            Optimized meal plan
        """
        # Get available dishes for this meal type
        available_dishes = self.dish_database.get(meal_type, [])
        if not available_dishes:
            logger.warning(f"No dishes available for {meal_type}")
            return None
        
        # Apply seasonal filter
        current_month = date.month
        seasonal_dishes = [d for d in available_dishes if current_month in d.seasonal_months]
        
        # Boost scores for seasonal dishes
        for dish in seasonal_dishes:
            dish.popularity_score *= (1 + self.seasonal_preference)
        
        # Apply repetition avoidance
        if previous_meals:
            recent_dishes = set()
            for meal in previous_meals:
                recent_dishes.update([d.name for d in meal.get_all_dishes()])
            
            # Reduce score for recently used dishes
            for dish in available_dishes:
                if dish.name in recent_dishes:
                    dish.popularity_score *= 0.7
        
        # Score all dishes
        for dish in available_dishes:
            dish.composite_score = self._calculate_dish_score(dish, meal_type)
        
        # Select dishes based on meal type requirements
        meal_plan = MealPlan(
            date=date,
            meal_type=meal_type,
            estimated_students=estimated_students
        )
        
        if meal_type == 'breakfast':
            # Breakfast typically has 2-3 items
            selected = self._select_dishes_optimized(
                available_dishes,
                num_to_select=min(3, len(available_dishes)),
                budget_limit=budget_remaining / 3 if budget_remaining else None
            )
            
            if len(selected) >= 1:
                meal_plan.main_dish = selected[0]
            if len(selected) >= 2:
                meal_plan.side_dish1 = selected[1]
            if len(selected) >= 3:
                meal_plan.beverage = selected[2]
        
        elif meal_type in ['lunch', 'dinner']:
            # Lunch/Dinner typically have more items
            # Select main dish (rice, chapati, etc.)
            main_dishes = [d for d in available_dishes if d.category in ['main_course', 'bread']]
            if main_dishes:
                meal_plan.main_dish = self._select_best_dish(main_dishes)
            
            # Select side dishes
            side_dishes = [d for d in available_dishes if d.category == 'side_dish']
            selected_sides = self._select_dishes_optimized(
                side_dishes,
                num_to_select=min(3, len(side_dishes)),
                exclude=[meal_plan.main_dish] if meal_plan.main_dish else []
            )
            
            if len(selected_sides) >= 1:
                meal_plan.side_dish1 = selected_sides[0]
            if len(selected_sides) >= 2:
                meal_plan.side_dish2 = selected_sides[1]
            
            # Select dessert if available
            desserts = [d for d in available_dishes if d.category == 'dessert']
            if desserts and random.random() < 0.3:  # 30% chance of dessert
                meal_plan.dessert = self._select_best_dish(desserts)
            
            # Select beverage
            beverages = [d for d in available_dishes if d.category == 'beverage']
            if beverages:
                meal_plan.beverage = self._select_best_dish(beverages)
        
        # Calculate meal metrics
        meal_plan.total_cost = meal_plan.calculate_total_cost()
        meal_plan.sustainability_score = meal_plan.calculate_sustainability_score()
        
        # Check budget constraint
        if budget_remaining and meal_plan.total_cost > budget_remaining:
            logger.warning(f"Meal cost {meal_plan.total_cost} exceeds budget {budget_remaining}")
            return self._adjust_to_budget(meal_plan, budget_remaining)
        
        self.menu_history.append(meal_plan)
        return meal_plan
    
    def _calculate_dish_score(self, dish: Dish, meal_type: str) -> float:
        """Calculate composite score for a dish"""
        # Base score
        score = (
            dish.popularity_score * self.popularity_weight +
            dish.sustainability_score * self.sustainability_weight +
            dish.nutritional_score * self.nutrition_weight
        )
        
        # Apply time-of-day preference
        if meal_type == 'breakfast':
            if 'light' in dish.name.lower() or dish.category in ['beverage']:
                score *= 1.1
        elif meal_type == 'dinner':
            if 'heavy' in dish.name.lower() or dish.category in ['main_course']:
                score *= 1.1
        
        # Apply waste penalty
        if dish.avg_waste_percentage > 20:
            score *= (1 - (dish.avg_waste_percentage - 20) / 100)
        
        return score
    
    def _select_best_dish(self, dishes: List[Dish]) -> Optional[Dish]:
        """Select the best dish from a list"""
        if not dishes:
            return None
        return max(dishes, key=lambda d: d.composite_score)
    
    def _select_dishes_optimized(self, 
                                dishes: List[Dish], 
                                num_to_select: int,
                                budget_limit: Optional[float] = None,
                                exclude: List[Dish] = None) -> List[Dish]:
        """
        Select optimal combination of dishes
        Uses greedy algorithm with knapsack-like approach
        """
        if not dishes:
            return []
        
        exclude_names = [d.name for d in exclude] if exclude else []
        available = [d for d in dishes if d.name not in exclude_names]
        
        if not available:
            return []
        
        # Sort by score/cost ratio for efficiency
        available.sort(key=lambda d: d.composite_score / (d.cost_per_serving + 0.01), reverse=True)
        
        selected = []
        total_cost = 0
        
        for dish in available:
            if len(selected) >= num_to_select:
                break
            
            if budget_limit and total_cost + dish.cost_per_serving > budget_limit:
                continue
            
            selected.append(dish)
            total_cost += dish.cost_per_serving
        
        return selected
    
    def _adjust_to_budget(self, meal_plan: MealPlan, budget_limit: float) -> MealPlan:
        """Adjust meal plan to fit within budget"""
        dishes = meal_plan.get_all_dishes()
        
        # Sort by cost descending to remove expensive items first
        dishes.sort(key=lambda d: d.cost_per_serving, reverse=True)
        
        while meal_plan.total_cost > budget_limit and len(dishes) > self.min_variety:
            # Remove most expensive non-essential dish
            removed = False
            for dish in dishes:
                if dish.category not in ['main_course', 'bread']:  # Keep essential items
                    # Remove this dish
                    if meal_plan.side_dish1 == dish:
                        meal_plan.side_dish1 = None
                    elif meal_plan.side_dish2 == dish:
                        meal_plan.side_dish2 = None
                    elif meal_plan.dessert == dish:
                        meal_plan.dessert = None
                    elif meal_plan.beverage == dish:
                        meal_plan.beverage = None
                    
                    removed = True
                    break
            
            if not removed:
                # If only essential items left, can't reduce further
                break
            
            meal_plan.total_cost = meal_plan.calculate_total_cost()
            dishes = meal_plan.get_all_dishes()
        
        return meal_plan
    
    def _generate_menu_recommendations(self, weekly_menu: List[Dict]) -> List[str]:
        """Generate recommendations based on menu analysis"""
        recommendations = []
        
        # Analyze variety
        all_dishes = set()
        dish_counts = defaultdict(int)
        
        for day in weekly_menu:
            for meal in day['meals'].values():
                for dish in meal.get_all_dishes():
                    all_dishes.add(dish.name)
                    dish_counts[dish.name] += 1
        
        # Check repetition
        repeated_dishes = [dish for dish, count in dish_counts.items() if count > 3]
        if repeated_dishes:
            recommendations.append(f"Consider reducing frequency of: {', '.join(repeated_dishes[:3])}")
        
        # Check variety
        if len(all_dishes) < 15:
            recommendations.append("Menu lacks variety. Consider adding more diverse dishes.")
        
        # Check seasonal items
        seasonal_count = sum(1 for day in weekly_menu 
                           for meal in day['meals'].values() 
                           for dish in meal.get_all_dishes() 
                           if datetime.now().month in dish.seasonal_months)
        
        if seasonal_count < 7:
            recommendations.append("Include more seasonal dishes for better sustainability and cost.")
        
        # Sustainability suggestions
        low_sustainability_dishes = []
        for day in weekly_menu:
            for meal in day['meals'].values():
                if meal.sustainability_score < 0.6:
                    for dish in meal.get_all_dishes():
                        if dish.sustainability_score < 0.6:
                            low_sustainability_dishes.append(dish.name)
        
        if low_sustainability_dishes:
            recommendations.append(f"Consider replacing low-sustainability items: {', '.join(set(low_sustainability_dishes[:3]))}")
        
        return recommendations
    
    def update_popularity_scores(self, consumption_data: Dict[str, float]) -> None:
        """
        Update dish popularity scores based on actual consumption
        
        Args:
            consumption_data: Dictionary mapping dish names to consumption rates
        """
        for meal_type, dishes in self.dish_database.items():
            for dish in dishes:
                if dish.name in consumption_data:
                    # Update with exponential moving average
                    new_score = consumption_data[dish.name]
                    dish.popularity_score = 0.7 * dish.popularity_score + 0.3 * new_score
                    
                    logger.debug(f"Updated popularity for {dish.name}: {dish.popularity_score:.2f}")
    
    def get_menu_statistics(self) -> Dict[str, Any]:
        """Get statistics about menu optimization"""
        total_dishes = sum(len(dishes) for dishes in self.dish_database.values())
        
        return {
            'total_dishes_available': total_dishes,
            'dishes_by_category': {
                meal_type: len(dishes) 
                for meal_type, dishes in self.dish_database.items()
            },
            'average_popularity': np.mean([
                d.popularity_score 
                for dishes in self.dish_database.values() 
                for d in dishes
            ]),
            'average_sustainability': np.mean([
                d.sustainability_score 
                for dishes in self.dish_database.values() 
                for d in dishes
            ]),
            'menu_history_size': len(self.menu_history),
            'optimization_params': self.config
        }
    
    def save_menu_plan(self, menu_plan: MealPlan, filepath: str) -> None:
        """Save menu plan to file"""
        try:
            import json
            
            data = {
                'date': menu_plan.date.isoformat(),
                'meal_type': menu_plan.meal_type,
                'dishes': [
                    {
                        'name': d.name,
                        'category': d.category,
                        'score': d.composite_score,
                        'cost': d.cost_per_serving
                    }
                    for d in menu_plan.get_all_dishes()
                ],
                'total_cost': menu_plan.total_cost,
                'sustainability_score': menu_plan.sustainability_score,
                'estimated_students': menu_plan.estimated_students
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Menu plan saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save menu plan: {e}")
    
    def load_menu_plan(self, filepath: str) -> Optional[MealPlan]:
        """Load menu plan from file"""
        try:
            import json
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Reconstruct meal plan (simplified - would need to match dishes)
            meal_plan = MealPlan(
                date=datetime.fromisoformat(data['date']),
                meal_type=data['meal_type'],
                total_cost=data['total_cost'],
                sustainability_score=data['sustainability_score'],
                estimated_students=data['estimated_students']
            )
            
            logger.info(f"Menu plan loaded from {filepath}")
            return meal_plan
            
        except Exception as e:
            logger.error(f"Failed to load menu plan: {e}")
            return None