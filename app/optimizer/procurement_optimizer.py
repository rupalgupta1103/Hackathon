import pulp
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class ProcurementOptimizer:
    def __init__(self):
        self.problem = None
        self.variables = {}
        
    def optimize_procurement(self, 
                            predicted_demand: Dict[str, Dict[str, float]],
                            ingredient_prices: Dict[str, float],
                            recipe_database: Dict[str, Dict[str, float]],
                            min_stock_levels: Dict[str, float] = None,
                            budget_constraint: float = None) -> Dict:
        """
        Optimize ingredient procurement using Linear Programming
        
        Args:
            predicted_demand: {meal: {dish: quantity}}
            ingredient_prices: {ingredient: price_per_kg}
            recipe_database: {dish: {ingredient: quantity_per_serving}}
            min_stock_levels: {ingredient: min_quantity}
            budget_constraint: maximum budget
        
        Returns:
            Optimal procurement quantities
        """
        
        # Calculate total ingredient requirements from predicted demand
        ingredient_requirements = self.calculate_ingredient_requirements(
            predicted_demand, recipe_database
        )
        
        # Create optimization problem
        self.problem = pulp.LpProblem("Procurement_Optimization", pulp.LpMinimize)
        
        # Create decision variables (quantity to procure for each ingredient)
        self.variables = {}
        for ingredient in ingredient_requirements.keys():
            self.variables[ingredient] = pulp.LpVariable(
                f"procure_{ingredient}",
                lowBound=0,
                cat='Continuous'
            )
        
        # Objective: Minimize total procurement cost
        self.problem += pulp.lpSum([
            self.variables[ing] * ingredient_prices.get(ing, 0)
            for ing in ingredient_requirements.keys()
        ]), "Total_Cost"
        
        # Constraints
        # 1. Meet demand requirements
        for ingredient, required_qty in ingredient_requirements.items():
            self.problem += self.variables[ingredient] >= required_qty, f"Min_{ingredient}"
        
        # 2. Minimum stock levels (if provided)
        if min_stock_levels:
            for ingredient, min_stock in min_stock_levels.items():
                if ingredient in self.variables:
                    self.problem += self.variables[ingredient] >= min_stock, f"MinStock_{ingredient}"
        
        # 3. Budget constraint (if provided)
        if budget_constraint:
            self.problem += pulp.lpSum([
                self.variables[ing] * ingredient_prices.get(ing, 0)
                for ing in ingredient_requirements.keys()
            ]) <= budget_constraint, "Budget_Constraint"
        
        # Solve the problem
        self.problem.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # Extract results
        optimal_procurement = {}
        total_cost = 0
        
        for ingredient, var in self.variables.items():
            qty = var.varValue
            if qty is not None and qty > 0:
                optimal_procurement[ingredient] = qty
                total_cost += qty * ingredient_prices.get(ingredient, 0)
        
        # Calculate savings compared to traditional procurement
        traditional_cost = self.calculate_traditional_cost(
            ingredient_requirements, ingredient_prices
        )
        
        savings = traditional_cost - total_cost if traditional_cost > total_cost else 0
        savings_percentage = (savings / traditional_cost * 100) if traditional_cost > 0 else 0
        
        return {
            'optimal_quantities': optimal_procurement,
            'total_cost': total_cost,
            'traditional_cost': traditional_cost,
            'savings': savings,
            'savings_percentage': savings_percentage,
            'status': pulp.LpStatus[self.problem.status]
        }
    
    def calculate_ingredient_requirements(self, 
                                        predicted_demand: Dict[str, Dict[str, float]],
                                        recipe_database: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate total ingredient requirements from predicted dish demand"""
        requirements = {}
        
        for meal, dishes in predicted_demand.items():
            for dish, quantity in dishes.items():
                if dish in recipe_database:
                    for ingredient, amount_per_serving in recipe_database[dish].items():
                        total_amount = amount_per_serving * quantity
                        requirements[ingredient] = requirements.get(ingredient, 0) + total_amount
        
        return requirements
    
    def calculate_traditional_cost(self, 
                                  requirements: Dict[str, float],
                                  prices: Dict[str, float],
                                  buffer_percentage: float = 0.2) -> float:
        """Calculate cost with traditional procurement (with buffer stock)"""
        total_cost = 0
        for ingredient, qty in requirements.items():
            # Traditional method adds buffer stock
            buffered_qty = qty * (1 + buffer_percentage)
            total_cost += buffered_qty * prices.get(ingredient, 0)
        
        return total_cost
    
    def get_procurement_recommendations(self, 
                                       optimal_procurement: Dict[str, float],
                                       current_stock: Dict[str, float] = None) -> Dict[str, Dict]:
        """Generate detailed procurement recommendations"""
        recommendations = {}
        
        for ingredient, optimal_qty in optimal_procurement.items():
            rec = {
                'optimal_quantity': optimal_qty,
                'unit': 'kg',
                'priority': 'High' if optimal_qty > 100 else 'Medium' if optimal_qty > 50 else 'Low'
            }
            
            if current_stock and ingredient in current_stock:
                rec['current_stock'] = current_stock[ingredient]
                rec['to_order'] = max(0, optimal_qty - current_stock[ingredient])
            
            recommendations[ingredient] = rec
        
        return recommendations