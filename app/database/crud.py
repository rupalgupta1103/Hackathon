"""
CRUD (Create, Read, Update, Delete) operations for database models
"""

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc
from datetime import datetime, timedelta, date
from typing import List, Optional, Dict, Any, Union
import logging

from app.database import models

logger = logging.getLogger(__name__)


class CRUDOperations:
    """
    Comprehensive CRUD operations for all database models
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    # ==================== MEAL CONSUMPTION CRUD ====================
    
    def create_meal(self, meal_data: Dict[str, Any]) -> models.MealConsumption:
        """
        Create a new meal consumption record
        """
        try:
            # Calculate waste if not provided
            if 'quantity_wasted' not in meal_data:
                meal_data['quantity_wasted'] = meal_data.get('quantity_prepared', 0) - meal_data.get('quantity_consumed', 0)
            
            # Set day of week and weekend flag
            if 'date' in meal_data:
                if isinstance(meal_data['date'], (datetime, date)):
                    meal_data['day_of_week'] = meal_data['date'].weekday()
                    meal_data['is_weekend'] = meal_data['day_of_week'] >= 5
                elif isinstance(meal_data['date'], str):
                    date_obj = datetime.fromisoformat(meal_data['date'])
                    meal_data['day_of_week'] = date_obj.weekday()
                    meal_data['is_weekend'] = date_obj.weekday() >= 5
            
            meal = models.MealConsumption(**meal_data)
            self.db.add(meal)
            self.db.commit()
            self.db.refresh(meal)
            logger.info(f"Created meal record: {meal.id}")
            return meal
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating meal: {e}")
            raise
    
    def get_meal(self, meal_id: int) -> Optional[models.MealConsumption]:
        """
        Get meal consumption by ID
        """
        return self.db.query(models.MealConsumption).filter(
            models.MealConsumption.id == meal_id
        ).first()
    
    def get_meal_by_date(self, 
                         target_date: Union[datetime, date], 
                         meal_type: Optional[str] = None,
                         dish_name: Optional[str] = None) -> List[models.MealConsumption]:
        """
        Get meals by date with optional filters
        """
        # Convert date to datetime range
        if isinstance(target_date, date) and not isinstance(target_date, datetime):
            start_date = datetime.combine(target_date, datetime.min.time())
            end_date = datetime.combine(target_date, datetime.max.time())
        else:
            start_date = target_date
            end_date = target_date + timedelta(days=1) - timedelta(microseconds=1)
        
        query = self.db.query(models.MealConsumption).filter(
            and_(
                models.MealConsumption.date >= start_date,
                models.MealConsumption.date <= end_date
            )
        )
        
        if meal_type:
            query = query.filter(models.MealConsumption.meal_type == meal_type)
        
        if dish_name:
            query = query.filter(models.MealConsumption.dish_name == dish_name)
        
        return query.all()
    
    def get_all_meals(self, 
                      skip: int = 0, 
                      limit: int = 100,
                      start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None,
                      meal_type: Optional[str] = None) -> List[models.MealConsumption]:
        """
        Get all meal consumption records with filters
        """
        query = self.db.query(models.MealConsumption)
        
        if start_date:
            query = query.filter(models.MealConsumption.date >= start_date)
        if end_date:
            query = query.filter(models.MealConsumption.date <= end_date)
        if meal_type:
            query = query.filter(models.MealConsumption.meal_type == meal_type)
        
        return query.order_by(desc(models.MealConsumption.date)).offset(skip).limit(limit).all()
    
    def update_meal(self, 
                    meal_id: int, 
                    updates: Dict[str, Any]) -> Optional[models.MealConsumption]:
        """
        Update meal consumption record
        """
        try:
            meal = self.get_meal(meal_id)
            if not meal:
                logger.warning(f"Meal {meal_id} not found for update")
                return None
            
            for key, value in updates.items():
                if hasattr(meal, key):
                    setattr(meal, key, value)
            
            # Recalculate waste if quantities changed
            if 'quantity_prepared' in updates or 'quantity_consumed' in updates:
                meal.quantity_wasted = meal.quantity_prepared - meal.quantity_consumed
            
            meal.updated_at = datetime.utcnow()
            self.db.commit()
            self.db.refresh(meal)
            logger.info(f"Updated meal: {meal_id}")
            return meal
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating meal {meal_id}: {e}")
            raise
    
    def delete_meal(self, meal_id: int) -> bool:
        """
        Delete meal consumption record
        """
        try:
            meal = self.get_meal(meal_id)
            if not meal:
                return False
            
            self.db.delete(meal)
            self.db.commit()
            logger.info(f"Deleted meal: {meal_id}")
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error deleting meal {meal_id}: {e}")
            raise
    
    # ==================== MENU ITEM CRUD ====================
    
    def create_menu_item(self, item_data: Dict[str, Any]) -> models.MenuItem:
        """
        Create a new menu item
        """
        try:
            # Handle ingredients JSON
            if 'ingredients' in item_data and isinstance(item_data['ingredients'], dict):
                item_data['ingredients'] = json.dumps(item_data['ingredients'])
            
            item = models.MenuItem(**item_data)
            self.db.add(item)
            self.db.commit()
            self.db.refresh(item)
            logger.info(f"Created menu item: {item.dish_name}")
            return item
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating menu item: {e}")
            raise
    
    def get_menu_item(self, item_id: int) -> Optional[models.MenuItem]:
        """
        Get menu item by ID
        """
        return self.db.query(models.MenuItem).filter(
            models.MenuItem.id == item_id
        ).first()
    
    def get_menu_item_by_name(self, dish_name: str) -> Optional[models.MenuItem]:
        """
        Get menu item by name
        """
        return self.db.query(models.MenuItem).filter(
            models.MenuItem.dish_name == dish_name
        ).first()
    
    def get_all_menu_items(self, 
                          category: Optional[str] = None,
                          vegetarian_only: bool = False,
                          skip: int = 0, 
                          limit: int = 100) -> List[models.MenuItem]:
        """
        Get all menu items with optional filters
        """
        query = self.db.query(models.MenuItem)
        
        if category:
            query = query.filter(models.MenuItem.category == category)
        
        if vegetarian_only:
            query = query.filter(models.MenuItem.is_vegetarian == True)
        
        return query.order_by(models.MenuItem.dish_name).offset(skip).limit(limit).all()
    
    def update_menu_item(self, 
                        item_id: int, 
                        updates: Dict[str, Any]) -> Optional[models.MenuItem]:
        """
        Update menu item
        """
        try:
            item = self.get_menu_item(item_id)
            if not item:
                return None
            
            # Handle ingredients JSON
            if 'ingredients' in updates and isinstance(updates['ingredients'], dict):
                updates['ingredients'] = json.dumps(updates['ingredients'])
            
            for key, value in updates.items():
                if hasattr(item, key):
                    setattr(item, key, value)
            
            item.updated_at = datetime.utcnow()
            self.db.commit()
            self.db.refresh(item)
            logger.info(f"Updated menu item: {item_id}")
            return item
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating menu item {item_id}: {e}")
            raise
    
    def update_popularity_score(self, item_id: int, score: float) -> Optional[models.MenuItem]:
        """
        Update popularity score of menu item
        """
        if 0 <= score <= 1:
            return self.update_menu_item(item_id, {'popularity_score': score})
        return None
    
    def delete_menu_item(self, item_id: int) -> bool:
        """
        Delete menu item
        """
        try:
            item = self.get_menu_item(item_id)
            if not item:
                return False
            
            self.db.delete(item)
            self.db.commit()
            logger.info(f"Deleted menu item: {item_id}")
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error deleting menu item {item_id}: {e}")
            raise
    
    # ==================== PROCUREMENT CRUD ====================
    
    def create_procurement(self, procurement_data: Dict[str, Any]) -> models.Procurement:
        """
        Create a new procurement record
        """
        try:
            # Calculate total cost if not provided
            if 'total_cost' not in procurement_data:
                procurement_data['total_cost'] = procurement_data.get('quantity', 0) * procurement_data.get('unit_price', 0)
            
            procurement = models.Procurement(**procurement_data)
            self.db.add(procurement)
            self.db.commit()
            self.db.refresh(procurement)
            logger.info(f"Created procurement: {procurement.ingredient_name} - {procurement.quantity}kg")
            return procurement
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating procurement: {e}")
            raise
    
    def get_procurement(self, procurement_id: int) -> Optional[models.Procurement]:
        """
        Get procurement by ID
        """
        return self.db.query(models.Procurement).filter(
            models.Procurement.id == procurement_id
        ).first()
    
    def get_procurement_by_date(self, 
                               target_date: Union[datetime, date],
                               ingredient: Optional[str] = None) -> List[models.Procurement]:
        """
        Get procurement records by date
        """
        if isinstance(target_date, date) and not isinstance(target_date, datetime):
            start_date = datetime.combine(target_date, datetime.min.time())
            end_date = datetime.combine(target_date, datetime.max.time())
        else:
            start_date = target_date
            end_date = target_date + timedelta(days=1) - timedelta(microseconds=1)
        
        query = self.db.query(models.Procurement).filter(
            and_(
                models.Procurement.date >= start_date,
                models.Procurement.date <= end_date
            )
        )
        
        if ingredient:
            query = query.filter(models.Procurement.ingredient_name == ingredient)
        
        return query.all()
    
    def get_procurement_by_ingredient(self,
                                     ingredient: str,
                                     start_date: Optional[datetime] = None,
                                     end_date: Optional[datetime] = None) -> List[models.Procurement]:
        """
        Get procurement records for a specific ingredient
        """
        query = self.db.query(models.Procurement).filter(
            models.Procurement.ingredient_name == ingredient
        )
        
        if start_date:
            query = query.filter(models.Procurement.date >= start_date)
        if end_date:
            query = query.filter(models.Procurement.date <= end_date)
        
        return query.order_by(desc(models.Procurement.date)).all()
    
    def get_recent_procurement(self, 
                              days: int = 7,
                              ingredient: Optional[str] = None) -> List[models.Procurement]:
        """
        Get procurement records from last N days
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        return self.get_procurement_by_ingredient(ingredient, start_date=cutoff_date)
    
    def update_procurement(self, 
                          procurement_id: int, 
                          updates: Dict[str, Any]) -> Optional[models.Procurement]:
        """
        Update procurement record
        """
        try:
            procurement = self.get_procurement(procurement_id)
            if not procurement:
                return None
            
            for key, value in updates.items():
                if hasattr(procurement, key):
                    setattr(procurement, key, value)
            
            # Recalculate total cost if quantity or price changed
            if 'quantity' in updates or 'unit_price' in updates:
                procurement.total_cost = procurement.quantity * procurement.unit_price
            
            procurement.updated_at = datetime.utcnow()
            self.db.commit()
            self.db.refresh(procurement)
            logger.info(f"Updated procurement: {procurement_id}")
            return procurement
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating procurement {procurement_id}: {e}")
            raise
    
    def update_actual_used(self, procurement_id: int, actual_used: float) -> Optional[models.Procurement]:
        """
        Update actual used quantity for procurement
        """
        return self.update_procurement(procurement_id, {'actual_used': actual_used})
    
    def delete_procurement(self, procurement_id: int) -> bool:
        """
        Delete procurement record
        """
        try:
            procurement = self.get_procurement(procurement_id)
            if not procurement:
                return False
            
            self.db.delete(procurement)
            self.db.commit()
            logger.info(f"Deleted procurement: {procurement_id}")
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error deleting procurement {procurement_id}: {e}")
            raise
    
    # ==================== WASTE LOG CRUD ====================
    
    def create_waste_log(self, waste_data: Dict[str, Any]) -> models.WasteLog:
        """
        Create a new waste log
        """
        try:
            # Calculate environmental impact if not provided
            if 'co2_footprint' not in waste_data:
                from app.config import config
                waste_data['co2_footprint'] = waste_data.get('waste_quantity', 0) * config.CO2_PER_KG_FOOD
                waste_data['water_footprint'] = waste_data.get('waste_quantity', 0) * config.WATER_PER_KG_FOOD
                waste_data['cost_loss'] = waste_data.get('waste_quantity', 0) * config.WASTE_COST_PER_KG
            
            waste = models.WasteLog(**waste_data)
            self.db.add(waste)
            self.db.commit()
            self.db.refresh(waste)
            logger.info(f"Created waste log: {waste.id}")
            return waste
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating waste log: {e}")
            raise
    
    def get_waste_log(self, waste_id: int) -> Optional[models.WasteLog]:
        """
        Get waste log by ID
        """
        return self.db.query(models.WasteLog).filter(
            models.WasteLog.id == waste_id
        ).first()
    
    def get_waste_logs_by_date_range(self,
                                     start_date: datetime,
                                     end_date: datetime,
                                     meal_type: Optional[str] = None,
                                     dish_name: Optional[str] = None) -> List[models.WasteLog]:
        """
        Get waste logs within date range with optional filters
        """
        query = self.db.query(models.WasteLog).filter(
            and_(
                models.WasteLog.date >= start_date,
                models.WasteLog.date <= end_date
            )
        )
        
        if meal_type:
            query = query.filter(models.WasteLog.meal_type == meal_type)
        
        if dish_name:
            query = query.filter(models.WasteLog.dish_name == dish_name)
        
        return query.order_by(desc(models.WasteLog.date)).all()
    
    def get_waste_by_reason(self, 
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> Dict[str, float]:
        """
        Get waste quantities grouped by reason
        """
        query = self.db.query(
            models.WasteLog.waste_reason,
            func.sum(models.WasteLog.waste_quantity).label('total_waste')
        )
        
        if start_date:
            query = query.filter(models.WasteLog.date >= start_date)
        if end_date:
            query = query.filter(models.WasteLog.date <= end_date)
        
        result = query.group_by(models.WasteLog.waste_reason).all()
        
        return {row.waste_reason: float(row.total_waste) for row in result}
    
    def get_waste_summary(self, 
                         days: int = 30,
                         group_by: str = 'meal_type') -> Dict[str, Dict[str, float]]:
        """
        Get waste summary grouped by specified field
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        group_column = getattr(models.WasteLog, group_by)
        
        query = self.db.query(
            group_column,
            func.sum(models.WasteLog.waste_quantity).label('total_waste'),
            func.sum(models.WasteLog.co2_footprint).label('total_co2'),
            func.sum(models.WasteLog.cost_loss).label('total_cost'),
            func.count(models.WasteLog.id).label('incident_count')
        ).filter(models.WasteLog.date >= cutoff_date
        ).group_by(group_column)
        
        result = {}
        for row in query.all():
            result[row[0]] = {
                'waste_quantity': float(row.total_waste),
                'co2_footprint': float(row.total_co2),
                'cost_loss': float(row.total_cost),
                'incident_count': row.incident_count
            }
        
        return result
    
    def delete_waste_log(self, waste_id: int) -> bool:
        """
        Delete waste log
        """
        try:
            waste = self.get_waste_log(waste_id)
            if not waste:
                return False
            
            self.db.delete(waste)
            self.db.commit()
            logger.info(f"Deleted waste log: {waste_id}")
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error deleting waste log {waste_id}: {e}")
            raise
    
    # ==================== BULK OPERATIONS ====================
    
    def bulk_create_meals(self, meals_data: List[Dict[str, Any]]) -> List[models.MealConsumption]:
        """
        Create multiple meal consumption records in bulk
        """
        try:
            meals = []
            for data in meals_data:
                # Calculate waste if needed
                if 'quantity_wasted' not in data:
                    data['quantity_wasted'] = data.get('quantity_prepared', 0) - data.get('quantity_consumed', 0)
                
                # Set day of week
                if 'date' in data:
                    if isinstance(data['date'], (datetime, date)):
                        data['day_of_week'] = data['date'].weekday()
                        data['is_weekend'] = data['day_of_week'] >= 5
                
                meals.append(models.MealConsumption(**data))
            
            self.db.add_all(meals)
            self.db.commit()
            
            for meal in meals:
                self.db.refresh(meal)
            
            logger.info(f"Bulk created {len(meals)} meal records")
            return meals
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error in bulk create meals: {e}")
            raise
    
    def bulk_create_waste_logs(self, waste_data_list: List[Dict[str, Any]]) -> List[models.WasteLog]:
        """
        Create multiple waste logs in bulk
        """
        try:
            from app.config import config
            
            wastes = []
            for data in waste_data_list:
                # Calculate environmental impact
                if 'co2_footprint' not in data:
                    data['co2_footprint'] = data.get('waste_quantity', 0) * config.CO2_PER_KG_FOOD
                    data['water_footprint'] = data.get('waste_quantity', 0) * config.WATER_PER_KG_FOOD
                    data['cost_loss'] = data.get('waste_quantity', 0) * config.WASTE_COST_PER_KG
                
                wastes.append(models.WasteLog(**data))
            
            self.db.add_all(wastes)
            self.db.commit()
            
            for waste in wastes:
                self.db.refresh(waste)
            
            logger.info(f"Bulk created {len(wastes)} waste logs")
            return wastes
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error in bulk create waste logs: {e}")
            raise
    
    # ==================== AGGREGATION & ANALYTICS ====================
    
    def get_daily_stats(self, target_date: Union[datetime, date]) -> Dict[str, Any]:
        """
        Get daily statistics for a specific date
        """
        meals = self.get_meal_by_date(target_date)
        
        if not meals:
            return {
                'date': target_date,
                'total_meals': 0,
                'total_students': 0,
                'total_prepared': 0,
                'total_consumed': 0,
                'total_wasted': 0,
                'waste_percentage': 0
            }
        
        total_prepared = sum(m.quantity_prepared for m in meals)
        total_consumed = sum(m.quantity_consumed for m in meals)
        total_wasted = sum(m.quantity_wasted for m in meals)
        total_students = sum(m.student_count for m in meals)
        
        return {
            'date': target_date,
            'total_meals': len(meals),
            'total_students': total_students,
            'total_prepared': total_prepared,
            'total_consumed': total_consumed,
            'total_wasted': total_wasted,
            'waste_percentage': (total_wasted / total_prepared * 100) if total_prepared > 0 else 0,
            'meals_breakdown': [
                {
                    'meal_type': m.meal_type,
                    'dish': m.dish_name,
                    'students': m.student_count,
                    'prepared': m.quantity_prepared,
                    'consumed': m.quantity_consumed,
                    'wasted': m.quantity_wasted,
                    'waste_pct': m.waste_percentage
                }
                for m in meals
            ]
        }
    
    def get_monthly_trends(self, year: int, month: int) -> Dict[str, Any]:
        """
        Get monthly trends for a specific month
        """
        start_date = datetime(year, month, 1)
        if month == 12:
            end_date = datetime(year + 1, 1, 1) - timedelta(microseconds=1)
        else:
            end_date = datetime(year, month + 1, 1) - timedelta(microseconds=1)
        
        meals = self.get_all_meals(start_date=start_date, end_date=end_date)
        
        if not meals:
            return {
                'year': year,
                'month': month,
                'total_days': 0,
                'total_meals': 0,
                'total_students': 0
            }
        
        # Group by day
        daily_stats = {}
        for meal in meals:
            day = meal.date.day
            if day not in daily_stats:
                daily_stats[day] = {
                    'total_students': 0,
                    'total_prepared': 0,
                    'total_wasted': 0
                }
            
            daily_stats[day]['total_students'] += meal.student_count
            daily_stats[day]['total_pre