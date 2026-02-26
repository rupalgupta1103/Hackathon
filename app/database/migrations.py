"""
Database migrations and schema management for Smart Mess Optimization System
Handles database versioning, schema updates, and data migrations
"""

from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from datetime import datetime
import logging
import json
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import hashlib

from app.database.database import engine, Base
from app.database import models
from app.config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Migration versions and their SQL scripts
MIGRATIONS = [
    {
        'version': 1,
        'name': 'initial_schema',
        'description': 'Initial database schema creation',
        'sql': """
            -- Initial schema is handled by SQLAlchemy Base.metadata.create_all()
        """
    },
    {
        'version': 2,
        'name': 'add_sustainability_columns',
        'description': 'Add sustainability tracking columns',
        'sql': """
            -- Add columns to waste_log table
            ALTER TABLE waste_log ADD COLUMN water_footprint FLOAT DEFAULT 0.0;
            ALTER TABLE waste_log ADD COLUMN cost_loss FLOAT DEFAULT 0.0;
            
            -- Add indexes for better performance
            CREATE INDEX IF NOT EXISTS idx_waste_log_date ON waste_log(date);
            CREATE INDEX IF NOT EXISTS idx_waste_log_meal_type ON waste_log(meal_type);
        """
    },
    {
        'version': 3,
        'name': 'add_menu_item_scores',
        'description': 'Add scoring columns to menu items',
        'sql': """
            -- Add score columns to menu_items
            ALTER TABLE menu_items ADD COLUMN nutritional_score FLOAT DEFAULT 0.0;
            ALTER TABLE menu_items ADD COLUMN sustainability_score FLOAT DEFAULT 0.0;
            
            -- Add dietary preference columns
            ALTER TABLE menu_items ADD COLUMN is_vegetarian BOOLEAN DEFAULT 1;
            ALTER TABLE menu_items ADD COLUMN is_vegan BOOLEAN DEFAULT 0;
            ALTER TABLE menu_items ADD COLUMN is_gluten_free BOOLEAN DEFAULT 0;
        """
    },
    {
        'version': 4,
        'name': 'add_procurement_quality',
        'description': 'Add quality tracking to procurement',
        'sql': """
            -- Add quality columns to procurement
            ALTER TABLE procurement ADD COLUMN quality_rating INTEGER;
            ALTER TABLE procurement ADD COLUMN delivery_delay INTEGER;
            ALTER TABLE procurement ADD COLUMN vendor_contact VARCHAR(50);
            
            -- Add index on ingredient_name
            CREATE INDEX IF NOT EXISTS idx_procurement_ingredient ON procurement(ingredient_name);
        """
    },
    {
        'version': 5,
        'name': 'add_user_feedback',
        'description': 'Add user feedback table for reinforcement learning',
        'sql': """
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATETIME NOT NULL,
                meal_type VARCHAR(50) NOT NULL,
                dish_name VARCHAR(100) NOT NULL,
                taste_rating INTEGER,
                quantity_rating INTEGER,
                overall_rating INTEGER NOT NULL,
                comments TEXT,
                user_id VARCHAR(100),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_user_feedback_date ON user_feedback(date);
            CREATE INDEX IF NOT EXISTS idx_user_feedback_dish ON user_feedback(dish_name);
        """
    },
    {
        'version': 6,
        'name': 'add_seasonal_tracking',
        'description': 'Add seasonal tracking to menu items',
        'sql': """
            ALTER TABLE menu_items ADD COLUMN seasonal_months VARCHAR(100);
            ALTER TABLE menu_items ADD COLUMN preparation_time INTEGER DEFAULT 30;
            
            -- Add index on category
            CREATE INDEX IF NOT EXISTS idx_menu_items_category ON menu_items(category);
        """
    }
]

class MigrationManager:
    """
    Manages database migrations and schema versions
    """
    
    def __init__(self):
        self.engine = engine
        self.migrations_table = 'schema_migrations'
        self.migrations = MIGRATIONS
    
    def create_migrations_table(self) -> None:
        """Create migrations tracking table if it doesn't exist"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS {self.migrations_table} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        version INTEGER NOT NULL UNIQUE,
                        name VARCHAR(100) NOT NULL,
                        description TEXT,
                        applied_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        checksum VARCHAR(64),
                        duration_ms INTEGER,
                        status VARCHAR(20) DEFAULT 'success'
                    )
                """))
                conn.commit()
                logger.info(f"Migrations table '{self.migrations_table}' created/verified")
        except Exception as e:
            logger.error(f"Failed to create migrations table: {e}")
            raise
    
    def get_applied_migrations(self) -> List[Dict[str, Any]]:
        """Get list of already applied migrations"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(f"""
                    SELECT version, name, applied_at, checksum, status
                    FROM {self.migrations_table}
                    ORDER BY version
                """))
                
                migrations = []
                for row in result:
                    migrations.append({
                        'version': row[0],
                        'name': row[1],
                        'applied_at': row[2],
                        'checksum': row[3],
                        'status': row[4]
                    })
                
                return migrations
        except Exception:
            # Table might not exist yet
            return []
    
    def calculate_checksum(self, sql: str) -> str:
        """Calculate checksum of migration SQL"""
        return hashlib.sha256(sql.encode('utf-8')).hexdigest()
    
    def migration_needs_rerun(self, migration: Dict, applied: Dict) -> bool:
        """Check if migration needs to be rerun due to changes"""
        if applied.get('status') == 'failed':
            return True
        
        current_checksum = self.calculate_checksum(migration['sql'])
        return current_checksum != applied.get('checksum')
    
    def backup_database(self) -> Optional[str]:
        """Create a backup of the database before migration"""
        try:
            # Get database path from engine URL
            db_url = str(self.engine.url)
            if 'sqlite' in db_url:
                db_path = db_url.replace('sqlite:///', '')
                if db_path.startswith('./'):
                    db_path = db_path[2:]
                
                # Create backups directory if it doesn't exist
                backup_dir = Path('backups')
                backup_dir.mkdir(exist_ok=True)
                
                # Create backup filename with timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_path = backup_dir / f"db_backup_{timestamp}.db"
                
                # Copy database file
                if os.path.exists(db_path):
                    shutil.copy2(db_path, backup_path)
                    logger.info(f"Database backed up to {backup_path}")
                    
                    # Keep only last 10 backups
                    self.cleanup_old_backups(backup_dir, keep=10)
                    
                    return str(backup_path)
            
            return None
        except Exception as e:
            logger.warning(f"Backup failed (continuing anyway): {e}")
            return None
    
    def cleanup_old_backups(self, backup_dir: Path, keep: int = 10) -> None:
        """Keep only the most recent N backups"""
        try:
            backups = sorted(backup_dir.glob("db_backup_*.db"), key=os.path.getmtime)
            if len(backups) > keep:
                for backup in backups[:-keep]:
                    backup.unlink()
                    logger.info(f"Removed old backup: {backup}")
        except Exception as e:
            logger.warning(f"Backup cleanup failed: {e}")
    
    def run_migration(self, migration: Dict, conn) -> bool:
        """Run a single migration"""
        version = migration['version']
        name = migration['name']
        
        logger.info(f"Running migration v{version}: {name}")
        
        try:
            # Start transaction
            with conn.begin():
                # Execute migration SQL
                if migration['sql'] and not migration['sql'].startswith('-- Initial'):
                    for statement in migration['sql'].split(';'):
                        if statement.strip():
                            conn.execute(text(statement.strip()))
                
                # Record migration
                checksum = self.calculate_checksum(migration['sql'])
                conn.execute(text(f"""
                    INSERT OR REPLACE INTO {self.migrations_table} 
                    (version, name, description, checksum, status)
                    VALUES (:version, :name, :description, :checksum, 'success')
                """), {
                    'version': version,
                    'name': name,
                    'description': migration['description'],
                    'checksum': checksum
                })
                
                logger.info(f"Migration v{version} completed successfully")
                return True
                
        except Exception as e:
            logger.error(f"Migration v{version} failed: {e}")
            
            # Record failure
            try:
                conn.execute(text(f"""
                    INSERT OR REPLACE INTO {self.migrations_table} 
                    (version, name, description, status)
                    VALUES (:version, :name, :description, 'failed')
                """), {
                    'version': version,
                    'name': name,
                    'description': migration['description']
                })
                conn.commit()
            except:
                pass
            
            return False
    
    def run_migrations(self, target_version: Optional[int] = None) -> Tuple[bool, List[str]]:
        """
        Run pending migrations up to target_version
        Returns: (success, list of messages)
        """
        messages = []
        
        try:
            # Ensure migrations table exists
            self.create_migrations_table()
            
            # Get applied migrations
            applied = self.get_applied_migrations()
            applied_versions = {m['version'] for m in applied if m['status'] == 'success'}
            failed_versions = {m['version'] for m in applied if m['status'] == 'failed'}
            
            # Determine which migrations to run
            to_run = []
            for migration in self.migrations:
                version = migration['version']
                
                if target_version and version > target_version:
                    continue
                
                if version in failed_versions:
                    # Re-run failed migrations
                    to_run.append(migration)
                    messages.append(f"Re-running failed migration v{version}")
                elif version not in applied_versions:
                    # Run new migrations
                    to_run.append(migration)
                    messages.append(f"New migration v{version} to run")
                else:
                    # Check if existing migration needs rerun
                    applied_migration = next(m for m in applied if m['version'] == version)
                    if self.migration_needs_rerun(migration, applied_migration):
                        to_run.append(migration)
                        messages.append(f"Migration v{version} changed, re-running")
            
            if not to_run:
                messages.append("No pending migrations")
                return True, messages
            
            # Create backup
            backup_path = self.backup_database()
            if backup_path:
                messages.append(f"Backup created at {backup_path}")
            
            # Run migrations
            with self.engine.connect() as conn:
                for migration in to_run:
                    success = self.run_migration(migration, conn)
                    if not success:
                        messages.append(f"Migration v{migration['version']} failed")
                        return False, messages
            
            messages.append("All migrations completed successfully")
            return True, messages
            
        except Exception as e:
            error_msg = f"Migration process failed: {e}"
            logger.error(error_msg)
            messages.append(error_msg)
            return False, messages
    
    def get_migration_status(self) -> Dict[str, Any]:
        """Get detailed migration status"""
        try:
            applied = self.get_applied_migrations()
            applied_versions = {m['version'] for m in applied}
            
            pending = []
            for migration in self.migrations:
                if migration['version'] not in applied_versions:
                    pending.append(migration)
            
            latest_version = max(applied_versions) if applied_versions else 0
            latest_migration = next(
                (m for m in self.migrations if m['version'] == latest_version),
                None
            )
            
            return {
                'database_url': str(self.engine.url),
                'total_migrations': len(self.migrations),
                'applied_migrations': len(applied),
                'pending_migrations': len(pending),
                'latest_version': latest_version,
                'latest_migration_name': latest_migration['name'] if latest_migration else None,
                'latest_applied_at': max((m['applied_at'] for m in applied), default=None),
                'pending': pending,
                'applied': applied
            }
        except Exception as e:
            logger.error(f"Failed to get migration status: {e}")
            return {'error': str(e)}
    
    def rollback_last_migration(self) -> Tuple[bool, str]:
        """Rollback the last migration (if possible)"""
        try:
            applied = self.get_applied_migrations()
            if not applied:
                return False, "No migrations to rollback"
            
            last_migration = applied[-1]
            version = last_migration['version']
            
            # Find corresponding migration
            migration = next(
                (m for m in self.migrations if m['version'] == version),
                None
            )
            
            if not migration:
                return False, f"Migration v{version} not found"
            
            # Create backup before rollback
            backup_path = self.backup_database()
            
            # For SQLite, we can't easily rollback schema changes
            # So we'll just mark it as rolled back and suggest restore from backup
            with self.engine.connect() as conn:
                conn.execute(text(f"""
                    UPDATE {self.migrations_table}
                    SET status = 'rolled_back'
                    WHERE version = :version
                """), {'version': version})
                conn.commit()
            
            msg = f"Migration v{version} marked as rolled back. "
            if backup_path:
                msg += f"Database backed up at {backup_path}. "
            msg += "Please restore from backup if needed."
            
            return True, msg
            
        except Exception as e:
            return False, f"Rollback failed: {e}"
    
    def verify_schema_integrity(self) -> Tuple[bool, List[str]]:
        """Verify that database schema matches models"""
        issues = []
        
        try:
            inspector = inspect(self.engine)
            
            # Get all tables from models
            model_tables = Base.metadata.tables.keys()
            
            # Get actual tables in database
            db_tables = inspector.get_table_names()
            
            # Check for missing tables
            for table in model_tables:
                if table not in db_tables and table != self.migrations_table:
                    issues.append(f"Missing table: {table}")
            
            # Check columns for each table
            for table in db_tables:
                if table == self.migrations_table:
                    continue
                
                model_columns = {}
                if table in Base.metadata.tables:
                    model_columns = {
                        col.name: col.type.python_type
                        for col in Base.metadata.tables[table].columns
                    }
                
                db_columns = {
                    col['name']: col['type']
                    for col in inspector.get_columns(table)
                }
                
                # Check for missing columns
                for col_name in model_columns:
                    if col_name not in db_columns:
                        issues.append(f"Missing column: {table}.{col_name}")
                
                # Check for extra columns (optional)
                for col_name in db_columns:
                    if col_name not in model_columns and col_name not in ['created_at', 'updated_at']:
                        issues.append(f"Extra column in DB: {table}.{col_name}")
            
            return len(issues) == 0, issues
            
        except Exception as e:
            return False, [f"Schema verification failed: {e}"]
    
    def export_schema(self, output_path: Optional[str] = None) -> str:
        """Export current database schema to SQL file"""
        if not output_path:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"schema_export_{timestamp}.sql"
        
        try:
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()
            
            with open(output_path, 'w') as f:
                f.write(f"-- Database Schema Export\n")
                f.write(f"-- Generated: {datetime.now()}\n")
                f.write(f"-- Database: {self.engine.url}\n\n")
                
                for table in tables:
                    f.write(f"-- Table: {table}\n")
                    
                    # Get create table statement
                    if 'sqlite' in str(self.engine.url):
                        # For SQLite, get CREATE statement
                        with self.engine.connect() as conn:
                            result = conn.execute(text(
                                f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table}'"
                            ))
                            row = result.fetchone()
                            if row and row[0]:
                                f.write(f"{row[0]};\n\n")
                    
                    # Get indexes
                    indexes = inspector.get_indexes(table)
                    for idx in indexes:
                        idx_name = idx['name']
                        idx_cols = ', '.join(idx['column_names'])
                        unique = "UNIQUE " if idx['unique'] else ""
                        f.write(f"CREATE {unique}INDEX IF NOT EXISTS {idx_name} ON {table}({idx_cols});\n")
                    
                    f.write("\n")
            
            logger.info(f"Schema exported to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Schema export failed: {e}")
            raise


# ==================== Public Functions ====================

migration_manager = MigrationManager()

def run_migrations(target_version: Optional[int] = None) -> Tuple[bool, List[str]]:
    """
    Run all pending migrations
    """
    logger.info("Starting database migrations...")
    return migration_manager.run_migrations(target_version)

def create_backup() -> Optional[str]:
    """
    Create a database backup
    """
    return migration_manager.backup_database()

def restore_from_backup(backup_path: str) -> bool:
    """
    Restore database from backup
    """
    try:
        # Get database path
        db_url = str(engine.url)
        if 'sqlite' in db_url:
            db_path = db_url.replace('sqlite:///', '')
            if db_path.startswith('./'):
                db_path = db_path[2:]
            
            # Restore from backup
            shutil.copy2(backup_path, db_path)
            logger.info(f"Database restored from {backup_path}")
            return True
        else:
            logger.error("Restore only supported for SQLite")
            return False
            
    except Exception as e:
        logger.error(f"Restore failed: {e}")
        return False

def get_migration_status() -> Dict[str, Any]:
    """
    Get current migration status
    """
    return migration_manager.get_migration_status()

def verify_schema() -> Tuple[bool, List[str]]:
    """
    Verify database schema integrity
    """
    return migration_manager.verify_schema_integrity()

def reset_database(confirm: bool = False) -> bool:
    """
    Reset database (drop all tables and recreate)
    Use with caution!
    """
    if not confirm:
        logger.warning("Reset cancelled: confirmation required")
        return False
    
    try:
        # Create backup before reset
        backup_path = create_backup()
        logger.info(f"Backup created at {backup_path}")
        
        # Drop all tables
        Base.metadata.drop_all(bind=engine)
        logger.info("All tables dropped")
        
        # Recreate tables
        Base.metadata.create_all(bind=engine)
        logger.info("All tables recreated")
        
        # Run migrations to ensure everything is up to date
        run_migrations()
        
        logger.info("Database reset completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Reset failed: {e}")
        return False


# ==================== CLI Interface ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Database migration tool")
    parser.add_argument('command', choices=['migrate', 'status', 'backup', 'verify', 'export', 'reset'])
    parser.add_argument('--version', type=int, help='Target migration version')
    parser.add_argument('--backup-file', type=str, help='Backup file path for restore')
    
    args = parser.parse_args()
    
    if args.command == 'migrate':
        success, messages = run_migrations(args.version)
        for msg in messages:
            print(f"- {msg}")
        print(f"\n{'✓' if success else '✗'} Migration {'succeeded' if success else 'failed'}")
        
    elif args.command == 'status':
        status = get_migration_status()
        print("\n📊 Migration Status")
        print("=" * 50)
        print(f"Database: {status.get('database_url')}")
        print(f"Total migrations: {status.get('total_migrations')}")
        print(f"Applied: {status.get('applied_migrations')}")
        print(f"Pending: {status.get('pending_migrations')}")
        print(f"Latest version: {status.get('latest_version')}")
        
        if status.get('pending'):
            print("\n⏳ Pending migrations:")
            for m in status['pending']:
                print(f"  v{m['version']}: {m['name']} - {m['description']}")
        
    elif args.command == 'backup':
        backup_path = create_backup()
        if backup_path:
            print(f"✓ Backup created: {backup_path}")
        else:
            print("✗ Backup failed")
            
    elif args.command == 'verify':
        success, issues = verify_schema()
        if success:
            print("✓ Schema integrity verified")
        else:
            print("✗ Schema issues found:")
            for issue in issues:
                print(f"  - {issue}")
                
    elif args.command == 'export':
        path = migration_manager.export_schema()
        print(f"✓ Schema exported to: {path}")
        
    elif args.command == 'reset':
        print("⚠️  This will DELETE ALL DATA and recreate the database!")
        confirm = input("Type 'yes' to confirm: ")
        if confirm == 'yes':
            success = reset_database(confirm=True)
            print(f"{'✓' if success else '✗'} Database reset {'completed' if success else 'failed'}")
        else:
            print("Reset cancelled")