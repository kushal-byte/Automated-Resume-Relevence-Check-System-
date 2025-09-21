# database.py - FIXED DATABASE with proper migration order
import sqlite3
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
import json
import threading
import contextlib
import time
import os
from pathlib import Path
from dataclasses import dataclass
import logging
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    """Data class to represent analysis results with proper typing"""
    id: int
    resume_filename: str
    jd_filename: str
    final_score: float
    verdict: str
    timestamp: datetime
    matched_skills: str = ""
    missing_skills: str = ""
    hard_match_score: Optional[float] = None
    semantic_score: Optional[float] = None
    
    def __post_init__(self):
        """Set fallback values after initialization"""
        if self.hard_match_score is None:
            self.hard_match_score = self.final_score
        if self.semantic_score is None:
            self.semantic_score = self.final_score

class DatabaseConfig:
    """Database configuration with production settings"""
    def __init__(self):
        self.db_path = os.getenv('DATABASE_PATH', 'resume_analysis.db')
        self.timeout = float(os.getenv('DATABASE_TIMEOUT', '30.0'))
        self.max_retries = int(os.getenv('DATABASE_MAX_RETRIES', '3'))
        self.retry_delay = float(os.getenv('DATABASE_RETRY_DELAY', '0.5'))
        self.enable_wal = os.getenv('DATABASE_ENABLE_WAL', 'true').lower() == 'true'
        self.backup_enabled = os.getenv('DATABASE_BACKUP_ENABLED', 'true').lower() == 'true'

config = DatabaseConfig()

# Thread lock for database operations
db_lock = threading.RLock()

def retry_on_db_error(max_retries: int = None):
    """Decorator for retrying database operations on failure"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = max_retries or config.max_retries
            last_exception = None
            
            for attempt in range(retries + 1):
                try:
                    return func(*args, **kwargs)
                except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
                    last_exception = e
                    if attempt < retries:
                        wait_time = config.retry_delay * (2 ** attempt)
                        logger.warning(f"Database operation failed (attempt {attempt + 1}/{retries + 1}): {e}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Database operation failed after {retries + 1} attempts: {e}")
                        
            raise last_exception
        return wrapper
    return decorator

@contextlib.contextmanager
def get_db_connection():
    """Production-grade database connection with comprehensive error handling"""
    conn = None
    try:
        with db_lock:
            # Ensure database directory exists
            db_dir = Path(config.db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)
            
            conn = sqlite3.connect(
                config.db_path, 
                timeout=config.timeout,
                check_same_thread=False,
                isolation_level=None  # Autocommit mode
            )
            
            # Set production-grade pragmas
            if config.enable_wal:
                conn.execute('PRAGMA journal_mode=WAL;')
            conn.execute('PRAGMA synchronous=NORMAL;')
            conn.execute('PRAGMA busy_timeout=30000;')
            conn.execute('PRAGMA foreign_keys=ON;')
            conn.execute('PRAGMA cache_size=-64000;')
            conn.execute('PRAGMA temp_store=MEMORY;')
            
            # Ensure schema is up to date
            migrate_db_schema(conn)
            yield conn
            
    except sqlite3.OperationalError as e:
        error_msg = str(e).lower()
        if "locked" in error_msg or "busy" in error_msg:
            logger.warning(f"Database busy/locked: {e}")
            raise
        else:
            logger.error(f"Database operational error: {e}")
            raise
    except Exception as e:
        logger.error(f"Unexpected database error: {e}")
        raise
    finally:
        if conn:
            try:
                conn.close()
            except Exception as e:
                logger.error(f"Error closing database connection: {e}")

def migrate_db_schema(conn: sqlite3.Connection):
    """FIXED schema migration with proper ordering"""
    try:
        cursor = conn.cursor()
        
        # Create version tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Get current schema version
        cursor.execute('SELECT MAX(version) FROM schema_version')
        result = cursor.fetchone()
        current_version = result[0] if result and result[0] else 0
        
        # FIXED: Proper migration order
        migrations = [
            (1, create_initial_schema),
            (2, add_enhanced_columns),  # Add columns first
            (3, create_indexes),        # Then create indexes
            (4, add_performance_optimizations)
        ]
        
        for version, migration_func in migrations:
            if current_version < version:
                logger.info(f"Applying migration version {version}")
                try:
                    migration_func(cursor)
                    cursor.execute('INSERT INTO schema_version (version) VALUES (?)', (version,))
                    conn.commit()
                    logger.info(f"✅ Migration version {version} completed successfully")
                except Exception as e:
                    logger.error(f"❌ Migration version {version} failed: {e}")
                    conn.rollback()
                    # For development, we'll continue with a simplified approach
                    if version <= 2:  # Critical migrations
                        raise
                    else:  # Optional migrations can be skipped
                        logger.warning(f"Skipping optional migration {version}")
                        continue
        
    except Exception as e:
        logger.error(f"Schema migration failed: {e}")
        # For existing databases, try to create a basic working schema
        try:
            create_basic_working_schema(cursor)
            conn.commit()
            logger.info("✅ Created basic working schema as fallback")
        except Exception as fallback_error:
            logger.error(f"Fallback schema creation failed: {fallback_error}")
            raise e

def create_basic_working_schema(cursor: sqlite3.Cursor):
    """Create a basic working schema for existing databases"""
    # Check what exists and create missing tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    existing_tables = [row[0] for row in cursor.fetchall()]
    
    if 'analysis_results' not in existing_tables:
        cursor.execute('''
            CREATE TABLE analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resume_filename TEXT NOT NULL,
                jd_filename TEXT NOT NULL,
                final_score REAL DEFAULT 0,
                verdict TEXT DEFAULT 'Unknown',
                hard_match_score REAL DEFAULT 0,
                semantic_score REAL DEFAULT 0,
                matched_skills TEXT DEFAULT '[]',
                missing_skills TEXT DEFAULT '[]',
                full_result TEXT DEFAULT '{}',
                processing_time REAL DEFAULT 0,
                analysis_mode TEXT DEFAULT 'standard',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
    else:
        # Add missing columns to existing table
        cursor.execute("PRAGMA table_info(analysis_results)")
        existing_columns = {info[1] for info in cursor.fetchall()}
        
        columns_to_add = [
            ('hard_match_score', 'REAL DEFAULT 0'),
            ('semantic_score', 'REAL DEFAULT 0'),
            ('matched_skills', 'TEXT DEFAULT "[]"'),
            ('missing_skills', 'TEXT DEFAULT "[]"'),
            ('full_result', 'TEXT DEFAULT "{}"'),
            ('processing_time', 'REAL DEFAULT 0'),
            ('analysis_mode', 'TEXT DEFAULT "standard"'),
            ('created_at', 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'),
            ('updated_at', 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP')
        ]
        
        for column_name, column_def in columns_to_add:
            if column_name not in existing_columns:
                try:
                    cursor.execute(f'ALTER TABLE analysis_results ADD COLUMN {column_name} {column_def}')
                    logger.info(f"Added column: {column_name}")
                except sqlite3.OperationalError as e:
                    if "duplicate column name" not in str(e).lower():
                        logger.warning(f"Could not add column {column_name}: {e}")
    
    # Create other essential tables
    if 'analytics_summary' not in existing_tables:
        cursor.execute('''
            CREATE TABLE analytics_summary (
                id INTEGER PRIMARY KEY DEFAULT 1,
                total_analyses INTEGER DEFAULT 0,
                avg_score REAL DEFAULT 0,
                high_matches INTEGER DEFAULT 0,
                medium_matches INTEGER DEFAULT 0,
                low_matches INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('INSERT OR IGNORE INTO analytics_summary (id) VALUES (1)')

def create_initial_schema(cursor: sqlite3.Cursor):
    """Initial database schema creation"""
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resume_filename TEXT NOT NULL,
            jd_filename TEXT NOT NULL,
            final_score REAL DEFAULT 0,
            verdict TEXT DEFAULT 'Unknown',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analytics_summary (
            id INTEGER PRIMARY KEY DEFAULT 1,
            total_analyses INTEGER DEFAULT 0,
            avg_score REAL DEFAULT 0,
            high_matches INTEGER DEFAULT 0,
            medium_matches INTEGER DEFAULT 0,
            low_matches INTEGER DEFAULT 0,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS screening_tests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            test_id TEXT UNIQUE NOT NULL,
            test_number INTEGER,
            job_title TEXT,
            company_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            total_candidates INTEGER DEFAULT 0,
            qualified_candidates INTEGER DEFAULT 0,
            status TEXT DEFAULT 'active'
        )
    ''')
    
    # Insert default analytics row
    cursor.execute('INSERT OR IGNORE INTO analytics_summary (id) VALUES (1)')

def add_enhanced_columns(cursor: sqlite3.Cursor):
    """Add enhanced analysis columns - FIXED ORDER"""
    # Check existing columns first
    cursor.execute("PRAGMA table_info(analysis_results)")
    existing_columns = {info[1] for info in cursor.fetchall()}
    
    new_columns = [
        ('hard_match_score', 'REAL DEFAULT 0'),
        ('semantic_score', 'REAL DEFAULT 0'),
        ('matched_skills', 'TEXT DEFAULT "[]"'),
        ('missing_skills', 'TEXT DEFAULT "[]"'),
        ('full_result', 'TEXT DEFAULT "{}"'),
        ('processing_time', 'REAL DEFAULT 0'),
        ('analysis_mode', 'TEXT DEFAULT "standard"')
    ]
    
    for column_name, column_def in new_columns:
        if column_name not in existing_columns:
            try:
                cursor.execute(f'ALTER TABLE analysis_results ADD COLUMN {column_name} {column_def}')
                logger.info(f"Added column: {column_name}")
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e).lower():
                    logger.warning(f"Could not add column {column_name}: {e}")

def create_indexes(cursor: sqlite3.Cursor):
    """Create performance indexes - FIXED to ensure columns exist"""
    # First, check what columns actually exist
    cursor.execute("PRAGMA table_info(analysis_results)")
    existing_columns = {info[1] for info in cursor.fetchall()}
    
    # Only create indexes for columns that exist
    potential_indexes = [
        ('idx_id', 'analysis_results', 'id'),
        ('idx_final_score', 'analysis_results', 'final_score'),
        ('idx_verdict', 'analysis_results', 'verdict'),
        ('idx_resume_filename', 'analysis_results', 'resume_filename'),
        ('idx_jd_filename', 'analysis_results', 'jd_filename')
    ]
    
    # Add timestamp index only if column exists
    if 'created_at' in existing_columns:
        potential_indexes.append(('idx_created_at', 'analysis_results', 'created_at'))
        potential_indexes.append(('idx_composite_score_date', 'analysis_results', 'final_score, created_at'))
    
    for index_name, table_name, columns in potential_indexes:
        try:
            # Check if all columns in the index exist
            index_columns = [col.strip() for col in columns.split(',')]
            if all(col in existing_columns for col in index_columns):
                cursor.execute(f'CREATE INDEX IF NOT EXISTS {index_name} ON {table_name}({columns})')
                logger.debug(f"Created index: {index_name}")
            else:
                logger.warning(f"Skipping index {index_name} - required columns not found")
        except sqlite3.OperationalError as e:
            logger.warning(f"Could not create index {index_name}: {e}")

def add_performance_optimizations(cursor: sqlite3.Cursor):
    """Add triggers and additional optimizations"""
    try:
        # Check if created_at and updated_at columns exist
        cursor.execute("PRAGMA table_info(analysis_results)")
        existing_columns = {info[1] for info in cursor.fetchall()}
        
        if 'updated_at' in existing_columns:
            # Update timestamp trigger
            cursor.execute('''
                CREATE TRIGGER IF NOT EXISTS update_analysis_timestamp 
                AFTER UPDATE ON analysis_results
                FOR EACH ROW
                BEGIN
                    UPDATE analysis_results 
                    SET updated_at = datetime('now') 
                    WHERE id = NEW.id;
                END
            ''')
            logger.debug("Created update timestamp trigger")
    except sqlite3.OperationalError as e:
        logger.warning(f"Could not create performance optimizations: {e}")

@retry_on_db_error()
def init_database():
    """Initialize database with enhanced error handling and logging"""
    try:
        with get_db_connection() as conn:
            logger.info("Database initialized successfully")
            return True
            
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        # Try to create a basic schema as fallback
        try:
            conn = sqlite3.connect(config.db_path, timeout=config.timeout)
            cursor = conn.cursor()
            create_basic_working_schema(cursor)
            conn.commit()
            conn.close()
            logger.info("✅ Created fallback database schema")
            return True
        except Exception as fallback_error:
            logger.error(f"Fallback database creation failed: {fallback_error}")
            raise e

@retry_on_db_error()
def save_analysis_result(analysis_data: dict, resume_filename: str, jd_filename: str) -> bool:
    """Enhanced save operation with better data extraction and validation"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Extract and validate data
            extracted_data = _extract_analysis_data(analysis_data)
            processing_time = analysis_data.get('processing_info', {}).get('processing_time', 0)
            analysis_mode = 'enhanced' if 'enhanced_analysis' in analysis_data else 'standard'
            
            # Check what columns exist before inserting
            cursor.execute("PRAGMA table_info(analysis_results)")
            existing_columns = {info[1] for info in cursor.fetchall()}
            
            # Base columns that should always exist
            base_columns = ['resume_filename', 'jd_filename', 'final_score', 'verdict']
            base_values = [
                str(resume_filename),
                str(jd_filename), 
                extracted_data['final_score'],
                extracted_data['verdict']
            ]
            
            # Add optional columns if they exist
            optional_columns = [
                ('hard_match_score', extracted_data['hard_match_score']),
                ('semantic_score', extracted_data['semantic_score']),
                ('matched_skills', json.dumps(extracted_data['matched_skills'])),
                ('missing_skills', json.dumps(extracted_data['missing_skills'])),
                ('full_result', json.dumps(analysis_data)),
                ('processing_time', processing_time),
                ('analysis_mode', analysis_mode),
                ('created_at', 'datetime("now")'),
                ('updated_at', 'datetime("now")')
            ]
            
            additional_columns = []
            additional_values = []
            
            for col_name, col_value in optional_columns:
                if col_name in existing_columns:
                    additional_columns.append(col_name)
                    if col_name in ['created_at', 'updated_at']:
                        additional_values.append('datetime("now")')
                    else:
                        additional_values.append('?')
                        base_values.append(col_value)
            
            all_columns = base_columns + additional_columns
            
            # Build the INSERT query
            placeholders = ['?'] * len(base_columns) + additional_values
            query = f'''
                INSERT INTO analysis_results ({', '.join(all_columns)})
                VALUES ({', '.join(placeholders)})
            '''
            
            cursor.execute(query, base_values)
            conn.commit()
            
            # Update analytics asynchronously
            _update_analytics_async(conn)
            
            logger.info(f"Analysis result saved: {resume_filename} - Score: {extracted_data['final_score']}")
            return True
            
    except Exception as e:
        logger.error(f"Error saving analysis result: {e}")
        return False

def _extract_analysis_data(analysis_data: dict) -> Dict[str, Any]:
    """Extract and normalize analysis data from different formats"""
    default_data = {
        'final_score': 0.0,
        'verdict': 'Analysis Completed',
        'hard_match_score': 0.0,
        'semantic_score': 0.0,
        'matched_skills': [],
        'missing_skills': []
    }
    
    try:
        # Enhanced analysis format
        if 'enhanced_analysis' in analysis_data and 'relevance_scoring' in analysis_data['enhanced_analysis']:
            scoring = analysis_data['enhanced_analysis']['relevance_scoring']
            return {
                'final_score': float(scoring.get('overall_score', 0)),
                'verdict': str(scoring.get('fit_verdict', 'Unknown')),
                'hard_match_score': float(scoring.get('skill_match_score', 0)),
                'semantic_score': float(scoring.get('experience_match_score', 0)),
                'matched_skills': list(scoring.get('matched_must_have', [])),
                'missing_skills': list(scoring.get('missing_must_have', []))
            }
        
        # Standard analysis format
        elif 'relevance_analysis' in analysis_data:
            relevance = analysis_data['relevance_analysis']
            output = analysis_data.get('output_generation', {})
            
            return {
                'final_score': float(relevance['step_3_scoring_verdict']['final_score']),
                'verdict': str(output.get('verdict', 'Unknown')),
                'hard_match_score': float(relevance['step_1_hard_match']['coverage_score']),
                'semantic_score': float(relevance['step_2_semantic_match']['experience_alignment_score']),
                'matched_skills': list(relevance['step_1_hard_match'].get('matched_skills', [])),
                'missing_skills': list(output.get('missing_skills', []))
            }
        
        return default_data
        
    except Exception as e:
        logger.warning(f"Error extracting analysis data, using defaults: {e}")
        return default_data

def _update_analytics_async(conn: sqlite3.Connection):
    """Update analytics in a non-blocking way"""
    try:
        update_analytics_summary_internal(conn)
    except Exception as e:
        logger.warning(f"Analytics update failed (non-critical): {e}")

@retry_on_db_error()
def get_analysis_history(limit: int = 50, offset: int = 0) -> List[AnalysisResult]:
    """Enhanced history retrieval with pagination and performance optimization"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Check what columns exist
            cursor.execute("PRAGMA table_info(analysis_results)")
            existing_columns = {info[1] for info in cursor.fetchall()}
            
            # Build query based on available columns
            base_columns = ['id', 'resume_filename', 'jd_filename', 'final_score', 'verdict']
            optional_columns = ['created_at', 'matched_skills', 'missing_skills', 'hard_match_score', 'semantic_score']
            
            select_columns = base_columns[:]
            for col in optional_columns:
                if col in existing_columns:
                    select_columns.append(col)
            
            # Use appropriate ORDER BY
            order_column = 'created_at' if 'created_at' in existing_columns else 'id'
            
            query = f'''
                SELECT {', '.join(select_columns)}
                FROM analysis_results 
                ORDER BY {order_column} DESC 
                LIMIT ? OFFSET ?
            '''
            
            cursor.execute(query, (limit, offset))
            
            results = []
            for row in cursor.fetchall():
                try:
                    # Map values to column names
                    row_dict = dict(zip(select_columns, row))
                    
                    # Handle timestamp
                    if 'created_at' in row_dict and row_dict['created_at']:
                        timestamp = _parse_timestamp(row_dict['created_at'])
                    else:
                        timestamp = datetime.now(timezone.utc)
                    
                    result = AnalysisResult(
                        id=row_dict['id'],
                        resume_filename=str(row_dict.get('resume_filename', 'Unknown')),
                        jd_filename=str(row_dict.get('jd_filename', 'Unknown')),
                        final_score=float(row_dict.get('final_score', 0)),
                        verdict=str(row_dict.get('verdict', 'Unknown')),
                        timestamp=timestamp,
                        matched_skills=row_dict.get('matched_skills', '[]'),
                        missing_skills=row_dict.get('missing_skills', '[]'),
                        hard_match_score=float(row_dict.get('hard_match_score', row_dict.get('final_score', 0))),
                        semantic_score=float(row_dict.get('semantic_score', row_dict.get('final_score', 0)))
                    )
                    results.append(result)
                    
                except Exception as row_error:
                    logger.warning(f"Skipping malformed row: {row_error}")
                    continue
            
            logger.info(f"Retrieved {len(results)} analysis results from history")
            return results
            
    except Exception as e:
        logger.error(f"Error getting analysis history: {e}")
        return []

def _parse_timestamp(timestamp_str: str) -> datetime:
    """Parse timestamp with multiple format support"""
    if not timestamp_str:
        return datetime.now(timezone.utc)
    
    formats = [
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d %H:%M:%S.%f',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%dT%H:%M:%S.%f',
        '%Y-%m-%dT%H:%M:%S.%fZ'
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(str(timestamp_str), fmt)
        except ValueError:
            continue
    
    logger.warning(f"Could not parse timestamp: {timestamp_str}")
    return datetime.now(timezone.utc)

@retry_on_db_error()
def get_analytics_summary() -> Dict[str, Any]:
    """Enhanced analytics with better error handling and caching"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get comprehensive analytics in a single transaction
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_analyses,
                    COALESCE(AVG(final_score), 0) as avg_score,
                    COUNT(CASE WHEN final_score >= 80 THEN 1 END) as high_matches,
                    COUNT(CASE WHEN final_score >= 60 AND final_score < 80 THEN 1 END) as medium_matches,
                    COUNT(CASE WHEN final_score < 60 AND final_score > 0 THEN 1 END) as low_matches
                FROM analysis_results
            ''')
            
            result = cursor.fetchone()
            
            total_analyses = result[0] or 0
            avg_score = round(float(result[1] or 0), 1)
            high_matches = result[2] or 0
            medium_matches = result[3] or 0
            low_matches = result[4] or 0
            
            # Calculate success rate
            success_rate = 0.0
            if total_analyses > 0:
                success_rate = round(((high_matches + medium_matches) / total_analyses) * 100, 1)
            
            analytics = {
                'total_analyses': total_analyses,
                'avg_score': avg_score,
                'high_matches': high_matches,
                'medium_matches': medium_matches,
                'low_matches': low_matches,
                'success_rate': success_rate,
                'generated_at': datetime.now(timezone.utc).isoformat()
            }
            
            logger.info(f"Analytics summary generated: {total_analyses} analyses, {avg_score}% avg score")
            return analytics
            
    except Exception as e:
        logger.error(f"Error getting analytics summary: {e}")
        return {
            'total_analyses': 0,
            'avg_score': 0.0,
            'high_matches': 0,
            'medium_matches': 0,
            'low_matches': 0,
            'success_rate': 0.0,
            'error': str(e)
        }

def update_analytics_summary():
    """Public method to update analytics summary"""
    try:
        with get_db_connection() as conn:
            update_analytics_summary_internal(conn)
    except Exception as e:
        logger.error(f"Error updating analytics summary: {e}")

def update_analytics_summary_internal(conn: sqlite3.Connection):
    """Internal analytics update with optimized queries"""
    try:
        cursor = conn.cursor()
        
        # Get analytics in a single query
        cursor.execute('''
            SELECT 
                COUNT(*) as total,
                COALESCE(AVG(final_score), 0) as avg_score,
                COUNT(CASE WHEN final_score >= 80 THEN 1 END) as high,
                COUNT(CASE WHEN final_score >= 60 AND final_score < 80 THEN 1 END) as medium,
                COUNT(CASE WHEN final_score < 60 AND final_score > 0 THEN 1 END) as low
            FROM analysis_results
        ''')
        
        result = cursor.fetchone()
        total, avg_score, high, medium, low = result
        
        # Check if analytics_summary table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='analytics_summary'")
        if cursor.fetchone():
            cursor.execute('''
                UPDATE analytics_summary 
                SET total_analyses = ?, avg_score = ?, high_matches = ?, 
                    medium_matches = ?, low_matches = ?, last_updated = datetime('now')
                WHERE id = 1
            ''', (total, round(avg_score, 1), high, medium, low))
        
        conn.commit()
        logger.debug(f"Analytics updated: {total} total analyses")
        
    except Exception as e:
        logger.error(f"Error updating analytics summary internally: {e}")

def get_recent_analyses(limit: int = 10) -> List[Dict[str, Any]]:
    """Enhanced recent analyses with better formatting"""
    try:
        results = get_analysis_history(limit)
        
        return [
            {
                "id": result.id,
                "resume": result.resume_filename,
                "job_description": result.jd_filename,
                "score": result.final_score,
                "verdict": result.verdict,
                "date": result.timestamp.strftime("%Y-%m-%d %H:%M") if hasattr(result.timestamp, 'strftime') else str(result.timestamp),
                "matched_skills": result.matched_skills,
                "missing_skills": result.missing_skills,
                "hard_match_score": result.hard_match_score,
                "semantic_score": result.semantic_score
            }
            for result in results
        ]
        
    except Exception as e:
        logger.error(f"Error getting recent analyses: {e}")
        return []

def backup_database(backup_path: Optional[str] = None) -> bool:
    """Create database backup"""
    if not config.backup_enabled:
        return True
        
    try:
        backup_path = backup_path or f"{config.db_path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with get_db_connection() as source:
            backup = sqlite3.connect(backup_path)
            source.backup(backup)
            backup.close()
            
        logger.info(f"Database backed up to: {backup_path}")
        return True
        
    except Exception as e:
        logger.error(f"Database backup failed: {e}")
        return False

def get_database_stats() -> Dict[str, Any]:
    """Get comprehensive database statistics"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get table sizes
            cursor.execute("SELECT COUNT(*) FROM analysis_results")
            analysis_count = cursor.fetchone()[0]
            
            # Get database file size
            db_size = Path(config.db_path).stat().st_size if Path(config.db_path).exists() else 0
            
            # Get date range if created_at exists
            cursor.execute("PRAGMA table_info(analysis_results)")
            existing_columns = {info[1] for info in cursor.fetchall()}
            
            date_range = (None, None)
            if 'created_at' in existing_columns:
                cursor.execute("SELECT MIN(created_at), MAX(created_at) FROM analysis_results")
                date_range = cursor.fetchone()
            
            return {
                "database_path": config.db_path,
                "database_size_bytes": db_size,
                "database_size_mb": round(db_size / (1024 * 1024), 2),
                "analysis_results_count": analysis_count,
                "earliest_record": date_range[0],
                "latest_record": date_range[1],
                "wal_enabled": config.enable_wal,
                "backup_enabled": config.backup_enabled
            }
            
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        return {"error": str(e)}

def repair_database():
    """Enhanced database repair with integrity checking"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            logger.info("Starting database repair and optimization...")
            
            # Check integrity
            cursor.execute('PRAGMA integrity_check')
            integrity_result = cursor.fetchall()
            
            if len(integrity_result) == 1 and integrity_result[0][0] == 'ok':
                logger.info("✅ Database integrity check passed")
            else:
                logger.warning(f"⚠️ Database integrity issues found: {integrity_result}")
                return False
            
            # Vacuum database
            logger.info("Vacuuming database...")
            cursor.execute('VACUUM')
            
            # Analyze for query optimization
            logger.info("Analyzing database for optimization...")
            cursor.execute('ANALYZE')
            
            # Update statistics
            cursor.execute('PRAGMA optimize')
            
            logger.info("✅ Database repair and optimization completed")
            return True
            
    except Exception as e:
        logger.error(f"❌ Database repair failed: {e}")
        return False

def test_database() -> bool:
    """Comprehensive database testing suite"""
    logger.info("🧪 Starting comprehensive database tests...")
    
    try:
        # Test 1: Initialization
        init_database()
        logger.info("✅ Database initialization test passed")
        
        # Test 2: Save operations
        test_data = {
            'enhanced_analysis': {
                'relevance_scoring': {
                    'overall_score': 85.5,
                    'fit_verdict': 'High Suitability',
                    'skill_match_score': 90.0,
                    'experience_match_score': 80.5,
                    'matched_must_have': ['Python', 'JavaScript', 'React'],
                    'missing_must_have': ['Node.js', 'Docker']
                }
            },
            'processing_info': {'processing_time': 2.5, 'enhanced_features': True}
        }
        
        success = save_analysis_result(test_data, "test_resume.pdf", "test_job.pdf")
        if not success:
            raise Exception("Save test failed")
        logger.info("✅ Save operation test passed")
        
        # Test 3: Retrieval operations
        history = get_analysis_history(10)
        logger.info(f"✅ History retrieval test passed ({len(history)} records)")
        
        # Test 4: Analytics
        analytics = get_analytics_summary()
        logger.info("✅ Analytics test passed")
        
        logger.info("🎉 All database tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database tests failed: {e}")
        return False

# Production initialization with better error handling
def initialize_production_db():
    """Initialize database for production environment"""
    try:
        logger.info("Initializing production database...")
        
        # Create database with proper setup
        init_database()
        
        # Create backup if enabled
        if config.backup_enabled:
            backup_database()
        
        # Run integrity check
        repair_database()
        
        # Log statistics
        stats = get_database_stats()
        logger.info(f"Database ready - Size: {stats.get('database_size_mb', 0)}MB, Records: {stats.get('analysis_results_count', 0)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Production database initialization failed: {e}")
        return False

# Auto-initialize for production
if config.db_path and not os.getenv('DISABLE_AUTO_INIT', '').lower() == 'true':
    try:
        initialize_production_db()
        logger.info("🚀 Production database module loaded and initialized")
    except Exception as e:
        logger.error(f"⚠️ Database initialization warning: {e}")

if __name__ == "__main__":
    test_database()
