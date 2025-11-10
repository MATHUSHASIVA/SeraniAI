import sqlite3
import os
from datetime import datetime
from typing import Optional

class DatabaseManager:
    def __init__(self, db_path: str = "database/assistant.db"):
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self):
        """Create and return a database connection."""
        return sqlite3.connect(self.db_path)
    
    def init_database(self):
        """Initialize database with required tables."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    preferences TEXT DEFAULT '{}'
                )
            """)
            
            # Tasks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    duration_minutes INTEGER,
                    status TEXT DEFAULT 'pending',
                    priority TEXT DEFAULT 'medium',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            conn.commit()
    
    def create_user(self, username: str) -> int:
        """Create a new user and return user ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (username) VALUES (?)", (username,))
            conn.commit()
            return cursor.lastrowid
    
    def get_user_by_username(self, username: str) -> Optional[dict]:
        """Get user by username."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            row = cursor.fetchone()
            if row:
                return {
                    'id': row[0],
                    'username': row[1],
                    'created_at': row[2],
                    'preferences': row[3]
                }
        return None
    
    def get_or_create_user(self, username: str) -> int:
        """Get existing user or create new one."""
        user = self.get_user_by_username(username)
        if user:
            return user['id']
        return self.create_user(username)
    
    def create_task(self, user_id: int, title: str, description: str = None,
                   start_time: datetime = None, duration_minutes: int = None,
                   priority: str = 'medium') -> int:
        """Create a new task."""
        end_time = None
        if start_time and duration_minutes:
            from datetime import timedelta
            end_time = start_time + timedelta(minutes=duration_minutes)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO tasks (user_id, title, description, start_time, end_time, 
                                 duration_minutes, priority)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (user_id, title, description, start_time, end_time, duration_minutes, priority))
            conn.commit()
            return cursor.lastrowid
    
    def get_user_tasks(self, user_id: int, status: str = None) -> list:
        """Get all tasks for a user, optionally filtered by status."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if status:
                cursor.execute("""
                    SELECT * FROM tasks WHERE user_id = ? AND status = ? 
                    ORDER BY created_at DESC
                """, (user_id, status))
            else:
                cursor.execute("""
                    SELECT * FROM tasks WHERE user_id = ? 
                    ORDER BY created_at DESC
                """, (user_id,))
            
            rows = cursor.fetchall()
            tasks = []
            for row in rows:
                tasks.append({
                    'id': row[0],
                    'user_id': row[1],
                    'title': row[2],
                    'description': row[3],
                    'start_time': row[4],
                    'end_time': row[5],
                    'duration_minutes': row[6],
                    'status': row[7],
                    'priority': row[8],
                    'created_at': row[9],
                    'updated_at': row[10]
                })
            return tasks
    
    def update_task_status(self, task_id: int, status: str):
        """Update task status."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE tasks SET status = ?, updated_at = CURRENT_TIMESTAMP 
                WHERE id = ?
            """, (status, task_id))
            conn.commit()
    
    def check_schedule_conflicts(self, user_id: int, start_time: datetime, 
                               end_time: datetime, exclude_task_id: int = None) -> list:
        """Check for scheduling conflicts with existing tasks."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = """
                SELECT * FROM tasks WHERE user_id = ? AND status = 'pending'
                AND start_time IS NOT NULL AND end_time IS NOT NULL
                AND (
                    (start_time <= ? AND end_time > ?) OR
                    (start_time < ? AND end_time >= ?) OR
                    (start_time >= ? AND end_time <= ?)
                )
            """
            params = [user_id, start_time, start_time, end_time, end_time, start_time, end_time]
            
            if exclude_task_id:
                query += " AND id != ?"
                params.append(exclude_task_id)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            conflicts = []
            for row in rows:
                conflicts.append({
                    'id': row[0],
                    'title': row[2],
                    'start_time': row[4],
                    'end_time': row[5]
                })
            return conflicts
    
