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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    due_date TEXT,
                    due_time TEXT,
                    reminder_date TEXT,
                    reminder_time TEXT,
                    status TEXT DEFAULT 'pending',
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
                   due_date: str = None, due_time: str = None, 
                   reminder_date: str = None, reminder_time: str = None,
                   status: str = 'pending') -> int:
        """Create a new task."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO tasks (user_id, title, description, due_date, due_time, 
                                 reminder_date, reminder_time, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (user_id, title, description, due_date, due_time, reminder_date, reminder_time, status))
            conn.commit()
            return cursor.lastrowid
    
    def get_user_tasks(self, user_id: int) -> list:
        """Get all tasks for a user."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
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
                    'created_at': row[4],
                    'due_date': row[5],
                    'due_time': row[6],
                    'reminder_date': row[7],
                    'reminder_time': row[8],
                    'status': row[9] if len(row) > 9 else 'pending'
                })
            return tasks
    
    def update_task(self, task_id: int, title: str = None, description: str = None,
                   due_date: str = None, due_time: str = None,
                   reminder_date: str = None, reminder_time: str = None,
                   status: str = None):
        """Update a task."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            updates = []
            params = []
            
            if title is not None:
                updates.append("title = ?")
                params.append(title)
            if description is not None:
                updates.append("description = ?")
                params.append(description)
            if due_date is not None:
                updates.append("due_date = ?")
                params.append(due_date)
            if due_time is not None:
                updates.append("due_time = ?")
                params.append(due_time)
            if reminder_date is not None:
                updates.append("reminder_date = ?")
                params.append(reminder_date)
            if reminder_time is not None:
                updates.append("reminder_time = ?")
                params.append(reminder_time)
            if status is not None:
                updates.append("status = ?")
                params.append(status)
            
            if updates:
                params.append(task_id)
                query = f"UPDATE tasks SET {', '.join(updates)} WHERE id = ?"
                cursor.execute(query, params)
                conn.commit()
    
    def delete_task(self, task_id: int):
        """Delete a task."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
            conn.commit()
    
    def check_schedule_conflicts(self, user_id: int, due_date: str, 
                               due_time: str, exclude_task_id: int = None) -> list:
        """Check for scheduling conflicts with existing tasks."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = """
                SELECT * FROM tasks WHERE user_id = ? 
                AND due_date = ? AND due_time = ?
            """
            params = [user_id, due_date, due_time]
            
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
                    'due_date': row[5],
                    'due_time': row[6]
                })
            return conflicts
    
    def get_tasks_by_status(self, user_id: int, status: str) -> list:
        """Get tasks filtered by status."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM tasks WHERE user_id = ? AND status = ?
                ORDER BY created_at DESC
            """, (user_id, status))
            
            rows = cursor.fetchall()
            tasks = []
            for row in rows:
                tasks.append({
                    'id': row[0],
                    'user_id': row[1],
                    'title': row[2],
                    'description': row[3],
                    'created_at': row[4],
                    'due_date': row[5],
                    'due_time': row[6],
                    'reminder_date': row[7],
                    'reminder_time': row[8],
                    'status': row[9] if len(row) > 9 else 'pending'
                })
            return tasks
    
    def update_task_status(self, task_id: int, status: str):
        """Update only the status of a task."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))
            conn.commit()
    
