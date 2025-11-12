import sqlite3
from datetime import datetime

# Connect to database
conn = sqlite3.connect('database/assistant.db')
cursor = conn.cursor()

# List all tables
print("\n" + "="*60)
print("DATABASE TABLES")
print("="*60)
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print("Available tables:", [t[0] for t in tables])
print()

# View Users table
print("\n" + "="*60)
print("USERS TABLE")
print("="*60)
cursor.execute("SELECT * FROM users")
users = cursor.fetchall()
print(f"{'ID':<5} {'Username':<20} {'Created At':<25}")
print("-"*60)
for row in users:
    print(f"{row[0]:<5} {row[1]:<20} {row[2]:<25}")

# View Tasks table
print("\n" + "="*110)
print("TASKS TABLE")
print("="*110)
cursor.execute("SELECT * FROM tasks ORDER BY created_at")
tasks = cursor.fetchall()
print(f"{'ID':<5} {'User':<6} {'Title':<20} {'Description':<30} {'Due Date':<12} {'Due Time':<10} {'Status':<10}")
print("-"*110)
for row in tasks:
    task_id, user_id, title, description, created_at, due_date, due_time, reminder_date, reminder_time, status = row
    
    # Format values
    desc_str = (description[:27] + "...") if description and len(description) > 30 else (description or "")
    due_date_str = due_date if due_date else "Not set"
    due_time_str = due_time if due_time else "Not set"
    status_str = status if status else "pending"
    
    print(f"{task_id:<5} {user_id:<6} {title[:18]:<20} {desc_str:<30} {due_date_str:<12} {due_time_str:<10} {status_str:<10}")
    
    # Show reminder info on next line if set
    if reminder_date and reminder_time:
        print(f"{'':>31} Reminder: {reminder_date} at {reminder_time}")

print("\n" + "="*60)
print(f"Total Users: {len(users)}")
print(f"Total Tasks: {len(tasks)}")
print("="*60)

conn.close()
