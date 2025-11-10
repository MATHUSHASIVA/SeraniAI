"""
Utility functions for the Contextual Personal Assistant.
"""

import re
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union
import dateparser

def parse_natural_time(time_str: str) -> Optional[datetime]:
    """
    Parse natural language time expressions into datetime objects.
    
    Examples:
    - "tomorrow evening" -> datetime object for tomorrow at 6 PM
    - "next Monday at 2 PM" -> datetime object for next Monday at 2 PM
    - "in 2 hours" -> datetime object 2 hours from now
    """
    try:
        # Use dateparser for initial parsing
        parsed_time = dateparser.parse(time_str)
        
        if parsed_time:
            # Handle relative expressions that might need adjustment
            now = datetime.now()
            
            # If just "evening" is mentioned, default to 6 PM
            if "evening" in time_str.lower() and parsed_time.hour < 12:
                parsed_time = parsed_time.replace(hour=18, minute=0, second=0)
            
            # If just "morning" is mentioned, default to 9 AM
            elif "morning" in time_str.lower() and parsed_time.hour > 12:
                parsed_time = parsed_time.replace(hour=9, minute=0, second=0)
            
            # If just "afternoon" is mentioned, default to 2 PM
            elif "afternoon" in time_str.lower() and (parsed_time.hour < 12 or parsed_time.hour > 18):
                parsed_time = parsed_time.replace(hour=14, minute=0, second=0)
            
            return parsed_time
        
        return None
        
    except Exception as e:
        print(f"Error parsing time '{time_str}': {e}")
        return None

def parse_duration(duration_str: str) -> Optional[int]:
    """
    Parse duration expressions into minutes.
    
    Examples:
    - "2 hours" -> 120
    - "30 minutes" -> 30
    - "1.5 hours" -> 90
    - "an hour" -> 60
    """
    try:
        duration_str = duration_str.lower().strip()
        
        # Pattern matching for common duration expressions
        patterns = [
            (r'(\d+\.?\d*)\s*h(?:ours?)?', 60),  # "2 hours", "1.5h"
            (r'(\d+)\s*m(?:in(?:utes?)?)?', 1),   # "30 minutes", "45 min"
            (r'an?\s+hour', 60),                   # "an hour", "a hour"
            (r'half\s+an?\s+hour', 30),           # "half an hour"
            (r'(\d+)\s*and\s*a\s*half\s*hours?', 90),  # "2 and a half hours"
        ]
        
        for pattern, multiplier in patterns:
            match = re.search(pattern, duration_str)
            if match:
                if pattern in [r'an?\s+hour', r'half\s+an?\s+hour']:
                    return multiplier
                else:
                    number = float(match.group(1))
                    return int(number * multiplier)
        
        # Try to extract just numbers
        numbers = re.findall(r'\d+\.?\d*', duration_str)
        if numbers:
            number = float(numbers[0])
            
            # Guess unit based on context
            if 'hour' in duration_str:
                return int(number * 60)
            elif 'min' in duration_str:
                return int(number)
            else:
                # Default to hours if number is small, minutes if large
                return int(number * 60) if number <= 8 else int(number)
        
        return None
        
    except Exception as e:
        print(f"Error parsing duration '{duration_str}': {e}")
        return None

def format_time_natural(dt: datetime) -> str:
    """
    Format datetime object into natural language.
    
    Examples:
    - Today at 6:00 PM
    - Tomorrow at 9:30 AM
    - Monday, December 4 at 2:15 PM
    """
    try:
        now = datetime.now()
        
        # Check if it's today
        if dt.date() == now.date():
            return f"today at {dt.strftime('%I:%M %p')}"
        
        # Check if it's tomorrow
        elif dt.date() == (now.date() + timedelta(days=1)):
            return f"tomorrow at {dt.strftime('%I:%M %p')}"
        
        # Check if it's within the next week
        elif dt.date() <= (now.date() + timedelta(days=7)):
            return f"{dt.strftime('%A')} at {dt.strftime('%I:%M %p')}"
        
        # Otherwise, use full date
        else:
            return dt.strftime('%A, %B %d at %I:%M %p')
        
    except Exception as e:
        print(f"Error formatting time: {e}")
        return str(dt)

def format_duration_natural(minutes: int) -> str:
    """
    Format duration in minutes into natural language.
    
    Examples:
    - 30 -> "30 minutes"
    - 90 -> "1 hour 30 minutes"
    - 120 -> "2 hours"
    """
    try:
        if minutes < 60:
            return f"{minutes} minute{'s' if minutes != 1 else ''}"
        
        hours = minutes // 60
        remaining_minutes = minutes % 60
        
        if remaining_minutes == 0:
            return f"{hours} hour{'s' if hours != 1 else ''}"
        else:
            return f"{hours} hour{'s' if hours != 1 else ''} {remaining_minutes} minute{'s' if remaining_minutes != 1 else ''}"
        
    except Exception as e:
        print(f"Error formatting duration: {e}")
        return f"{minutes} minutes"

def extract_task_keywords(text: str) -> List[str]:
    """
    Extract task-related keywords from text for better intent recognition.
    """
    task_keywords = [
        # Action words
        'remind', 'schedule', 'plan', 'book', 'set', 'create', 'add', 'make',
        'prepare', 'work on', 'start', 'finish', 'complete', 'do',
        
        # Task types
        'meeting', 'appointment', 'call', 'presentation', 'project', 'task',
        'deadline', 'event', 'session', 'break', 'lunch', 'workout',
        
        # Time indicators
        'tomorrow', 'today', 'tonight', 'morning', 'afternoon', 'evening',
        'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
        'next week', 'this week', 'later', 'soon', 'in an hour', 'in minutes'
    ]
    
    text_lower = text.lower()
    found_keywords = []
    
    for keyword in task_keywords:
        if keyword in text_lower:
            found_keywords.append(keyword)
    
    return found_keywords

def clean_text_for_embedding(text: str) -> str:
    """
    Clean and normalize text for embedding generation.
    """
    try:
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-]', '', text)
        
        # Normalize case (keep original for now, may lowercase later)
        return text
        
    except Exception as e:
        print(f"Error cleaning text: {e}")
        return text

def validate_openai_api_key(api_key: str) -> bool:
    """
    Validate OpenAI API key format.
    """
    if not api_key:
        return False
    
    # OpenAI API keys typically start with 'sk-' and are 51 characters long
    if api_key.startswith('sk-') and len(api_key) == 51:
        return True
    
    return False

def safe_json_parse(json_str: str, default: dict = None) -> dict:
    """
    Safely parse JSON string with fallback to default.
    """
    try:
        # Clean up common JSON formatting issues
        json_str = json_str.strip()
        if json_str.startswith('```json'):
            json_str = json_str[7:]
        if json_str.startswith('```'):
            json_str = json_str[3:]
        if json_str.endswith('```'):
            json_str = json_str[:-3]
        
        return json.loads(json_str.strip())
        
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        return default or {}
    except Exception as e:
        print(f"Unexpected error parsing JSON: {e}")
        return default or {}

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to specified length with optional suffix.
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)].rsplit(' ', 1)[0] + suffix

def get_time_of_day_greeting() -> str:
    """
    Get appropriate greeting based on current time of day.
    """
    hour = datetime.now().hour
    
    if 5 <= hour < 12:
        return "Good morning"
    elif 12 <= hour < 17:
        return "Good afternoon"
    elif 17 <= hour < 22:
        return "Good evening"
    else:
        return "Hello"