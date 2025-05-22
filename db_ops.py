import shutil
import sqlite3
import logging
import os
import json
import numpy as np
from typing import List
from vietnamese_error_simulation import generate_vietnamese_errors

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = "db/labels.db"

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy types."""
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def init_db():
    """Initialize SQLite database with required tables."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS categories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS labels (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        value TEXT NOT NULL,
        category_id INTEGER NOT NULL,
        is_dynamic INTEGER NOT NULL,
        FOREIGN KEY (category_id) REFERENCES categories(id),
        UNIQUE(value, category_id)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS corrections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ocr_output TEXT NOT NULL,
        corrected TEXT NOT NULL
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS abbreviations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        abbr TEXT NOT NULL,
        full_form TEXT NOT NULL,
        category_id INTEGER NOT NULL,
        FOREIGN KEY (category_id) REFERENCES categories(id),
        UNIQUE(abbr, category_id)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS training_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        start_time TEXT NOT NULL,
        end_time TEXT NOT NULL,
        labels TEXT NOT NULL,
        validation_results TEXT NOT NULL,
        model_path TEXT NOT NULL
    )''')
    conn.commit()
    conn.close()
    logger.info(f"Database initialized at {DB_PATH}")

def get_categories():
    """Retrieve all categories."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name FROM categories")
    categories = [{"id": row[0], "name": row[1]} for row in c.fetchall()]
    conn.close()
    return categories

def get_labels_by_category(category_id):
    """Retrieve labels for a specific category."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, value, is_dynamic FROM labels WHERE category_id = ?", (category_id,))
    labels = [{"id": row[0], "value": row[1], "is_dynamic": row[2]} for row in c.fetchall()]
    conn.close()
    return labels

def add_category(name):
    """Add a new category."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO categories (name) VALUES (?)", (name,))
        conn.commit()
        logger.info(f"Added category: {name}")
    except sqlite3.IntegrityError:
        logger.warning(f"Category '{name}' already exists")
    conn.close()

def add_label(value, category_id, is_dynamic=0):
    """Add a new label to a category."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO labels (value, category_id, is_dynamic) VALUES (?, ?, ?)",
                  (value, category_id, is_dynamic))
        conn.commit()
        logger.info(f"Added label: {value}")
    except sqlite3.IntegrityError:
        logger.warning(f"Label '{value}' already exists in category {category_id}")
    conn.close()

def get_all_labels():
    """Retrieve all labels with their categories."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT l.value, c.name FROM labels l JOIN categories c ON l.category_id = c.id")
    labels = [{"label": row[0], "category": row[1]} for row in c.fetchall()]
    conn.close()
    return labels

def save_correction(ocr_output, corrected):
    """Save an OCR correction."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO corrections (ocr_output, corrected) VALUES (?, ?)", (ocr_output, corrected))
    conn.commit()
    conn.close()
    logger.info(f"Saved correction: {ocr_output} -> {corrected}")

def get_corrections():
    """Retrieve all corrections."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT ocr_output, corrected FROM corrections")
    corrections = [{"ocr_output": row[0], "corrected": row[1]} for row in c.fetchall()]
    conn.close()
    return corrections

def get_abbreviations():
    """Retrieve all abbreviations."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT a.id, a.abbr, a.full_form, c.name FROM abbreviations a JOIN categories c ON a.category_id = c.id")
    abbrs = [{"id": row[0], "abbr": row[1], "full_form": row[2], "category": row[3]} for row in c.fetchall()]
    conn.close()
    return abbrs

def add_abbreviation(abbr, full_form, category_id):
    """Add a new abbreviation."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO abbreviations (abbr, full_form, category_id) VALUES (?, ?, ?)",
                  (abbr, full_form, category_id))
        conn.commit()
        logger.info(f"Added abbreviation: {abbr} -> {full_form}")
    except sqlite3.IntegrityError:
        logger.warning(f"Abbreviation '{abbr}' already exists in category {category_id}")
    conn.close()

def update_abbreviation(abbr_id, abbr, full_form, category_id):
    """Update an existing abbreviation."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE abbreviations SET abbr = ?, full_form = ?, category_id = ? WHERE id = ?",
              (abbr, full_form, category_id, abbr_id))
    conn.commit()
    conn.close()
    logger.info(f"Updated abbreviation ID {abbr_id}")

def delete_abbreviation(abbr_id):
    """Delete an abbreviation."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM abbreviations WHERE id = ?", (abbr_id,))
    conn.commit()
    conn.close()
    logger.info(f"Deleted abbreviation ID {abbr_id}")

def preprocess_input(text):
    """Preprocess input using abbreviation mappings and normalization."""
    text = text.lower()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT abbr, full_form FROM abbreviations")
    abbr_map = {row[0].lower(): row[1] for row in c.fetchall()}
    conn.close()
    return abbr_map.get(text, text)

def save_training_session(start_time, end_time, labels, validation_results, model_path):
    """Save a training session to the database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        labels_json = json.dumps(labels, ensure_ascii=False)
        validation_results_json = json.dumps(validation_results, ensure_ascii=False)
        c.execute("INSERT INTO training_sessions (start_time, end_time, labels, validation_results, model_path) VALUES (?, ?, ?, ?, ?)",
                  (start_time, end_time, labels_json, validation_results_json, model_path))
        conn.commit()
        logger.info(f"Saved training session: {model_path}")
    except Exception as e:
        logger.error(f"Failed to save training session: {str(e)}")
        raise
    finally:
        conn.close()

def get_training_sessions():
    """Retrieve all training sessions."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, start_time, end_time, labels, validation_results, model_path FROM training_sessions")
    sessions = [{"id": row[0], "start_time": row[1], "end_time": row[2], 
                "labels": json.loads(row[3]), "validation_results": json.loads(row[4]), 
                "model_path": row[5]} for row in c.fetchall()]
    conn.close()
    return sessions

def delete_training_session(session_id):
    """Delete a training session and its model directory."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT model_path FROM training_sessions WHERE id = ?", (session_id,))
    result = c.fetchone()
    if result:
        model_path = result[0]
        try:
            if os.path.exists(model_path):
                shutil.rmtree(model_path, ignore_errors=True)
                logger.info(f"Deleted model directory: {model_path}")
            c.execute("DELETE FROM training_sessions WHERE id = ?", (session_id,))
            conn.commit()
            logger.info(f"Deleted training session ID {session_id}")
        except Exception as e:
            logger.error(f"Failed to delete training session ID {session_id}: {str(e)}")
    conn.close()