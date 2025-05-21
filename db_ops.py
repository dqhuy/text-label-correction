import sqlite3
import logging
import os
import random
import re
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = "db/labels.db"

def generate_vietnamese_errors(text: str, num_variants: int = 1) -> str:
    """
    Tạo một biến thể có lỗi của chuỗi tiếng Việt đầu vào, mô phỏng lỗi gõ phím và OCR
    
    Args:
        text: Chuỗi tiếng Việt đầu vào
        num_variants: Số lượng biến thể lỗi cần tạo (mặc định 1 để tương thích với dataset generation)
    
    Returns:
        Một chuỗi có chứa lỗi
    """
    # Các cặp ký tự dễ nhầm lẫn trong tiếng Việt
    VIETNAMESE_ERROR_PAIRS = [
        ('ă', 'aw'), ('ă', 'a'), ('ă', 'â'),
        ('â', 'aa'), ('â', 'a'), ('â', 'ă'),
        ('đ', 'dd'), ('đ', 'd'),
        ('ê', 'ee'), ('ê', 'e'),
        ('ô', 'oo'), ('ô', 'o'),
        ('ơ', 'ow'), ('ơ', 'o'), ('ơ', 'ô'),
        ('ư', 'uw'), ('ư', 'u'),
        ('s', 'x'), ('x', 's'),
        ('ch', 'c'), ('ch', 'tr'), ('tr', 'ch'),
        ('gi', 'd'), ('gi', 'r'), ('d', 'gi'), ('r', 'gi'),
        ('n', 'l'), ('l', 'n'),
        ('i', 'y'), ('y', 'i'),
        ('c', 'k'), ('k', 'c'), ('q', 'k'),
        ('ph', 'f'), ('f', 'ph'),
        ('g', 'gh'), ('gh', 'g'),
        ('ng', 'ngh'), ('ngh', 'ng'),
    ]
    
    # Các ký tự dễ nhầm lẫn trong OCR
    OCR_ERROR_PAIRS = [
        ('a', 'o'), ('o', 'a'), ('o', 'c'), ('c', 'o'),
        ('e', 'c'), ('c', 'e'), ('b', 'h'), ('h', 'b'),
        ('d', 'cl'), ('cl', 'd'), ('m', 'n'), ('n', 'm'),
        ('u', 'v'), ('v', 'u'), ('w', 'vv'), ('i', 'l'),
        ('t', 'f'), ('f', 't'), ('g', 'q'), ('q', 'g'),
    ]
    
    variants = []
    
    for _ in range(num_variants):
        # Chọn ngẫu nhiên số lỗi (1-3 lỗi mỗi chuỗi)
        num_errors = random.randint(1, 3)
        corrupted = list(text.lower())
        
        for __ in range(num_errors):
            if len(corrupted) == 0:
                break
                
            # Chọn ngẫu nhiên vị trí để thêm lỗi
            pos = random.randint(0, len(corrupted)-1)
            char = corrupted[pos]
            
            # 50% lỗi tiếng Việt, 50% lỗi OCR
            if random.random() < 0.5:
                # Lỗi tiếng Việt
                for pair in VIETNAMESE_ERROR_PAIRS:
                    if char in pair:
                        # Thay thế bằng ký tự dễ nhầm
                        replacement = pair[1] if pair[0] == char else pair[0]
                        # Đôi khi thêm/ghép ký tự
                        if random.random() < 0.3 and len(replacement) == 1:
                            replacement = replacement * random.randint(1, 2)
                        corrupted[pos] = replacement
                        break
            else:
                # Lỗi OCR
                for pair in OCR_ERROR_PAIRS:
                    if char in pair:
                        replacement = pair[1] if pair[0] == char else pair[0]
                        corrupted[pos] = replacement
                        break
                
            # Đôi khi xóa hoặc thêm ký tự (10% xác suất)
            if random.random() < 0.1:
                if random.random() < 0.5 and len(corrupted) > 1:
                    # Xóa ký tự
                    del corrupted[pos]
                else:
                    # Thêm ký tự ngẫu nhiên
                    random_char = random.choice(['a', 'e', 'o', 'd', 'm', 'n', 'g', 'h'])
                    corrupted.insert(pos, random_char)
        
        # Ghép lại thành chuỗi và thêm vào danh sách
        variant = ''.join(corrupted)
        
        # Đôi khi thêm khoảng trắng ngẫu nhiên (10% xác suất)
        if random.random() < 0.1:
            space_pos = random.randint(1, len(variant)-1)
            variant = variant[:space_pos] + ' ' + variant[space_pos:]
        
        variants.append(variant)
    
    # Trả về một biến thể ngẫu nhiên để tương thích với generate_dataset
    return random.choice(variants) if variants else text

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
    # Normalization dictionary for common variations
    normalization_map = {
        'viet nam': 'Việt Nam',
        'vietnam': 'Việt Nam',
        'ha noi': 'Hà Nội',
        'hanoi': 'Hà Nội',
        'can cuoc cong dan': 'Căn cước công dân',
        'chung minh nhan dan': 'Chứng minh nhân dân',
        'hop dong dan su': 'Hợp đồng dân sự',
    }
    # Check normalization first
    normalized_text = normalization_map.get(text, text)
    # Then check abbreviations from database
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT abbr, full_form FROM abbreviations")
    abbr_map = {row[0].lower(): row[1] for row in c.fetchall()}
    conn.close()
    return abbr_map.get(normalized_text, normalized_text)