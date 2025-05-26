import streamlit as st
import logging
import faiss
import sqlite3
import os
from ui_demo import demo_page
from ui_data_management import data_management_page
from ui_model_management import model_management_page
from train_ops import load_or_train_model, update_faiss_index, get_latest_model_path
from db_ops import init_db, get_all_labels, DB_PATH

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_ROOT = "trained_model"

def initialize_session_state():
    """Initialize model and FAISS index with progress display."""
    if 'model' not in st.session_state:
        with st.status("Initializing OCR Correction System...", expanded=True) as status:
            st.write("Loading or training model...")
            try:
                st.session_state.model = load_or_train_model(st_placeholder=st.empty())
                st.session_state.model_path = get_latest_model_path()
            except Exception as e:
                st.error(f"Failed to initialize model: {str(e)}. Check logs and ensure write permissions for {MODEL_ROOT} and db/labels.db.")
                logger.error(f"Model initialization failed: {str(e)}")
                status.update(label="Initialization failed!", state="error")
                return
            dimension = 384
            st.write("Setting up FAISS index...")
            st.session_state.index = faiss.IndexFlatL2(dimension)
            valid_values = get_all_labels()
            st.session_state.valid_values = valid_values
            st.write("Updating FAISS index...")
            st.session_state.labels = update_faiss_index(st.session_state.model, st.session_state.index, valid_values)
            status.update(label="Initialization complete!", state="complete")

def verify_database_schema():
    """Verify that required tables exist in the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='categories'")
        if not c.fetchone():
            logger.error("Categories table does not exist")
            raise sqlite3.OperationalError("Categories table not found in database")
        logger.info("Database schema verified")
    except Exception as e:
        logger.error(f"Database schema verification failed: {str(e)}")
        raise
    finally:
        conn.close()

def main():
    st.set_page_config(page_title="OCR Correction System", layout="wide")
    st.title("OCR Correction System v1.3")
    
    init_db()
    verify_database_schema()
    initialize_session_state()
    if 'model' not in st.session_state:
        return

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Demo", "Data Management", "Model Management"])

    if page == "Demo":
        demo_page()
    elif page == "Data Management":
        data_management_page()
    elif page == "Model Management":
        model_management_page()

if __name__ == "__main__":
    logger.info(f"Using database at: {DB_PATH}")
    try:
        init_db()
        verify_database_schema()
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        initial_categories = ["Tài liệu", "Quốc tịch", "Tỉnh/Thành phố", "Dân tộc"]
        for cat in initial_categories:
            c.execute("INSERT OR IGNORE INTO categories (name) VALUES (?)", (cat,))
        initial_labels = [
            ("Căn cước công dân", "Tài liệu", 0),
            ("Chứng minh nhân dân", "Tài liệu", 0),
            ("Hợp đồng lao động", "Tài liệu", 0),
        ]
        for value, cat, is_dynamic in initial_labels:
            c.execute("SELECT id FROM categories WHERE name = ?", (cat,))
            cat_id = c.fetchone()
            if cat_id:
                c.execute("INSERT OR IGNORE INTO labels (value, category_id, is_dynamic) VALUES (?, ?, ?)",
                          (value, cat_id[0], is_dynamic))
            else:
                logger.error(f"Category '{cat}' not found for label '{value}'")
        initial_abbrs = [
            ("CCCD", "Căn cước công dân", "Tài liệu"),
            ("CMND", "Chứng minh nhân dân", "Tài liệu"),
            ("HDLD", "Hợp đồng lao động", "Tài liệu"),
        ]
        for abbr, full_form, cat in initial_abbrs:
            c.execute("SELECT id FROM categories WHERE name = ?", (cat,))
            cat_id = c.fetchone()
            if cat_id:
                c.execute("INSERT OR IGNORE INTO abbreviations (abbr, full_form, category_id) VALUES (?, ?, ?)",
                          (abbr, full_form, cat_id[0]))
            else:
                logger.error(f"Category '{cat}' not found for abbreviation '{abbr}'")
        conn.commit()
        logger.info("Default data inserted successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database with default data: {str(e)}")
        raise
    finally:
        conn.close()
    
    main()