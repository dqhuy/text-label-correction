import streamlit as st
import sqlite3
import random
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import faiss
from sklearn.model_selection import train_test_split
import logging
import torch
import os
import shutil
import tempfile
import time
from collections import Counter
import pandas as pd
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global constants
DB_PATH = "labels.db"
MODEL_PATH = "trained_model"

def init_db():
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

# Initialize database
init_db()

# Simulate OCR errors
def simulate_ocr_errors(text):
    errors = [
        lambda x: x.replace("công", "cuong"),
        lambda x: x.replace("cước", "cuoc"),
        lambda x: x.replace("dân", "dan"),
        lambda x: x.replace("hợp", "hơp"),
        lambda x: x + " s",
        lambda x: x[:-1],
        lambda x: x.replace(" ", ""),
        lambda x: x.lower(),
        lambda x: x.replace("Việt", "Viet"),
        lambda x: "cccd" if x == "Căn cước công dân" else x,
        lambda x: "cmnd" if x == "Chứng minh nhân dân" else x,
        lambda x: "vn" if x == "Việt Nam" else x,
    ]
    return random.choice(errors)(text)

# Generate simulated dataset
def generate_dataset(n_per_class=50):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT value FROM labels WHERE is_dynamic = 0")
    static_labels = [row[0] for row in c.fetchall()]
    conn.close()
    
    dataset = []
    for value in static_labels[:10]:
        for _ in range(n_per_class):
            ocr_output = simulate_ocr_errors(value) if random.random() > 0.1 else value
            dataset.append({"ocr_output": ocr_output, "corrected": value})
    logger.info(f"Generated {len(dataset)} examples")
    distribution = Counter(item["corrected"] for item in dataset)
    logger.info(f"Dataset distribution: {dict(distribution)}")
    return dataset

# Rule-based preprocessing using database abbreviations
def preprocess_input(text):
    text = text.lower()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT abbr, full_form FROM abbreviations")
    abbr_map = {row[0].lower(): row[1] for row in c.fetchall()}
    conn.close()
    return abbr_map.get(text, text)

# Automatic abbreviation detection
def detect_abbreviation(term, label):
    term = term.upper().replace(" ", "")
    words = re.findall(r'\b\w+\b', label)
    first_letters = ''.join(word[0].upper() for word in words if word)
    return term == first_letters

# Database operations
def get_categories():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name FROM categories")
    categories = [{"id": row[0], "name": row[1]} for row in c.fetchall()]
    conn.close()
    return categories

def get_labels_by_category(category_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, value, is_dynamic FROM labels WHERE category_id = ?", (category_id,))
    labels = [{"id": row[0], "value": row[1], "is_dynamic": row[2]} for row in c.fetchall()]
    conn.close()
    return labels

def add_category(name):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO categories (name) VALUES (?)", (name,))
        conn.commit()
    except sqlite3.IntegrityError:
        st.error(f"Category '{name}' already exists.")
    conn.close()

def add_label(value, category_id, is_dynamic=0):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO labels (value, category_id, is_dynamic) VALUES (?, ?, ?)",
                  (value, category_id, is_dynamic))
        conn.commit()
    except sqlite3.IntegrityError:
        st.warning(f"Label '{value}' already exists in this category.")
    conn.close()

def get_all_labels():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT l.value, c.name FROM labels l JOIN categories c ON l.category_id = c.id")
    labels = [{"label": row[0], "category": row[1]} for row in c.fetchall()]
    conn.close()
    return labels

def save_correction(ocr_output, corrected):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO corrections (ocr_output, corrected) VALUES (?, ?)", (ocr_output, corrected))
    conn.commit()
    conn.close()

def get_corrections():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT ocr_output, corrected FROM corrections")
    corrections = [{"ocr_output": row[0], "corrected": row[1]} for row in c.fetchall()]
    conn.close()
    return corrections

def get_abbreviations():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT a.id, a.abbr, a.full_form, c.name FROM abbreviations a JOIN categories c ON a.category_id = c.id")
    abbrs = [{"id": row[0], "abbr": row[1], "full_form": row[2], "category": row[3]} for row in c.fetchall()]
    conn.close()
    return abbrs

def add_abbreviation(abbr, full_form, category_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO abbreviations (abbr, full_form, category_id) VALUES (?, ?, ?)",
                  (abbr, full_form, category_id))
        conn.commit()
    except sqlite3.IntegrityError:
        st.warning(f"Abbreviation '{abbr}' already exists in this category.")
    conn.close()

def update_abbreviation(abbr_id, abbr, full_form, category_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE abbreviations SET abbr = ?, full_form = ?, category_id = ? WHERE id = ?",
              (abbr, full_form, category_id, abbr_id))
    conn.commit()
    conn.close()

def delete_abbreviation(abbr_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM abbreviations WHERE id = ?", (abbr_id,))
    conn.commit()
    conn.close()

# Custom callback for logging training progress
class TrainingCallback:
    def __call__(self, score, epoch, steps):
        logger.info(f"Epoch {epoch}, Steps {steps}, Loss: {score:.4f}")

# Fine-tune MiniLM
def fine_tune_model(model, train_data, epochs=1, retrain=False):
    logger.info("Starting fine-tuning of MiniLM model...")
    train_examples = []
    for item in train_data:
        repeats = 5 if retrain else 1
        for _ in range(repeats):
            train_examples.append(InputExample(texts=[item["ocr_output"], item["corrected"]], label=1.0))
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=1 if len(train_examples) <= 5 else 16)
    train_loss = losses.CosineSimilarityLoss(model)
    
    try:
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=10 if len(train_examples) <= 5 else 100,
            optimizer_params={'lr': 5e-5 if retrain else 2e-5},
            show_progress_bar=True,
            callback=TrainingCallback()
        )
        logger.info("Fine-tuning completed")
        
        temp_dir = tempfile.mkdtemp()
        for attempt in range(3):
            try:
                model.save(temp_dir, safe_serialization=True)
                if os.path.exists(MODEL_PATH):
                    shutil.rmtree(MODEL_PATH, ignore_errors=True)
                shutil.move(temp_dir, MODEL_PATH)
                logger.info(f"Saved model to {MODEL_PATH}")
                model = SentenceTransformer(MODEL_PATH)
                logger.info(f"Reloaded model from {MODEL_PATH}")
                return model
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed to save model: {str(e)}. Retrying...")
                time.sleep(1)
        logger.error("Failed to save model after 3 attempts.")
        raise RuntimeError("Model saving failed")
    except Exception as e:
        logger.error(f"Error during fine-tuning: {str(e)}")
        raise

# Load or fine-tune model
def load_or_train_model(train_data, model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
    if os.path.exists(MODEL_PATH):
        logger.info(f"Loading pre-trained model from {MODEL_PATH}")
        try:
            model = SentenceTransformer(MODEL_PATH)
        except Exception as e:
            logger.warning(f"Failed to load model from {MODEL_PATH}: {str(e)}. Fine-tuning new model.")
            model = SentenceTransformer(model_name)
            model = fine_tune_model(model, train_data, epochs=3)
    else:
        logger.info(f"No pre-trained model found at {MODEL_PATH}, fine-tuning new model")
        model = SentenceTransformer(model_name)
        model = fine_tune_model(model, train_data, epochs=3)
    return model

# Update FAISS index
def update_faiss_index(model, index, valid_values):
    index.reset()
    labels = [v["label"] for v in valid_values]
    embeddings = model.encode(labels, show_progress_bar=True)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    logger.info(f"FAISS index updated with {len(labels)} labels")
    return labels

# Streamlit app
def main():
    st.title("OCR Correction System v1.1")
    
    # Initialize session state
    if 'model' not in st.session_state:
        dataset = generate_dataset()
        train_data, _ = train_test_split(dataset, test_size=0.2, random_state=42)
        st.session_state.model = load_or_train_model(train_data)
        dimension = 384
        valid_values = get_all_labels()
        nlist = max(1, min(len(valid_values), 1000))  # Adjust nlist based on number of labels
        quantizer = faiss.IndexFlatIP(dimension)
        st.session_state.index = faiss.IndexIVFPQ(quantizer, dimension, nlist, 8, 8)
        st.session_state.valid_values = valid_values
        st.session_state.index.train(st.session_state.model.encode([v["label"] for v in valid_values[:10000]]))
        st.session_state.labels = update_faiss_index(st.session_state.model, st.session_state.index, valid_values)

    # Assign variables after session state initialization
    model = st.session_state.model
    index = st.session_state.index
    valid_values = st.session_state.valid_values
    labels = st.session_state.labels

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Demo", "Admin"])

    if page == "Demo":
        st.header("Demo: Predict Corrected Value")
        input_text = st.text_input("Enter OCR text (e.g., 'Căn cuoc', 'vn')")
        if st.button("Predict"):
            if input_text:
                start_time = time.time()
                processed_input = preprocess_input(input_text)
                ocr_embedding = model.encode([processed_input])[0]
                ocr_embedding = np.array([ocr_embedding], dtype=np.float32)
                faiss.normalize_L2(ocr_embedding)
                distances, indices = index.search(ocr_embedding, k=1)
                predicted_value = labels[indices[0][0]]
                confidence = min(max(distances[0][0] * 100, 0.0), 100.0)
                elapsed_time = time.time() - start_time
                st.success(f"**Predicted**: {predicted_value} (Confidence: {confidence:.2f}%)")
                st.write(f"**Prediction Time**: {elapsed_time:.4f} seconds")
                
                # Auto-detect abbreviation
                detected = [(v["label"], v["category"]) for v in valid_values if detect_abbreviation(input_text, v["label"])]
                if detected:
                    st.subheader("Detected Abbreviations")
                    for label, category in detected:
                        st.write(f"**{input_text}** may be an abbreviation for **{label}** ({category})")
                        if st.button(f"Add {input_text} → {label} to Abbreviations", key=f"add_abbr_{label}"):
                            conn = sqlite3.connect(DB_PATH)
                            c = conn.cursor()
                            c.execute("SELECT id FROM categories WHERE name = ?", (category,))
                            category_id = c.fetchone()[0]
                            add_abbreviation(input_text.upper(), label, category_id)
                            conn.close()
                            st.success(f"Added abbreviation: {input_text} → {label}")
            else:
                st.error("Please enter text to predict.")

    elif page == "Admin":
        st.header("Admin: Manage Categories, Labels, and Abbreviations")
        admin_task = st.selectbox("Select Task", ["View Categories", "Add Category", "Add Label", "Manage Abbreviations"])

        if admin_task == "View Categories":
            st.subheader("Categories")
            categories = get_categories()
            if categories:
                df = pd.DataFrame(categories)
                st.dataframe(df)
                category_id = st.selectbox("Select Category to View Labels", 
                                         [c["id"] for c in categories], 
                                         format_func=lambda x: next(c["name"] for c in categories if c["id"] == x))
                labels = get_labels_by_category(category_id)
                if labels:
                    st.subheader(f"Labels in {next(c['name'] for c in categories if c['id'] == category_id)}")
                    df_labels = pd.DataFrame(labels)
                    df_labels["is_dynamic"] = df_labels["is_dynamic"].map({0: "Static", 1: "Dynamic"})
                    st.dataframe(df_labels)
                else:
                    st.write("No labels in this category.")
            else:
                st.write("No categories available.")

        elif admin_task == "Add Category":
            st.subheader("Add New Category")
            category_name = st.text_input("Category Name")
            if st.button("Add Category"):
                if category_name:
                    add_category(category_name)
                    st.success(f"Added category: {category_name}")
                else:
                    st.error("Please enter a category name.")

        elif admin_task == "Add Label":
            st.subheader("Add Label to Category")
            categories = get_categories()
            if categories:
                category_id = st.selectbox("Select Category", 
                                         [c["id"] for c in categories], 
                                         format_func=lambda x: next(c["name"] for c in categories if c["id"] == x))
                label_value = st.text_input("Label Value")
                is_dynamic = st.checkbox("Dynamic Label", value=True)
                if st.button("Add Label"):
                    if label_value:
                        add_label(label_value, category_id, 1 if is_dynamic else 0)
                        valid_values.append({"label": label_value, "category": next(c["name"] for c in categories if c["id"] == category_id)})
                        st.session_state.valid_values = valid_values
                        st.session_state.labels = update_faiss_index(model, index, valid_values)
                        st.success(f"Added label: {label_value} to category")
                    else:
                        st.error("Please enter a label value.")
            else:
                st.error("No categories available. Add a category first.")

        elif admin_task == "Manage Abbreviations":
            st.subheader("Manage Abbreviations")
            abbrs = get_abbreviations()
            if abbrs:
                df = pd.DataFrame(abbrs)
                st.dataframe(df)
                
                st.subheader("Add Abbreviation")
                categories = get_categories()
                if categories:
                    category_id = st.selectbox("Select Category for Abbreviation", 
                                             [c["id"] for c in categories], 
                                             format_func=lambda x: next(c["name"] for c in categories if c["id"] == x))
                    abbr = st.text_input("Abbreviation (e.g., VN)")
                    full_form = st.text_input("Full Form (e.g., Việt Nam)")
                    if st.button("Add Abbreviation"):
                        if abbr and full_form:
                            add_abbreviation(abbr.upper(), full_form, category_id)
                            st.success(f"Added abbreviation: {abbr} → {full_form}")
                        else:
                            st.error("Please enter both abbreviation and full form.")
                
                st.subheader("Edit/Delete Abbreviation")
                abbr_id = st.selectbox("Select Abbreviation to Edit/Delete", 
                                     [a["id"] for a in abbrs], 
                                     format_func=lambda x: next(f"{a['abbr']} → {a['full_form']}" for a in abbrs if a["id"] == x))
                selected_abbr = next(a for a in abbrs if a["id"] == abbr_id)
                new_abbr = st.text_input("New Abbreviation", value=selected_abbr["abbr"])
                new_full_form = st.text_input("New Full Form", value=selected_abbr["full_form"])
                new_category_id = st.selectbox("New Category", 
                                             [c["id"] for c in categories], 
                                             index=[c["id"] for c in categories].index(selected_abbr["category_id"]), 
                                             format_func=lambda x: next(c["name"] for c in categories if c["id"] == x))
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Update Abbreviation"):
                        if new_abbr and new_full_form:
                            update_abbreviation(abbr_id, new_abbr.upper(), new_full_form, new_category_id)
                            st.success(f"Updated abbreviation: {new_abbr} → {new_full_form}")
                        else:
                            st.error("Please enter both abbreviation and full form.")
                with col2:
                    if st.button("Delete Abbreviation"):
                        delete_abbreviation(abbr_id)
                        st.success(f"Deleted abbreviation: {selected_abbr['abbr']}")
            else:
                st.write("No abbreviations available.")

if __name__ == "__main__":
    # Populate initial data
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    initial_categories = ["Tài liệu", "Quốc tịch", "Tỉnh/Thành phố", "Dân tộc"]
    for cat in initial_categories:
        c.execute("INSERT OR IGNORE INTO categories (name) VALUES (?)", (cat,))
    initial_labels = [
        ("Căn cước công dân", "Tài liệu", 0),
        ("Chứng minh nhân dân", "Tài liệu", 0),
        ("Hợp đồng dân sự", "Tài liệu", 0),
        ("Việt Nam", "Quốc tịch", 0),
        ("Hà Nội", "Tỉnh/Thành phố", 0),
        ("Kinh", "Dân tộc", 0),
        ("Tày", "Dân tộc", 0),
    ]
    for value, cat, is_dynamic in initial_labels:
        c.execute("SELECT id FROM categories WHERE name = ?", (cat,))
        cat_id = c.fetchone()[0]
        c.execute("INSERT OR IGNORE INTO labels (value, category_id, is_dynamic) VALUES (?, ?, ?)",
                  (value, cat_id, is_dynamic))
    initial_abbrs = [
        ("CCCD", "Căn cước công dân", "Tài liệu"),
        ("CMND", "Chứng minh nhân dân", "Tài liệu"),
        ("VN", "Việt Nam", "Quốc tịch"),
    ]
    for abbr, full_form, cat in initial_abbrs:
        c.execute("SELECT id FROM categories WHERE name = ?", (cat,))
        cat_id = c.fetchone()[0]
        c.execute("INSERT OR IGNORE INTO abbreviations (abbr, full_form, category_id) VALUES (?, ?, ?)",
                  (abbr, full_form, cat_id))
    conn.commit()
    conn.close()
    
    main()