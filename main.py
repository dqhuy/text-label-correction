import streamlit as st
import numpy as np
import faiss
import pandas as pd
import re
import time
import shutil
import os
from db_ops import *
from train_ops import load_or_train_model, update_faiss_index, test_model, generate_custom_dataset, fine_tune_model
from sklearn.model_selection import train_test_split
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_PATH = "trained_model"
DEFAULT_N_PER_CLASS = 100

def detect_abbreviation(term, label):
    """Detect if term is an abbreviation of label."""
    term = term.upper().replace(" ", "")
    words = re.findall(r'\b\w+\b', label)
    first_letters = ''.join(word[0].upper() for word in words if word)
    return term == first_letters

def initialize_session_state():
    """Initialize model and FAISS index with progress display."""
    if 'model' not in st.session_state:
        with st.status("Initializing OCR Correction System...", expanded=True) as status:
            st.write("Loading or training model...")
            try:
                st.session_state.model = load_or_train_model(st_placeholder=st.empty())
            except Exception as e:
                st.error(f"Failed to initialize model: {str(e)}. Please check logs and ensure write permissions for {MODEL_PATH}.")
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

def main():
    st.title("OCR Correction System v1.1")
    
    initialize_session_state()
    if 'model' not in st.session_state:
        return

    model = st.session_state.model
    index = st.session_state.index
    valid_values = st.session_state.valid_values
    labels = st.session_state.labels

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Demo", "Admin"])

    if page == "Demo":
        st.header("Demo: Predict Corrected Value")
        input_text = st.text_input("Enter incorected OCR text (e.g., 'Căn cuoc cong dam')")
        if st.button("Predict"):
            if input_text:
                start_time = time.time()
                processed_input = preprocess_input(input_text)
                ocr_embedding = model.encode([processed_input])[0]
                ocr_embedding = np.array([ocr_embedding], dtype=np.float32)
                faiss.normalize_L2(ocr_embedding)
                distances, indices = index.search(ocr_embedding, k=1)
                predicted_value = labels[indices[0][0]]
                confidence = min(max((1 - distances[0][0] / 2) * 100, 0.0), 100.0)
                elapsed_time = time.time() - start_time
                logger.info(f"Prediction: Input='{input_text}' -> Predicted='{predicted_value}' "
                           f"(Confidence: {confidence:.2f}%, Time: {elapsed_time:.4f}s)")
                st.success(f"**Predicted**: {predicted_value} (Confidence: {confidence:.2f}%)")
                st.write(f"**Prediction Time**: {elapsed_time:.4f} seconds")
                
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
        admin_task = st.selectbox("Select Task", [
            "View Categories", 
            "Add Category", 
            "Add Label", 
            "Manage Abbreviations", 
            "Test Model", 
            "Clear Training Data",
            "Retrain with Custom Label"
        ])

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
                                     format_func=lambda x: next(f"{a['abbr']} -> {a['full_form']}" for a in abbrs if a["id"] == x))
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
                            st.success(f"Updated abbreviation: {new_abbr} -> {new_full_form}")
                        else:
                            st.error("Please enter both abbreviation and full form.")
                with col2:
                    if st.button("Delete Abbreviation"):
                        delete_abbreviation(abbr_id)
                        st.success(f"Deleted abbreviation: {selected_abbr['abbr']}")
            else:
                st.write("No abbreviations available.")

        elif admin_task == "Test Model":
            st.subheader("Test Model Accuracy")
            if st.button("Run Tests"):
                with st.spinner("Running tests..."):
                    results, accuracy = test_model(model, index, labels)
                    st.write(f"**Test Accuracy**: {accuracy:.2f}%")
                    df_results = pd.DataFrame(results)
                    st.dataframe(df_results)
                    for result in results:
                        if result["correct"]:
                            st.success(f"Input: {result['input']} -> Predicted: {result['predicted']} (Correct)")
                        else:
                            st.error(f"Input: {result['input']} -> Predicted: {result['predicted']} "
                                    f"(Expected: {result['expected']})")

        elif admin_task == "Clear Training Data":
            st.subheader("Clear Training Data")
            st.warning("This will delete the trained model and require retraining.")
            if st.button("Clear Training Data"):
                if os.path.exists(MODEL_PATH):
                    shutil.rmtree(MODEL_PATH, ignore_errors=True)
                    logger.info(f"Deleted trained model at {MODEL_PATH}")
                for key in ['model', 'index', 'valid_values', 'labels']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.success("Training data cleared. Please retrain the model.")
                initialize_session_state()
                if 'model' in st.session_state:
                    st.session_state.model = st.session_state.model
                    st.session_state.index = st.session_state.index
                    st.session_state.valid_values = st.session_state.valid_values
                    st.session_state.labels = st.session_state.labels

        elif admin_task == "Retrain with Custom Label":
            st.subheader("Retrain with Custom Label")
            custom_label = st.text_input("Enter Label (e.g., Hợp đồng lao động)")
            n_per_class = st.number_input("Number of Samples per Label", min_value=10, max_value=100, value=DEFAULT_N_PER_CLASS)
            categories = get_categories()
            if categories:
                category_id = st.selectbox("Select Category for Label", 
                                         [c["id"] for c in categories], 
                                         format_func=lambda x: next(c["name"] for c in categories if c["id"] == x))
                if st.button("Retrain Model"):
                    if custom_label:
                        with st.status("Retraining Model...", expanded=True) as status:
                            try:
                                st.write("Generating dataset for new label...")
                                dataset = generate_custom_dataset(custom_label, n_per_class=n_per_class, only_new_label=True)
                                train_data, _ = train_test_split(dataset, test_size=0.2, random_state=42)
                                st.write("Fine-tuning model with new label...")
                                st.session_state.model = fine_tune_model(
                                    st.session_state.model, 
                                    train_data, 
                                    epochs=1,
                                    retrain=False,
                                    st_placeholder=st.empty()
                                )
                                st.write("Updating FAISS index...")
                                add_label(custom_label, category_id, is_dynamic=0)
                                valid_values = get_all_labels()
                                st.session_state.valid_values = valid_values
                                st.session_state.labels = update_faiss_index(
                                    st.session_state.model, 
                                    st.session_state.index, 
                                    valid_values
                                )
                                status.update(label="Retraining complete!", state="complete")
                                st.success(f"Model fine-tuned with {n_per_class} samples for '{custom_label}'")
                            except Exception as e:
                                st.error(f"Failed to retrain model: {str(e)}. Please check logs and ensure write permissions for {MODEL_PATH}.")
                                logger.error(f"Retraining failed: {str(e)}")
                                status.update(label="Retraining failed!", state="error")
                                return
                            model = st.session_state.model
                            index = st.session_state.index
                            valid_values = st.session_state.valid_values
                            labels = st.session_state.labels
                    else:
                        st.error("Please enter a label.")
            else:
                st.error("No categories available. Add a category first.")

if __name__ == "__main__":
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    initial_categories = ["Tài liệu", "Quốc tịch", "Tỉnh/Thành phố", "Dân tộc"]
    for cat in initial_categories:
        c.execute("INSERT OR IGNORE INTO categories (name) VALUES (?)", (cat,))
    initial_labels = [
        ("Căn cước công dân", "Tài liệu", 0),
        ("Chứng minh nhân dân", "Tài liệu", 0),
    ]
    for value, cat, is_dynamic in initial_labels:
        c.execute("SELECT id FROM categories WHERE name = ?", (cat,))
        cat_id = c.fetchone()[0]
        c.execute("INSERT OR IGNORE INTO labels (value, category_id, is_dynamic) VALUES (?, ?, ?)",
                  (value, cat_id, is_dynamic))
    initial_abbrs = [
        ("CCCD", "Căn cước công dân", "Tài liệu"),
        ("CMND", "Chứng minh nhân dân", "Tài liệu"),
    ]
    for abbr, full_form, cat in initial_abbrs:
        c.execute("SELECT id FROM categories WHERE name = ?", (cat,))
        cat_id = c.fetchone()[0]
        c.execute("INSERT OR IGNORE INTO abbreviations (abbr, full_form, category_id) VALUES (?, ?, ?)",
                  (abbr, full_form, cat_id))
    conn.commit()
    conn.close()
    
    main()