import sqlite3
import streamlit as st
import pandas as pd
import numpy as np
import faiss
import time
import shutil
import os
from db_ops import DB_PATH, get_categories, get_training_sessions, delete_training_session, add_label, get_all_labels, preprocess_input, save_training_session
from train_ops import test_model, generate_custom_dataset, fine_tune_model, load_or_train_model, update_faiss_index, get_latest_model_path, generate_vietnamese_errors
from collections import Counter
import logging

logger = logging.getLogger(__name__)

MODEL_ROOT = "trained_model"
DEFAULT_N_PER_CLASS = 50

def model_management_page():
    """Render the Model Management interface."""
    st.header("Model Management: View, Test, and Train Models")
    
    admin_task = st.selectbox("Select Task", [
        "View and Manage Models",
        "Test Model",
        "Retrain with Custom Label",
        "Clear Training Data"
    ])

    if admin_task == "View and Manage Models":
        st.subheader("Trained Models")
        sessions = get_training_sessions()
        if sessions:
            df = pd.DataFrame(sessions)
            df["labels"] = df["labels"].apply(lambda x: ", ".join(f"{k}: {v}" for k, v in x.items()))
            df["validation_accuracy"] = df["validation_results"].apply(lambda x: f"{x['accuracy']:.2f}%")
            st.dataframe(df[["id", "start_time", "end_time", "labels", "validation_accuracy", "model_path"]])
            
            session_id = st.selectbox("Select Model to Load or Delete",
                                      [s["id"] for s in sessions],
                                      format_func=lambda x: next(f"{s['start_time']} ({s['model_path']})" for s in sessions if s["id"] == x))
            selected_session = next(s for s in sessions if s["id"] == session_id)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Load Model"):
                    try:
                        st.session_state.model = load_or_train_model(model_name=selected_session["model_path"])
                        st.session_state.model_path = selected_session["model_path"]
                        st.session_state.labels = update_faiss_index(st.session_state.model, st.session_state.index, st.session_state.valid_values)
                        st.success(f"Loaded model from {selected_session['model_path']}")
                    except Exception as e:
                        st.error(f"Failed to load model: {str(e)}")
                        logger.error(f"Failed to load model {selected_session['model_path']}: {str(e)}")
            with col2:
                if st.button("Delete Model"):
                    delete_training_session(session_id)
                    st.success(f"Deleted model ID {session_id}")
                    if selected_session["model_path"] == st.session_state.model_path:
                        del st.session_state.model
                        del st.session_state.model_path
                        initialize_session_state()
                    st.experimental_rerun()
        else:
            st.write("No trained models available.")

    elif admin_task == "Test Model":
        st.subheader("Test Model Accuracy")
        test_mode = st.radio("Test Mode", ["Test All Labels", "Test Specific Labels"])
        
        if test_mode == "Test All Labels":
            if st.button("Run Tests"):
                with st.spinner("Running tests..."):
                    results, accuracy = test_model(st.session_state.model, st.session_state.index, st.session_state.labels)
                    st.write(f"**Test Accuracy**: {accuracy:.2f}%")
                    df_results = pd.DataFrame(results)
                    st.dataframe(df_results)
                    for result in results:
                        if result["correct"]:
                            st.success(f"Input: {result['input']} -> Predicted: {result['predicted']} (Correct)")
                        else:
                            st.error(f"Input: {result['input']} -> Predicted: {result['predicted']} "
                                    f"(Expected: {result['expected']})")
        
        else:
            valid_values = st.session_state.valid_values
            label_options = [v["label"] for v in valid_values]
            selected_labels = st.multiselect("Select Labels to Test", label_options)
            custom_input = st.text_input("Or Enter Custom Test Input (e.g., 'can cuoc')")
            
            if st.button("Run Tests"):
                if selected_labels or custom_input:
                    with st.status("Running tests...", expanded=True) as status:
                        results = []
                        test_cases = []
                        
                        # Generate test cases from selected labels
                        for label in selected_labels:
                            variants = generate_vietnamese_errors(label, 5)
                            test_cases.extend([{"input": v, "expected": label} for v in variants])
                        
                        # Add custom input
                        if custom_input:
                            test_cases.append({"input": custom_input, "expected": None})
                        
                        # Run tests
                        model = st.session_state.model
                        index = st.session_state.index
                        labels = st.session_state.labels
                        progress_bar = st.progress(0)
                        for i, case in enumerate(test_cases):
                            input_text = case["input"]
                            expected = case["expected"]
                            processed_input = preprocess_input(input_text)
                            ocr_embedding = model.encode([processed_input])[0]
                            ocr_embedding = np.array([ocr_embedding], dtype=np.float32)
                            faiss.normalize_L2(ocr_embedding)
                            distances, indices = index.search(ocr_embedding, k=1)
                            predicted_value = labels[indices[0][0]]
                            confidence = float(min(max((1 - distances[0][0] / 2) * 100, 0.0), 100.0))
                            correct = predicted_value == expected if expected else None
                            results.append({
                                "input": input_text,
                                "predicted": predicted_value,
                                "expected": expected,
                                "confidence": confidence,
                                "correct": correct
                            })
                            logger.info(f"Test: Input='{input_text}' -> Predicted='{predicted_value}' "
                                       f"(Expected='{expected}', Confidence: {confidence:.2f}%, Correct: {correct})")
                            progress_bar.progress((i + 1) / len(test_cases))
                        
                        accuracy = sum(r["correct"] for r in results if r["correct"] is not None) / len([r for r in results if r["correct"] is not None]) * 100 if any(r["correct"] is not None for r in results) else 0
                        st.write(f"**Test Accuracy**: {accuracy:.2f}%")
                        df_results = pd.DataFrame(results)
                        st.dataframe(df_results)
                        for result in results:
                            if result["correct"] is None:
                                st.info(f"Input: {result['input']} -> Predicted: {result['predicted']} (Confidence: {result['confidence']:.2f}%)")
                            elif result["correct"]:
                                st.success(f"Input: {result['input']} -> Predicted: {result['predicted']} (Correct)")
                            else:
                                st.error(f"Input: {result['input']} -> Predicted: {result['predicted']} "
                                        f"(Expected: {result['expected']})")
                        status.update(label="Testing complete!", state="complete")
                else:
                    st.error("Please select at least one label or enter a custom input.")

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
                            train_data, test_data = generate_custom_dataset(custom_label, n_per_class=n_per_class, only_new_label=True)
                            st.write("Fine-tuning model with new label...")
                            st.session_state.model, validation_results, start_time, end_time = fine_tune_model(
                                st.session_state.model,
                                train_data,
                                test_data,
                                epochs=1,
                                retrain=False,
                                st_placeholder=st.empty()
                            )
                            st.write("Updating FAISS index...")
                            add_label(custom_label, category_id, is_dynamic=0)
                            st.session_state.valid_values = get_all_labels()
                            st.session_state.labels = update_faiss_index(
                                st.session_state.model,
                                st.session_state.index,
                                st.session_state.valid_values
                            )
                            st.session_state.model_path = get_latest_model_path()
                            if end_time:
                                labels = dict(Counter(item["corrected"] for item in train_data))
                                try:
                                    save_training_session(start_time, end_time, labels, validation_results, st.session_state.model_path)
                                    logger.info(f"Saved training session for '{custom_label}' to database")
                                    conn = sqlite3.connect(DB_PATH)
                                    c = conn.cursor()
                                    c.execute("SELECT id FROM training_sessions WHERE start_time = ?", (start_time,))
                                    session_id = c.fetchone()
                                    conn.close()
                                    if session_id:
                                        logger.info(f"Verified training session ID {session_id[0]} in database")
                                    else:
                                        logger.error(f"Training session for '{custom_label}' not found in database")
                                except Exception as e:
                                    logger.error(f"Failed to save training session for '{custom_label}': {str(e)}")
                                    st.error(f"Failed to save training session to database: {str(e)}")
                            status.update(label="Retraining complete!", state="complete")
                            st.success(f"Model fine-tuned with {n_per_class} samples for '{custom_label}'")
                            st.write(f"Validation Accuracy: {validation_results['accuracy']:.2f}%")
                        except Exception as e:
                            st.error(f"Failed to retrain model: {str(e)}. Please check logs and ensure write permissions for {MODEL_ROOT}.")
                            logger.error(f"Retraining failed: {str(e)}")
                            status.update(label="Retraining failed!", state="error")
                            return
                else:
                    st.error("Please enter a label.")
        else:
            st.error("No categories available. Add a category first.")

    elif admin_task == "Clear Training Data":
        st.subheader("Clear Training Data")
        st.warning("This will delete all trained models and require retraining.")
        if st.button("Clear Training Data"):
            if os.path.exists(MODEL_ROOT):
                shutil.rmtree(MODEL_ROOT, ignore_errors=True)
                logger.info(f"Deleted all models at {MODEL_ROOT}")
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("DELETE FROM training_sessions")
            conn.commit()
            conn.close()
            logger.info("Cleared training_sessions table")
            for key in ['model', 'index', 'valid_values', 'labels', 'model_path']:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("Training data cleared. Please retrain the model.")
            initialize_session_state()
            st.experimental_rerun()

def initialize_session_state():
    """Initialize model and FAISS index."""
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