import time
import streamlit as st
import numpy as np
import faiss
import sqlite3
import re
from db_ops import add_abbreviation, DB_PATH, preprocess_input
import logging

logger = logging.getLogger(__name__)

def detect_abbreviation(term, label):
    """Detect if term is an abbreviation of label."""
    term = term.upper().replace(" ", "")
    words = re.findall(r'\b\w+\b', label)
    first_letters = ''.join(word[0].upper() for word in words if word)
    return term == first_letters

def demo_page():
    """Render the Demo interface."""
    st.header("Demo: Predict Corrected Value")
    
    # Display model information
    st.subheader("Loaded Model Information")
    if 'model_path' in st.session_state and st.session_state.model_path:
        st.write(f"**Model Path**: {st.session_state.model_path}")
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT start_time, end_time, labels FROM training_sessions WHERE model_path = ?",
                  (st.session_state.model_path,))
        session = c.fetchone()
        conn.close()
        if session:
            start_time, end_time, labels = session
            st.write(f"**Trained On**: {start_time} - {end_time}")
            st.write(f"**Labels Trained**: {labels}")
        else:
            st.write("No training session details available.")
    else:
        st.write("No model loaded.")

    # Prediction input
    input_text = st.text_input("Enter OCR text (e.g., 'Căn cuoc')")
    if st.button("Predict"):
        if input_text:
            model = st.session_state.model
            index = st.session_state.index
            labels = st.session_state.labels
            valid_values = st.session_state.valid_values

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