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
import time
from datetime import datetime
from collections import Counter
from typing import List
from db_ops import get_all_labels, save_training_session
from vietnamese_error_simulation import generate_vietnamese_errors
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_ROOT = "trained_model"
DEFAULT_N_PER_CLASS = 50

def get_latest_model_path():
    """Find the latest model directory by timestamp."""
    if not os.path.exists(MODEL_ROOT):
        return None
    model_dirs = [d for d in os.listdir(MODEL_ROOT) if os.path.isdir(os.path.join(MODEL_ROOT, d))]
    if not model_dirs:
        return None
    latest_dir = sorted(model_dirs, reverse=True)[0]
    return os.path.join(MODEL_ROOT, latest_dir)

def generate_dataset(n_per_class=DEFAULT_N_PER_CLASS):
    """Generate simulated dataset for training."""
    valid_values = get_all_labels()
    static_labels = [v["label"] for v in valid_values if isinstance(v["label"], str)]
    
    dataset = []
    test_dataset = []
    for value in static_labels[:10]:
        # Generate n_per_class variants
        variants = generate_vietnamese_errors(value, n_per_class)
        if not variants:
            logger.warning(f"No variants generated for '{value}'. Using original value {n_per_class} times.")
            variants = [value] * n_per_class
        elif not all(isinstance(v, str) for v in variants):
            logger.warning(f"Invalid variants for '{value}': {variants}. Using original value.")
            variants = [value] * n_per_class
        else:
            # Pad with original value if fewer than n_per_class variants
            while len(variants) < n_per_class:
                variants.append(value)
            logger.debug(f"Generated {len(variants)} variants for '{value}': {variants[:5]}...")

        # Split variants into train (80%) and test (20%)
        for ocr_output in variants:
            data_point = {"ocr_output": ocr_output, "corrected": value}
            if random.random() < 0.2:
                test_dataset.append(data_point)
            else:
                dataset.append(data_point)
    
    logger.info(f"Generated {len(dataset)} training examples and {len(test_dataset)} test examples")
    distribution = Counter(item["corrected"] for item in dataset)
    logger.info(f"Training dataset distribution: {dict(distribution)}")
    test_distribution = Counter(item["corrected"] for item in test_dataset)
    logger.info(f"Test dataset distribution: {dict(test_distribution)}")
    return dataset + test_dataset  # Return combined dataset for train_test_split

def generate_custom_dataset(custom_label, n_per_class=DEFAULT_N_PER_CLASS, only_new_label=False):
    """Generate dataset and test set for a custom label."""
    if not isinstance(custom_label, str):
        logger.error(f"Invalid custom_label: {custom_label}. Must be a string.")
        return [], []
    
    dataset = []
    test_dataset = []
    
    if only_new_label:
        # Generate n_per_class variants for custom label
        variants = generate_vietnamese_errors(custom_label, n_per_class)
        if not variants:
            logger.warning(f"No variants generated for '{custom_label}'. Using original value {n_per_class} times.")
            variants = [custom_label] * n_per_class
        elif not all(isinstance(v, str) for v in variants):
            logger.warning(f"Invalid variants for '{custom_label}': {variants}. Using original value.")
            variants = [custom_label] * n_per_class
        else:
            while len(variants) < n_per_class:
                variants.append(custom_label)
            logger.debug(f"Generated {len(variants)} variants for '{custom_label}': {variants[:5]}...")
        
        for ocr_output in variants:
            data_point = {"ocr_output": ocr_output, "corrected": custom_label}
            if random.random() < 0.2:
                test_dataset.append(data_point)
            else:
                dataset.append(data_point)
    else:
        valid_values = get_all_labels()
        static_labels = [v["label"] for v in valid_values if isinstance(v["label"], str)]
        if custom_label not in static_labels:
            static_labels.append(custom_label)
        
        for value in static_labels:
            # Generate n_per_class variants
            variants = generate_vietnamese_errors(value, n_per_class)
            if not variants:
                logger.warning(f"No variants generated for '{value}'. Using original value {n_per_class} times.")
                variants = [value] * n_per_class
            elif not all(isinstance(v, str) for v in variants):
                logger.warning(f"Invalid variants for '{value}': {variants}. Using original value.")
                variants = [value] * n_per_class
            else:
                while len(variants) < n_per_class:
                    variants.append(value)
                logger.debug(f"Generated {len(variants)} variants for '{value}': {variants[:5]}...")
            
            for ocr_output in variants:
                data_point = {"ocr_output": ocr_output, "corrected": value}
                if random.random() < 0.2:
                    test_dataset.append(data_point)
                else:
                    dataset.append(data_point)
    
    logger.info(f"Generated {len(dataset)} training examples and {len(test_dataset)} test examples for custom label '{custom_label}'")
    distribution = Counter(item["corrected"] for item in dataset)
    logger.info(f"Training dataset distribution: {dict(distribution)}")
    test_distribution = Counter(item["corrected"] for item in test_dataset)
    logger.info(f"Test dataset distribution: {dict(test_distribution)}")
    return dataset, test_dataset

class TrainingCallback:
    """Custom callback for logging training progress and updating Streamlit UI."""
    def __init__(self, st_placeholder=None):
        self.st_placeholder = st_placeholder

    def __call__(self, score, epoch, steps):
        log_message = f"Epoch {epoch}, Steps {steps}, Loss: {score:.4f}"
        logger.info(log_message)
        if self.st_placeholder:
            self.st_placeholder.write(log_message)

def fine_tune_model(model, train_data, test_data, epochs=1, retrain=False, st_placeholder=None):
    """Fine-tune MiniLM model and validate on test data."""
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Starting {'retraining' if retrain else 'fine-tuning'} of MiniLM model at {start_time}")
    
    train_examples = []
    for item in train_data:
        if not (isinstance(item.get("ocr_output"), str) and isinstance(item.get("corrected"), str)):
            logger.warning(f"Invalid training item: {item}. Skipping.")
            continue
        repeats = 5 if retrain else 1
        for _ in range(repeats):
            try:
                train_examples.append(InputExample(texts=[item["ocr_output"], item["corrected"]], label=1.0))
                negative_candidates = [d for d in train_data if d["corrected"] != item["corrected"] and isinstance(d["corrected"], str)]
                negative = random.choice(negative_candidates) if negative_candidates else item
                train_examples.append(InputExample(texts=[item["ocr_output"], negative["corrected"]], label=0.0))
            except Exception as e:
                logger.warning(f"Failed to create InputExample for item: {item}. Error: {str(e)}")
                continue
    
    if not train_examples:
        logger.error("No valid training examples generated. Aborting fine-tuning.")
        raise ValueError("No valid training examples available.")
    
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=1 if len(train_examples) <= 5 else 8)
    train_loss = losses.CosineSimilarityLoss(model)
    
    try:
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=10 if len(train_examples) <= 5 else 50,
            optimizer_params={'lr': 1e-5},
            show_progress_bar=True,
            callback=TrainingCallback(st_placeholder)
        )
        logger.info("Fine-tuning completed")
        
        validation_results = validate_model(model, test_data)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        model_path = os.path.join(MODEL_ROOT, timestamp)
        if os.path.exists(model_path):
            shutil.rmtree(model_path, ignore_errors=True)
        os.makedirs(model_path, exist_ok=True)
        logger.info(f"Created directory: {model_path}")
        
        if not os.access(model_path, os.W_OK):
            logger.error(f"No write permissions for {model_path}")
            return model, validation_results, start_time, None
        
        for attempt in range(3):
            try:
                logger.debug(f"Attempt {attempt + 1} to save model to {model_path}")
                model.save(model_path, safe_serialization=False)
                logger.info(f"Saved model to {model_path}")
                model = SentenceTransformer(model_path)
                logger.info(f"Reloaded model from {model_path}")
                end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                return model, validation_results, start_time, end_time
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed to save model: {str(e)}")
                time.sleep(2)
        logger.warning("Failed to save model after 3 attempts. Returning in-memory model.")
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return model, validation_results, start_time, end_time
    except Exception as e:
        logger.error(f"Error during fine-tuning: {str(e)}")
        raise

def validate_model(model, test_data):
    """Validate model on test data."""
    results = []
    labels = list(set(item["corrected"] for item in test_data if isinstance(item["corrected"], str)))
    if not labels:
        logger.error("No valid labels for validation.")
        return {"accuracy": 0.0, "details": []}
    
    index = faiss.IndexFlatL2(384)
    label_embeddings = model.encode(labels, show_progress_bar=False)
    faiss.normalize_L2(label_embeddings)
    index.add(label_embeddings)
    
    for case in test_data:
        if not (isinstance(case.get("ocr_output"), str) and isinstance(case.get("corrected"), str)):
            logger.warning(f"Invalid test case: {case}. Skipping.")
            continue
        input_text = case["ocr_output"]
        expected = case["corrected"]
        ocr_embedding = model.encode([input_text])[0]
        ocr_embedding = np.array([ocr_embedding], dtype=np.float32)
        faiss.normalize_L2(ocr_embedding)
        distances, indices = index.search(ocr_embedding, k=1)
        predicted_value = labels[indices[0][0]]
        confidence = float(min(max((1 - distances[0][0] / 2) * 100, 0.0), 100.0))
        correct = predicted_value == expected
        results.append({
            "input": input_text,
            "predicted": predicted_value,
            "expected": expected,
            "confidence": confidence,
            "correct": correct
        })
        logger.info(f"Validation: Input='{input_text}' -> Predicted='{predicted_value}' "
                   f"(Expected='{expected}', Confidence: {confidence:.2f}%, Correct: {correct})")
    
    accuracy = sum(r["correct"] for r in results) / len(results) * 100 if results else 0
    validation_results = {"accuracy": accuracy, "details": results}
    logger.info(f"Validation Accuracy: {accuracy:.2f}%")
    return validation_results

def load_or_train_model(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', st_placeholder=None):
    """Load the latest model or fine-tune a new one."""
    try:
        latest_model_path = get_latest_model_path()
        if latest_model_path:
            logger.info(f"Loading latest model from {latest_model_path}")
            try:
                model = SentenceTransformer(latest_model_path)
                logger.info("Successfully loaded model from disk")
                return model
            except Exception as e:
                logger.warning(f"Failed to load model from {latest_model_path}: {str(e)}. Fine-tuning new model.")
        
        logger.info(f"No pre-trained model found, loading model from Hugging Face")
        try:
            model = SentenceTransformer(model_name)
            logger.info(f"Loaded model {model_name} from Hugging Face")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise
        
        dataset = generate_dataset(n_per_class=DEFAULT_N_PER_CLASS)
        if not dataset:
            logger.error("Failed to generate dataset. Aborting.")
            raise ValueError("No valid dataset generated.")
        train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
        model, validation_results, start_time, end_time = fine_tune_model(model, train_data, test_data, epochs=1, st_placeholder=st_placeholder)
        if end_time:
            labels = dict(Counter(item["corrected"] for item in train_data))
            try:
                save_training_session(start_time, end_time, labels, validation_results, get_latest_model_path())
                logger.info(f"Saved initial training session: start={start_time}")
            except Exception as e:
                logger.error(f"Failed to save initial training session: {str(e)}")
        return model
    except Exception as e:
        logger.error(f"Error in load_or_train_model: {str(e)}")
        raise

def update_faiss_index(model, index, valid_values):
    """Update FAISS index with label embeddings."""
    index.reset()
    labels = [v["label"] for v in valid_values if isinstance(v["label"], str)]
    if not labels:
        logger.error("No valid labels for FAISS index.")
        return []
    embeddings = model.encode(labels, show_progress_bar=True)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    logger.info(f"FAISS index updated with {len(labels)} labels")
    return labels

def test_model(model, index, labels):
    """Test model accuracy on known inputs."""
    test_cases = [
        {"input": "can cuoc cong dan", "expected": "Căn cước công dân"},
        {"input": "chung minh nhan dan", "expected": "Chứng minh nhân dân"},
        {"input": "hop dong lao dong", "expected": "Hợp đồng lao động"},
    ]
    results = []
    for case in test_cases:
        input_text = case["input"]
        expected = case["expected"]
        processed_input = input_text
        ocr_embedding = model.encode([processed_input])[0]
        ocr_embedding = np.array([ocr_embedding], dtype=np.float32)
        faiss.normalize_L2(ocr_embedding)
        distances, indices = index.search(ocr_embedding, k=1)
        predicted_value = labels[indices[0][0]]
        confidence = float(min(max((1 - distances[0][0] / 2) * 100, 0.0), 100.0))
        correct = predicted_value == expected
        results.append({
            "input": input_text,
            "predicted": predicted_value,
            "expected": expected,
            "confidence": confidence,
            "correct": correct
        })
        logger.info(f"Test: Input='{input_text}' -> Predicted='{predicted_value}' "
                   f"(Expected='{expected}', Confidence: {confidence:.2f}%, Correct: {correct})")
    accuracy = sum(r["correct"] for r in results) / len(results) * 100
    logger.info(f"Test Accuracy: {accuracy:.2f}%")
    return results, accuracy