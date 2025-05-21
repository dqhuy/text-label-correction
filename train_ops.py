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
from db_ops import get_all_labels, generate_vietnamese_errors

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_PATH = "trained_model"
DEFAULT_N_PER_CLASS = 100
def generate_dataset(n_per_class=DEFAULT_N_PER_CLASS):
    """Generate simulated dataset for training."""
    valid_values = get_all_labels()
    static_labels = [v["label"] for v in valid_values]
    
    dataset = []
    for value in static_labels[:10]:
        for _ in range(n_per_class):
            ocr_output = generate_vietnamese_errors(value) if random.random() > 0.1 else value
            dataset.append({"ocr_output": ocr_output, "corrected": value})
    logger.info(f"Generated {len(dataset)} examples")
    distribution = Counter(item["corrected"] for item in dataset)
    logger.info(f"Dataset distribution: {dict(distribution)}")
    return dataset

def generate_custom_dataset(custom_label, n_per_class=100, only_new_label=False):
    """Generate dataset for a custom label, optionally only for the new label."""
    dataset = []
    
    if only_new_label:
        # Generate data only for the custom label
        for _ in range(n_per_class):
            ocr_output = generate_vietnamese_errors(custom_label) if random.random() > 0.1 else custom_label
            dataset.append({"ocr_output": ocr_output, "corrected": custom_label})
    else:
        # Generate data for custom label and existing labels
        valid_values = get_all_labels()
        static_labels = [v["label"] for v in valid_values]
        if custom_label not in static_labels:
            static_labels.append(custom_label)
        
        for value in static_labels:
            for _ in range(n_per_class):
                ocr_output = generate_vietnamese_errors(value) if random.random() > 0.1 else value
                dataset.append({"ocr_output": ocr_output, "corrected": value})
    
    logger.info(f"Generated {len(dataset)} examples for custom label '{custom_label}'")
    distribution = Counter(item["corrected"] for item in dataset)
    logger.info(f"Custom dataset distribution: {dict(distribution)}")
    return dataset

class TrainingCallback:
    """Custom callback for logging training progress and updating Streamlit UI."""
    def __init__(self, st_placeholder=None):
        self.st_placeholder = st_placeholder

    def __call__(self, score, epoch, steps):
        log_message = f"Epoch {epoch}, Steps {steps}, Loss: {score:.4f}"
        logger.info(log_message)
        if self.st_placeholder:
            self.st_placeholder.write(log_message)

def fine_tune_model(model, train_data, epochs=3, retrain=False, st_placeholder=None):
    """Fine-tune MiniLM model with hard negative mining."""
    logger.info(f"Starting {'retraining' if retrain else 'fine-tuning'} of MiniLM model...")
    train_examples = []
    for item in train_data:
        repeats = 5 if retrain else 1
        for _ in range(repeats):
            # Positive example
            train_examples.append(InputExample(texts=[item["ocr_output"], item["corrected"]], label=1.0))
            # Hard negative: pair with a different correct label
            negative = random.choice([d for d in train_data if d["corrected"] != item["corrected"]]) if len(set(d["corrected"] for d in train_data)) > 1 else item
            train_examples.append(InputExample(texts=[item["ocr_output"], negative["corrected"]], label=0.0))
    
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=1 if len(train_examples) <= 5 else 16)
    train_loss = losses.CosineSimilarityLoss(model)
    
    try:
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=10 if len(train_examples) <= 5 else 100,
            optimizer_params={'lr': 1e-5},  # Lower LR for better convergence
            show_progress_bar=True,
            callback=TrainingCallback(st_placeholder)
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

def load_or_train_model(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', st_placeholder=None):
    """Load or fine-tune MiniLM model, generating dataset only when needed."""
    if os.path.exists(MODEL_PATH):
        logger.info(f"Loading pre-trained model from {MODEL_PATH}")
        try:
            model = SentenceTransformer(MODEL_PATH)
        except Exception as e:
            logger.warning(f"Failed to load model from {MODEL_PATH}: {str(e)}. Fine-tuning new model.")
            dataset = generate_dataset(n_per_class=DEFAULT_N_PER_CLASS)
            train_data, _ = train_test_split(dataset, test_size=0.2, random_state=42)
            model = SentenceTransformer(model_name)
            model = fine_tune_model(model, train_data, epochs=3, st_placeholder=st_placeholder)
    else:
        logger.info(f"No pre-trained model found at {MODEL_PATH}, fine-tuning new model")
        dataset = generate_dataset(n_per_class=DEFAULT_N_PER_CLASS)
        train_data, _ = train_test_split(dataset, test_size=0.2, random_state=42)
        model = SentenceTransformer(model_name)
        model = fine_tune_model(model, train_data, epochs=3, st_placeholder=st_placeholder)
    return model

def update_faiss_index(model, index, valid_values):
    """Update FAISS index with label embeddings."""
    index.reset()
    labels = [v["label"] for v in valid_values]
    embeddings = model.encode(labels, show_progress_bar=True)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    logger.info(f"FAISS index updated with {len(labels)} labels")
    return labels

def test_model(model, index, labels):
    """Test model accuracy on known inputs."""
    test_cases = [
        
        {"input": "căn cuoc công dan", "expected": "Căn cước công dân"},
    ]
    results = []
    for case in test_cases:
        input_text = case["input"]
        expected = case["expected"]
        processed_input = input_text  # Preprocessing handled by model
        ocr_embedding = model.encode([processed_input])[0]
        ocr_embedding = np.array([ocr_embedding], dtype=np.float32)
        faiss.normalize_L2(ocr_embedding)
        distances, indices = index.search(ocr_embedding, k=1)
        predicted_value = labels[indices[0][0]]
        confidence = min(max((1 - distances[0][0] / 2) * 100, 0.0), 100.0)
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