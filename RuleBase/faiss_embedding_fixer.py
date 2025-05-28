
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import pickle

class FassisEmbeddingFixer:
    def __init__(self, model_path="trained_model", embedding_model="keepitreal/vietnamese-sbert"):
        self.model_path = model_path
        self.model = SentenceTransformer(embedding_model)
        self.indexes = defaultdict(lambda: faiss.IndexFlatIP(768))
        self.human_values = defaultdict(list)

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self._load_all()

    def learn(self, field, human_value):
        if not human_value:
            return
        emb = self.model.encode([human_value])[0]
        self.indexes[field].add(np.array([emb], dtype=np.float32))
        self.human_values[field].append(human_value)

    def suggest(self, field, ocr_value, threshold=0.7):
        if not self.human_values[field]:
            return None
        emb = self.model.encode([ocr_value])[0].astype(np.float32)
        D, I = self.indexes[field].search(np.array([emb]), k=1)
        if D[0][0] > threshold:
            return self.human_values[field][I[0][0]]
        return None

    def save_all(self):
        for field in self.indexes:
            index_file = os.path.join(self.model_path, f"{field}.index")
            data_file = os.path.join(self.model_path, f"{field}_values.pkl")
            faiss.write_index(self.indexes[field], index_file)
            with open(data_file, "wb") as f:
                pickle.dump(self.human_values[field], f)

    def _load_all(self):
        for file in os.listdir(self.model_path):
            if file.endswith(".index"):
                field = file.replace(".index", "")
                index_file = os.path.join(self.model_path, file)
                data_file = os.path.join(self.model_path, f"{field}_values.pkl")
                try:
                    self.indexes[field] = faiss.read_index(index_file)
                    with open(data_file, "rb") as f:
                        self.human_values[field] = pickle.load(f)
                except Exception as e:
                    print(f"Failed to load model for field {field}: {e}")
