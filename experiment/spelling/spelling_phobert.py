import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from pyvi import ViTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "vinai/phobert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
model.eval()

def is_oov(word):
    """Kiểm tra từ có trong vocab hay không"""
    # Bỏ dấu _ (pyvi tokenizes thành dạng có dấu gạch dưới)
    plain_word = word.replace("_", "")
    return plain_word.lower() not in tokenizer.get_vocab()

def correct_spelling(text, top_k=5, conf_threshold=3.0):
    segmented_text = ViTokenizer.tokenize(text)
    words = segmented_text.split()
    corrected_words = words.copy()

    for i, word in enumerate(words):
        if not is_oov(word):
            continue

        masked_words = words.copy()
        masked_words[i] = tokenizer.mask_token  # '<mask>'
        masked_sentence = " ".join(masked_words)

        inputs = tokenizer(masked_sentence, return_tensors="pt").to(device)
        mask_index = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1].item()

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, mask_index]
            probs = torch.softmax(logits, dim=0)
            topk = torch.topk(probs, top_k)

        top_tokens = topk.indices.cpu().tolist()
        top_scores = topk.values.cpu().tolist()

        # Lấy token gốc nếu có trong vocab, không phải mask token
        original_token_id = tokenizer.convert_tokens_to_ids(word.replace("_", ""))
        original_score = probs[original_token_id].item() if original_token_id in tokenizer.get_vocab().values() else 0

        # Lấy token thay thế tốt nhất (khác token gốc)
        for token_id, score in zip(top_tokens, top_scores):
            candidate_token = tokenizer.decode([token_id]).strip()
            if candidate_token.lower() != word.replace("_", "").lower():
                # Nếu xác suất candidate gấp conf_threshold lần xác suất từ gốc thì thay
                if score > conf_threshold * original_score:
                    print(f"[!] {word} → {candidate_token}")
                    corrected_words[i] = candidate_token
                    break

    # Ghép lại thành câu, thay dấu gạch dưới thành khoảng trắng
    return " ".join(corrected_words).replace("_", " ")

# Test
input_text = "toi đang họk tiếng việt"
print("Câu gốc:     ", input_text)
corrected = correct_spelling(input_text)
print("Câu sau sửa:", corrected)
