import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import Dataset, DataLoader
import random

# ==== 1. Bộ dữ liệu mẫu cho huấn luyện và kiểm thử ====

data = [
    # Địa chỉ viết tắt/lỗi -> đúng
    ("123 Dg Trần Hưng Đa", "123 Đường Trần Hưng Đạo"),
    ("56 dg Phan Đình Phùng", "56 Đường Phan Đình Phùng"),
    ("09 ngo 2 PĐP", "09 Ngõ 2 Phan Đình Phùng"),
    ("1 duong Nguyen Du", "1 Đường Nguyễn Du"),
    ("7 Dg Hai Ba Trung", "7 Đường Hai Bà Trưng"),
    ("số 2 ngo 55 Le Duan", "số 2 Ngõ 55 Lê Duẩn"),
    ("14 ngo 3 KĐT Mỹ Dình", "14 Ngõ 3 Khu đô thị Mỹ Đình"),
    ("25 duong 3/2", "25 Đường 3 Tháng 2"),
]

random.shuffle(data)
train_data = data[:6]
val_data = data[6:]

# ==== 2. Khởi tạo model và tokenizer ====

MODEL_NAME = "VietAI/vit5-base"  # Nhẹ và tối ưu cho tiếng Việt
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ==== 3. Dataset PyTorch ====

class AddressCorrectionDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        source, target = self.pairs[idx]
        input_text = f"Sửa địa chỉ: {source}"
        input_ids = tokenizer(input_text, padding='max_length', truncation=True, max_length=64, return_tensors="pt").input_ids.squeeze()
        labels = tokenizer(target, padding='max_length', truncation=True, max_length=64, return_tensors="pt").input_ids.squeeze()
        labels[labels == tokenizer.pad_token_id] = -100
        return {"input_ids": input_ids, "labels": labels}

train_dataset = AddressCorrectionDataset(train_data)
val_dataset = AddressCorrectionDataset(val_data)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1)

# ==== 4. Huấn luyện ====

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

print("Bắt đầu huấn luyện...")

for epoch in range(5):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"[Epoch {epoch+1}] Train loss: {total_loss/len(train_loader):.4f}")

# ==== 5. Đánh giá và In kết quả ====

model.eval()
print("\n== Kết quả kiểm thử ==")

for input_text, expected in val_data:
    full_input = f"Sửa địa chỉ: {input_text}"
    input_ids = tokenizer(full_input, return_tensors="pt").input_ids.to(device)
    output_ids = model.generate(input_ids, max_length=64)
    predicted = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"OCR: {input_text}")
    print(f"✔ Kỳ vọng: {expected}")
    print(f"🤖 Gợi ý:   {predicted}")
    print("-" * 40)
