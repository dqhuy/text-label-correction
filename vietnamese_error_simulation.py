import random
import re
from typing import List

def generate_vietnamese_errors(text: str, num_variants: int = 5) -> List[str]:
    """
    Tạo các biến thể có lỗi của chuỗi tiếng Việt đầu vào, mô phỏng lỗi gõ phím và OCR
    
    Args:
        text: Chuỗi tiếng Việt đầu vào
        num_variants: Số lượng biến thể lỗi cần tạo
    
    Returns:
        Danh sách các chuỗi có chứa lỗi
    """
    # Các cặp ký tự dễ nhầm lẫn trong tiếng Việt
    VIETNAMESE_ERROR_PAIRS = [
        ('ă', 'aw'), ('ă', 'a'), ('ă', 'â'),
        ('â', 'aa'), ('â', 'a'), ('â', 'ă'),
        ('đ', 'dd'), ('đ', 'd'),
        ('ê', 'ee'), ('ê', 'e'),
        ('ô', 'oo'), ('ô', 'o'),
        ('ơ', 'ow'), ('ơ', 'o'), ('ơ', 'ô'),
        ('ư', 'uw'), ('ư', 'u'),
        ('s', 'x'), ('x', 's'),
        ('ch', 'c'), ('ch', 'tr'), ('tr', 'ch'),
        ('gi', 'd'), ('gi', 'r'), ('d', 'gi'), ('r', 'gi'),
        ('n', 'l'), ('l', 'n'),
        ('i', 'y'), ('y', 'i'),
        ('c', 'k'), ('k', 'c'), ('q', 'k'),
        ('ph', 'f'), ('f', 'ph'),
        ('g', 'gh'), ('gh', 'g'),
        ('ng', 'ngh'), ('ngh', 'ng'),
    ]
    
    # Các ký tự dễ nhầm lẫn trong OCR
    OCR_ERROR_PAIRS = [
        ('a', 'o'), ('o', 'a'), ('o', 'c'), ('c', 'o'),
        ('e', 'c'), ('c', 'e'), ('b', 'h'), ('h', 'b'),
        ('d', 'cl'), ('cl', 'd'), ('m', 'n'), ('n', 'm'),
        ('u', 'v'), ('v', 'u'), ('w', 'vv'), ('i', 'l'),
        ('t', 'f'), ('f', 't'), ('g', 'q'), ('q', 'g'),
    ]
    
    variants = []
    
    for _ in range(num_variants):
        # Chọn ngẫu nhiên số lỗi (1-3 lỗi mỗi chuỗi)
        num_errors = random.randint(1, 3)
        corrupted = list(text.lower())
        
        for __ in range(num_errors):
            if len(corrupted) == 0:
                break
                
            # Chọn ngẫu nhiên vị trí để thêm lỗi
            pos = random.randint(0, len(corrupted)-1)
            char = corrupted[pos]
            
            # 50% lỗi tiếng Việt, 50% lỗi OCR
            if random.random() <= 0.5:
                # Lỗi tiếng Việt
                for pair in VIETNAMESE_ERROR_PAIRS:
                    if char in pair:
                        # Thay thế bằng ký tự dễ nhầm
                        replacement = pair[1] if pair[0] == char else pair[0]
                        # Đôi khi thêm/ghép ký tự
                        if random.random() < 0.3 and len(replacement) == 1:
                            replacement = replacement * random.randint(1, 2)
                        corrupted[pos] = replacement
                        break
            else:
                # Lỗi OCR
                for pair in OCR_ERROR_PAIRS:
                    if char in pair:
                        replacement = pair[1] if pair[0] == char else pair[0]
                        corrupted[pos] = replacement
                        break
                
            # Đôi khi xóa hoặc thêm ký tự (30% xác suất)
            if random.random() <= 0.3:
                if random.random() < 0.5 and len(corrupted) > 1:
                    # Xóa ký tự
                    del corrupted[pos]
                else:
                    # Thêm ký tự ngẫu nhiên
                    random_char = random.choice(['a', 'e', 'o', 'd', 'm', 'n', 'g', 'h'])
                    corrupted.insert(pos, random_char)
        
        # Ghép lại thành chuỗi và thêm vào danh sách
        variant = ''.join(corrupted)
        
        # Đôi khi thêm khoảng trắng ngẫu nhiên (10% xác suất)
        if random.random() <= 0.2:
            space_pos = random.randint(1, len(variant)-1)
            variant = variant[:space_pos] + ' ' + variant[space_pos:]
        
        variants.append(variant)
    
    return variants

# Ví dụ sử dụng
if __name__ == "__main__":
    correct_text = "căn cước công dân"
    errors = generate_vietnamese_errors(correct_text, 10)
    
    print(f"Chuỗi đúng: {correct_text}")
    print("Các biến thể lỗi:")
    for i, error in enumerate(errors, 1):
        print(f"{i}. {error}")