# synthetic_error_generator.py
import random
import unicodedata

# --------------------- Synthetic Error Generator ---------------------
def generate_synthetic_errors(text: str, n_errors: int = 5):
    vowels = "aeiou"
    replacements = {
        "ă": "a", "â": "a", "ê": "e", "ô": "o", "ơ": "o", "ư": "u",
        "đ": "d", "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
        "à": "a", "è": "e", "ù": "u", "ò": "o", "ì": "i",
        "ã": "a", "õ": "o", "ĩ": "i", "ũ": "u", "ỹ": "y"
    }

    def normalize(text):
        return unicodedata.normalize('NFKC', text)

    variants = set()
    text = normalize(text)
    for _ in range(n_errors):
        corrupted = list(text)
        for i in range(len(corrupted)):
            char = corrupted[i].lower()
            if char in replacements and random.random() < 0.5:
                corrupted[i] = replacements[char]
            elif char in vowels and random.random() < 0.2:
                corrupted[i] = random.choice(vowels)
            elif random.random() < 0.1:
                corrupted[i] = ''
        corrupted_text = ''.join(corrupted)
        if corrupted_text and corrupted_text != text:
            variants.add(corrupted_text)

    return list(variants)
