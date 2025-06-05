import random
import re
import unicodedata
import logging

logger = logging.getLogger(__name__)

def generate_vietnamese_errors(text, n_samples=5):
    """
    Generate error variants for Vietnamese text covering multiple error types.
    Ensures exactly 1 variant for 'remove_tones' and at least 5 variants for each other error type.
    Preserves the original case of the input text.
    
    Args:
        text (str): The input text to generate errors for (e.g., 'Căn cước công dân').
        n_samples (int): Number of error variants to generate (default: 5, minimum 31).
    
    Returns:
        list: List of unique error variants.
    """
    if not text or n_samples < 1:
        logger.warning(f"Invalid input: text='{text}', n_samples={n_samples}")
        return []
    
    variants = set()
    text = text.strip()
    
    # Vietnamese tone marks and their base characters (includes both cases)
    tone_map = {
        'À': 'A', 'Á': 'A', 'Ả': 'A', 'Ã': 'A', 'Ạ': 'A',
        'à': 'a', 'á': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
        'È': 'E', 'É': 'E', 'Ẻ': 'E', 'Ẽ': 'E', 'Ẹ': 'E',
        'è': 'e', 'é': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e',
        'Ì': 'I', 'Í': 'I', 'Ỉ': 'I', 'Ĩ': 'I', 'Ị': 'I',
        'ì': 'i', 'í': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
        'Ò': 'O', 'Ó': 'O', 'Ỏ': 'O', 'Õ': 'O', 'Ọ': 'O',
        'ò': 'o', 'ó': 'o', 'ỏ': 'o', 'õ': 'o', 'ọ': 'o',
        'Ù': 'U', 'Ú': 'U', 'Ủ': 'U', 'Ũ': 'U', 'Ụ': 'U',
        'ù': 'u', 'ú': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u',
        'Ỳ': 'Y', 'Ý': 'Y', 'Ỷ': 'Y', 'Ỹ': 'Y', 'Ỵ': 'Y',
        'ỳ': 'y', 'ý': 'y', 'ỷ': 'y', 'ỹ': 'y', 'ỵ': 'y',
        'Â': 'A', 'Ê': 'E', 'Ô': 'O',
        'â': 'a', 'ê': 'e', 'ô': 'o',
        'Ằ': 'A', 'Ắ': 'A', 'Ẳ': 'A', 'Ẵ': 'A', 'Ặ': 'A',
        'ằ': 'a', 'ắ': 'a', 'ẳ': 'a', 'ẵ': 'a', 'ặ': 'a',
        'Ờ': 'O', 'Ớ': 'O', 'Ở': 'O', 'Ỡ': 'O', 'Ợ': 'O',
        'ờ': 'o', 'ớ': 'o', 'ở': 'o', 'ỡ': 'o', 'ợ': 'o',
        'Ư': 'U', 'Ừ': 'U', 'Ứ': 'U', 'Ử': 'U', 'Ữ': 'U', 'Ự': 'U',
        'ư': 'u', 'ừ': 'u', 'ứ': 'u', 'ử': 'u', 'ữ': 'u', 'ự': 'u',
        'Đ': 'D', 'đ': 'd'
    }
    
    # Inverse tone map for changing tones
    tone_groups = {}
    for toned, base in tone_map.items():
        if base not in tone_groups:
            tone_groups[base] = []
        tone_groups[base].append(toned)
    
    # Similar-looking character substitutions (both cases)
    visual_similar = {
        'A': ['Ă', 'Â', '4', '@'], 'a': ['ă', 'â', '4', '@'],
        'D': ['Đ', 'B', 'P'], 'd': ['đ', 'b', 'p'],
        'O': ['Ô', 'Ơ', '0', 'Q'], 'o': ['ô', 'ơ', '0', 'q'],
        'U': ['Ư', 'V', 'W'], 'u': ['ư', 'v', 'w'],
        'I': ['1', 'L', '!'], 'i': ['1', 'l', '!'],
        'S': ['5', 'Z', '$'], 's': ['5', 'z', '$'],
        'G': ['9', 'Q'], 'g': ['9', 'q'],
        'T': ['7', '+'], 't': ['7', '+'],
        'N': ['M', 'H'], 'n': ['m', 'h'],
        'E': ['3', '€'], 'e': ['3', '€']
    }
    
    # Common Vietnamese spelling mistakes (both cases)
    spelling_mistakes = {
        'C': ['K'], 'c': ['k'],
        'CH': ['TR'], 'ch': ['tr'],
        'TR': ['CH'], 'tr': ['ch'],
        'GI': ['D', 'R'], 'gi': ['d', 'r'],
        'D': ['GI', 'Z'], 'd': ['gi', 'z'],
        'PH': ['F'], 'ph': ['f'],
        'U': ['Ô', 'O'], 'u': ['ô', 'o'],
        'Ô': ['U', 'O'], 'ô': ['u', 'o'],
        'NG': ['N', 'M'], 'ng': ['n', 'm'],
        'N': ['NG', 'M'], 'n': ['ng', 'm'],
        'NH': ['N'], 'nh': ['n'],
        'T': ['TH'], 't': ['th'],
        'TH': ['T'], 'th': ['t']
    }
    
    # Noise characters for handwriting/font errors
    noise_chars = ['~', '`', '^', '*', '#', '(', ')']
    
    def remove_tones(s):
        """Remove all tone marks (e.g., 'Căn cước' → 'Can cuoc')."""
        result = ''
        for char in unicodedata.normalize('NFD', s):
            if char in tone_map:
                result += tone_map[char]
            else:
                result += char
        return unicodedata.normalize('NFC', result)
    
    def random_tone_change(s):
        """Randomly change or remove tone marks (e.g., 'Căn' → 'Cẫn', 'Can')."""
        chars = list(s)
        for i, char in enumerate(chars):
            if char in tone_map and random.random() < 0.5:
                base_char = tone_map[char]
                if random.random() < 0.5:
                    chars[i] = base_char  # Remove tone
                else:
                    possible_tones = [t for t in tone_groups.get(base_char, []) if t != char]
                    if possible_tones:
                        chars[i] = random.choice(possible_tones)  # Change tone
        return ''.join(chars)
    
    def space_error(s):
        """Add or remove spaces randomly (e.g., 'Căn cước' → 'Căncuoc', 'Căn  cước')."""
        words = s.split()
        if not words:
            return s
        if random.random() < 0.5:
            # Remove spaces
            return ''.join(words)
        else:
            # Add extra spaces
            result = []
            for word in words:
                result.append(word)
                if random.random() < 0.4:
                    result.append(' ' * random.randint(1, 2))
            return ''.join(result).strip()
    
    def spelling_error(s):
        """Introduce common Vietnamese spelling mistakes, preserving case."""
        for orig, repls in spelling_mistakes.items():
            def replacement(match):
                repl = random.choice(repls)
                # Preserve case of the first character
                if match.group(0)[0].isupper():
                    return repl[0].upper() + repl[1:].lower()
                return repl
            if random.random() < 0.3:
                s = re.sub(r'\b' + orig + r'\b', replacement, s)
        return s
    
    def visual_similar_error(s):
        """Substitute characters with visually similar ones (e.g., 'o' → '0')."""
        chars = list(s)
        for i, char in enumerate(chars):
            if char in visual_similar and random.random() < 0.25:
                chars[i] = random.choice(visual_similar[char])
        return ''.join(chars)
    
    def handwriting_font_error(s):
        """Simulate handwriting or complex font errors with noise (e.g., 'Căn' → 'Căn~')."""
        chars = list(s)
        for i in range(len(chars)):
            if random.random() < 0.15:
                chars[i] = random.choice(noise_chars + [chars[i]])
            elif random.random() < 0.1:
                chars.insert(i, random.choice(noise_chars))
        return ''.join(chars)
    
    def missing_chars(s):
        """Randomly remove characters (e.g., 'Căn cước công dân' → 'Căn cướ công dâ')."""
        chars = list(s)
        if len(chars) <= 3:  # Avoid excessive removal for short strings
            return s
        num_to_remove = random.randint(1, max(1, len(chars) // 4))
        indices = random.sample(range(len(chars)), num_to_remove)
        for idx in sorted(indices, reverse=True):
            del chars[idx]
        return ''.join(chars) if chars else s
    
    # List of error functions
    error_types = [
        remove_tones,
        random_tone_change,
        space_error,
        spelling_error,
        visual_similar_error,
        handwriting_font_error,
        missing_chars
    ]
    
    # Track samples per error type
    samples_by_type = {func.__name__: [] for func in error_types}
    
    # Step 1: Generate exactly 1 sample for remove_tones
    variant = remove_tones(text)
    if variant != text and variant not in variants:
        variants.add(variant)
        samples_by_type['remove_tones'].append(variant)
    
    # Step 2: Generate at least 5 samples for each other error type
    other_error_types = [f for f in error_types if f != remove_tones]
    for error_func in other_error_types:
        attempts = 0
        max_attempts = 50
        while len(samples_by_type[error_func.__name__]) < 5 and attempts < max_attempts:
            variant = error_func(text)
            if random.random() < 0.35:
                extra_error = random.choice([f for f in error_types if f != error_func])
                variant = extra_error(variant)
            if variant != text and variant not in variants:
                variants.add(variant)
                samples_by_type[error_func.__name__].append(variant)
            attempts += 1
    
    # Step 3: Generate additional samples to reach n_samples
    while len(variants) < min(n_samples, 31) and any(len(samples_by_type[func.__name__]) < 10 for func in other_error_types):
        error_func = random.choice(other_error_types)
        variant = error_func(text)
        if random.random() < 0.35:
            extra_error = random.choice([f for f in error_types if f != error_func])
            variant = extra_error(variant)
        if variant != text and variant not in variants:
            variants.add(variant)
            samples_by_type[error_func.__name__].append(variant)
    
    # Step 4: If n_samples > 31, fill with random combinations
    while len(variants) < n_samples:
        variant = text
        num_errors = random.randint(1, 3)
        applied_errors = random.sample(error_types, num_errors)
        for error_func in applied_errors:
            variant = error_func(variant)
        if variant != text and variant not in variants:
            variants.add(variant)
            random_func = random.choice(error_types)
            samples_by_type[random_func.__name__].append(variant)
    
    # Log sample distribution
    for func_name, samples in samples_by_type.items():
        logger.debug(f"Error type '{func_name}': {len(samples)} samples")
    
    result = list(variants)[:n_samples]
    logger.debug(f"Generated {len(result)} error variants for '{text}': {result}")
    return result
# Ví dụ sử dụng
if __name__ == "__main__":
    correct_text = "căn cước công dân"
    errors = generate_vietnamese_errors(correct_text, 10)
    
    print(f"Chuỗi đúng: {correct_text}")
    print("Các biến thể lỗi:")
    for i, error in enumerate(errors, 1):
        print(f"{i}. {error}")