from symspellpy import SymSpell, Verbosity
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

# ------------------ SETUP ------------------

# Load SymSpell for English spell checking
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
sym_spell.load_dictionary("frequency_dictionary_en.txt", term_index=0, count_index=1)

# Load Indic NLP Normalizer
normalizer_ta = IndicNormalizerFactory().get_normalizer("ta")
normalizer_kn = IndicNormalizerFactory().get_normalizer("kn")
normalizer_te = IndicNormalizerFactory().get_normalizer("te")


# ---------------- SPELL CHECKING (ENGLISH ONLY) ----------------

def spell_check(text, lang):
    if lang == "en":
        suggestions = sym_spell.lookup(text, Verbosity.CLOSEST, max_edit_distance=2)
        return suggestions[0].term if suggestions else text
    return text  # No spell correction for Tamil, Kannada, Telugu


# ---------------- TEXT NORMALIZATION ----------------

def normalize_text(text, lang):
    if lang == "ta":
        return normalizer_ta.normalize(text)
    elif lang == "kn":
        return normalizer_kn.normalize(text)
    elif lang == "te":
        return normalizer_te.normalize(text)
    return text  # No normalization for unsupported languages


# ---------------- MAIN PROCESS ----------------

def process_text(text, lang):
    # print(f"Original Text ({lang}): {text}")
    
    # Step 1: Spell Check (Only for English)
    corrected_text = spell_check(text, lang)
    # print(f"Spell Checked: {corrected_text}")

    # Step 2: Normalize Text (Tamil, Kannada, Telugu)
    normalized_text = normalize_text(corrected_text, lang)
    # print(f"Normalized: {normalized_text}")

    return normalized_text

def main(text,src):
    # t
    t = process_text(text, src)
    return t
