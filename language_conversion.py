import processingLanguage
from indic_transliteration import sanscript

def language_conversion(text, src):
    """Convert text from an Indian language script to ITRANS."""
    
    source_lang = {
        'te': sanscript.TELUGU,
        'kn': sanscript.KANNADA,
        'ta': sanscript.TAMIL,
    }

    if src not in source_lang:
        raise ValueError(f"Unsupported source language: {src}")
    
    temp_source = source_lang[src]
    print("hello")
    print(sanscript.transliterate(text, temp_source, sanscript.ITRANS).capitalize())
    return sanscript.transliterate(text, temp_source, sanscript.ITRANS).capitalize()
