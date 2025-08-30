import string
import nltk
from langdetect import detect
from nltk.corpus import stopwords

# Download stopwords if not already present
nltk.download('stopwords')

# Define supported languages
SUPPORTED_LANGUAGES = ["english", "french", "spanish", "italian", "russian", "catalan"]
STOPWORDS_DICT = {lang: set(stopwords.words(lang)) for lang in SUPPORTED_LANGUAGES}
PUNCTUATION = set(string.punctuation)

def detect_language(text):
    """Detects the language of the given text."""
    try:
        lang_code = detect(text)
        lang_mapping = {
            "en": "english",
            "fr": "french",
            "es": "spanish",
            "it": "italian",
            "ru": "russian",
            "ca": "catalan"
        }
        return lang_mapping.get(lang_code, "english")
    except:
        return "english"
