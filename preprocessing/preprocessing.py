"""
preprocessing.py
    Contains preprocessing methods to apply on the dataset.

Dependencies
!pip install numpy pandas pyspellchecker spacy nltk
!python -m spacy download en_core_web_sm
"""

import string
import numpy as np
import pandas as pd
import re
from spellchecker import SpellChecker
spell = SpellChecker()
import spacy
nlp = spacy.load("en_core_web_sm")
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop = stopwords.words('english')

emoticons_happy = {':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}', ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D', '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P', 'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)', '<3'}
emoticons_sad = {':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<', ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c', ':c', ':{', '>:\\', ';('}
emotes = emoticons_happy.union(emoticons_sad)

# ===== TRANSFORM METHODS =====
def tokenize(text):
    """Given string, apply Spacy's nlp then return list of text"""
    return [token.text for token in nlp(text)]

def spellcorrect(text):
    """Given string, list-split, apply SpellChecker's correction,
    return space-delimited list and no. of misspelt words"""
    original_text = text.split()
    corrected_text = [spell.correction(word) for word in original_text]
    return " ".join(corrected_text)

def remove_url(text):
    """Given string, remove url by regex."""
    # url = re.compile(r'https?://\S+|www\.\S+')  # Axel
    url = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')  # Tom
    return url.sub(r'',text)

def remove_html(text):
    """Given string, remove html by regex."""
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)

def remove_digits(text):
    """Given string, remove digits."""
    text = ''.join([i for i in text if not i.isdigit()])
    return text

def remove_punctuations(text):
    """Given string, remove punctuations."""
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)

def transform_lower_chars(text):
    """Given string, transform into lower characters."""
    return str(text).lower()

def remove_emojis(text):
    """Given text, remove emojis."""
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# ===== COLLECT METHODS =====

def collect_url(string):
    """Given string, collect urls by regex"""
    text = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',string)
    return "".join(text)

def collect_stopwords(tokens):
    """Given list of words, collect only NLTK stopwords"""
    return [token for token in tokens if token in stop]

def collect_punctuations(text):
    """Given list of words, collect only string punctuations"""
    return [c for c in text if c in string.punctuation]

def collect_digits(text):
    """Given string, collect only digits"""
    return " ".join([c for c in text if c.isdigit()])

def collect_uppercase_words(tokens):
    """Given list of tokens, collect only uppercase words"""
    return [1 for token in tokens if token.isupper()]

def collect_uppercase_chars(text):
    """Given string, collect only uppercase characters"""
    return [1 for c in text if c.isupper()]

def collect_url(string):
    """Given string, collect urls by regex."""
    text = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',string)
    return "".join(text)

def collect_at_mentions(text):
    """Given string, collect @mentions by regex."""
    line=re.findall(r'(?<=@)\w+',text)
    return " ".join(line)

def collect_hashtags(text):
    """Given string, collect #hashtags by regex."""
    line=re.findall(r'(?<=#)\w+',text)
    return " ".join(line)

def collect_numbers(text):
    """Given string, collect raw numbers by regex."""
    line=re.findall(r'[0-9]+',text)
    return " ".join(line)

def collect_entities(text):
    """Given list of tokens, collect entities using Spacy."""
    return [token.text for token in nlp(text).ents]


# ===== NUMERIC METHODS =====
def num_words(tokens):
    """Given list of words, return no. of words (int)"""
    return len(tokens)

def num_chars(text):
    """Given string, return no. of characters (int)"""
    return len(text)

def num_stopwords(tokens):
    """Given list of words, return no. of NLTK stopwords (int)"""
    return len(collect_stopwords(tokens))

def num_special_chars(text):
    """Given string, return no. of punctuation characters (int)"""
    return len(collect_punctuations(text))

def num_numeric(text):
    """Given string, return no. of digits (int)"""
    return len(collect_digits(text))

def num_uppercase_words(tokens):
    """Given list of words, return no. of uppercase words (int)"""
    return len(collect_uppercase_words(tokens))

def num_uppercase_chars(text):
    """Given string, return no. of uppercase characters (int)"""
    return len(collect_uppercase_chars(text))

def num_misspelt_words(text):
    """Given string, return no. of misspelt words."""
    original_text = text.split()
    corrected_text = spellcorrect(text)
    return sum([1 for o, c in zip(original_text, corrected_text) if o != c])


# ===== DERIVED FEATURES =====
def sum_word_len(tokens):
    """Given list of words, return sum of length of words (int)"""
    return sum([len(token) for token in tokens])

def avg_word_len(tokens):
    """Given list of words, return average word length (int)"""
    return sum_word_len(tokens) / num_words(tokens)

def ratio_uppercase_chars(text):
    """Given text, return ratio of uppercase words (float)"""
    return num_uppercase_chars(text) / num_chars(text)

# ===== BOOLEAN METHODS =====
def is_emote(tokens):
    return [1 for token in tokens if token in emotes]


"""
preprocess(df) creates columns of preprocessed data in the DataFrame in-place.
"""
def preprocess(df):
    # Transformations
    df['text'] = df['text'].apply(remove_html)
    df['num_misspelt_words'] = df['text'].apply(num_misspelt_words)
    df['text'] = df['text'].apply(spellcorrect)
    df['location'].fillna(0, inplace=True)
    df['keyword'].fillna(0, inplace=True)

    # Feature creation
    df['tokens'] = df['text'].apply(tokenize)
    df['url'] = df['text'].apply(collect_url)
    df['at_mentions'] = df['text'].apply(collect_at_mentions)
    df['hashtags'] = df['text'].apply(collect_hashtags)
    df['numbers'] = df['text'].apply(collect_numbers)
    df['digits'] = df['text'].apply(collect_digits)

    # Numeric features
    df['num_special_chars'] = df['text'].apply(num_special_chars)
    df['num_chars'] = df['text'].apply(num_chars)
    df['num_words'] = df['tokens'].apply(num_words)
    df['num_stopwords'] = df['tokens'].apply(num_stopwords)
    df['num_numeric'] = df['text'].apply(num_numeric)
    df['num_uppercase_words'] = df['tokens'].apply(num_uppercase_words)
    df['num_uppercase_chars'] = df['text'].apply(num_uppercase_chars)
    df['length'] = df['text'].apply(len)
    df['num_hashtags'] = df['text'].apply(lambda x: len([c for c in str(x) if c == '#']))
    df['num_mentions'] = df['text'].apply(lambda x: len([c for c in str(x) if c == '@']))
    df['count_capital_letters'] = df['text'].apply(lambda x: len(re.findall(r'[A-Z]', x)))
    df['ratio_capital_letters'] = df['length'] / df['count of capital letters']
    df['external_url'] = df['text'].apply(collect_url)

    # Derived features
    df['sum_word_len'] = df['tokens'].apply(sum_word_len)
    df['avg_word_len'] = df['tokens'].apply(avg_word_len)
    df['ratio_uppercase_chars'] = df['text'].apply(ratio_uppercase_chars)

    # Final text cleaning
    df['text'] = df['text'].apply(remove_url)
    df['text'] = df['text'].apply(transform_lower_chars)
    df['text'] = df['text'].apply(remove_digits)
    df['text'] = df['text'].apply(remove_punctuations)
    df['text'] = df['text'].apply(remove_emojis)
