"""
preprocessing.py
    Contains preprocessing methods to apply on the dataset.

Dependencies
!pip install numpy pandas pyspellchecker spacy nltk
!python -m spacy download en_core_web_sm
"""


import numpy as np
import pandas as pd
from spellchecker import SpellChecker
spell = SpellChecker()
import spacy
nlp = spacy.load("en_core_web_sm")
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop = stopwords.words('english')

emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
    ])
emoticons_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
    ])
emotes = emoticons_happy.union(emoticons_sad)
# ===== TRANSFORM METHODS =====
def tokenize(text):
    """Given string, apply Spacy's nlp then return list of text"""
    return [token.text for token in nlp(text)]

def spellcorrect(text):
    """Given string, list-split, apply SpellChecker's correction,
    return space-delimited list"""
    return " ".join([spell.correction(word) for word in text.split()])

def collect_stopwords(tokens):
    """Given list of words, collect only NLTK stopwords"""
    return [token for token in tokens if token in stop]

def collect_punctuations(tokens):
    """Given list of words, collect only string punctuations"""
    return [c for c in text if c in string.punctuation]

def collect_digits(text):
    """Given string, collect only digits"""
    return [c for c in text if c.isdigit()]

def collect_uppercase_words(tokens):
    """Given list of tokens, collect only uppercase words"""
    return [1 for token in tokens if token.isupper()]

def collect_uppercase_chars(text):
    """Given string, collect only uppercase characters"""
    return [1 for c in text if c.isupper()]

def has_emote(tokens):
    return [1 for token in tokens if token in emotes]

# ===== NUMERIC METHODS =====
def num_words(tokens):
    """Given list of words, return no. of words (int)"""
    return len(tokens)

def num_chars(tokens):
    """Given string, return no. of characters (int)"""
    return num_chars = lambda text: len(text)

def num_stopwords(tokens):
    """Given list of words, return no. of NLTK stopwords (int)"""
    return len(filter_stopwords(tokens))

def num_special_chars(text):
    """Given string, return no. of punctuation characters (int)"""
    return len(collect_punctuation(tokens))

def num_numeric(text):
    """Given string, return no. of digits (int)"""
    return len(collect_digits(text))

def num_uppercase_words(tokens):
    """Given list of words, return no. of uppercase words (int)"""
    return len(collect_uppercase_words(tokens))

def num_uppercase_chars(text):
    """Given string, return no. of uppercase characters (int)"""
    return len(collect_uppercase_chars(text))

def sum_word_len(tokens):
    """Given list of words, return sum of length of words (int)"""
    return sum([len(token) for token in tokens])

def avg_word_len(tokens):
    """Given list of words, return average word length (int)"""
    return sum_word_len(tokens) / num_words(tokens)


"""
preprocess(df) creates columns of preprocessed data in the DataFrame in-place.
"""
def preprocess(df):
  df['tokens'] = df['text'].apply(tokenize)
  df['num_words'] = df['tokens'].apply(num_words)
  df['num_chars'] = df['text'].apply(num_chars)
  df['avg_word_len'] = df['tokens'].apply(avg_word_len)
  df['num_stopwords'] = df['tokens'].apply(num_stopwords)
  df['num_special_chars'] = df['text'].apply(num_special_chars)
  df['num_numeric'] = df['text'].apply(num_numeric)
  df['num_uppercase_words'] = df['tokens'].apply(num_uppercase_words)
  df['num_uppercase_chars'] = df['text'].apply(num_uppercase_chars)
  df['length'] = df['text'].apply(len)
  df['text'].apply(lambda x: len([c for c in str(x) if c == '#']))
  df['count of capital letters'] = df['text'].apply(lambda x: len(re.findall(r'[A-Z]', x)))
  df['ratio of capital letters'] = df['length'] / df['count of capital letters']