# ==== Necessary Imports ====

import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from spellchecker import SpellChecker
import spacy
nlp = spacy.load("en_core_web_sm")
import nltk
nltk.download('stopwords')
stop = stopwords.words('english')
import string
import re


# ==== Remove ID Column ====

# df.drop('id', axis=1, inplace=True)

# ==== Preprocessing methods before SpellCorrection ====
# def tokenize(text):
#     """Given string, apply Spacy's nlp then return list of text"""
#     return [token.text for token in nlp(text)]

# def find_url(string):
#     text = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',string)
#     return "".join(text) # converting return value from list to string
# New Feature for extracting URLs

# def remove_URL(text):
#     url = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
#     return url.sub(r'',text)
# # Can remove from text after extracting them, especially if text is not going to be used in the model.
#
# def remove_html(text):
#     html=re.compile(r'<.*?>')
#     return html.sub(r'',text)
# # HTML Tags are most probably useless in our evaluation. Can just remove
#
# def num_uppercase_words(tokens):
#     return len(collect_uppercase_words(tokens))
# # New Feature for number of uppercase words

# def num_uppercase_chars(text):
#     return len(collect_uppercase_chars(text))
# # New Feature for number of uppercase letters

# def find_at(text):
#     line=re.findall(r'(?<=@)\w+',text)
#     return " ".join(line)
# # New Feature for number of @mentions
#
# def find_hash(text):
#     line=re.findall(r'(?<=#)\w+',text)
#     return " ".join(line)
# # New feature for hashtags

# def find_number(text):
#     line=re.findall(r'[0-9]+',text)
#     return " ".join(line)
# New Column with only numbers

# def remove_numbers(text):
#     text = ''.join([i for i in text if not i.isdigit()])
#     return text
# Removing Numbers

# def remove_punct(text):
#     table=str.maketrans('','',string.punctuation)
#     return text.translate(table)
# # Removing Punctuation
#
# def lower_text(text):
#     text = str(text).lower()
#     return text


# ==== Spell Correction and adding feature of No. of Misspelled Words ====

# spell = SpellChecker()
# def correct_spellings(text):
#     corrected_text = []
#     misspelled_words = spell.unknown(text.split())
#     for word in text.split():
#         if word in misspelled_words:
#             corrected_text.append(spell.correction(word))
#         else:
#             corrected_text.append(word)
#     return (" ".join(corrected_text), len(misspelled_words))


# # ==== Preprocessing methods after SpellCorrection ====
# def collect_stopwords(tokens):
#     return [token for token in tokens if token in stop]

# Need one more function for named entity recognition

# def preprocess_before(df):
# 	df['text']=df['text'].apply(lambda x : remove_html(x))
# 	df['tokens'] = df['spelled_text'].apply(tokenize)
# 	df['url']=df['text'].apply(lambda x:find_url(x))
# 	df['text']=df['text'].apply(lambda x : remove_URL(x))
# 	df['num_uppercase_chars'] = df['text'].apply(num_uppercase_chars)
# 	df['num_uppercase_words'] = df['tokens'].apply(num_uppercase_words)
# 	df['at_mention']=df['text'].apply(lambda x: find_at(x))
# 	df['hash']=df['text'].apply(lambda x: find_hash(x))
# 	df['number']=df['text'].apply(lambda x: find_number(x))
# 	df['text']=df['text'].apply(lambda x: remove_numbers(x))
# 	df['text']=df['text'].apply(lambda x : remove_punct(x))
#
#
# def preprocess_after(df):
# 	df['spelled_text']=df['text'].apply(lambda x : correct_spellings(x)[0])
# 	df['spelled_tokens']=df['tokens'].apply(lambda x : correct_spellings(x)[0])
# 	df['number_misspelled']=df['text'].apply(lambda x : correct_spellings(x)[1])
# 	df['stopwords']=df['spelled_tokens'].apply(lambda x : collect_stopwords(x))
# 	''' Function to extract all named entities from the spelled tokens'''
