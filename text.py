import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import os
from collections import defaultdict
import spacy
badwords = set(stopwords.words('english'))
PATH = os.path.join(os.path.abspath(''), 'data')
raw = pd.read_csv(PATH+'/train.csv')
valid= pd.read_csv(PATH+'/test.csv')
