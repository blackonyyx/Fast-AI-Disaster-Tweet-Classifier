# -*- coding: utf-8 -*-
"""tweetAnalysis.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GdpCCNSsZxpK9KT30pCuyF-f94VHafoh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import spacy
import re

#from google.colab import drive
#drive.mount('/content/gdrive')

#root_path = 'gdrive/My Drive/Stats study group/'
#Add wherever your file is..
tweet_df = pd.read_csv('gdrive/My Drive/Stats study group/real_fake.csv')

tweet_df.columns

tweet_df.head()

((tweet_df.isnull() | tweet_df.isna()).sum() * 100 / tweet_df.index.size).round(2)

#top 20 keywords
sns.barplot(y=tweet_df['keyword'].value_counts()[:20].index,x=tweet_df['keyword'].value_counts()[:20],
            orient='h')

#top few locations
sns.barplot(y=tweet_df['location'].value_counts()[:5].index,x=tweet_df['location'].value_counts()[:5],
            orient='h')

#add length of tweet into df
tweet_df['length'] = tweet_df['text'].apply(len)
#count of hashtags
tweet_df['hashtag_count'] = tweet_df['text'].apply(lambda x: len([c for c in str(x) if c == '#']))
#count of mentions
tweet_df['mention_count'] = tweet_df['text'].apply(lambda x: len([c for c in str(x) if c == '@']))
#count capital letters
tweet_df['count of capital letters'] = tweet_df['text'].apply(lambda x: len(re.findall(r'[A-Z]',x)))
#Ratio of capital letters
tweet_df['ratio of capital letters'] = tweet_df['length']/tweet_df['count of capital letters']
#missing location
tweet_df['location'].fillna(0, inplace=True)
#missing keyword
tweet_df['keyword'].fillna(0, inplace=True)

"""Most of the tweets have a keyword, while location is not tagged to 1/3 of them. Could check for 
Presence of # and @ vs true/False, length vs true false
"""

#extract #  @, external links
import re

def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)
    
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

#remove HTML and links and emoji
tweet_df['cleaned_text'] = tweet_df['text'].apply(lambda x: remove_emoji(remove_URL(remove_html(x))))

true_df = tweet_df[tweet_df.target.eq(1)]
false_df = tweet_df[tweet_df.target.eq(0)]

#wordcloud for fake and real tweets
from wordcloud import WordCloud, ImageColorGenerator

#fake
text = " ".join(str(each) for each in false_df['cleaned_text'])
wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="black").generate(text)
plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='Bilinear')
plt.axis("off")
plt.figure(1,figsize=(12, 12))
plt.show()

#true
text = " ".join(str(each) for each in true_df['cleaned_text'])
wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="black").generate(text)
plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='Bilinear')
plt.axis("off")
plt.figure(1,figsize=(12, 12))
plt.show()

#correlation graphs

import en_core_web_sm
nlp = en_core_web_sm.load()
import nltk

#stemming
tweet_df['tokenised_text'] = tweet_df['cleaned_text'].apply(lambda x: nlp.tokenizer(x))

#entity recognition
#idk what is happening here
tweet_df['entities'] = tweet_df['tokenised_text'].text

#vectorizing