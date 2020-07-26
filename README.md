# Fast-AI-Disaster-Tweet-Classifier

NLP Project as part of the FastAI study group conducted by NUS Statistics Society in Summer 2020

## Project Intro

Introductory NLP project looking into the classification of tweets made during a disaster.

Find more info from here. https://www.kaggle.com/c/nlp-getting-started

### Contributing members

- New Jun Jie (Jet)
- Tom Joju
- Stephen
- Axel Lau

## Pre-processing

All 4 of us had prior experience with the more commonly used models in scikit-learn. So we started off from there, approaching this task with the assumption of using structured data. We did some research on what other Kagglers had done for preprocessing.

Similar projects dealing with the detection of fake tweets had datasets with more features than what we had. They usually had the profile details as well as no. of replies to the tweet. This made it easy for them to use RandomForests due to the presence of structured data.

We merely had the location of the tweet and the tweet itself. We thus settled on using Spacy and NLTK's tokenizer for its Named Entity Recognition (NER) capabilities. Apart from that our preprocessing steps included:

1. Reducing characters to lowercase
2. Removing mentions (@) and hashtags (#)
3. Removing external links to other websites
4. Removing unicode for emojis

Using these tabular data and the tweet text as a bag of words, we were going to try out Machine Learning Models such as Naive Bayes and Random Forest Classifiers on the dataset. However, given the unstructured data of the tweets and the lackluster performance of non-deep learning models on the task we looked at kaggle notebooks and found that many kaggle notebooks on the same competition used Deep Learning models such as BERT to great success.

As such we were tempted to find out more about such models and why they were so successful on the task.

## Evaluation Models

Given Jet's prior knowledge of developments in NLP, we initially researched on 3 different models, BERT, RoBERTa and GPT2. From here, we realised the presence of the Simple Transformers package which has greatly reduced the amount of code that we had to write.

### GPT2

Upon further research into the 3 different models. We realised that each of them had their own advantages and shortcomings. One notable example was the OpenAI's [GPT2](https://openai.com/blog/better-language-models/) model. There were a few reasons why it did not seem to be effective at detecting machine generated tweets.

1. GPT2 is good at detecting mahine generated tweets made by GPT2, but its performance is lacklustre when using it against different language models. Tweets in the datset were not generated using GPT2 as GPT2 did not exist yet.
2. GPT2 is more often used in creating text content, which is why its creators were initialy reluctant towards releasing the full model due to the fear of it enabling malicious activity.

### BERT

BERT is a model that worked extremely well for us. It stands for Bidirectional Encoder Representations from Transformers. It is a single model that is trained on a large unlabelled dataset, able to perform many individual NLP tasks with state-of-the-art results. To understand why BERT produced the results it did and to assess its suitability as our model of choice, we went on to do a little bit of research. Here are our main findings.

1. All NLP tasks face a major obstacle that is a lack of training data. Although there is an enormous amount of text data available, we end up with too few training examples when this data is split into the different diverse fields to create task-specific datasets. These datasets then become too small to perform well - NLP models perform much better when trained on millions or billions of labelled data. BERT is an example of a general purpose model that is made to overcome this issue. It is pretrained on a large corpus of unlabelled text (including the entire Wikipedia and Book Corpus) and it can then be further fine-tuned to perform tasks such as natural language inference, sentiment analysis, question answering, paraphrase detection and linguistic acceptability.

2. BERT is also bidirectionally trained which means that it learns information from both the left and right side of a token at the same time, which gives us a deeper sense of the context and flow as compared to single direction models. Traditionally, we had language models either trained to predict the next word in a sentence (right-to-left context used in GPT) or language models that were trained on a left-to-right context. These models were susceptible to errors due to loss in information. The bi-directional aspect of BERT greatly reduces this error.

3. BERT can be easily fine-tuned by adding just a couple of output layers to create the model for our specific task. The simple transformers library allowed us to leverage this to create a BERT model specific for a text classification task. Since our project required us to essentially do a binary text classification, this model fit into our project well.

As mentioned, the Simple Transformers package allowed us to run the BERT model with very few lines of code. The base model before fine-tuning produced an impressive accuracy score a 0.835 and an f1-score of 0.797.

### Towards [RoBERTa](https://arxiv.org/abs/1907.11692):

RoBERTa is a improvement on the baseline BERT model discovered in the paper [Attention is All You Need](https://arxiv.org/abs/1706.03762) that makes several key improvements to pre-training and training strategies that lead to better downstream task performances and lead better into transfer learning than the original BERT model.

Note: RoBERTa uses THE SAME archietecture as BERT, however instead goes into certain design choices and pretraining steps and refines them to produce better performance on most downstream tasks.

_Why does the model perform so well?_

The pretraining machine learning task that baseline BERT is trained to perform is to perform Masked Language Models (Masking a proportion of words in text and using the other words to predict the mask) and Next Sentence Prediction(NSP) to predict whether 2 segments follow each other or are from different documents.

1. Dynamic Masking Patterns vs Static Masking Patterns in pre-training

- RoBERTa uses dynamic masking for the MLM task where masking for each sentence is dynamically generated each time rather than with a static set of patterns, resulting in better gains in metrics of NLP learning like SQuAD 2.0, MNLI-m and SST-2.

2. Different training objectives

- BERT training objective and thus how it's training loss is calculated and optimised, uses NSP while RoBERTa uses instead FULL-SENTENCES without NSP where inputs are packed with sentences sampled from one or more documents until input of the model is reached. Hence removing the NSP prediction task improves the performance of the model on downstream tasks.

3. More data and larger batch sizes can improve the performance of BERT models on downstream tasks

- More data was used and larger batch sizes of around 2K improved performance on their tasks.

Application for RoBERTa:

As BERT/RoBERTa Models require only text, and such text in lowercase format. We clean the text to get rid of all punctuations, weird symbols, htmls and other such things before running it through simpletransformers.

Then, we apply the RoBERTa tokenizer and use the simpletransformers library to perform the task easily with minimal boilerplate code.

Surprisingly BERT (0.797) seemed to perform marginally better on the Data Set as compared to RoBERTa (0.784), however this could possible be attributed to not enough training, or random error.

## Further Extension

We could explore a hybrid model of using both unstructured and structured data to train the model. [TABERTA](https://ai.facebook.com/research/publications/tabert-pretraining-for-joint-understanding-of-textual-and-tabular-data/)

## Thoughts and Reflections

Preprocessing social media comments requires good understanding of the platform in order to convey the context of comments accurately

Hard for NLP models to detect sarcasm. Even humans have problems picking it up on the internet.

Simple Transformers Package is really simple to use for NLP. Makes NLP simple to use and has relatively high accuracy even on a base model.
