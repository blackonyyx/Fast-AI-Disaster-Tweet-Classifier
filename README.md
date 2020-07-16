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

Given the unstructured data of the tweets, many kaggle notebooks on the same competition used Deep Learning models such as BERT to great success. As such we were tempted to find out more.

## Evaluation Models

Given Jet's prior knowledge of developments in NLP, we initially researched on 3 different models, BERT, RoBERTa and GPT2. From here, we realised the presence of the Simple Transformers package which has greatly reduced the amount of code that we had to write.

Upon further research into the 3 different models. We realised that each of them had their own advantages and shortcomings. One notable example was the OpenAI's GPT2 model. There were a few reasons why it did not seem to be effective at detecting machine generated tweets.

1. GPT2 is good at detecting mahine generated tweets made by GPT2, but its performance is lacklustre when using it against different language models. Tweets in the datset were not generated using GPT2 as GPT2 did not exist yet.
2. GPT2 is more often used in creating text content, which is why its creators were initialy reluctant towards releasing the full model due to the fear of it enabling malicious activity.

BERT is a model that worked extremely well for us. It stands for Bidirectional Encoder Representations from Transformers. It is a single model that is trained on a large unlabelled dataset, able to perform many individual NLP tasks with state-of-the-art results. To understand why BERT produced the results it did and to assess its suitability as our model of choice, we went on to do a little bit of research. Here are our main findings.

1. Why is Bert needed?

All NLP tasks face a major obstacle that is a lack of training data. Although there is an enormous amount of text data available, we end up with too few training examples when this data is split into the different diverse fields to create task-specific datasets. These datasets then become too small to perform well - NLP models perform much better when trained on millions or billions of labelled data. BERT is an example of a general purpose model that is made to overcome this issue. It is pretrained on a large corpus of unlabelled text (including the entire Wikipedia and Book Corpus) and it can then be further fine-tuned to perform tasks such as natural language inference, sentiment analysis, question answering, paraphrase detection and linguistic acceptability.

2. BERT is also bidirectionally trained which means that it learns information from both the left and right side of a token at the same time, which gives us a deeper sense of the context and flow as compared to single direction models. Traditionally, we had language models either trained to predict the next word in a sentence (right-to-left context used in GPT) or language models that were trained on a left-to-right context. These models were susceptible to errors due to loss in information. The bi-directional aspect of BERT greatly reduces this error.

3. BERT can be easily fine-tuned by adding just a couple of output layers to create the model for our specific task. The simple transformers library allowed us to leverage this to create a BERT model specific for a text classification task. Since our project required us to essentially do a binary text classification, this model fit into our project well.

As mentioned, the Simple Transformers package allowed us to run the BERT model with very few lines of code. The base model before fine-tuning produced an impressive accuracy score a 0.845.

~~ Insert more comments about the final results ~~

## Further Extension

We could explore a hybrid model of using both unstructured and structured data to train the model.

## Thoughts and Reflections

Preprocessing social media comments requires good understanding of the platform in order to convey the context of comments accurately

Hard for NLP models to detect sarcasm. Even humans have problems picking it up on the internet.

Simple Transformers Package is really simple to use for NLP. Makes NLP simple to use and has relatively high accuracy even on the base model.
