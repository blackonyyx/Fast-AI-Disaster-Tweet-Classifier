# Fast-AI-Disaster-Tweet-Classifier
NLP Project as part of the FastAI study group conducted by NUS Statistics Society in Summer 2020

## Project Intro
Introductory NLP project looking into the classification of tweets made during a disaster. 

Find more info from here. https://www.kaggle.com/c/nlp-getting-started

### Contributing members
* New Jun Jie (Jet)
* Tom Joju
* Stephen
* Axel Lau

## Pre-processing

All 4 of us had prior experience with the more commonly used models in scikit-learn. So we started off from there, approaching this task with the assumption of using structured data. We did some research on what other Kagglers had done for preprocessing.

Similar projects dealing with the detection of fake tweets had datasets with more features than what we had. They usually had the profile details as well as no. of replies to the tweet. This made it easy for them to use RandomForests due to the presence of structured data.

We merely had the location of the tweet and the tweet itself. We thus settled on using Spacy and NLTK's tokenizer for its Named Entity Recognition (NER) capabilities. Apart from that our preprocessing steps included:

1. Reducing characters to lowercase
2. Removing @ and #
3. Removing external links to other websites
4. Removing unicode for emojis

Given the unstructured data of the tweets, many kaggle notebooks on the same competition used Deep Learning models such as BERT to great success. As such we were tempted to find out more.

## Evaluation Models

Given Jet's prior knowledge of developments in NLP, we initially researched on 3 different models, BERT, RoBERTa and GPT2. From here, we realised the presence of the Simple Transformers package which has greatly reduced the amount of code that we had to write. 

Upon further research into the 3 different models. We realised that each of them had their own advantages and shortcomings. One notable example was the OpenAI's GPT2 model. There were a few reasons why it did not seem to be effective at detecting machine generated tweets.

1. GPT2 is good at detecting mahine generated tweets made by GPT2, but its performance is lacklustre when using it against different language models. Tweets in the datset were not generated using GPT2 as GPT2 did not exist yet.
2. GPT2 is more often used in creating text content, which is why its creators were initialy reluctant towards releasing the full model due to the fear of it enabling malicious activity.

Why BERT and roBERTa were effective at detecting fake tweets

~~ Tom and Stephen fill up ~~


~~ Insert more comments about the final results ~~

## Further Extension

We could explore a hybrid model of using both unstructured and structured data to train the model.

## Thoughts and Reflections

Preprocessing social media comments requires good understanding of the platform in order to convey the context of comments accurately
Hard for NLP models to detect sarcasm. Even humans have problems picking it up on the internet.
Simple Transformers Package is really simple to use for NLP. Makes NLP simple to use and has relatively high accuracy even on the base model.

