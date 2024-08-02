# Coronavirus Tweet Sentiment Analysis

## Capstone Project

### Presented By
1. Nitheesh Kani K - Sethu Institute of Technology - CSD

---

## Table of Contents
- [Problem Statement](#problem-statement)
- [Proposed Solution](#proposed-solution)
- [System Development Approach](#system-development-approach)
- [Algorithm & Deployment](#algorithm--deployment)
- [Results](#results)
- [Conclusion](#conclusion)
- [Future Scope](#future-scope)
- [References](#references)

---

## Problem Statement
Build a classification model to predict the sentiment of COVID-19 tweets. The tweets have been pulled from Twitter and manually tagged. Names and usernames have been coded to avoid any privacy concerns.

## Proposed Solution
The proposed system aims to accurately classify the sentiment of COVID-19 tweets using natural language processing (NLP) and machine learning techniques.

### Components:
- **Data Collection**:
  - Use pre-collected and manually tagged tweets to ensure privacy and data integrity.
- **Data Preprocessing**:
  - Clean the tweets by removing irrelevant information, such as special characters and URLs.
  - Perform tokenization, stopword removal, and stemming/lemmatization.
- **Feature Engineering**:
  - Extract relevant features from the tweets, such as n-grams, TF-IDF values, and sentiment scores from lexicons.
- **Machine Learning Algorithm**:
  - Implement classification algorithms such as Logistic Regression, Naive Bayes, and Support Vector Machines (SVM) to predict tweet sentiments.
  - Use cross-validation to evaluate model performance.
- **Deployment**:
  - Develop a user-friendly interface for real-time sentiment analysis.
  - Deploy the solution on a scalable platform, ensuring efficient processing of new tweets.

## System Development Approach
### System Requirements
- Python programming language
- Libraries: pandas, numpy, scikit-learn, NLTK, and Flask (for deployment)

## Algorithm & Deployment
### Chosen Algorithm:
- Multiclass Classification using IBM Watson Studio AI.

### Data Input:
- Preprocessed tweet text, extracted features (such as n-grams and TF-IDF values), and sentiment scores from lexicons.

### Training Process:
- The model was trained using the IBM Watson Studio AI platform.
- Cross-validation and hyperparameter tuning were employed to optimize the model's performance.

### Deployment:
- The trained model was deployed on IBM Cloud, leveraging IBM Watson Studio AI's capabilities for efficient handling and processing of new tweets.
- The deployment ensures real-time sentiment analysis, allowing the model to classify tweets into sentiment categories (e.g., positive, negative) with high confidence.

## Results
The machine learning model successfully predicted the sentiment of COVID-19 tweets with high confidence.

- **Prediction Type**: Multiclass Classification
- **Prediction Percentage**: Based on 100 records, the sentiments are evenly split between negative (neg) and positive (pos) sentiments.

## Conclusion
- The classification model for predicting the sentiment of COVID-19 tweets has demonstrated excellent performance, with predictions made confidently and accurately.
- **Effectiveness**: The model accurately classifies tweets into positive and negative sentiments, providing valuable insights into public sentiment during the COVID-19 pandemic.
- **Challenges**: Handling ambiguous tweets and ensuring the model generalizes well to unseen data remain challenges.
- **Importance**: Understanding public sentiment can aid policymakers, health organizations, and the general public in making informed decisions and addressing concerns during the pandemic.
- The successful implementation and high performance of the sentiment analysis model underscore its potential utility in real-world applications for social media sentiment analysis.

## Future Scope
- Incorporate additional data sources to enhance sentiment prediction.
- Optimize the algorithm for better performance and real-time processing.
- Expand the system to analyze tweets in multiple languages and cover broader topics beyond COVID-19.

## References
- IBM Watson Studio Documentation
- IBM Cloud Documentation
- Natural Language Processing with Python by Steven Bird, Edward Loper, and Ewan Klein
- Scikit-Learn Documentation
- Text Classification Algorithms by Charu C. Aggarwal and ChengXiang Zhai
