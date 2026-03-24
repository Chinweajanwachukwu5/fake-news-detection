# Fake News Detection

A machine learning project that detects fake news articles using classical text classification techniques. Built in Python using Google Colab.

## Project Overview

This project implements a fake news detection pipeline trained on a Kaggle dataset containing 20,000 labelled news articles. The goal is to classify articles as either real or fake using traditional machine learning approaches.

## Methods Used

- Text Preprocessing — cleaning and preparing raw news text
- Feature Extraction — TF-IDF (Term Frequency-Inverse Document Frequency) to convert text into numerical features
- Models trained and compared:
  - Rule-based classifier
  - Multinomial Naive Bayes
  - Logistic Regression

## Results

| Model | Accuracy | F1-Score |
|---|---|---|
| Logistic Regression | 0.510 | 0.510 |
| Naive Bayes | 0.506 | 0.501 |

The ROC-AUC score of 0.507 indicates that fake and real news articles in this dataset share very similar language patterns, making classification challenging for frequency-based models.

## Key Finding

Traditional word frequency models struggle to distinguish fake from real news when both classes use similar vocabulary. This highlights
