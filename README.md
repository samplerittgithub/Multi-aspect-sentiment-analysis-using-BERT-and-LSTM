Multi-Aspect Sentiment Analysis using BERT & LSTM
Project Overview

This project develops a multi-aspect sentiment analysis framework capable of identifying and classifying sentiment across multiple dimensions within customer reviews. Unlike traditional single-label sentiment classification, this approach captures granular sentiment signals related to specific aspects such as service, food quality, ambience, and pricing.

The system integrates contextual BERT embeddings with LSTM-based sequential modelling to enhance representation of semantic context and long-range dependencies in text data.

Problem Statement

Traditional sentiment analysis models assign a single polarity label (positive, negative, neutral) to an entire review. However, real-world customer feedback often contains mixed sentiments across multiple aspects within the same text.

Example:

“The food was excellent, but the service was slow.”

A single-label classifier fails to distinguish aspect-specific sentiment.

This project addresses this limitation by building a multi-aspect classification system that:

Extracts aspect-level sentiment signals

Handles contextual dependencies

Improves precision-recall balance in imbalanced sentiment distributions

Objectives

Identify limitations of conventional single-aspect sentiment models.

Explore advanced NLP architectures (BERT + LSTM) for contextual modelling.

Collect and preprocess real-world review datasets.

Develop a hybrid deep learning architecture combining contextual embeddings and sequential modelling.

Evaluate model performance using classification metrics and compare against traditional ML baselines.

Analyse performance trade-offs and interpret results for business applicability.

Dataset

The dataset consists of customer reviews collected from hospitality-focused platforms (e.g., OpenTable, Yelp).

Preprocessing steps included:

Text normalization

Tokenization

Stop-word removal

Lemmatization

Handling class imbalance

Aspect labelling

The final dataset was structured to support multi-label classification.

Technical Architecture
Model Design

The architecture combines:

Pre-trained BERT embeddings for contextual semantic representation

LSTM layers to model sequential dependencies

Dense classification layers for multi-aspect prediction

Pipeline:

Input text tokenization

BERT embedding extraction

LSTM sequence modelling

Fully connected classification layer

Multi-label output

This hybrid structure enables:

Better contextual understanding

Improved handling of long-range dependencies

Aspect-specific sentiment classification

Model Training

Training strategy included:

Train-validation split

Cross-validation for robustness

Learning rate scheduling

Threshold optimisation for multi-label output

Early stopping to prevent overfitting

Loss function was adapted to handle multi-label classification.

Evaluation Metrics

Model performance was evaluated using:

Accuracy

Precision

Recall

F1-Score

Confusion matrix analysis

Precision-Recall trade-off curves

Special emphasis was placed on recall optimisation to reduce false negatives in negative sentiment detection.

Benchmark comparison was conducted against:

Logistic Regression

Random Forest

Traditional single-label classifiers

The BERT + LSTM model demonstrated improved F1-score and better generalisation across aspects.

Results & Insights

Key outcomes:

Improved aspect-level classification consistency

Better handling of mixed sentiment reviews

Stronger contextual understanding compared to baseline models

Balanced precision-recall trade-offs after threshold tuning

The model demonstrated meaningful improvements in multi-aspect sentiment granularity over traditional approaches.

Business Relevance

This framework enables:

Granular customer experience analysis

Targeted service improvement strategies

Automated review monitoring at scale

Data-driven operational decision-making

Applications include:

Hospitality analytics

Brand reputation monitoring

Customer experience optimisation

Product/service feedback intelligence

Repository Structure
├── data/
├── notebooks/
├── models/
├── src/
├── evaluation/
├── README.md

Technologies Used

Python

PyTorch / TensorFlow

Hugging Face Transformers

Scikit-learn

Pandas

NumPy

Matplotlib / Seaborn

Future Improvements

Attention-based aspect extraction

Explainability using SHAP / LIME

Deployment via REST API

Model compression for production environments

Conclusion

This project demonstrates the practical implementation of advanced NLP architectures for aspect-based sentiment classification. By integrating contextual embeddings with sequential modelling, the framework enhances interpretability and classification precision in real-world review analytics.
