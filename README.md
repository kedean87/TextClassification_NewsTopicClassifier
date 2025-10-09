# AG News Text Classification — TF-IDF + Logistic Regression

## Project Overview
This project demonstrates a **text classification pipeline** using the [AG News dataset](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset).  
The goal is to automatically classify news articles into **four categories**:
- World
- Sports
- Business
- Science/Technology

This project implements a **classic NLP workflow** using:
- **TF-IDF vectorization** for text feature extraction
- **Logistic Regression** for supervised classification

---

## Objectives
- Understand how to preprocess and vectorize text
- Train and evaluate a machine learning model on textual data
- Build a lightweight, interpretable baseline model before using deep learning

---

## Dataset
- **Dataset:** AG News Dataset  
- **Description:** Each record contains a short news title and description, labeled with one of four topics  
- **Labels:**
  - 0 → World  
  - 1 → Sports  
  - 2 → Business  
  - 3 → Science/Tech  

The dataset can be accessed automatically via the `datasets` library from Hugging Face.

---
