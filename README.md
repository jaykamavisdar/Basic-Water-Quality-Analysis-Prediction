# Basic-Water-Quality-Analysis-Prediction
## Overview

This project leverages data analytics and **machine learning** to analyze and predict the *potability* of water based on various chemical and physical features.

---

## Table of Contents

- [Features](#features)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Training & Evaluation](#model-training--evaluation)
- [Exporting Clean Data](#exporting-clean-data)

---

## Features

- Data cleaning and preprocessing
- Exploratory data analysis (EDA) with visualizations
- Feature correlation analysis
- Machine learning classification (Random Forest, Naive Bayes)
- Model evaluation (accuracy, confusion matrix, classification report)
- Comparison between algorithms
- Export of cleaned dataset for Power BI or other tools

---

## Dataset

- The dataset is loaded from your Google Drive:
/content/drive/MyDrive/data.csv

text
- Required features include: `PH`, `Turbidity`, `temperature`, `Potability`, etc.

---

## Requirements

- Python 3.13
- Google Colab (or compatible local environment)
- Packages:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- ydata-profiling

Install missing packages using:
!pip install numpy pandas matplotlib seaborn scikit-learn ydata-profiling --upgrade

text

---

## Project Structure

├── data.csv # Source data (on Google Drive)

├── [Notebook/script].ipynb # Both your analysis and code

├── cleaned_water_quality.csv # Cleaned output data

└── README.md # Project documentation

text

---

## Usage

### 1. Connect Google Drive and Load Data
from google.colab import drive

drive.mount('/content/drive')

import pandas as pd

input_data = pd.read_csv('/content/drive/MyDrive/data.csv')


### 2. Data Cleaning
cleaned_df = input_data.dropna()

cleaned_df.columns = [col.strip().replace(" ", "_") for col in cleaned_df.columns]


### 3. Exploratory Data Analysis

- Pie chart of potability classes
- Distribution/KDE plots for features
- Correlation heatmap

Example

import seaborn as sns

import matplotlib.pyplot as plt


cleaned_df.groupby('Potability').size().plot(kind='pie')

sns.distplot(a=cleaned_df['PH'], kde=False)

sns.kdeplot(data=cleaned_df['PH'], shade=True)


corr = cleaned_df.corr()

plt.subplots(figsize=(5,5))

sns.heatmap(corr, annot = True)

plt.show()


### 4. Data Profiling

from ydata_profiling import ProfileReport

profile = ProfileReport(cleaned_df, title="Report for our Project", explorative=True)

profile.to_notebook_iframe()


### 5. Model Training & Evaluation

#### Random Forest Classifier

#### Naive Bayes

#### Model Evaluation (Example for Confusion Matrix)


## Exporting Clean Data

cleaned_df.to_csv("cleaned_water_quality.csv", index=False)

from google.colab import files

files.download("cleaned_water_quality.csv")

---

<!--
TIP: Update paths, dataset features, and examples to match your actual use case as necessary.
-->
