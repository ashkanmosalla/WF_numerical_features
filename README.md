# Wildfire Prediction Using Deep Learning Models

![Project Cover](images/WF2.webp)


This project implements deep learning models to predict the probabilities of wildfires in Canada using historical wildfire data and influencing environmental factors. Models such as Deep Artificial Neural Networks (DNN), Recurrent Neural Networks (RNN), Elman RNN, and Encoder-Decoder Bi-directional RNN (BiRNN) have been optimized using Grey Wolf Optimizer (GWO) for hyperparameter tuning.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Running the Models](#running-the-models)
- [Results](#results)
- [Acknowledgments](#Acknowledgments)

## Introduction

This project is based on historical wildfire events and environmental factors. The main objective is to predict the probability of wildfires in unseen sub-areas using deep learning models. The models have been optimized using the Grey Wolf Optimizer (GWO), and feature selection has been performed using Boruta and CART algorithms.

## Project Structure

The repository is organized as follows:

├── data/ # Directory containing datasets │ ├── WF2_numeric.csv # Wildfire dataset (labeled data) │ ├── WF2_Export1.csv # Unlabeled data for sub-area 1 │ ├── WF2_Export2.csv # Unlabeled data for sub-area 2 ├── models/ # Saved models after training ├── outputs/ # Output files such as predictions and model performance ├── wildfire_prediction.py # Main Python script for running predictions ├── requirements.txt # Python dependencies ├── README.md # Project documentation

### Datasets

- **`WF2_numeric.csv`**: Contains historical wildfire data and influencing environmental factors.
- **`WF2_Export1.csv`**: Unseen data from sub-area 1 for wildfire prediction.
- **`WF2_Export2.csv`**: Unseen data from sub-area 2 for wildfire prediction.
## Setup

### Prerequisites

To run the project, you will need:

- Python 3.7+
- TensorFlow
- Pandas
- Scikit-learn

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/ashkanmosalla/WF_numerical_features.git
   cd WF_numerical_features

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

pip install -r requirements.txt

## Running the models
To train and test the models, execute the following command:

python wildfire_prediction.py

## Results

The predicted wildfire probabilities for sub-area 1 and sub-area 2 are saved as CSV files in the outputs/ folder. You can use these probabilities to assess the likelihood of wildfires in these areas.

In addition, the model evaluation metrics such as accuracy, precision, recall, and F1-score are also saved and displayed. ROC curves are generated to visualize model performance.

## Acknowledgements

### Key Additions:
- **Acknowledgment for Dr. Khabat Khosravi**: Thanking him for gathering data, insights, and his vision for the project.
- **Acknowledgment for Alireza Shahvaran**: Recognizing his contribution in making maps and providing insights.
