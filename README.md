# Diabetes Prediction Model

## Overview

This project is a machine learning model designed to predict whether a person is diabetic based on certain health metrics. The model uses a [Support Vector Machine (SVM)](https://en.wikipedia.org/wiki/Support-vector_machine) with a linear kernel to classify individuals as diabetic or non-diabetic. The dataset used for training and testing the model is the Pima Indians Diabetes Database, which is a well-known dataset in the field of [machine learning](https://en.wikipedia.org/wiki/Machine_learning) and healthcare analytics.

## Dataset

The dataset used in this project is the `diabetes.csv` file, which contains several health-related attributes for each individual, including:

- Number of pregnancies
- Glucose level
- Blood pressure
- Skin thickness
- Insulin level
- Body mass index (BMI)
- Diabetes pedigree function
- Age
- Outcome (1 for diabetic, 0 for non-diabetic)

## Project Structure

- **Data Preprocessing**: The data is first loaded and checked for any missing values. The features are then standardized using `StandardScaler` to ensure that each feature contributes equally to the distance calculations in the SVM algorithm.

- **Model Training**: The dataset is split into training and testing sets. An SVM with a linear kernel is trained on the training data.

- **Model Evaluation**: The model's performance is evaluated on both the training and testing datasets, with accuracy scores calculated to assess how well the model generalizes to unseen data.

- **Prediction**: The model can predict the diabetic status of a new individual based on their health metrics. The input data is standardized using the same scaler fitted on the training data before making predictions.

## Requirements

To run this project, you will need the following Python libraries:

- [`numpy`](https://numpy.org/doc/stable/reference/generated/numpy.array.html)
- [`pandas`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)
- [`scikit-learn`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

## Usage

1. **Clone the repository**: Clone this repository to your local machine using `git clone`.

2. **Prepare the dataset**: Ensure that the `diabetes.csv` file is in the same directory as the script (Dataset downloaded from Kaggle.)

3. **Run the script**: Execute the script to train the model and make predictions. You can modify the `input_data` variable to test the model with different inputs.

4. **Interpret the results**: The script will output whether the person is diabetic or not based on the input data.

## Conclusion

This project demonstrates the use of a Support Vector Machine for binary classification in a healthcare context. The model provides a simple yet effective way to predict diabetes, which can be a valuable tool for early diagnosis and management of the condition.
  
