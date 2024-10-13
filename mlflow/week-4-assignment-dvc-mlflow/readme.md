# Advertising Sales Classification Model

This project implements a Logistic Regression model to predict sales based on advertising data. The dataset includes information on TV, radio, and newspaper advertising expenditures, and the model aims to classify whether the sales are high or low based on a specified threshold.

## Table of Contents
- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Configuration](#configuration)
- [Model Training](#model-training)
- [Logging with MLflow](#logging-with-mlflow)
- [How to Run the Code](#how-to-run-the-code)

## Project Overview

The project uses the `pandas` library for data manipulation, `scikit-learn` for implementing the Logistic Regression model, and `mlflow` for tracking experiments and managing models. The primary goal is to predict whether sales will be above or below a certain threshold based on advertising spending.

## Requirements

Make sure to have the following libraries installed:

- pandas
- numpy
- scikit-learn
- mlflow
- PyYAML

You can install the required libraries using pip:

```bash
pip install pandas numpy scikit-learn mlflow pyyaml