# AutoML with PyCaret
=====================

## Overview
------------

This code uses the PyCaret library to create an automated machine learning (AutoML) pipeline. The pipeline includes data preprocessing, model selection, hyperparameter tuning, and model evaluation.

## Importing Libraries
----------------------

The code starts by importing the necessary libraries:
```python
import streamlit as st
import pandas as pd
import pycaret.classification as cls
import pycaret.regression as reg
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import os