# Lecture 1 homework

## Use case

**User:** A data scientist or machine learning engineer maintaining a production purchase propensity model.

**Problem:** Unexplained changes in prediction distributions may result from upstream data quality failures, distribution drift, or scoring pipeline issues. Finding relevant evidence currently requires manually examining metrics, logs, and documentation.

**Goal:** Build an AI investigator that eventually searches documents and invokes tools to diagnose incidents.

## Synthetic dataset

Code to generate the fake dataset can be found here: src/arxiv_curator/monitoring_data.py

## Model experiment observations

First model service experiment can be found here: lecture_1/model_experiments.ipynb

Main lmitations: The first model experiment only uses the context within the prompt and does not retrieve real Delta data, search documentation, or use any other tools to solve the problem.
