# London City Bike Demand Forecasting

Predicting bike hire and return demand in the London City Bike system using time series modeling with PyTorch LSTM.

---

## Project Overview

This project builds a deep learning model to forecast the number of hired and returned bikes in London's bike hire system. Using LSTM (Long Short-Term Memory) networks from PyTorch, the model aims to predict demand in 30-minute intervals across geographical clusters of bike stations.

---

## Data Source

The raw data is provided by [Transport for London (TfL) Cycling Data](https://cycling.data.tfl.gov.uk/). It contains timestamped records of bike hires and returns across London.
Weather data comes from [Open-meteo](https://open-meteo.com/).

---

## Key Features

- **Data Collection**:  
  - Web scraping with **Beautiful Soup** and **Selenium** to fetch TfL cycling data.

- **Preprocessing**:
  - Time-based aggregation to 30-minute intervals.
  - Station clustering using **KNN (scikit-learn)** to group stations into 30 geographic clusters.
  - Efficient data manipulation using **Polars** and **NumPy**.

- **Modeling**:
  - Sequence prediction using **PyTorch LSTM** for time series forecasting.
  - Separate predictions for bike hires and returns per cluster.

- **Experiment Tracking**:
  - Experiment management and comparison using **MLflow**.

---

## Example Use Case

Forecasting bike demand for each cluster can help optimize bike distribution logistics, reduce shortages or surpluses, and ultimately improve the efficiency of the system.

---

## Future Plans

- **Model Deployment**:
  - Serve the model with **FastAPI** and containerize it using **Docker**.

- **Visualization & Interface**:
  - Build a user-friendly dashboard using **Streamlit** and **Plotly**.

- **Automation & Pipelines**:
  - Implement end-to-end ML pipelines to:
    - Automatically fetch new data
    - Retrain the model
    - Generate fresh predictions
    - Store predictions in a **database**

---

## Tech Stack

- **Languages**: Python
- **ML/DL**: PyTorch, scikit-learn
- **Data Handling**: Polars, NumPy
- **Scraping**: Selenium, BeautifulSoup
- **Tracking**: MLflow
- **Deployment (Planned)**: FastAPI, Docker
- **Visualization (Planned)**: Streamlit, Plotly
- **Pipelines (Planned)**: Airflow
- **DB (Planned)**: PostgreSQL / SQLite

---
