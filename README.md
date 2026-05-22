# 🖥️ Server Failure Prediction System

## Project Overview
A machine learning system that predicts server failures 
in real-time using 11 server health metrics.

## 🔗 Live Application
👉 [Click here to open the app](YOUR_STREAMLIT_URL_HERE)

## Dataset
- 100,000 server monitoring records
- 5,000 failure records (5%)
- 11 health features

## Models Trained
| Model | Type | F1-Score | Recall |
|---|---|---|---|
| Logistic Regression | ML | 0.8398 | 0.9990 |
| Decision Tree | ML | 0.9838 | 0.9990 |
| **Random Forest** | **ML** | **0.9847** | **1.0000** |
| LSTM | DL | — | — |
| GRU | DL | — | — |
| Autoencoder | DL | — | — |

## Best Model — Random Forest (100K)
| Metric | Score |
|---|---|
| Accuracy | 99.84% |
| F1-Score | 98.47% |
| Recall | 100.00% |
| Precision | 96.99% |
| FN (Missed Failures) | **0** |

## Risk Levels
- 🟢 LOW    : Probability < 30%
- 🟡 MEDIUM : Probability 30–60%  
- 🔴 HIGH   : Probability > 60%

## Project — BITS ZC229T Design Project
