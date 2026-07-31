---
title: Predictive Maintenance Prediction
emoji: 🔧
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
pinned: false
---

# Predictive Maintenance Prediction

Streamlit app that predicts whether an engine requires maintenance from live sensor readings, using an XGBoost model registered on the Hugging Face Hub (`noormd100/predictive-maintenance-model`).

Note: the live app is deployed on Streamlit Community Cloud (HF Spaces hosting hit an account license/limit). This README/Dockerfile are kept so the `hosting.py` push still produces a working HF Space if/when Spaces hosting is available.
