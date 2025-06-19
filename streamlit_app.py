import streamlit as st
import torch
import numpy as np
import pandas as pd
from model import TwinTowerModel
import warnings
warnings.filterwarnings("ignore")


@st.cache_data()
def load_model():
    model = torch.load('twin_tower_model.pth', map_location=torch.device('cpu'))
    model.eval()
    return model

@st.cache_resource()
def load_encoders():
    import pickle
    with open('le_ad.pkl', 'rb') as f:
        le_ad = pickle.load(f)
    with open('le_gender.pkl', 'rb') as f:
        le_gender = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return le_ad, le_gender, scaler

model = load_model()
le_ad, le_gender, scaler = load_encoders()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Your full ad list
unique_ads = le_ad.classes_

def recommend_topk_ads_for_input(user_feats, gender, model, top_k=5, candidate_k=100):
    model.eval()
    with torch.no_grad():
        # Normalize user_feats: must match training scaling
        numerical_user_feats_scaled = scaler.transform([user_feats])
        user_feats_scaled = np.append(numerical_user_feats_scaled.flatten(), gender_encoded).astype(float)

        user_feats_tensor = torch.tensor(user_feats_scaled, dtype=torch.float32).to(device)
        user_feats_tensor = user_feats_tensor.repeat(candidate_k, 1)

        candidate_ads = np.random.choice(unique_ads, size=candidate_k, replace=False)
        candidate_ad_ids = torch.tensor(le_ad.transform(candidate_ads), dtype=torch.long).to(device)

        scores = model(user_feats_tensor, candidate_ad_ids)
        top_indices = torch.topk(scores, top_k).indices.cpu().numpy()
        top_ads = [candidate_ads[i] for i in top_indices]
        top_scores = scores[top_indices].cpu().numpy()

        return list(zip(top_ads, top_scores))


st.title("Interactive Ad Ranking Demo")

age = st.slider("Age", 18, 70, 30)
gender = st.selectbox("Gender", ['Male', 'Female'])
daily_time_spent = st.slider("Daily Time Spent on Site (normalized)", 0.0, 1.0, 0.5)
area_income = st.slider("Area Income (normalized)", 0.0, 1.0, 0.5)
daily_internet_usage = st.slider("Daily Internet Usage (normalized)", 0.0, 1.0, 0.5)

# Encode gender
gender_encoded = le_gender.transform([gender])[0]

user_features = [daily_time_spent, age / 70, area_income, daily_internet_usage]

if st.button("Recommend Ads"):
    recommendations = recommend_topk_ads_for_input(user_features, gender_encoded, model)
    st.write("### Top Recommended Ads:")
    for i, (ad, score) in enumerate(recommendations, 1):
        st.write(f"{i}. {ad} — Score: {score:.4f}")
