# 🧠 Project: Ad Ranking System

### 🎯 Goal  
Recommend the most relevant ads to users based on their personal and browsing data — inspired by Pinterest’s multi-stage ranking architecture.

---

## 🧩 Dataset

We used a synthetic advertising dataset with the following columns:

- `Daily Time Spent on Site`  
- `Age`  
- `Area Income`  
- `Daily Internet Usage`  
→ *Numerical user features*

- `Gender`  
→ *Categorical user feature*

- `Ad Topic Line`  
→ *Simulated ad identity*

- `Clicked on Ad`  
→ *Label (1 = user clicked, 0 = user ignored)*

**Source:** [swekerr/click-through-rate-prediction](https://www.kaggle.com/datasets/swekerr/click-through-rate-prediction)
---

## ⚙️ Core Idea: Twin-Tower Neural Network

We implemented a **two-tower model** similar to what Pinterest uses:

- **User tower**: Maps user features into a dense embedding  
- **Ad tower**: Embedding layer learns vector representations for each ad  
- **Scoring**: Compute dot product between user and ad embeddings  
- **Training**: Binary classification using click labels (1 = clicked, 0 = skipped)

---

## ✅ Step-by-Step Overview

### 1. Preprocessing

- Encoded `Gender` using `LabelEncoder`
- Normalized numerical features with `MinMaxScaler`
- Encoded `Ad Topic Line` into integer IDs (10,000+ unique ads)

---

### 2. Training Data Creation

For each user:

- Created a **positive sample** → (user, clicked ad, label = 1)  
- Generated **negative samples** by pairing the user with random, unclicked ads → (user, random ad, label = 0)  
- This creates a **contrastive setup** to help the model learn both relevance and irrelevance.

---

### 3. DataLoader & Dataset

- Wrapped the training samples into a PyTorch `Dataset`  
- Used `DataLoader` for efficient mini-batch training

---

### 4. Model Architecture

- **User tower**: 2-layer feedforward network  
- **Ad tower**: `nn.Embedding` layer for ads  
- **Output**: `sigmoid(dot(user_embedding, ad_embedding))` → click probability

---

### 5. Training

- Optimizer: `Adam`  
- Loss function: Binary Cross-Entropy (`BCELoss`)  
- Evaluation metric: AUC (Area Under Curve)

---

### 6. Two-Stage Ranking Pipeline

#### 🔹 Stage 1: Candidate Retrieval  
Randomly sample 100 ads from the full ad pool (simulates recall step used in real-world systems)

#### 🔹 Stage 2: Neural Ranking  
Use the trained twin-tower model to score and rank the 100 candidate ads for a given user.

---

### 7. Streamlit UI

We built an optional **interactive web app** using Streamlit:

- Users input their features (age, gender, browsing time, etc.)
- The app:
  - Encodes and scales the input
  - Retrieves 100 random ads
  - Ranks them using the trained model
  - Returns the top-N recommended ads and their scores

---

## 💡 Why This Architecture Is Realistic

- Pinterest, TikTok, Meta, and YouTube use **two-tower systems** for large-scale ad/content recommendation  
- Negative sampling is standard in training click models  
- Multi-stage ranking (recall → ranking) improves efficiency and performance

---

## 📁 Artifacts We Saved

- `twin_tower_model.pth` → Trained PyTorch model  
- `le_ad.pkl` → LabelEncoder for ad topic lines  
- `le_gender.pkl` → LabelEncoder for gender  
- `scaler.pkl` → MinMaxScaler for numeric features  

These are loaded into the Streamlit app for real-time inference.

---
## 🛠️ Step-by-Step Guide to Run the Project

---

### ✅ 1. Clone the Repository

```
git clone https://github.com/your-username/ad-ranking-system.git
cd ad-ranking-system
```

### ✅ 2. Download the Dataset
Download the dataset from Kaggle:
swekerr/click-through-rate-prediction

### ✅ 3. Run the Jupyter Notebook (Model Training)

You can use Jupyter or Google Colab.

Open the notebook:
notebooks/train_twin_tower_model.ipynb

Execute each cell step-by-step:

- `Preprocessing (label encoding, scaling)`
- `Generating positive & negative samples`
- `Model training`
- `AUC evaluation`
- `Save model and encoders`

Outputs:

- `twin_tower_model.pth` → Trained PyTorch model  
- `le_ad.pkl` → LabelEncoder for ad topic lines  
- `le_gender.pkl` → LabelEncoder for gender  
- `scaler.pkl` → MinMaxScaler for numeric features

### ✅ 4. Run the Streamlit App (Optional UI)
Use the trained model in an interactive web app.
```
streamlit run streamlit_app.py
```

### ✅ 5.  Re-Train or Extend
Replace random ad sampling with semantic retrieval (e.g., Faiss)
