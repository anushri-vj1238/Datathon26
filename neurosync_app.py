import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

st.title("🧠 NeuroSync: Adaptive Music Optimization")

# ----------------------------
# 1. SIMULATE DATASET
# ----------------------------

np.random.seed(42)
n = 1000

tasks = np.random.choice(['math','writing','reading','coding'], n)
emotions = np.random.choice(['calm','neutral','stressed','excited'], n)
bpm = np.random.randint(60, 161, n)

# Simulated EEG signals
eeg_alpha = np.random.normal(10, 2, n)  # relaxed focus
eeg_beta = np.random.normal(20, 5, n)   # active thinking
heart_rate = np.random.normal(75, 10, n)

# Productivity formula (structured pattern)
productivity = (
    0.03 * bpm +
    0.5 * eeg_alpha -
    0.02 * eeg_beta +
    np.where(tasks == 'math', 2, 0) +
    np.where(emotions == 'stressed', -2, 0) +
    np.random.normal(0, 1, n)
)

df = pd.DataFrame({
    "task": tasks,
    "emotion": emotions,
    "bpm": bpm,
    "eeg_alpha": eeg_alpha,
    "eeg_beta": eeg_beta,
    "heart_rate": heart_rate,
    "productivity": productivity
})

# ----------------------------
# 2. EDA
# ----------------------------

st.header("📊 Exploratory Data Analysis")

if st.checkbox("Show Correlation Heatmap"):
    plt.figure(figsize=(8,6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    st.pyplot(plt)

if st.checkbox("Show BPM vs Productivity Scatter"):
    plt.figure()
    sns.scatterplot(x=df["bpm"], y=df["productivity"])
    st.pyplot(plt)

if st.checkbox("Show Productivity by Task (Boxplot)"):
    plt.figure()
    sns.boxplot(x="task", y="productivity", data=df)
    st.pyplot(plt)

if st.checkbox("Show Time-Series Trend"):
    df["time"] = np.arange(len(df))
    plt.figure()
    sns.lineplot(x="time", y="productivity", data=df)
    st.pyplot(plt)

# ----------------------------
# 3. MODEL TRAINING
# ----------------------------

st.header("🤖 Machine Learning Model")

# Encode categorical
le_task = LabelEncoder()
le_emotion = LabelEncoder()

df["task_enc"] = le_task.fit_transform(df["task"])
df["emotion_enc"] = le_emotion.fit_transform(df["emotion"])

X = df[["task_enc","emotion_enc","bpm","eeg_alpha","eeg_beta","heart_rate"]]
y = df["productivity"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_score = r2_score(y_test, lr_pred)

# Random Forest
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_score = r2_score(y_test, rf_pred)

st.write("Linear Regression R²:", round(lr_score,3))
st.write("Random Forest R²:", round(rf_score,3))

# ----------------------------
# 4. LIVE DEMO (PERSONALIZATION ENGINE)
# ----------------------------

st.header("🎧 Live Music Recommendation Engine")

task_input = st.selectbox("Select Task", ['math','writing','reading','coding'])
emotion_input = st.selectbox("Select Emotional State", ['calm','neutral','stressed','excited'])
bpm_input = st.slider("Select BPM", 60,160,100)

alpha_input = st.slider("EEG Alpha Level", 5,15,10)
beta_input = st.slider("EEG Beta Level", 10,30,20)
hr_input = st.slider("Heart Rate", 60,120,75)

if st.button("Predict Productivity"):
    input_df = pd.DataFrame({
        "task_enc":[le_task.transform([task_input])[0]],
        "emotion_enc":[le_emotion.transform([emotion_input])[0]],
        "bpm":[bpm_input],
        "eeg_alpha":[alpha_input],
        "eeg_beta":[beta_input],
        "heart_rate":[hr_input]
    })

    prediction = rf.predict(input_df)[0]

    st.success(f"Predicted Productivity Score: {round(prediction,2)}")

    # Reinforcement Logic
    if prediction < 5:
        st.warning("⚡ Try increasing BPM for higher stimulation.")
    elif prediction > 8:
        st.success("🔥 Optimal BPM range detected!")
    else:
        st.info("🔄 Slight adjustments may improve focus.")
        