import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import time  # for progress animations

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


# -----------------------
# Helper functions
# -----------------------
@st.cache_data
def load_data():
    ds = sklearn.datasets.load_breast_cancer()
    X = pd.DataFrame(ds.data, columns=ds.feature_names)
    y = pd.Series(ds.target, name='label')
    df = pd.concat([X, y], axis=1)
    return df, ds

@st.cache_resource
def build_model(input_dim, hidden_size=16, lr=0.001):
    model = Sequential([
        Dense(hidden_size, activation='relu', input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(hidden_size, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Breast Cancer Classifier", layout="wide")

# Dark theme CSS
st.markdown("""
    <style>
    body { 
        background-color: #000000; 
        color: #FFFFFF; 
    }
    .stApp {
        background-color: #000000;
        color: #FFFFFF;
    }
    .stTabs [role="tab"] {
        background: #111111;
        color: #FFFFFF;
        padding: 10px;
        border-radius: 5px;
    }
    .stTabs [aria-selected="true"] {
        background: #FF4B4B;
        color: white;
    }
    .stMetric {
        background: #111111;
        padding: 15px;
        border-radius: 10px;
        color: #FFFFFF;
    }
    .stMetric .stMetricValue {
        color: #FFFFFF !important;
    }
    </style>
""", unsafe_allow_html=True)


# Banner (smaller size)
st.image("banner.png", width=500, use_container_width=False)

st.title("🎗️ Breast Cancer Classification — Interactive App")

# Load data
df, ds = load_data()
FEATURES = list(ds.feature_names)

# Tabs
tabs = st.tabs(["📊 Data Overview", "🧠 Train Model", "🔮 Predict", "📥 Export"])

# -----------------------
# 1) Data Overview
# -----------------------
with tabs[0]:
    st.header("📊 Dataset Overview")
    st.metric("Samples", df.shape[0])
    st.metric("Features", df.shape[1]-1)
    st.dataframe(df.head(20))

    st.subheader("Class Distribution")
    fig = px.histogram(df, x='label', color='label', title='Benign vs Malignant',
                       labels={'label': '0=Malignant, 1=Benign'})
    st.plotly_chart(fig, use_container_width=True)

# -----------------------
# 2) Train Model
# -----------------------
with tabs[1]:
    st.header("🧠 Train Your Model")
    if not TF_AVAILABLE:
        st.error("TensorFlow/Keras not available in this environment.")
    else:
        epochs = st.slider('Epochs', 5, 50, 10)
        batch_size = st.slider('Batch Size', 8, 128, 16)
        hidden = st.slider('Hidden Layer Size', 8, 128, 32)
        lr = st.slider('Learning Rate', 0.0001, 0.01, 0.001, format="%f")

        if st.button("🚀 Train Model"):
            with st.spinner("Training model... Please wait ⏳"):
                # Fake progress bar
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.02)
                    progress.progress(i + 1)

                # Prepare data
                X = df[FEATURES]
                y = df['label']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)

                # Train model
                model = build_model(X_train.shape[1], hidden, lr)
                history = model.fit(X_train_s, y_train, validation_data=(X_test_s, y_test),
                                    epochs=epochs, batch_size=batch_size, verbose=0)

                preds = (model.predict(X_test_s) >= 0.5).astype(int)
                acc = accuracy_score(y_test, preds)

            st.success(f"✅ Accuracy: {acc*100:.2f}%")
            st.balloons()  # 🎈 Celebration animation

            # Loss curve
            fig = px.line(y=[history.history['loss'], history.history['val_loss']],
                          labels={'index': 'Epoch', 'value': 'Loss'},
                          title='Training vs Validation Loss')
            st.plotly_chart(fig, use_container_width=True)

            # Confusion matrix
            cm = confusion_matrix(y_test, preds)
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                               labels=dict(x="Predicted", y="Actual"))
            st.plotly_chart(fig_cm, use_container_width=True)

            st.session_state['trained_model'] = model
            st.session_state['scaler'] = scaler
            st.session_state['features'] = FEATURES

# -----------------------
# 3) Predict
# -----------------------
with tabs[2]:
    st.header("🔮 Make Predictions")
    if 'trained_model' not in st.session_state:
        st.warning("Please train a model first.")
    else:
        model = st.session_state['trained_model']
        scaler = st.session_state['scaler']
        cols = st.session_state['features']

        inputs = {}
        # Only top 5 features editable
        for f in cols[:5]:
            inputs[f] = st.number_input(f, float(df[f].min()), float(df[f].max()), float(df[f].mean()))

        # Fill remaining with mean values
        for f in cols[5:]:
            inputs[f] = float(df[f].mean())

        with st.spinner("Making prediction 🔮..."):
            Xraw = pd.DataFrame([inputs])[cols]
            pred = (model.predict(scaler.transform(Xraw)) >= 0.5).astype(int)[0][0]
            label = '✅ Benign' if pred == 1 else '⚠️ Malignant'

        st.subheader("Prediction Result")
        st.metric("Outcome", label)

        # Save predictions (append mode)
        result_df = Xraw.copy()
        result_df["Prediction"] = label
        if "last_predictions" in st.session_state:
            st.session_state["last_predictions"] = pd.concat(
                [st.session_state["last_predictions"], result_df],
                ignore_index=True
            )
        else:
            st.session_state["last_predictions"] = result_df

# -----------------------
# 4) Export
# -----------------------
with tabs[3]:
    st.header("📥 Export Results")

    if 'last_predictions' in st.session_state:
        st.dataframe(st.session_state['last_predictions'])
        csv = st.session_state['last_predictions'].to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Predictions", csv, "predictions.csv", "text/csv")
        st.success("✅ Predictions saved and ready to export.")
    else:
        st.info("No predictions available yet.")
