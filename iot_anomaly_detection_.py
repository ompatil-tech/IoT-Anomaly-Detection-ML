import os

# --- CRITICAL FIX FOR SEGMENTATION FAULTS ---
# Disable GPU/Metal to prevent Mac/M1 crash
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# --------------------------------------------

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

# ==========================================
# CONFIGURATION
# ==========================================
SEQUENCE_LENGTH = 10  
PCA_COMPONENTS = 0.95 
EPOCHS = 15
BATCH_SIZE = 32
RANDOM_SEED = 42

# Set seeds
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

def preprocess_data(df):
    print("[INFO] Preprocessing data...")
    df = df.ffill() # Handle missing values
    
    if 'label' in df.columns:
        y = df['label'].values
        X = df.drop(columns=['label'])
    else:
        y = None
        X = df
        
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    print(f"[INFO] Applying PCA (Variance: {PCA_COMPONENTS})...")
    pca = PCA(n_components=PCA_COMPONENTS)
    X_pca = pca.fit_transform(X_scaled)
    print(f"[INFO] Features reduced from {X.shape[1]} to {X_pca.shape[1]}")
    
    return X_pca, y

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(64, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_isolation_forest(X_train):
    print("[INFO] Training Isolation Forest...")
    iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=RANDOM_SEED)
    iso_forest.fit(X_train)
    return iso_forest

def create_sequences(X, y, time_steps=SEQUENCE_LENGTH):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def main():
    # --- 1. Data Loading (Sequential Synthetic Data) ---
    print("[WARNING] Generating synthetic IoT data (Demo Mode)...")
    
    # Generate data in BLOCKS to preserve Time-Series structure
    # Block 1: Normal Traffic
    X_norm_1 = np.random.normal(loc=0, scale=1, size=(3000, 20))
    y_norm_1 = np.zeros(3000)

    # Block 2: Cyberattack (More realistic noise)
    # Reduced 'loc' difference and increased 'scale' to make overlap harder
    X_anom = np.random.normal(loc=2.5, scale=2.0, size=(500, 20))
    y_anom = np.ones(500)

    # Block 3: Normal Traffic
    X_norm_2 = np.random.normal(loc=0, scale=1, size=(2000, 20))
    y_norm_2 = np.zeros(2000)
    
    X_combined = np.vstack([X_norm_1, X_anom, X_norm_2])
    y_combined = np.hstack([y_norm_1, y_anom, y_norm_2])
    
    df = pd.DataFrame(X_combined)
    df['label'] = y_combined

    # --- 2. Preprocessing ---
    X_pca, y = preprocess_data(df)
    
    # --- 3. Model 1: Isolation Forest ---
    X_train_iso, X_test_iso, y_train_iso, y_test_iso = train_test_split(X_pca, y, test_size=0.2, random_state=RANDOM_SEED)
    
    iso_model = train_isolation_forest(X_train_iso)
    y_pred_iso = iso_model.predict(X_test_iso)
    y_pred_iso = [1 if x == -1 else 0 for x in y_pred_iso]
    
    print("\n--- Isolation Forest Results ---")
    print(classification_report(y_test_iso, y_pred_iso, target_names=['Normal', 'Anomaly']))

    # --- 4. Model 2: LSTM ---
    print("\n[INFO] Preparing sequences for LSTM...")
    X_lstm, y_lstm = create_sequences(X_pca, y)
    
    X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=RANDOM_SEED)
    
    # Compute class weights
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_seq),
        y=y_train_seq
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    lstm_model = build_lstm_model(input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]))
    
    print("[INFO] Training LSTM model...")
    lstm_model.fit(
        X_train_seq, y_train_seq, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        validation_split=0.1, 
        verbose=1,
        class_weight=class_weight_dict
    )
    
    y_pred_prob = lstm_model.predict(X_test_seq)
    y_pred_lstm = (y_pred_prob > 0.5).astype(int)
    
    print("\n--- LSTM Results ---")
    print(classification_report(y_test_seq, y_pred_lstm, target_names=['Normal', 'Anomaly']))
    
    f1 = f1_score(y_test_seq, y_pred_lstm)
    print(f"LSTM Anomaly F1 Score: {f1:.4f}")
    print("[NOTE] This run used synthetic data. Real paper results (F1=0.92) were obtained on UNSW-NB15.")

if __name__ == "__main__":
    main()