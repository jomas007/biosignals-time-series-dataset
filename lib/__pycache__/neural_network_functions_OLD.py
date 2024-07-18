# Import necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, MaxPooling1D, Flatten, Layer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from keras.regularizers import l2, l1
from keras.callbacks import EarlyStopping
import optuna

from tensorflow.keras.optimizers import Adam

from imblearn.under_sampling import RandomUnderSampler

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

# Training neural net functions

def define_model(timesteps, num_features, num_classes, conv_filters, kernel_size, lstm_units, dropout_conv, dropout_lstm, learning_rate, kernel_regularizer_l1, kernel_regularizer_l2):
    model = Sequential()
    
    # Conv1D layer
    model.add(Conv1D(filters=conv_filters, kernel_size=kernel_size, activation='relu', input_shape=(timesteps, num_features), kernel_regularizer=l1(kernel_regularizer_l1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(dropout_conv))
    
    # LSTM layer
    model.add(LSTM(lstm_units, activation='relu', recurrent_activation='sigmoid', return_sequences=True, kernel_regularizer=l2(kernel_regularizer_l2)))
    model.add(Dropout(dropout_lstm))
    
    # Flatten layer
    model.add(Flatten())
    
    # Output layer
    model.add(Dense(num_classes, activation='sigmoid'))
    
    # Compile model
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def define_model_simple(timesteps, num_features, num_classes, lstm_units, dropout_lstm, learning_rate):
    model = Sequential()
    model.add(LSTM(lstm_units, activation='relu', input_shape=(timesteps, num_features), kernel_regularizer=l2(0.01)))
    model.add(Dropout(dropout_lstm))
    model.add(Dense(num_classes, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def normalize_data(X):
    # Criar uma nova lista para armazenar as matrizes normalizadas
    X_normalized = []
    
    for matrix in X:
        # Converter a matriz em um array numpy, caso não seja
        matrix = np.array(matrix)
        
        # Normalizar a matriz entre 0 e 1
        matrix_min = matrix.min()
        matrix_max = matrix.max()
        matrix_normalized = (matrix - matrix_min) / (matrix_max - matrix_min)
        
        # Adicionar a matriz normalizada à lista
        X_normalized.append(matrix_normalized)
    
    return np.array(X_normalized)

def balance_data(X, y, random_state=42):
    # Reshape X para 2D (n_samples, n_features)
    n_samples = X.shape[0]
    n_features = X.shape[1] * X.shape[2]
    X_reshaped = X.reshape(n_samples, n_features)

    # Aplicar RandomUnderSampler
    rus = RandomUnderSampler(random_state=random_state)
    X_resampled, y_resampled = rus.fit_resample(X_reshaped, y)

    # Reshape X_resampled de volta para 3D
    X_resampled = X_resampled.reshape(X_resampled.shape[0], X.shape[1], X.shape[2])

    return X_resampled, y_resampled

def plot_learning_curves(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.show()

def evaluate_model_performance(model, X_train, y_train, X_test, y_test):
    y_train_pred_proba = model.predict(X_train)
    y_train_pred = (y_train_pred_proba > 0.5).astype(int).flatten()  # Binário
    
    y_test_pred_proba = model.predict(X_test)
    y_test_pred = (y_test_pred_proba > 0.5).astype(int).flatten() # Binário
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    
    # Print metrics
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1-Score: {f1:.2f}')
    print('\nConfusion Matrix:')
    print(conf_matrix)
    
    # ROC AUC Score (only for binary classification)
    if len(np.unique(y_train)) == 2:
        # Calculating ROC and AUC for the training set
        fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred_proba)
        roc_auc_train = roc_auc_score(y_train, y_train_pred_proba)
        
        # Calculating ROC and AUC for the test set
        fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred_proba)
        roc_auc_test = roc_auc_score(y_test, y_test_pred_proba)
        
        # Plotting the ROC curves
        plt.figure()
        plt.plot(fpr_train, tpr_train, color='blue', lw=2, label='Train ROC curve (AUC = %0.2f)' % roc_auc_train)
        plt.plot(fpr_test, tpr_test, color='darkorange', lw=2, label='Test ROC curve (AUC = %0.2f)' % roc_auc_test)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()
    
    return accuracy, precision, recall, f1, conf_matrix
