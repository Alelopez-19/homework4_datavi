import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train_models(X_train, X_test, y_train, y_test):
    
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)

    
    nn_model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1) 
    ])
    nn_model.compile(optimizer='adam', loss='mse')
    nn_model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=0)
    y_pred_nn = nn_model.predict(X_test).flatten()

  
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)

    mae_nn = mean_absolute_error(y_test, y_pred_nn)
    mse_nn = mean_squared_error(y_test, y_pred_nn)
    r2_nn = r2_score(y_test, y_pred_nn)

    return (y_pred_lr, y_pred_nn, mae_lr, mse_lr, r2_lr, mae_nn, mse_nn, r2_nn)


def load_and_prepare_data():
    #
    df = pd.read_csv("NintendoGames.csv")
    
    
    df = df.dropna(subset=['meta_score'])

    
    df['genres'] = df['genres'].apply(lambda x: x.strip("[]").replace("'", "").split(", ") if isinstance(x, str) else [])

   
    genres_encoded = df['genres'].explode().str.get_dummies().groupby(level=0).sum()

   
    categorical_columns = ['developers', 'platform']
    df_encoded = pd.get_dummies(df[categorical_columns])

    
    X_encoded = pd.concat([df_encoded, genres_encoded], axis=1)

    
    y = df['meta_score']

    
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def display_results(mae_lr, mse_lr, r2_lr, mae_nn, mse_nn, r2_nn, y_test, y_pred_lr, y_pred_nn):
    st.title("Model Performance Comparison")

    st.subheader("Linear Regression Results")
    st.write(f"MAE: {mae_lr:.2f}, MSE: {mse_lr:.2f}, R²: {r2_lr:.2f}")

    st.subheader("Neural Network Results")
    st.write(f"MAE: {mae_nn:.2f}, MSE: {mse_nn:.2f}, R²: {r2_nn:.2f}")

    
    lr_comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_lr})
    fig_lr = px.scatter(lr_comparison, x='Actual', y='Predicted', title="Linear Regression: Actual vs Predicted")
    st.plotly_chart(fig_lr)

    
    nn_comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_nn})
    fig_nn = px.scatter(nn_comparison, x='Actual', y='Predicted', title="Neural Network: Actual vs Predicted")
    st.plotly_chart(fig_nn)


def main():
    X_train_scaled, X_test_scaled, y_train, y_test = load_and_prepare_data()
    
    
    (y_pred_lr, y_pred_nn, mae_lr, mse_lr, r2_lr, mae_nn, mse_nn, r2_nn) = train_models(X_train_scaled, X_test_scaled, y_train, y_test)

    
    display_results(mae_lr, mse_lr, r2_lr, mae_nn, mse_nn, r2_nn, y_test, y_pred_lr, y_pred_nn)

if __name__ == "__main__":
    main()