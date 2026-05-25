import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

st.set_page_config(
    page_title="Iris Species Classification",
    layout="wide"
)

st.title("Iris Species Classification Dashboard")

st.write(
    "Proyecto de clasificación de flores Iris usando "
    "Machine Learning y Random Forest."
)

iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

y = iris.target

species_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test,
    predictions,
    average='weighted'
)

recall = recall_score(
    y_test,
    predictions,
    average='weighted'
)

f1 = f1_score(
    y_test,
    predictions,
    average='weighted'
)

st.sidebar.header("Flower Measurements")

sepal_length = st.sidebar.slider(
    "Sepal Length (cm)",
    4.0,
    8.0,
    5.4
)

sepal_width = st.sidebar.slider(
    "Sepal Width (cm)",
    2.0,
    5.0,
    3.4
)

petal_length = st.sidebar.slider(
    "Petal Length (cm)",
    1.0,
    7.0,
    1.3
)

petal_width = st.sidebar.slider(
    "Petal Width (cm)",
    0.1,
    3.0,
    0.2
)

input_data = pd.DataFrame([[
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
]], columns=X.columns)

prediction = model.predict(input_data)

predicted_species = species_names[prediction[0]]

st.subheader("Prediction")

st.success(
    f"Predicted Species: {predicted_species.upper()}"
)

st.subheader("Model Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Accuracy",
    f"{accuracy:.2f}"
)

col2.metric(
    "Precision",
    f"{precision:.2f}"
)

col3.metric(
    "Recall",
    f"{recall:.2f}"
)

col4.metric(
    "F1 Score",
    f"{f1:.2f}"
)

st.subheader("Dataset Preview")

st.dataframe(
    X.head(),
    width='stretch'
)

st.subheader("Petal Length Distribution")

fig, ax = plt.subplots(figsize=(10, 5))

sns.histplot(
    X['petal length (cm)'],
    kde=True,
    ax=ax
)

ax.set_xlabel("Petal Length (cm)")
ax.set_ylabel("Frequency")

st.pyplot(fig)

st.subheader("Scatter Plot")

fig2, ax2 = plt.subplots(figsize=(8, 6))

scatter = ax2.scatter(
    X['sepal length (cm)'],
    X['petal length (cm)'],
    c=y
)

ax2.set_xlabel("Sepal Length (cm)")
ax2.set_ylabel("Petal Length (cm)")

st.pyplot(fig2)

st.subheader("User Input")

input_df = pd.DataFrame({
    "Sepal Length": [sepal_length],
    "Sepal Width": [sepal_width],
    "Petal Length": [petal_length],
    "Petal Width": [petal_width]
})

st.dataframe(
    input_df,
    width='stretch'
)

X_plot = X.copy()

X_plot["species"] = y.astype(str)

st.subheader("3D Iris Visualization")

fig3d = px.scatter_3d(
    X_plot,
    x='sepal length (cm)',
    y='sepal width (cm)',
    z='petal length (cm)',
    color='species',
    title='3D Iris Dataset Visualization'
)

st.plotly_chart(
    fig3d,
    width='stretch'
)