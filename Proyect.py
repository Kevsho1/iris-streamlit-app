import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# CONFIGURATION
st.set_page_config(
    page_title="Iris Species Classification",
    layout="wide"
)

st.title("Iris Species Classification Dashboard")

st.write("""
Machine Learning project for classifying Iris flower species
using a Random Forest classifier.
""")

# WORKFLOW
st.subheader("Project Workflow")

with st.expander("1. Data Understanding"):
    st.write("""
    The Iris dataset contains 150 samples with 4 numerical features:
    - Sepal length
    - Sepal width
    - Petal length
    - Petal width
    
    The objective is to classify 3 flower species.
    """)

with st.expander("2. Data Preparation"):
    st.write("""
    - No missing values in the dataset
    - No advanced preprocessing required
    - Data split into training (80%) and testing (20%)
    """)

with st.expander("3. Model Training"):
    st.write("""
    A Random Forest Classifier is used due to:
    - High accuracy on small datasets
    - Ability to handle non-linear relationships
    - Robust and stable performance
    """)

with st.expander("4. Evaluation"):
    st.write("""
    The model is evaluated using:
    - Accuracy
    - Precision
    - Recall
    - F1-score
    """)

# DATA
iris = load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
species_names = iris.target_names

# TRAIN MODEL
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

# METRICS
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='weighted')
recall = recall_score(y_test, predictions, average='weighted')
f1 = f1_score(y_test, predictions, average='weighted')

# USER INPUT
st.sidebar.header("Flower Measurements")

sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.4)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 5.0, 3.4)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.3)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 3.0, 0.2)

input_data = pd.DataFrame([[
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
]], columns=X.columns)

prediction = model.predict(input_data)
predicted_species = species_names[prediction[0]]

# PREDICTION RESULT
st.subheader("Prediction Result")

st.success(f"The model predicts: {predicted_species.upper()}")

st.info("""
This prediction is generated using a trained Random Forest model
that compares the input values with patterns learned from the dataset.
""")

# METRICS
st.subheader("Model Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{accuracy:.2f}")
col2.metric("Precision", f"{precision:.2f}")
col3.metric("Recall", f"{recall:.2f}")
col4.metric("F1 Score", f"{f1:.2f}")

# DATA PREVIEW
st.subheader("Dataset Preview")
st.dataframe(X.head())

# HISTOGRAM
st.subheader("Petal Length Distribution")

fig, ax = plt.subplots()
sns.histplot(X['petal length (cm)'], kde=True, ax=ax)
st.pyplot(fig)

# SCATTER PLOT
st.subheader("Scatter Plot")

fig2, ax2 = plt.subplots()
ax2.scatter(
    X['sepal length (cm)'],
    X['petal length (cm)'],
    c=y
)
ax2.set_xlabel("Sepal Length (cm)")
ax2.set_ylabel("Petal Length (cm)")
st.pyplot(fig2)

# 3D VISUALIZATION
st.subheader("3D Visualization with User Prediction")

X_plot = X.copy()
X_plot["species"] = y.astype(str)

fig3d = px.scatter_3d(
    X_plot,
    x='sepal length (cm)',
    y='sepal width (cm)',
    z='petal length (cm)',
    color='species',
    title="Iris Dataset in 3D Space"
)

# USER POINT
fig3d.add_scatter3d(
    x=[sepal_length],
    y=[sepal_width],
    z=[petal_length],
    mode='markers',
    marker=dict(size=8, color='black'),
    name="User Input"
)

st.plotly_chart(fig3d, use_container_width=True)

# USER INPUT DISPLAY
st.subheader("User Input Data")
st.dataframe(input_data)