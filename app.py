import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="wide", initial_sidebar_state="expanded")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "EDA", "Model Performance", "Predict Price"])

@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\user\Videos\MY PORTFOLIO\MACHIENE LEARNING PORFOLIO\HOUSE PRICE PREDICTION MODEL\house_price_prediction_.csv")
    df = df.drop("id", axis=1)
    cat_cols = ["location", "has_garage"]
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col])
    return df

df = load_data()

x = df.drop("price", axis=1)
y = df["price"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = r2_score(y_test, y_pred) * 100

if page == "Home":
    st.title("🏠 House Price Prediction App")
    st.write("This app allows you to explore the dataset, check model performance, and predict house prices interactively.")
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    
elif page == "EDA":
    st.title("📊 Exploratory Data Analysis")
    
    if st.checkbox("Show Correlation Heatmap"):
        plt.figure(figsize=(10,6))
        sns.heatmap(df.corr(), annot=True, cmap="YlGnBu")
        st.pyplot(plt)
    
    num_cols = ["area_sqft", "bedrooms", "bathrooms", "floors", "year_built"]
    if st.checkbox("Show Scatter Plots of Numeric Features vs Price"):
        plt.figure(figsize=(15,8))
        for i, col in enumerate(num_cols):
            plt.subplot(2, 3, i+1)
            sns.scatterplot(x=df[col], y=df["price"], alpha=0.5, color="#007ACC")
            plt.xlabel(col)
            plt.ylabel("Price")
        plt.tight_layout()
        st.pyplot(plt)
    
    if st.checkbox("Boxplot for Location"):
        plt.figure(figsize=(10,5))
        sns.boxplot(x=df["location"], y=df["price"], palette="Blues")
        st.pyplot(plt)
    
    if st.checkbox("Boxplot for Has Garage"):
        plt.figure(figsize=(6,5))
        sns.boxplot(x=df["has_garage"], y=df["price"], palette="coolwarm")
        st.pyplot(plt)

elif page == "Model Performance":
    st.title("📈 Model Performance")
    st.write(f"**Linear Regression R² Score:** {accuracy:.2f}%")
    
    if st.checkbox("Predicted vs Actual Prices"):
        plt.figure(figsize=(10,6))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, color="#007ACC")
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel("Actual Price")
        plt.ylabel("Predicted Price")
        plt.grid(True)
        st.pyplot(plt)
    
    if st.checkbox("Residuals Histogram"):
        residuals = y_test - y_pred
        plt.figure(figsize=(10,6))
        sns.histplot(residuals, kde=True, color="#00BFA6")
        st.pyplot(plt)

elif page == "Predict Price":
    st.title("🖊️ Predict House Price")
    st.write("Input the details below to predict the house price.")

    with st.form("prediction_form"):
        area_sqft = st.number_input("Area (sqft)", min_value=100, max_value=10000, value=1000, step=50)
        bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
        floors = st.number_input("Floors", min_value=1, max_value=5, value=1)
        year_built = st.number_input("Year Built", min_value=1900, max_value=2025, value=2020)
        location = st.selectbox("Location (encoded)", df["location"].unique())
        has_garage = st.selectbox("Has Garage (encoded)", df["has_garage"].unique())
        
        submitted = st.form_submit_button("Predict Price")
        if submitted:
            input_data = [[area_sqft, bedrooms, bathrooms, floors, year_built, location, has_garage]]
            prediction = model.predict(input_data)[0]
            st.success(f"Predicted House Price: ${prediction:,.2f}")