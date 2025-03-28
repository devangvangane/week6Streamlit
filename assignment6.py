import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("historical_orders.csv")

data = load_data()

# Prepare the model
@st.cache_resource
def train_model(data):
    data_encoded = pd.get_dummies(data, columns=["Product Category", "Shipping Method"], drop_first=True)
    X = data_encoded.drop(columns=["Delivery Time"])
    y = data_encoded["Delivery Time"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_train.columns

model, feature_names = train_model(data)

# Title of the app
st.title("🚚 Delivery Time Prediction App")

st.header("📦 Enter Order Details")

product_category = st.selectbox("Select Product Category", data["Product Category"].unique())
customer_location = st.text_input("Enter Customer Location (City, State)")
shipping_method = st.selectbox("Select Shipping Method", data["Shipping Method"].unique())
order_weight = st.number_input("Enter Order Weight (kg)", min_value=0.1, step=0.1)
distance = st.number_input("Enter Distance to Customer (km)", min_value=1, step=1)

if not customer_location:
    st.warning("⚠️ Please enter a valid customer location.")
else:
    if st.button("Predict"):
        # Convert input into a DataFrame
        input_data = pd.DataFrame(columns=feature_names)  # Ensure same feature names
        input_data.loc[0] = 0  # Initialize with zeros

        # Assign user inputs
        input_data["Order Weight"] = order_weight
        input_data["Distance"] = distance

        # Encode categorical features
        cat_feature_product = "Product Category_" + product_category
        cat_feature_shipping = "Shipping Method_" + shipping_method

        if cat_feature_product in feature_names:
            input_data[cat_feature_product] = 1
        if cat_feature_shipping in feature_names:
            input_data[cat_feature_shipping] = 1

        # input data matches training feature columns
        input_data = input_data[feature_names]

        # predict
        prediction = model.predict(input_data)[0]
        st.success(f"🚀 Expected delivery time: {prediction:.1f} days")

st.sidebar.info("Powered by Streamlit")
