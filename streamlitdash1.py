# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Real estate.csv")  # Make sure this file is in the same folder

# Rename columns to user-friendly names (optional)
# Load dataset
df = pd.read_csv("Real estate.csv")

# Check how many columns were loaded
print(df.shape)  # Should print something like (n_rows, 8)

# Rename columns correctly — make sure this list has exactly 8 names!
df.columns = [
    'sno',
    'transaction_date',
    'house_age',
    'distance_to_mrt',
    'convenience_stores',
    'latitude',
    'longitude',
    'price_per_unit'
]

# Feature selection
features = ['house_age', 'distance_to_mrt', 'convenience_stores', 'latitude', 'longitude']
X = df[features]
y = df['price_per_unit']

# Split and train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Streamlit UI
st.title("🏠 Real Estate Price Prediction Dashboard")

st.sidebar.header("🔧 Input Property Features")
house_age = st.sidebar.slider("House Age (years)", 0, 50, 10)
distance_to_mrt = st.sidebar.slider("Distance to MRT (meters)", 0, 10000, 500)
convenience_stores = st.sidebar.slider("Number of Nearby Convenience Stores", 0,10,0)
latitude = st.sidebar.number_input("Latitude", value=24.98)
longitude = st.sidebar.number_input("Longitude", value=121.54)

# Predict using input
input_data = np.array([[house_age, distance_to_mrt, convenience_stores, latitude, longitude]])
predicted_price = model.predict(input_data)[0]

st.subheader("📈 Predicted Price per Unit Area")
st.success(f"NT${predicted_price:.2f}")

# Show performance metrics
st.subheader("📊 Model Performance on Test Data")
st.write(f"**R² Score:** {r2:.3f}")
st.write(f"**Mean Squared Error:** {mse:.2f}")

# Plot feature coefficients
coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_
})

print(coef_df)
coef_df['abs_coef'] = coef_df['Coefficient'].abs()
coef_df = coef_df.sort_values(by='abs_coef')



# Apply dark style
plt.style.use("dark_background")

fig, ax = plt.subplots(figsize=(7, 4))

sns.barplot(
    x='Coefficient', 
    y='Feature', 
    data=coef_df, 
    palette='coolwarm',
    ax=ax, 
    legend=False
)

# Adjust text & labels for readability
ax.set_title("Feature Impact on Price", color="white", fontsize=14)
ax.set_xlabel("Coefficient", color="white")
ax.set_ylabel("Feature", color="white")

# Make ticks visible on dark background
ax.tick_params(colors="white")

st.pyplot(fig)


# Optional: show raw data
if st.checkbox("Show Raw Dataset Overview"):

    st.dataframe((df.drop(columns=['sno']).head(10)))


