%%writefile streamlit_app.py

import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


# Load existing model and dataset

reg_model = pickle.load(open('best_diamond_model.pkl', 'rb'))
df = pd.read_csv("diamonds.csv")  


# Encode categorical variables g)

cut_mapping = {"Fair":0, "Good":1, "Very Good":2, "Premium":3, "Ideal":4}
color_mapping = {"J":0,"I":1,"H":2,"G":3,"F":4,"E":5,"D":6}
clarity_mapping = {"I1":0,"SI2":1,"SI1":2,"VS2":3,"VS1":4,"VVS2":5,"VVS1":6,"IF":7}

df['cut_enc'] = df['cut'].map(cut_mapping)
df['color_enc'] = df['color'].map(color_mapping)
df['clarity_enc'] = df['clarity'].map(clarity_mapping)

# PCA features
cluster_features = ['carat','cut_enc','color_enc','clarity_enc','depth','table','x','y','z']
X_cluster = df[cluster_features]

# Scale features for PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(X_scaled)


# Streamlit UI

st.title("Diamond Dynamics: Price Prediction & Market Segmentation ")
st.subheader("Enter Diamond Characteristics")

# Numeric Inputs
carat = st.number_input("Carat", value=1.0)
depth = st.number_input("Depth (%)", value=60.0)
table = st.number_input("Table (%)", value=57.0)
x = st.number_input("X (mm)", value=5.5)
y = st.number_input("Y (mm)", value=5.7)
z = st.number_input("Z (mm)", value=3.5)

# Categorical Inputs
cut = st.selectbox("Cut", ["Fair","Good","Very Good","Premium","Ideal"])
color = st.selectbox("Color", ["J","I","H","G","F","E","D"])
clarity = st.selectbox("Clarity", ["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"])


# Predict Price

if st.button("Predict Price"):
    cut_enc = cut_mapping[cut]
    color_enc = color_mapping[color]
    clarity_enc = clarity_mapping[clarity]

    # Derived / placeholder features
    volume = x * y * z
    price_per_carat = 0      # Placeholder, cannot compute before prediction
    dimension_ratio = x / y
    carat_category = 0       # Placeholder
    placeholder_extra = 0    # Placeholder for 14th feature if needed

    # Prepare 14-feature array
    features = np.array([[carat, cut_enc, color_enc, clarity_enc, depth, table, 
                          x, y, z, volume, price_per_carat, dimension_ratio, carat_category, placeholder_extra]])

    # Predict price
    price = reg_model.predict(features)[0]
    st.success(f"Predicted Price: ${price:.2f}")
-
# PCA Market Segmentation Plot

st.subheader("Diamond Market Segmentation (PCA Visualization)")

fig, ax = plt.subplots(figsize=(8,6))
scatter = ax.scatter(pca_data[:,0], pca_data[:,1], c=df['carat'], cmap='viridis', alpha=0.6)
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.set_title("Diamond Clusters based on Features")
cbar = plt.colorbar(scatter)
cbar.set_label("Carat Size")
st.pyplot(fig)




