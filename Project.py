# Mengimport library
import pickle
import numpy as np
import streamlit as st

# Load the knn model from file
DTR = pickle.load(open('dtr.pkl', 'rb'))
scaler = pickle.load(open('Sc.pkl', 'rb'))

# Judul Web

st.title("Prediksi SwimIcon")

# Mapping

siteType = st.radio("siteType", ['River', 'Coastal', 'Lake'])
if siteType == 'River':
    siteType = 0
elif siteType == 'Coastal':
    siteType = 1
elif siteType == 'Lake':
    siteType = 2

Latitude = st.text_input("Latitude")
if Latitude != '':
    Latitude = float(Latitude)  # Convert to float

Longitude = st.text_input("Longitude")
if Longitude != '':
    Longitude = float(Longitude)  # Convert to float

Property = st.radio("Property", ['E-coli', 'Enterococci', 'Cyanobacteria'])
if Property == 'E-coli':
    Property = 0
elif Property == 'Enterococci':
    Property = 1
elif Property == 'Cyanobacteria':
    Property = 2

Value = st.text_input("Value")
if Value != '':
    Value = float(Value)  # Convert to float

# Kode untuk prediksi

Prediksi_SwimIcon = ''
if st.button("Prediksi Sekarang"):
    # Mengubah argumen menjadi array numpy dua dimensi
    data = np.array([[siteType, Latitude, Longitude, Property, Value]])
    # Melakukan scaling pada data input
    scaled_data = scaler.transform(data)
    # Melakukan prediksi dengan Decision Tree
    Prediksi = DTR.predict(scaled_data)

    if Prediksi[0] == 0:
        Prediksi_SwimIcon = "Green (Boleh Berenang)"
    elif Prediksi[0] == 1:
        Prediksi_SwimIcon = "Red (Tidak Boleh Berenang)"
    elif Prediksi[0] == 2:
        Prediksi_SwimIcon = "Amber (Perlu Hati-Hati Saat Berenang)"
    else:
        Prediksi_SwimIcon = "No Data"

st.success(Prediksi_SwimIcon)
st.write("Prediksi :", Prediksi_SwimIcon)

