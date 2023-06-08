import pickle
import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler

# Menjalankan model dan scaler
air_model = pickle.load(open('air_model.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))

# Judul
st.title('Prediksi Wisata Air')

Latitude = st.text_input('Input Nilai Latitude')
Longitude = st.text_input('Input Nilai Longitude')
Value = st.text_input('Input Nilai Value')
Property = st.text_input('Input Nilai Property')

# Kode Prediksi
wisata_air_prediction = ''

# Membuat prediksi dan tombol button
if st.button('Prediksi Wisata Air'):
    if Latitude and Longitude and Value and Property:
        try:
            Latitude = float(Latitude)
            Longitude = float(Longitude)
            Value = float(Value)
            Property = float(Property)

            # membuat input data array
            input_data = np.array([Latitude, Longitude, Value, Property]).reshape(1, -1)

            # Melakukan scaling pada data yang di input
            std_data = scaler.transform(input_data)

            # Membuat prediksi
            prediction = air_model.predict(std_data)

            if prediction[0] == 0:
                wisata_air_prediction = 'Anda boleh berenang'
            elif prediction[0] == 1:
                wisata_air_prediction = 'Anda tidak boleh berenang'
            else:
                wisata_air_prediction = 'Berhati-hati'
        except ValueError:
            wisata_air_prediction = 'Input tidak valid, harap masukkan angka'
    else:
        wisata_air_prediction = 'Harap lengkapi semua input'

st.success(wisata_air_prediction)
