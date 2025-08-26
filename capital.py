import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load the model and scalers
model_path = 'C:/Users/faiq/PycharmProjects/pythonProject1/streamlit_project/model_test.joblib'
scaler_path = 'C:/Users/faiq/PycharmProjects/pythonProject1/streamlit_project/scaler.joblib'
encoder_path = 'C:/Users/faiq/PycharmProjects/pythonProject1/streamlit_project/encoder.joblib'

# Load the pre-trained model, scaler, and encoder
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
encoder = joblib.load(encoder_path)

# Title for the web app
st.title("Financial Capital Prediction App")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file
    input_data = pd.read_csv(uploaded_file)
    if input_data.empty:
        st.error("Uploaded file is empty. Please upload a valid CSV file.")
    else:
        # Display the uploaded data
        st.write("### Uploaded Data Preview:")
        st.write(input_data.head())

        # Ensure correct data types and calculations
        input_data['total_hutang'] = input_data['total_hutang'].astype(int)
        input_data['loan_to_value_ratio'] = input_data['total_hutang'] / input_data['Nilai_Agunan']
        input_data['Penghasilan_setelah_dipotong_pajak'] = input_data['Penghasilan_setelah_dipotong_pajak'].astype(int)
        input_data['Pengeluaran_dalam_Sebulan'] = input_data['Pengeluaran_dalam_Sebulan'].str.replace('.', '').astype(int)
        input_data['Pendapatan_bulanan_tersedia'] = input_data['Penghasilan_setelah_dipotong_pajak'] - input_data['Pengeluaran_dalam_Sebulan']
        input_data['Pendapatan_bulanan_tersedia'] = input_data['Pendapatan_bulanan_tersedia'].clip(lower=0)  # Set negative values to 0

        # Drop unnecessary columns
        input_data = input_data.drop(columns=['aset'])

        # Handle categorical and numerical variables
        catVar = ['modal', 'Networking(Jaringan)', 'surat_berharga', 'keluarga_besar', 'status_kepemilikan_aset', 'aset.1', 'Status_Lunas_Agunan']
        numVar = ['Penghasilan_setelah_dipotong_pajak', 'Nilai_Agunan', 'Jumlah_Agunan', 'harga_aset_lainnya', 'loan_to_value_ratio', 'Pendapatan_bulanan_tersedia']
        fixed = ['jumlah_aset', 'frekuensi_hutang', 'Modal_barang', 'Aset_Maya', 'tabungan', 'Mutasi_Rekening', 'Laporan_keuangan_aset', 'hutang_1', 'hutang_2', 'hutang_lainnya', 'total_hutang', 'aset_lainnya', 'total_asset', 'aset_mobil', 'aset_motor']

        # Process categorical and numerical data
        df_cat = input_data[catVar]
        df_num = input_data[numVar]
        df_fixed = input_data[fixed]

        # Apply previously trained scaler and encoder
        numerik_1_scaled = scaler.transform(df_num)  # Use transform, not fit_transform
        numerik_1_scaled = pd.DataFrame(numerik_1_scaled, columns=df_num.columns)

        categorical_1_encoded = encoder.transform(df_cat)  # Use transform, not fit_transform
        categorical_1_encoded = pd.DataFrame(categorical_1_encoded, columns=encoder.get_feature_names_out(df_cat.columns))

        # Concatenate numerical, fixed, and categorical features
        df_cap = pd.concat([numerik_1_scaled, df_fixed, categorical_1_encoded], axis=1)

        # Drop any NaN values
        df_cap = df_cap.dropna()

        # Make predictions
        st.write("### Predictions:")
        predictions = model.predict(df_cap)

        # Display predictions
        prediction_df = pd.DataFrame(predictions, columns=["Prediction"])
        st.write(prediction_df)

        # Allow the user to download the prediction results
        prediction_df_csv = prediction_df.to_csv(index=False)
        st.download_button(label="Download Predictions as CSV",
                           data=prediction_df_csv,
                           file_name="predictions.csv",
                           mime="text/csv")
