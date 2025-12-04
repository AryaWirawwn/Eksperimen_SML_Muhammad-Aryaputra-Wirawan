import pandas as pd
import os

DATA_URL = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vT5qaBV39KxL2ViGJdv1_8J6zOj-U59NGL6BbfxRW_0Mf5mGAWkat7o25CNGKaLJGyry9BAOOaXgiD7/pub?gid=352973935&single=true&output=csv'
OUTPUT_FILENAME = 'dataset_mesin_membangun_sistem_machine_learning_preprocessing.csv'
OUTPUT_DIR = 'preprocessing'

def preprocess_data(data_path: str) -> pd.DataFrame:
    """
    Melakukan preprocessing data secara otomatis.
    """
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"Error saat memuat dataset: {e}")
        return pd.DataFrame()

    print("1. Memuat dataset berhasil.")

    columns_to_fix = ['Rotational speed [rpm]', 'Torque [Nm]']
    
    for col in columns_to_fix:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df[col] = df[col].apply(lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x))
    
    print("2. Penanganan outlier pada Rotational speed [rpm] dan Torque [Nm] berhasil.")
    
    df = df[~((df['Target'] == 0) & (df['Failure Type'] != 'No Failure'))]
    df = df[~((df['Target'] == 1) & (df['Failure Type'] == 'No Failure'))]
    
    df.reset_index(drop=True, inplace=True)
    
    print(f"3. Penanganan inconsistent values berhasil. Jumlah data akhir: {len(df)}.")

    if 'UDI' in df.columns and 'Product ID' in df.columns:
        df = df.drop(['UDI', 'Product ID'], axis=1)
        print("4. Kolom UDI dan Product ID berhasil dihapus.")
    else:
        print("4. Kolom UDI dan Product ID tidak ditemukan atau sudah dihapus.")
        
    typeMapping = {
        'L': 1,
        'M': 2,
        'H': 3
    }
    
    df.loc[:, 'Type'] = df['Type'].map(typeMapping)
    
    print("5. Encoding kolom 'Type' berhasil.")
    
    return df

if __name__ == "__main__":
    os.makedirs('preprocessing', exist_ok=True)
    
    print("Memulai proses preprocessing data untuk GitHub Actions...")
    preprocessed_df = preprocess_data(DATA_URL)

    if not preprocessed_df.empty:
        output_path = os.path.join('preprocessing', OUTPUT_FILENAME)
        preprocessed_df.to_csv(output_path, index=False)
        print(f"\nPreprocessing selesai. Data siap dilatih disimpan di: {output_path}")