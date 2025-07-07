import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

# --- Configuration ---
# Get the directory of the current script (assuming this script is in the root of your project)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'medicine_details.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Ensure the 'models' directory exists
os.makedirs(MODEL_DIR, exist_ok=True)


# --- Data Loading and Cleaning ---
def load_and_clean_data_for_ml(file_path):
    """
    Loads the CSV, cleans column names, handles missing values, and removes duplicates.
    This function is a simplified version for ML preparation, focusing on relevant columns.
    """
    try:
        df = pd.read_csv(file_path)

        # Clean column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('[^a-z0-9_]', '', regex=True)

        # Ensure essential text columns are present and filled
        essential_text_cols = ['uses', 'medicine_name', 'composition', 'side_effects']
        for col in essential_text_cols:
            if col not in df.columns:
                print(f"Error: Essential column '{col}' not found in CSV. Please check your CSV header.")
                return pd.DataFrame()  # Return empty DataFrame if essential column is missing
            df[col] = df[col].fillna('').astype(str).str.lower().str.strip()

        # Handle duplicates based on 'medicine_name' and 'composition' for uniqueness
        # Assuming a medicine is unique by its name and composition
        df.drop_duplicates(subset=['medicine_name', 'composition'], inplace=True)

        return df
    except FileNotFoundError:
        print(f"Error: Dataset not found at {file_path}. Please ensure 'medicine_details.csv' is in the 'data' folder.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred during data loading or cleaning for ML: {e}")
        return pd.DataFrame()


# --- Main Execution for Model Training ---
if __name__ == "__main__":
    print("--- Starting ML Model Training and Saving ---")

    # 1. Load and Clean Data
    df_cleaned = load_and_clean_data_for_ml(DATA_PATH)

    if df_cleaned.empty:
        print("Aborting model training: Cleaned DataFrame is empty or essential columns are missing.")
    else:
        # 2. Feature Engineering: TF-IDF Vectorization
        # We will vectorize the 'uses' column as it's most relevant for symptom matching.
        # 'stop_words='english'' removes common English words like 'the', 'is', 'and'.
        # 'max_features' limits the number of unique words (features) to consider, improving performance.
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

        # Fit the vectorizer to the 'uses' text and transform it into a TF-IDF matrix
        # This matrix represents each medicine's 'uses' as a vector of numbers
        medicine_uses_tfidf_matrix = tfidf_vectorizer.fit_transform(df_cleaned['uses'])

        print(f"TF-IDF Vectorizer fitted. Vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")
        print(f"Medicine Uses TF-IDF Matrix shape: {medicine_uses_tfidf_matrix.shape}")

        # 3. Save the Trained Components
        # Save the TF-IDF Vectorizer: This is crucial because you'll need it to transform
        # new user symptom queries into the same numerical space as your medicine data.
        vectorizer_path = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(tfidf_vectorizer, f)
        print(f"TF-IDF Vectorizer saved to: {vectorizer_path}")

        # Save the TF-IDF Matrix of medicine uses: This matrix will be used to calculate
        # similarity between a user's symptom and all medicines.
        matrix_path = os.path.join(MODEL_DIR, 'medicine_uses_tfidf_matrix.pkl')
        with open(matrix_path, 'wb') as f:
            pickle.dump(medicine_uses_tfidf_matrix, f)
        print(f"Medicine Uses TF-IDF Matrix saved to: {matrix_path}")

        # Save the cleaned DataFrame itself: It's good practice to save the cleaned data
        # because you'll need the original medicine names, compositions, etc., in your Flask app
        # to display the recommendations.
        # We'll save it as a new CSV file to avoid overwriting the original.
        cleaned_data_path = os.path.join(BASE_DIR, 'data', 'medicine_details_cleaned.csv')
        df_cleaned.to_csv(cleaned_data_path, index=False)
        print(f"Cleaned DataFrame saved to: {cleaned_data_path}")

        print("--- ML Model Training and Saving Complete ---")
