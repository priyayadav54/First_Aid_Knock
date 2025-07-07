import pandas as pd
import nltk
from nltk.corpus import stopwords
from collections import Counter
import os

# --- Configuration ---
# Get the directory of the current script (assuming this script is in the root of your project)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'medicine_details.csv')

# --- NLTK Downloads (Run these once if you haven't already) ---
# You might need to run these lines the very first time you use NLTK
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')

# Load English stopwords
stop_words = set(stopwords.words('english'))


# --- Data Loading ---
def load_and_clean_data(file_path):
    """Loads the CSV, cleans column names, handles missing values, and removes duplicates."""
    try:
        df = pd.read_csv(file_path)
        print(f"Original data shape: {df.shape}")
        print("Original columns:", df.columns.tolist())

        # Clean column names: lowercase, replace spaces with underscores, remove special chars
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('[^a-z0-9_]', '', regex=True)
        print("Cleaned columns:", df.columns.tolist())

        # Identify and fill missing values for relevant text columns
        text_cols_to_clean = ['uses', 'side_effects', 'composition', 'medicine_name', 'category']
        for col in text_cols_to_clean:
            if col in df.columns:
                df[col] = df[col].fillna('').astype(str).str.lower().str.strip()
            else:
                print(f"Warning: Column '{col}' not found in DataFrame.")

        # Handle duplicate entries
        initial_duplicates = df.duplicated().sum()
        df.drop_duplicates(inplace=True)
        print(f"Removed {initial_duplicates - df.duplicated().sum()} duplicate rows.")
        print(f"Cleaned data shape: {df.shape}")

        return df
    except FileNotFoundError:
        print(f"Error: Dataset not found at {file_path}. Please ensure 'medicine_details.csv' is in the 'data' folder.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred during data loading or initial cleaning: {e}")
        return pd.DataFrame()


# --- Text Analysis Function ---
def analyze_text_column(df_series, num_words=20):
    """
    Analyzes a pandas Series containing text data to find the most common words.
    Performs tokenization, lowercasing, punctuation removal, and stop word removal.
    """
    all_words = []
    for text in df_series:
        # Ensure text is string and convert to lowercase
        tokens = nltk.word_tokenize(str(text).lower())
        # Filter out non-alphabetic tokens and stopwords
        filtered_words = [word for word in tokens if word.isalpha() and word not in stop_words]
        all_words.extend(filtered_words)
    return Counter(all_words).most_common(num_words)


# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting Data Cleaning and EDA ---")

    # Load and clean the data
    df = load_and_clean_data(DATA_PATH)

    if not df.empty:
        print("\n--- Data Summary after Cleaning ---")
        print(df.info())
        print("\nFirst 5 rows of cleaned data:")
        print(df.head())

        # --- Exploratory Data Analysis (EDA) ---
        print("\n--- EDA: Value Counts for 'category' ---")
        if 'category' in df.columns:
            print(df['category'].value_counts().head(10))
        else:
            print("Column 'category' not found for value counts.")

        print("\n--- EDA: Most Common Words in 'uses' column ---")
        if 'uses' in df.columns:
            common_uses_words = analyze_text_column(df['uses'])
            for word, count in common_uses_words:
                print(f"  {word}: {count}")
        else:
            print("Column 'uses' not found for text analysis.")

        print("\n--- EDA: Most Common Words in 'side_effects' column ---")
        if 'side_effects' in df.columns:
            common_side_effects_words = analyze_text_column(df['side_effects'])
            for word, count in common_side_effects_words:
                print(f"  {word}: {count}")
        else:
            print("Column 'side_effects' not found for text analysis.")

        print("\n--- Data Cleaning and EDA Complete ---")
    else:
        print("\nDataframe is empty. Cannot proceed with EDA.")

