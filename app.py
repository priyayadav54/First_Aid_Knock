# app.py

from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity  # Import cosine_similarity

app = Flask(__name__)

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'medicine_details.csv')
CLEANED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'medicine_details_cleaned.csv')  # Path to your cleaned CSV
MODEL_DIR = os.path.join(BASE_DIR, 'models')
TFIDF_VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
TFIDF_MATRIX_PATH = os.path.join(MODEL_DIR, 'medicine_uses_tfidf_matrix.pkl')

# --- Global Variables for Data and Model Components ---
# These will be loaded once when the app starts
medicine_df_cleaned = None
tfidf_vectorizer = None
medicine_uses_tfidf_matrix = None


# --- Loading ML Components and Cleaned Data ---
def load_ml_components():
    """Loads the cleaned DataFrame, TF-IDF vectorizer, and TF-IDF matrix."""
    global medicine_df_cleaned, tfidf_vectorizer, medicine_uses_tfidf_matrix
    try:
        # Load the cleaned DataFrame
        medicine_df_cleaned = pd.read_csv(CLEANED_DATA_PATH)
        print(f"Successfully loaded cleaned data from: {CLEANED_DATA_PATH}")

        # Load the TF-IDF Vectorizer
        with open(TFIDF_VECTORIZER_PATH, 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        print(f"Successfully loaded TF-IDF Vectorizer from: {TFIDF_VECTORIZER_PATH}")

        # Load the TF-IDF Matrix
        with open(TFIDF_MATRIX_PATH, 'rb') as f:
            medicine_uses_tfidf_matrix = pickle.load(f)
        print(f"Successfully loaded Medicine Uses TF-IDF Matrix from: {TFIDF_MATRIX_PATH}")

    except FileNotFoundError as e:
        print(
            f"ERROR: File not found during ML component loading: {e}. Make sure you have run train_model.py first and paths are correct.")
        # Set components to None to indicate failure
        medicine_df_cleaned = None
        tfidf_vectorizer = None
        medicine_uses_tfidf_matrix = None
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during loading ML components: {e}")
        medicine_df_cleaned = None
        tfidf_vectorizer = None
        medicine_uses_tfidf_matrix = None


# --- ML-based Recommendation Logic Function ---
def get_ml_recommendations(symptom_query, top_n=10):
    """
    Generates medicine recommendations based on symptom similarity using TF-IDF and Cosine Similarity.
    """
    if medicine_df_cleaned is None or tfidf_vectorizer is None or medicine_uses_tfidf_matrix is None:
        print("ML components not loaded. Cannot provide recommendations.")
        return []

    # Preprocess the symptom query using the *trained* TF-IDF vectorizer
    symptom_tfidf = tfidf_vectorizer.transform([symptom_query.lower()])

    # Calculate cosine similarity between the symptom vector and all medicine uses vectors
    cosine_similarities = cosine_similarity(symptom_tfidf, medicine_uses_tfidf_matrix).flatten()

    # Get the indices of medicines sorted by similarity score in descending order
    related_medicine_indices = cosine_similarities.argsort()[::-1]

    recommendations = []
    # Iterate through sorted indices, but only add if similarity is above a certain threshold (e.g., > 0)
    for i in related_medicine_indices:
        if cosine_similarities[i] > 0:  # Only include if there's any similarity
            medicine_info = medicine_df_cleaned.iloc[i]
            recommendations.append({
                'name': medicine_info.get('medicine_name', 'N/A').title(),
                'composition': medicine_info.get('composition', 'N/A').title(),
                'uses': medicine_info.get('uses', 'N/A').capitalize(),
                'side_effects': medicine_info.get('side_effects', 'No common side effects listed.').capitalize(),
                'image_url': medicine_info.get('image_url', 'https://placehold.co/100x100/cccccc/000000?text=No+Image')
            })
            if len(recommendations) >= top_n:
                break

    return recommendations


# --- Rule-based Multi-System Advice Logic Function ---
def get_multi_system_advice(symptom_query):
    """
    Provides multi-system advice (Allopathy, Ayurveda, Homeopathy) based on the symptom query.
    This is currently a rule-based system.
    """
    symptom_query_lower = symptom_query.lower()

    advice = {
        'allopathy': 'No specific Allopathy advice found for this symptom. Please consult a doctor.',
        'ayurvedic': 'No specific Ayurvedic advice found for this symptom. Please consult a doctor.',
        'homeopathy': 'No specific Homeopathy advice found for this symptom. Please consult a doctor.'
    }

    if "headache" in symptom_query_lower:
        advice[
            'allopathy'] = 'For a headache, consider taking a common pain reliever like <span class="font-semibold">Acetaminophen (e.g., Tylenol)</span> or <span class="font-semibold">Ibuprofen (e.g., Advil, Motrin)</span>. Rest in a quiet, dark room.'
        advice[
            'ayurvedic'] = 'For headaches, try applying a paste of <span class="font-semibold">sandalwood or ginger</span> on the forehead. Consuming <span class="font-semibold">ginger tea</span> can also help. Practice Shirodhara (oil drip therapy) for chronic cases.'
        advice[
            'homeopathy'] = 'For headaches, common homeopathic remedies include <span class="font-semibold">Belladonna</span> (for throbbing pain), <span class="font-semibold">Bryonia</span> (for pain worse with movement), or <span class="font-semibold">Nux Vomica</span> (for headaches from indigestion/stress).';
    elif "fever" in symptom_query_lower:
        advice[
            'allopathy'] = 'For fever, use <span class="font-semibold">Acetaminophen (e.g., Tylenol)</span> or <span class="font-semibold">Ibuprofen (e.g., Advil, Motrin)</span> to reduce temperature and discomfort. Drink plenty of fluids.'
        advice[
            'ayurvedic'] = 'For fever, consume <span class="font-semibold">Giloy Kadha</span> (Tinospora cordifolia decoction) or <span class="font-semibold">Tulsi (Holy Basil) tea</span>. Light, easily digestible foods and adequate rest are recommended.'
        advice[
            'homeopathy'] = 'For fever, homeopathic remedies like <span class="font-semibold">Aconite</span> (for sudden onset, high fever), <span class="font-semibold">Belladonna</span> (for red face, throbbing), or <span class="font-semibold">Gelsemium</span> (for chills, weakness) may be considered.'
    elif "cough" in symptom_query_lower:
        advice[
            'allopathy'] = 'For a cough, a <span class="font-semibold">cough suppressant (e.g., Dextromethorphan)</span> for dry coughs or an <span class="font-semibold">expectorant (e.g., Guaifenesin)</span> for productive coughs might help. Honey and warm liquids can also soothe the throat.'
        advice[
            'ayurvedic'] = 'For cough, use <span class="font-semibold">ginger and honey syrup</span>. Steam inhalation with <span class="font-semibold">eucalyptus oil</span> and consuming warm herbal teas with licorice or basil can provide relief.'
        advice[
            'homeopathy'] = 'For cough, homeopathic options include <span class="font-semibold">Bryonia</span> (for dry, painful cough), <span class="font-semibold">Pulsatilla</span> (for loose cough, worse in warm room), or <span class="font-semibold">Drosera</span> (for spasmodic cough).'
    elif "sore throat" in symptom_query_lower or "sore_throat" in symptom_query_lower:
        advice[
            'allopathy'] = 'For a sore throat, try <span class="font-semibold">throat lozenges, gargling with warm salt water,</span> or using a <span class="font-semibold">pain reliever like Acetaminophen or Ibuprofen</span>. Stay hydrated.'
        advice[
            'ayurvedic'] = 'For sore throat, gargle with warm water mixed with <span class="font-semibold">turmeric and salt</span>. Consume warm herbal teas with ginger, honey, and black pepper. Avoid cold foods and drinks.'
        advice[
            'homeopathy'] = 'For sore throat, remedies like <span class="font-semibold">Belladonna</span> (for sudden, red, inflamed throat), <span class="font-semibold">Mercurius Solubilis</span> (for raw, burning pain, bad breath), or <span class="font-semibold">Hepar Sulphuris</span> (for splinter-like pain) can be helpful.'
    elif "stomach ache" in symptom_query_lower or "stomach_ache" in symptom_query_lower:
        advice[
            'allopathy'] = 'For a mild stomach ache, an <span class="font-semibold">antacid (e.g., Tums, Rolaids)</span> or <span class="font-semibold">Bismuth Subsalicylate (e.g., Pepto-Bismol)</span> can provide relief. Avoid heavy or spicy foods.';
        advice[
            'ayurvedic'] = 'For stomach ache, consume <span class="font-semibold">ginger tea</span> or a mixture of <span class="font-semibold">cumin powder and warm water</span>. Light, easily digestible meals and avoiding cold beverages are advised.';
        advice[
            'homeopathy'] = 'For stomach ache, consider <span class="font-semibold">Colocynthis</span> (for cramping pain relieved by pressure), <span class="font-semibold">Nux Vomica</span> (for indigestion, bloating), or <span class="font-semibold">Magnesia Phosphorica</span> (for spasmodic pain).';
    elif "cold" in symptom_query_lower or "flu" in symptom_query_lower or "cold_flu" in symptom_query_lower:
        advice[
            'allopathy'] = 'For cold/flu symptoms, consider an <span class="font-semibold">over-the-counter cold and flu medication</span> that combines pain relievers, decongestants, and cough suppressants. Rest and hydration are key.';
        advice[
            'ayurvedic'] = 'For cold/flu, drink warm <span class="font-semibold">Tulsi-ginger tea</span>. Inhale steam with a few drops of <span class="font-semibold">eucalyptus oil</span>. Consume warm, light, and nourishing foods like vegetable soup.';
        advice[
            'homeopathy'] = 'For cold/flu, remedies like <span class="font-semibold">Aconite</span> (for sudden onset, dry cold), <span class="font-semibold">Arsenicum Album</span> (for restless, anxious cold), or <span class="font-semibold">Gelsemium</span> (for sluggish, heavy cold) are commonly used.';
    elif "muscle pain" in symptom_query_lower or "muscle_pain" in symptom_query_lower:
        advice[
            'allopathy'] = 'For muscle pain, <span class="font-semibold">Ibuprofen (e.g., Advil)</span> or <span class="font-semibold">Naproxen (e.g., Aleve)</span> can help reduce inflammation and pain. Applying a hot or cold pack can also provide relief.';
        advice[
            'ayurvedic'] = 'For muscle pain, apply <span class="font-semibold">Mahanarayan oil</span> or <span class="font-semibold">Mahanarayan Taila</span> (herbal oil) externally and gently massage. Consuming warm turmeric milk can also help reduce inflammation.';
        advice[
            'homeopathy'] = 'For muscle pain, homeopathic remedies include <span class="font-semibold">Arnica Montana</span> (for bruising, soreness from injury), <span class="font-semibold">Rhus Tox</span> (for stiffness, worse on first movement), or <span class="font-semibold">Bryonia</span> (for pain worse with movement).';

    return advice


# --- Flask Routes ---

@app.route('/')
def index():
    """
    Renders the main input page for symptom selection.
    Passes unique 'uses' from the cleaned data for search suggestions.
    """
    unique_uses = []
    if medicine_df_cleaned is not None and 'uses' in medicine_df_cleaned.columns:
        all_uses_text = ' '.join(medicine_df_cleaned['uses'].tolist())
        unique_uses = sorted(list(set(all_uses_text.split())))
        unique_uses = unique_uses[:500]  # Adjust as needed

    return render_template('index.html', unique_uses=unique_uses)


@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Receives symptom input, uses the ML model for recommendation,
    and displays the results.
    """
    symptom = request.form.get('symptom')
    if not symptom:
        empty_recommendations = []
        default_advice = {
            'allopathy': 'Please enter a symptom to get advice.',
            'ayurvedic': 'Please enter a symptom to get advice.',
            'homeopathy': 'Please enter a symptom to get advice.'
        }
        return render_template('result.html', symptom="No symptom entered", recommendations=empty_recommendations,
                               advice=default_advice)

    recommendations = get_ml_recommendations(symptom)
    multi_system_advice = get_multi_system_advice(symptom)

    return render_template('result.html', symptom=symptom, recommendations=recommendations, advice=multi_system_advice)


@app.route('/oxygen')
def oxygen():
    """Renders the Oxygen section page."""
    return render_template('oxygen.html')


@app.route('/about')
def about():
    """Renders the About Us section page."""
    return render_template('about.html')


@app.route('/contact')
def contact():
    """Renders a placeholder Contact page."""
    # This page was not fully developed or themed at the 'awesome' state.
    # Providing a basic, non-themed version.
    return render_template('contact.html')


# --- Run the App ---
if __name__ == '__main__':
    # Load ML components and cleaned data when the Flask app starts
    load_ml_components()
    app.run(debug=True)
