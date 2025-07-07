# First_Aid_Knock
"First Aid Knock" is a web application designed to provide immediate, general first-cure advice on common ailments and over-the-counter (OTC) medicine recommendations.
 It leverages machine learning for medicine suggestions and offers advice, alongside educational content on vital health topics like oxygen and breathing.

Features
Symptom-Based Medicine Recommendations: Users can input symptoms to receive suggestions for relevant over-the-counter medicines, powered by a machine learning model (TF-IDF and Cosine Similarity).

Multi-System First-Cure Advice: Provides general first aid advice categorized by Allopathy, Ayurveda, and Homeopathy for common symptoms.

Interactive Organ Information: Learn about key body systems (Brain, Heart, Lungs, Digestive System) and how to maintain their health through interactive modals.

Oxygen & Breathing Section: Dedicated page with benefits of optimal oxygen levels and various breathing techniques for wellness.

About Us Page: Information about the project's mission and how it helps users.

Contact Us Page: A form for users to send questions, feedback, or suggestions.

Responsive Design: Optimized for seamless viewing and interaction across various devices (desktop, tablet, mobile).

Light/Dark Theme Toggle: (If currently implemented) Allows users to switch between light and dark modes for improved readability and user preference.

Technologies Used
Backend: Flask (Python Web Framework)

Machine Learning:

Python

Pandas (for data manipulation)

Scikit-learn (for TF-IDF Vectorization and Cosine Similarity)

Pickle (for model serialization)

Frontend:

HTML5

Tailwind CSS (for rapid styling and responsiveness)

JavaScript (for interactive elements like modals and theme toggle)

Setup Instructions
Follow these steps to set up and run the project locally:

Clone the Repository:

git clone <repository_url>
cd First_Aid_Knock

Create a Virtual Environment (Recommended):

python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

Install Dependencies:

pip install -r requirements.txt

(Make sure you have a requirements.txt file with Flask, pandas, scikit-learn listed. If not, create it: pip freeze > requirements.txt)

Prepare Data and Train ML Model:

Ensure you have your medicine_details.csv dataset in the data/ directory.

Run the train_model.py script to preprocess data and generate the TF-IDF vectorizer and matrix. This will create medicine_details_cleaned.csv in data/ and tfidf_vectorizer.pkl, medicine_uses_tfidf_matrix.pkl in models/.

python train_model.py

(Note: If train_model.py is not provided, you'll need to create it based on the ML logic discussed.)

Run the Flask Application:

python app.py

Access the Application:
Open your web browser and go to http://127.0.0.1:5000/

How to Use
Home Page: Enter a symptom in the search bar and click "Get Advice" to receive medicine recommendations and multi-system advice. You can also explore information about different organ systems by clicking on their respective icons.

Navigation Bar: Use the navigation links (Home, Oxygen, About Us, Contact) to explore different sections of the website.

Theme Toggle: (If implemented) Click the sun/moon icon in the header to switch between light and dark themes.

Contact Form: Fill out the form on the "Contact Us" page to send a message. The message details will be printed to the Flask console.

Disclaimer
This application provides general information and is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for any medical concerns, persistent symptoms, or before starting any new medication.

Future Enhancements (Potential Ideas)
Database Integration: Store contact messages and potentially expand medicine data in a proper database (e.g., SQLite, PostgreSQL).

User Accounts: Implement user registration and login for personalized experiences.

Advanced ML Models: Explore more sophisticated machine learning models for improved recommendation accuracy.

Search Functionality: Enhance search on the Oxygen and About Us pages.

Admin Panel: Create an admin interface to manage medicine data and view contact messages.

Accessibility Improvements: Further enhance accessibility features for all users.
