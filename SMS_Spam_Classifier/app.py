import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

# Download only stopwords (not punkt)
nltk.download('stopwords')

# Load model and vectorizer
ps = PorterStemmer()
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Text preprocessing
def transform_text(text):
    text = text.lower()
    text = text.split()
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(ps.stem(i))
    return " ".join(y)

# --- Streamlit UI ---
st.markdown(
    """
    <style>
        h1 {
            font-weight: 700;
            font-size: 3rem;
            color: #111827;
            margin-bottom: 0.25rem;
            text-align: center;
        }
        .subheading {
            font-weight: 500;
            font-size: 1.125rem;
            color: #6b7280;
            text-align: center;
            margin-bottom: 2rem;
        }
        .stTextInput > label {
            font-weight: 600;
            color: #374151;
        }
        .stTextInput > div > div > input {
            background-color: #f9fafb;
            border: 1px solid #d1d5db;
            border-radius: 10px;
            padding: 0.75rem 1rem;
            font-size: 1rem;
            color: #111827;
            transition: border-color 0.3s;
        }
        .stTextInput > div > div > input:focus {
            border-color: #2563eb;
            outline: none;
            box-shadow: 0 0 0 3px rgb(37 99 235 / 0.4);
        }
        .stButton > button {
            font-weight: 600;
            font-size: 1rem;
            padding: 0.6rem 1.4rem;
            border-radius: 8px;
            border: none;
            cursor: pointer;
            transition: background-color 0.3s ease;
            min-width: 120px;
            background-color: #2563eb;
            color: white;
        }
        .stButton > button:hover {
            background-color: #1e40af;
        }
        .result-message {
            font-weight: 700;
            font-size: 1.25rem;
            text-align: center;
            padding: 1rem 1.5rem;
            border-radius: 12px;
            max-width: 480px;
            margin-left: auto;
            margin-right: auto;
            box-shadow: 0 2px 12px rgb(0 0 0 / 0.05);
        }
        .spam {
            color: #b91c1c;
            background-color: #fee2e2;
            border: 1px solid #fca5a5;
        }
        .not-spam {
            color: #166534;
            background-color: #d1fae5;
            border: 1px solid #6ee7b7;
        }
        details {
            max-width: 480px;
            margin: 2rem auto 0 auto;
            font-size: 0.9rem;
            color: #4b5563;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 1rem 1.5rem;
            background-color: #f9fafb;
            box-shadow: 0 2px 6px rgb(0 0 0 / 0.03);
            cursor: pointer;
            transition: background-color 0.3s;
        }
        details[open] {
            background-color: #ffffff;
        }
        details summary {
            font-weight: 600;
            outline: none;
        }
        footer {
            text-align: center;
            font-size: 0.8rem;
            color: #9ca3af;
            margin-top: 3rem;
            margin-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1>📩 SMS Spam Classifier</h1>", unsafe_allow_html=True)
st.markdown("<div class='subheading'>Enter a message below to determine if it's spam or not.</div>", unsafe_allow_html=True)

input_sms = st.text_input("Your message here", placeholder="e.g., Win a free iPhone by clicking this link!")

if st.button('🚀 Predict'):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a valid message.")
    else:
        with st.spinner("Analyzing..."):
            # Preprocess
            transformed_text = transform_text(input_sms)
            # Vectorize
            vec_input = tfidf.transform([transformed_text])
            # Predict
            result = model.predict(vec_input)[0]

            if result == 1:
                st.markdown(
                    "<div class='result-message spam'>❗ This message is <strong>Spam</strong>.</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div class='result-message not-spam'>✅ This message is <strong>Not Spam</strong>.</div>",
                    unsafe_allow_html=True,
                )

# How does this classifier work?
st.markdown(
    """
    <details>
        <summary>How does this classifier work?</summary>
        <p>The SMS Spam Classifier uses a series of steps to determine whether a message is spam or not:</p>
        <ol>
            <li><strong>Data Cleaning:</strong> The dataset is cleaned by removing unnecessary columns and handling missing values.</li>
            <li><strong>Exploratory Data Analysis (EDA):</strong> The distribution of spam and ham messages is analyzed, revealing an imbalance in the dataset.</li>
            <li><strong>Text Preprocessing:</strong> This includes lowercasing, tokenization, removing special characters, stop words, and stemming.</li>
            <li><strong>Model Building:</strong> Various models were tested, including Naive Bayes, Logistic Regression, SVM, Decision Trees, and more.</li>
            <li><strong>Model Evaluation:</strong> The Multinomial Naive Bayes model with TF-IDF vectorization was selected due to its high precision (100%).</li>
            <li><strong>Performance Comparison:</strong> A comparison of multiple classifiers was conducted, showing the performance metrics for accuracy and precision.</li>
        </ol>
    </details>
    """,
    unsafe_allow_html=True,
)

# Footer
st.markdown(
    "<footer>© 2024 Tanishak Bansal | <a href='https://github.com/your-repo' target='_blank'>GitHub Repository</a></footer>",
    unsafe_allow_html=True,
)
