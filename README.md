📩 SMS Spam Classifier
A web-based SMS spam classifier built using Streamlit, scikit-learn, and Natural Language Processing (NLP) techniques. The app allows users to input a text message and predicts whether it is spam or not.

🚀 Features
Classifies messages as Spam or Not Spam

Clean and interactive Streamlit UI

Preprocessing with:

Lowercasing

Removing stopwords

Stemming

Model trained on TF-IDF vectorized text

Built with a pre-trained machine learning model (pickle)

🛠️ Technologies Used
Python 🐍

Streamlit 📱

scikit-learn 🤖

NLTK 🧠

Pickle 🧃

📦 How to Run Locally
Clone this repo:

bash
Copy
Edit
git clone https://github.com/your-username/sms-spam-classifier.git
cd sms-spam-classifier
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Download NLTK stopwords (if not already installed):

python
Copy
Edit
import nltk
nltk.download('stopwords')
Run the app:

bash
Copy
Edit
streamlit run code.ipynb
📁 Project Structure
bash
Copy
Edit
├── model.pkl               # Trained ML model
├── vectorizer.pkl          # TF-IDF vectorizer
├── code.ipynb              # Main Streamlit notebook
├── requirements.txt        # Required Python libraries
└── README.md               # Project documentation
📸 Demo

🙌 Acknowledgements
Streamlit

scikit-learn

NLTK

Spam dataset (SMS Spam Collection from UCI repository)

📬 Contact
Created with ❤️ by Tanishak Basnal
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/tanishak-bansal007)


