
# 📩 SMS Spam Classifier

A web-based SMS spam classifier built using **Streamlit**, **scikit-learn**, and **Natural Language Processing (NLP)** techniques. The app allows users to input a text message and predicts whether it is spam or not.

---

## 🚀 Features

- Classifies messages as **Spam** or **Not Spam**
- Clean and interactive **Streamlit UI**
- Preprocessing with:
  - Lowercasing  
  - Removing stopwords  
  - Stemming  
- Model trained on **TF-IDF vectorized** text  
- Built with a **pre-trained machine learning model** (Pickle)

---

## 🛠️ Technologies Used

- 🐍 Python  
- 🧮 Streamlit  
- 🤖 scikit-learn  
- 🧠 NLTK  
- 🧃 Pickle  

---

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/sms-spam-classifier.git
   cd sms-spam-classifier
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**:
   ```bash
   streamlit run code.ipynb
   ```

4. **(Optional)** If stopwords don’t work, download them using:
   ```python
   import nltk
   nltk.download('stopwords')
   ```

---

## 📁 Project Structure

```
├── model.pkl               # Trained ML model
├── vectorizer.pkl          # TF-IDF vectorizer
├── code.ipynb              # Streamlit notebook
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 📬 Contact

Created with ❤️ by Tanishak Bansal (https://github.com/tanishak0000007777)
🔗 [Connect on LinkedIn](https://www.linkedin.com/in/tanishak-bansal007)
