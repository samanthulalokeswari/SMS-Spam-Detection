# SMS Shield: Intelligent SMS Spam Detection

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange" />
  <img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

An intelligent SMS classification application that identifies spam and legitimate messages using Machine Learning and an interactive Streamlit interface.

## 🚀 Overview

SMS Shield analyzes text messages and predicts whether they are **Spam** or **Ham (Not Spam)**. The project uses text feature extraction and a trained classification model to help users identify potentially unwanted or fraudulent messages.

## ✨ Features

- 📩 SMS spam and non-spam classification
- 🤖 Machine Learning-based text analysis
- ⚡ Fast predictions using a saved model
- 💻 Interactive Streamlit web interface
- 📊 Model evaluation using classification metrics
- 🔄 Reproducible model training pipeline

## 🛠️ Tech Stack

- **Language:** Python
- **Data Processing:** Pandas
- **Feature Extraction:** TF-IDF
- **Machine Learning:** Scikit-learn, Multinomial Naive Bayes
- **Web Framework:** Streamlit
- **Visualization:** Matplotlib, Seaborn
- **Model Persistence:** Joblib

## 📂 Project Structure

```text
SMS-Spam-Detection/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── spam_model.pkl
├── metrics.json
└── data/
    ├── SMSSpamCollection
    └── readme
```

## ⚙️ Installation and Usage

**1. Clone the repository**

```bash
git clone https://github.com/YOUR-USERNAME/SMS-Spam-Detection.git
cd SMS-Spam-Detection
```

**2. Create and activate a virtual environment**

```powershell
python -m venv venv
.\venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the application**

```bash
streamlit run app.py
```

## 📈 Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 96.71% |
| Spam Precision | 100.00% |
| Spam Recall | 73.44% |
| Spam F1-Score | 84.68% |

*Results are from the current held-out test set. Performance may vary on new messages.*

## 📚 Dataset

**SMS Spam Collection Dataset**

Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)

The dataset contains labeled SMS messages used to train and evaluate the classification model.

## 🔮 Future Enhancements

- Improve detection of sophisticated scam messages.
- Add suspicious-link and keyword explanations.
- Support multilingual SMS classification.
- Deploy the application for public access.

## 👩‍💻 Author

**Your Name**

Machine Learning Enthusiast | Aspiring Software Developer

GitHub: [Your GitHub Profile](https://github.com/YOUR-USERNAME)

---

<p align="center">
  <strong>Built with Python and Machine Learning ❤️</strong>
</p>