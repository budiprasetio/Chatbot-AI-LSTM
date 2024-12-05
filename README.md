# 🤖 Chatbot AI with LSTM  

*A Deep Learning-powered chatbot application built using LSTM in Python.*

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)  
![Flask](https://img.shields.io/badge/Flask-Framework-black?style=for-the-badge&logo=flask)  
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)  

---

## 📌 Overview  
This project is an **AI chatbot** application powered by **Long Short-Term Memory (LSTM)** networks. The chatbot is designed to classify user intents and provide appropriate responses using pre-trained data. It includes a web interface developed with **Flask** for user interaction and supports dynamic retraining.

---

## ⚙️ Features  
- **AI-Powered**: Leverages LSTM for natural language understanding.  
- **Web Interface**: A clean and interactive UI powered by Flask.  
- **Customizable**: Easily add or update intents via `intents.json`.  
- **Lightweight Database**: Uses SQLite for efficient data storage.  
- **Extensible**: Supports further customization and integration with APIs.  

---

## 📂 Project Structure  
```plaintext
📦 chatbot-ai-lstm
├── app.py                # Flask server script
├── intents.json          # Dataset of intents, patterns, and responses
├── bot_model.h5          # Trained LSTM model file
├── classes.pkl           # Encoded class labels
├── words.pkl             # Tokenized words
├── finalchatbot.sql      # MySQL database
├── requirements.txt      # Python dependencies
├── trainingupdate.py     # Model training script
├── templates/            # Frontend HTML files
├── static/               # Static assets (CSS, JS)
└── .gitignore            # Files to ignore in version control
```
---

## 🚀 Getting Started  

Follow these steps to set up and run the project on your local machine:

---

### Prerequisites  
Make sure you have the following installed:  
- **Python** (3.8 or higher)  
- **Git**  

---

### Setup  

1. **Clone the Repository**  
   Use the following command to clone the project:  
   ```bash
   git clone https://github.com/your-repo/chatbot-ai-lstm.git
   cd chatbot-ai-lstm
   ```
2. **Install Dependencies**
   Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. **Train or Update the Model**
   To train the LSTM model or update it with new data, run:
    ```bash
   python trainingupdate.py
    ```
4. **Run the Application**
    ```bash
   python app.py
    ```
    The chatbot application will be accessible at http://127.0.0.1:5000.

---
## 💻 How It Works
Component	Description
Intents	intents.json contains predefined intents, patterns, and responses for training.
Model	The LSTM model (bot_model.h5) is trained to classify user inputs based on the patterns provided.
Backend	Flask manages API requests and handles the interaction between the model and the web interface.
Frontend	HTML, CSS, and JavaScript power the chatbot UI served through Flask templates.
Database	MySQL (finalchatbot.sql) is used for lightweight data storage.

---

🖥️ Demo
Web Interface
Below is a preview of the chatbot's user interface:

---

## 🔮 Planned Enhancements
Future updates and improvements for this project:

Multi-language support to cater to diverse audiences.
API integration for dynamic and real-time responses.
Improved UI/UX design with modern frameworks.
Expanded dataset for better chatbot performance.

---

🛠️ Tech Stack
Here are the main technologies used in this project:

Backend: Flask, TensorFlow, Keras
<br/>
Frontend: HTML, CSS, JavaScript
<br/>
Database: MySQL
<br/>
Languages: Python

---
   
