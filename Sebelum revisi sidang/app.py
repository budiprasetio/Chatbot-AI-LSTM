# app.py
from flask import Flask, jsonify, render_template, request
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model # type: ignore
import numpy as np
import json
import random
import pickle
from flask_mysqldb import MySQL
import MySQLdb.cursors
from dotenv import load_dotenv # type: ignore
import os
from flask_cors import CORS



# Muat variabel lingkungan dari file .env
load_dotenv()

def create_app():
    app = Flask(__name__)
    app.secret_key = os.getenv("SECRET_KEY")

    # Konfigurasi MySQL
    app.config['MYSQL_HOST'] = os.getenv("MYSQL_HOST")
    app.config['MYSQL_USER'] = os.getenv("MYSQL_USER")
    app.config['MYSQL_PASSWORD'] = os.getenv("MYSQL_PASSWORD")
    app.config['MYSQL_DB'] = os.getenv("MYSQL_DB")

    mysql = MySQL(app)

    # Mengaktifkan CORS untuk seluruh rute dalam aplikasi
    CORS(app)

    # Download nltk data
    nltk.download('punkt')
    nltk.download('wordnet')

    # Inisialisasi lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Load model dan data
    try:
        words = pickle.load(open('words.pkl', 'rb'))
        classes = pickle.load(open('classes.pkl', 'rb'))
        model = load_model('bot_model.h5')
        with open('intents.json', 'r', encoding='utf-8') as f:
            intents = json.load(f)
    except Exception as e:
        print(f"Error loading resources: {e}")
        exit()

    def clean_up_sentence(sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    def bow(sentence, words):
        sentence_words = clean_up_sentence(sentence)
        bag = [0] * len(words)
        for s in sentence_words:
            if s in words:
                bag[words.index(s)] = 1
        return np.array(bag)

    def predict_class(sentence, model):
        p = bow(sentence, words)
        p = np.expand_dims(p, axis=0)  # Ubah bentuk menjadi (1, len(words))
        p = np.expand_dims(p, axis=1)  # Tambah dimensi time_steps menjadi (1, 1, len(words))
        res = model.predict(p)[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return_list = []
        if results:
            for r in results:
                return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
        else:
            return_list.append({"intent": "no-response", "probability": "1.0"})
        
        return return_list

    def get_response(intents_list, intents_json):
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                return random.choice(i['responses'])
        return "Sorry, I didn't understand that."

    def save_to_database(user_message, bot_response):
        try:
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute('INSERT INTO chat_history (user_message, bot_response) VALUES (%s, %s)', (user_message, bot_response))
            mysql.connection.commit()
            cursor.close()
        except MySQLdb.Error as e:
            print(f"Error saving to database: {e}")

    @app.route('/')
    def home():
        return render_template('index.html')

    @app.route('/chat_history')
    def chat_history():
        try:
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute('SELECT * FROM chat_history')
            messages = cursor.fetchall()
            cursor.close()
            return render_template('chat_history.html', messages=messages)
        except MySQLdb.Error as e:
            print(f"Error fetching chat history: {e}")
            return render_template('chat_history.html', messages=[])

    @app.route('/get_response', methods=['POST'])
    def get_bot_response():
        try:
            message = request.json.get('message', '')
            if not message:
                return jsonify({'error': 'No message provided.'}), 400
            
            ints = predict_class(message, model)
            response = get_response(ints, intents)
            save_to_database(message, response)
            return jsonify({'response': response})
        except Exception as e:
            print(f"Error processing request: {e}")
            return jsonify({'error': 'An error occurred during processing the request.'}), 500

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
