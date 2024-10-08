import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Inisialisasi
lemmatizer = WordNetLemmatizer()
ignore_words = ['?', '!']

# Load data
data_file = open('intents.json', 'r', encoding='utf-8').read()
intents = json.loads(data_file)

words = []
classes = []
documents = []

# Preprocessing
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

print(len(documents), "dokumen")
print(len(classes), "kelas", classes)
print(len(words), "kata unik yang telah dilematisasi", words)

training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
train_x = np.array([t[0] for t in training])
train_y = np.array([t[1] for t in training])

print("Data training telah dibuat")

# Pisahkan data menjadi data pelatihan dan data uji
train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

# Ubah data input menjadi bentuk yang sesuai untuk LSTM
# LSTM membutuhkan input berbentuk (samples, time_steps, features)
# Di sini kita anggap time_steps = 1
train_x = np.expand_dims(train_x, axis=1)
test_x = np.expand_dims(test_x, axis=1)

model = Sequential()

# Lapisan LSTM untuk menangkap dependensi sekuensial
model.add(LSTM(128, input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences=True))
model.add(Dropout(0.5))

model.add(LSTM(64))
model.add(Dropout(0.5))

# Lapisan output
model.add(Dense(len(train_y[0]), activation='softmax'))

adam = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Latih model dengan data pelatihan dan validasi dengan data uji
hist = model.fit(train_x, train_y, epochs=450, batch_size=8, verbose=2, validation_split=0.1, callbacks=[early_stopping])

# Evaluasi model dengan data uji
loss, accuracy = model.evaluate(test_x, test_y, verbose=0)
print(f"Loss: {loss}") 
print(f"Accuracy: {accuracy}") #kalau mau dalam bentuk persen > #accuracy_percent = accuracy * 100

model.save('bot_model.h5')
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

print("Training Model Selesai")

# Plot hasil pelatihan
plt.figure(figsize=(12, 4))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(hist.history['loss'], label='Training Loss')
plt.plot(hist.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

# Plot akurasi
plt.subplot(1, 2, 2)
plt.plot(hist.history['accuracy'], label='Training Accuracy')
plt.plot(hist.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.show()
