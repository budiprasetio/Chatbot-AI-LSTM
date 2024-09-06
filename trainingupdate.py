import random
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import Sequential # type: ignore
from keras.layers import Dense, Dropout, LSTM # type: ignore
from keras.optimizers import Adam # type: ignore
from keras.callbacks import EarlyStopping, Callback # type: ignore
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt

# Custom callback untuk menghitung precision, recall, f1-score, dan accuracy pada setiap epoch
class MetricsCallback(Callback):
    def __init__(self, validation_data):
        super(MetricsCallback, self).__init__()
        self.validation_data = validation_data
        self.val_precision = []
        self.val_recall = []
        self.val_f1s = []
        self.val_accuracy = []

    def on_epoch_end(self, epoch, logs=None):
        val_predict = np.argmax(self.model.predict(self.validation_data[0]), axis=1)
        val_targ = np.argmax(self.validation_data[1], axis=1)
        _val_precision, _val_recall, _val_f1, _ = precision_recall_fscore_support(val_targ, val_predict, average='weighted')
        _val_accuracy = accuracy_score(val_targ, val_predict)

        self.val_precision.append(_val_precision)
        self.val_recall.append(_val_recall)
        self.val_f1s.append(_val_f1)
        self.val_accuracy.append(_val_accuracy)

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

# Pisahkan data menjadi data pelatihan (train), validasi (val), dan uji (test)
train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.1, random_state=42)

# Ubah data input menjadi bentuk yang sesuai untuk LSTM
train_x = np.expand_dims(train_x, axis=1)
val_x = np.expand_dims(val_x, axis=1)
test_x = np.expand_dims(test_x, axis=1)

# Hitung class weights untuk menangani imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(train_y.argmax(axis=1)), y=train_y.argmax(axis=1))
class_weights = dict(enumerate(class_weights))

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
metrics_callback = MetricsCallback(validation_data=(val_x, val_y))

# Latih model dengan data pelatihan dan validasi
hist = model.fit(train_x, train_y, epochs=450, batch_size=8, verbose=2, validation_data=(val_x, val_y), callbacks=[early_stopping, metrics_callback], class_weight=class_weights)

# Evaluasi model dengan data uji
loss, accuracy = model.evaluate(test_x, test_y, verbose=0)
print(f"Loss: {loss}") 
print(f"Accuracy: {accuracy:.2%}")

# Hitung metrik tambahan untuk data uji
y_pred = model.predict(test_x)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(test_y, axis=1)

precision, recall, f1, _ = precision_recall_fscore_support(y_true_classes, y_pred_classes, average='weighted')
test_accuracy = accuracy_score(y_true_classes, y_pred_classes)

# Cetak metrik dalam bentuk persen
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1-Score: {f1:.2%}")
print(f"Test Accuracy: {test_accuracy:.2%}")

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

# Plot Precision
plt.figure(figsize=(8, 6))
plt.plot(np.array(metrics_callback.val_precision) * 100, label='Precision', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Precision (%)')
plt.ylim(0, 100)
plt.grid(True)
plt.title('Precision Over Epochs')
plt.legend()
plt.show()

# Plot Recall
plt.figure(figsize=(8, 6))
plt.plot(np.array(metrics_callback.val_recall) * 100, label='Recall', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Recall (%)')
plt.ylim(0, 100)
plt.grid(True)
plt.title('Recall Over Epochs')
plt.legend()
plt.show()

# Plot F1-Score
plt.figure(figsize=(8, 6))
plt.plot(np.array(metrics_callback.val_f1s) * 100, label='F1-Score', color='green')
plt.xlabel('Epochs')
plt.ylabel('F1-Score (%)')
plt.ylim(0, 100)
plt.grid(True)
plt.title('F1-Score Over Epochs')
plt.legend()
plt.show()

# Plot Validation Accuracy
plt.figure(figsize=(8, 6))
plt.plot(np.array(metrics_callback.val_accuracy) * 100, label='Validation Accuracy', color='red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
plt.grid(True)
plt.title('Validation Accuracy Over Epochs')
plt.legend()
plt.show()

# Hitung confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)

# Dapatkan label unik dari prediksi dan data uji
unique_labels = np.unique(np.concatenate((y_true_classes, y_pred_classes)))

# Filter kelas yang relevan
filtered_classes = [classes[i] for i in unique_labels]

# Cetak confusion matrix di terminal
print("Confusion Matrix:")
print(cm)

# Tampilkan metrik lain
print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=filtered_classes))

# Plot confusion matrix
plt.figure(figsize=(8, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=filtered_classes)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()
