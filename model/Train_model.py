import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from keras import Sequential
from keras import layers
from keras.layers import Dense
from keras import utils
from keras.utils import to_categorical
from sklearn.metrics import precision_score, recall_score, f1_score
# Charger les données
df = pd.read_excel('./stress_data.xlsx')
# Préparer les données
X = df.drop(columns=['Stress_State', 'Stress_Score'])
y = df['Stress_State']
# Encoder les étiquettes de classe
le = LabelEncoder()
y = le.fit_transform(y)
# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Convertir les étiquettes de classe en catégories one-hot
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# Créer un modèle
model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation='softmax')) # 2 pour 'stress' et 'non stress'
# Compiler le modèle
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Entraîner le modèle
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)
# Évaluer le modèle
loss, accuracy = model.evaluate(X_test, y_test)
print('Test accuracy:', accuracy)
# Obtenir les prédictions du modèle
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1) # convertir les catégories one-hot en étiquettes
# Convertir y_test en étiquettes
y_test_labels = np.argmax(y_test, axis=1)
# Calculer les métriques
precision = precision_score(y_test_labels, y_pred, average='macro')
recall = recall_score(y_test_labels, y_pred, average='macro')
fscore = f1_score(y_test_labels, y_pred, average='macro')
print('Macro Precision:', precision)
print('Macro Recall:', recall)
print('Macro F-score:', fscore)
# Calculer les métriques avec l'average='micro'
micro_precision = precision_score(y_test_labels, y_pred, average='micro')
micro_recall = recall_score(y_test_labels, y_pred, average='micro')
micro_fscore = f1_score(y_test_labels, y_pred, average='micro')
print('Micro Precision:', micro_precision)
print('Micro Recall:', micro_recall)
print('Micro F-score:', micro_fscore)
