# -*- coding: utf-8 -*-
"""
Created on 2023-08-13 20:00:00

@author: Rajeeth-A
"""
### Importation des bibliothèques

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from tensorflow import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping

##########################
### Création du modèle ###
##########################

# Charger les données
file_name = "test.csv"
file_name_train = "train.csv"
df = pd.read_csv(file_name_train, sep=',')
test_df = pd.read_csv(file_name, sep=',')

# Préparation des données
print("5 premières lignes des données d'entraînement:")
print(df.head())
print("Taille des données d'entraînement:", df.shape)
print("Taille des données de test:", test_df.shape)
print("Valeurs manquantes dans les données d'entraînement:")
print(df.isnull().sum())
print("Valeurs manquantes dans les données de test:")
print(test_df.isnull().sum())

# Préparer les étiquettes et les caractéristiques
X = df.drop(["label"], axis=1)
y = df["label"]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# Normaliser les caractéristiques
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Remodeler les données pour le CNN
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
x_train = x_train.values.reshape(-1, 28, 28, 1)
x_test = x_test.values.reshape(-1, 28, 28, 1)

# Visualiser la répartition des labels
print("Répartition des labels:")
print(y.value_counts())
sns.countplot(y)
plt.show()

# Visualiser une seule image
image_unique = x_train[0][:, :, 0]
plt.imshow(image_unique, cmap='gray')
plt.show()

# Création du modèle

model = Sequential()


model.add(Conv2D(filters=32, kernel_size=(5,5), input_shape=(28, 28, 1), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(5,5), activation='relu',))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu',))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu',))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(BatchNormalization())

model.add(Flatten())

model.add(Dense(32, activation='relu'))

model.add(Dense(10, activation='softmax')) 


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#Paramètre du modèle
model.summary() 

# Callback pour arrêt anticipé
early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience = 3,
    verbose=1,
)

############################
### Evaluation du modèle ###
############################

# Entraînement du modèle
Entrainement = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=30,
    batch_size=128,
    validation_steps=x_test.shape[0] // 128,
    steps_per_epoch=x_train.shape[0] // 128,
    callbacks=[early_stopping]
)

# Métrique de performences
pertes = pd.DataFrame(Entrainement.history)
print(pertes)

# Courbe d'apprentissage
pertes[['accuracy', 'val_accuracy']].plot()
plt.show()

pertes[['loss', 'val_loss']].plot()
plt.show()

# Evaluation du modèle
score = model.evaluate(x_test,y_test,verbose=1)

print("Les paramètres associées à notre modèles : ", model.metrics_names)
print("La valeur de perte est : ", score[0])
print("La valeur d'exactitude est : ", score[1])

# Prédiction sur les données de test
test_df = test_df.values.reshape(-1, 28, 28, 1)
predictions = model.predict(test_df)

valeur_predite = np.argmax(predictions, axis=1)

print("prediction simple : ", predictions[0])
print(valeur_predite)

#Vérification
plt.imshow(test_df[0], cmap='gray')
plt.show()

