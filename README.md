# Projet Kaggle : Digit Recognizer | Realisé par Rajeeth-A

## Description :

Site du projet : https://www.kaggle.com/competitions/digit-recognizer

Dans un monde où de nombreuses données sont disponibles et où l'écriture manuscrite continue de jouer un rôle important, la capacité des machines à reconnaître et à interpréter les caractères manuscrits est cruciale. Imaginez un système capable de déchiffrer des notes manuscrites, de trier automatiquement des chèques bancaires ou de numériser des documents historiques. C'est dans ce contexte que nous nous plongeons dans le projet fascinant de la reconnaissance des chiffres manuscrits. 

Dans ce projet, notre objectif est d’identifier correctement des nombres parmi une collection de dizaines de milliers
d’images manuscrites. Tout au long de ce projet, nous utiliserons Keras qui semble être un outil simple, polyvalent et
puissant pour le Deep Learning et plus spécifiquement pour les réseaux de neurones convolutifs. Keras dispose également
de nombreux guides de documentation et de développement. On présente dans ce projet la base de données MNIST
("Modified National Institute of Standards and Technology"). Depuis sa publication en 1999, ce jeu de données classique
dans le domaine du Deep Learning et est composé d’images manuscrites a été utilisé comme référence pour évaluer les
performances des algorithmes de classification. L’ensemble des données MNIST contient des chiffres manuscrits uniques
de 0 à 9.

## Réseaux de Neurones Convolutifs pour la Classification d'Images
### Contenue:
1. `Analyse des données`
2. `Traitement des données`
3. `Visualisation des Données Images`
4. `Remodelage des Données`
5. `Création du Modèle`
6. `Entraînement du Modèle`
7. `Évaluation du Modèle`
8. `Prédiction d'une image donnée`

### Pré-requis

Keras qui semble être un outil simple, polyvalent et puissant pour le Deep Learning et plus spécifiquement pour les réseaux de neurones convolutifs. 
Keras dispose également de nombreux guides de documentation et de développement.

Documentation Keras : https://keras.io

### Analyse des données

- Représentation des 5 premiers lignes du Dataset
- Vérication des données (NAN)
- Vérification de la forme des données

### Traitement des données

- Définition des caractéristiques et des labels.
- Normalisation des données
- Encodage One-Hot

### Visualisation des Données Images

- Représentation d'une image unique

### Création du modèle

- Couche de convolution
- Couche de Pooling
- Applatissement de l'image
- CallBack

### Entraînement du modèle

- Evaluation des métriques de performences
- Représentation des coubes d'apprentissage
- Evaluation du modèle

## Conclusion

Notre réseau de neurones convolutifs semble fonctionner avec une précision de 99.4% de classification correcte.