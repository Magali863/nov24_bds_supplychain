
# Supply Chain Satisfaction des clients

L'objectif de ce projet est de développer un modèle de machine learning capable d'analyser et de prédire la satisfaction client à partir des commentaires des utilisateurs. En analysant les données textuelles des avis et les indicateurs pertinents qui émergent des commentaires, nous cherchons à identifier des tendances positives ou négatives et à évaluer le niveau de satisfaction des clients.

Nous nous interrogeons notamment sur la manière d'améliorer la satisfaction client en procédant à une analyse rapide des commentaires des utilisateurs et en détectant les évolutions de la satisfaction à travers des indicateurs clés.


## Fonctionnalités du Projet

- Analyse des Commentaires : Extraction et traitement des données textuelles issues des avis clients.

- Prédiction de la Satisfaction : Utilisation de modèles de machine learning pour prédire le niveau de satisfaction des clients.
## Données Sources

Nous avons recueilli des informations sur le site Trustpilot, une plateforme dédiée aux avis sur les entreprises. Ainsi, nous avons constitué un ensemble de données comprenant 33 563 avis concernant trois sociétés de la catégorie ordinateur et téléphone qui sont les suivantes : Materiel.net, Recommerce.com et Rebuy.

La constitution de notre jeu de données a été effectuée à l'aide de la méthode de web scraping.
## Mode Opératoire du Projet

Cette procédure opérationnelle décrit les étapes pour créer un DataFrame à partir de données scrappées, effectuer le préprocessing, réaliser des visualisations, puis appliquer des techniques de machine learning et de deep learning afin de trouver la meilleure modélisation.

 
      Étape 1 : Création du DataFrame - Webscraping
      
- Collecte des données :
  > Utilisation de webscraping BeautifulSoup pour extraire les données souhaitées à partir d'une source web.
  > 
  > Convertir les données extraites en un DataFrame à l'aide de pandas.
>   

     Étape 2 : Préprocessing

- Nettoyage :
  > Supprimer les doublons et les valeurs manquantes.
  > 
  > Corriger les types de données (par exemple, convertir les dates en format datetime).

- Exploration de la base scrappée :
  > Utilisation des statistiques descriptives pour explorer les données.
>

        Étape 3 : Data Visualization

- Visualisation des distributions des variables.

- WordCloud pour visualiser les mots les plus fréquents dans les commentaires selon les notes de satisfaction client :
  
  > Notes de satisfaction client = 1 et 2
  > 
  > Note de satisfaction client = 3
  > 
  > Notes de satisfaction client = 4 et 5
> 

        Étape 4 : Feature Engineering

- Suppression des colonnes inutiles :
 
    > Identifier et supprimer les colonnes qui ne seront pas utiles pour la modélisation.
- Création de nouvelles caractéristiques :
  
     > Développer des nouvelles variables qui pourraient améliorer la performance du modèle.
- Analyse de Corrélation :
  
     > Analyser la corrélation entre les variables explicatives et la variable cible
 >

        Étape 5 : Modélisation 1

- Préparation des données :
  > Diviser les données en ensembles d'entraînement, de test et de validation.
  > Normaliser et encoder les données.
  > Equilibrer l’ensemble d’entrainement si le jeu de données est déséquilibré
   
- Modélisation :
  >Tester plusieurs modèles de classification : 
     RandomForestClassifier,
     LightGBM Classifier,
     Régression logistique.
  >
  > Optimiser les hyperparamètres à l'aide de RandomizedSearchCV.
  > 
  > Effectuer une validation croisée pour évaluer la robustesse du modèle.
  > 
  > Analyser les erreurs et interpréter les résultats
>

     Étape 6 : Modélisation 2


- Intégration de nouvelles caractéristiques NLP :
  >
  >Caractéristique pour détecter la négation dans le titre du commentaire.
  >
  > Caractéristique pour classer les sentiments en utilisant un modèle BERT ("nlptown/bert-base-multilingual-uncased-sentiment")

- Répéter la préparation et la modélisation :
  > Suivre les mêmes étapes de préparation des données et de modélisation que précédemment
>
            Étape 7 : Modélisation 3 Binaire

- Transformation de la variable cible en un modèle binaire :
  > Valeur 1 pour les notes 1 et 2.
  > 
  > Valeur 2 pour les notes 3, 4 et 5

- Modélisation :
  > Tester plusieurs modèles de classification : 
      RandomForestClassifier, 
      LightGBM Classifier
  > 
  > Effectuer l'optimisation des hyperparamètres, la validation croisée, et l'analyse des erreurs comme précédemment
> 

            Étape 8 : Modélisation Deep learning

-	Appliquer un modèle de réseau de neurones récurrent avec une couche LSTM (Long Short-Term Memory)
 
    >à l'aide de la bibliothèque Keras de TensorFlow. 



## Principales Bibliothèques et Outils Utilisés

Python : langage de programmation

Pandas : bibliothèque Python pour la manipulation et l'analyse de données

Scikit-learn : bibliothèque Python pour le machine learning

NLTK / SpaCy: bibliothèques Python pour le traitement du langage naturel (NLP).

Matplotlib / Seaborn : bibliothèques Python de visualisation de données en Python.

BeautifulSoup : bibliothèques Python utilisée pour le web scraping

WordCloud : bibliothèque Python utilisée pour générer des nuages de mots à partir de textes, permettant de visualiser les mots les plus fréquents dans un corpus.

Statsmodels : bibliothèque Python utilisée pour l'estimation de modèles statistiques et l'analyse des données.

Imbalanced-learn : bibliothèque Python pour le traitement des ensembles de données déséquilibrés.

Scipy : Utilisé pour des opérations scientifiques et techniques, y compris des outils pour l'optimisation et les statistiques.

GridSearchCV / RandSearchCV : Outil de Scikit-learn pour l'optimisation des hyperparamètres.
## Auteurs

- [@Magali864](https://www.github.com/Magali864)
- [@mcdieye](https://github.com/mcdieye)
- [@DonaBN](https://github.com/DonaBN)
- [@yassinetazit](https://github.com/yassinetazit)

  
