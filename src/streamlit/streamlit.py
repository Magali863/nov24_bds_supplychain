
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from scipy.stats import chi2_contingency 
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#from imblearn.metrics import classification_report_imbalanced  
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler, SMOTE
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
import lightgbm as lgb
from lightgbm import LGBMClassifier
import shap
import joblib
import os
from PIL import UnidentifiedImageError, Image
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import confusion_matrix
le = LabelEncoder()
le2 = LabelEncoder()
vectorizer = TfidfVectorizer()
vectorizer2 = TfidfVectorizer()




st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">', unsafe_allow_html=True)


#scraping
#df_scraping=pd.read_csv ("C:/Users/magal/Documents/ProjetStreamlit/df_projet_truspilot2_etape2.csv", sep=',')
df_scraping=pd.read_csv ("df_projet_truspilot2_etape2.csv", sep=',')


#base definitive
# df=pd.read_csv("C:/Users/magal/Documents/ProjetStreamlit/derniere_base.csv", sep=';')
# df_binaire=pd.read_csv("C:/Users/magal/Documents/ProjetStreamlit/derniere_base.csv", sep=';')
df=pd.read_csv("derniere_base.csv", sep=';')
df_binaire=pd.read_csv("derniere_base.csv", sep=';')


#base scrapée certideal pour démo
# df_demo=pd.read_csv("C:/Users/magal/Documents/ProjetStreamlit/demo_certideal.csv", sep=';')
# df_demo_binaire=pd.read_csv("C:/Users/magal/Documents/ProjetStreamlit/demo_certideal.csv", sep=';')
df_demo=pd.read_csv("demo_certideal.csv", sep=';')
df_demo_binaire=pd.read_csv("demo_certideal.csv", sep=';')


#base pour Feature Engineering
#df_feat = pd.read_csv("C:/Users/magal/Documents/ProjetStreamlit/df2.projet_truspilot2_etape3.csv", sep=',')
df_feat = pd.read_csv("df2.projet_truspilot2_etape3.csv", sep=',')




st.sidebar.image("logo4.png",width=180)
st.sidebar.write("")
st.sidebar.markdown("<h1 style='color: #00008B;font-size: 30px;'>Projet Supply Chain Satisfaction Client</h1>", unsafe_allow_html=True)
st.sidebar.write("")
#st.sidebar.image("logo3.png", width=200,use_column_width='auto')
st.sidebar.title("Sommaire")
# Ajouter une image au sommaire

pages=["Contexte","Objectif","Web Scraping","Exploration", "Feature engineering","Data Visualisation", "Modélisation","Démo","Auteurs"]

page=st.sidebar.radio("Aller vers", pages)
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.image("Truspilot.png", width=200,use_container_width='auto')

########################  PAGE 0 #########################################

if page == pages[0] : 
  
  st.markdown("""

<h2 style='color: #1f487e;font-size: 30px; text-align: center;'>Contexte</h2>
<p style='text-align: center;'></p>
<hr style="border: 2px solid #1f487e;">
<p>
Dans un environnement commercial de plus en plus compétitif, <strong>la satisfaction client</strong> est essentielle pour les entreprises souhaitant se démarquer.

<strong>Les avis et les évaluations</strong> des utilisateurs représentent une source d'information essentielle pour les entreprises, leur permettant de mieux cerner les attentes, les préférences et les insatisfactions de leurs clients.
En analysant les commentaires et en suivant les notes de satisfaction, les entreprises peuvent s'ajuster aux attentes des clients, améliorer l'expérience utilisateur et optimiser leurs services, tout en réduisant le taux d'attrition.               
L'amélioration continue de la satisfaction client favorise la fidélisation, ce qui conduit le plus souvent à une augmentation des revenus à long terme de l’entreprise.
<p>          
La satisfaction client est donc un enjeu majeur pour toute entreprise. Cependant, traiter cette masse de données textuelles de manière efficace et rapide représente un défi.
<p>
         
<br>
</p>
""", unsafe_allow_html=True)
  
  #Ce projet vise à utiliser des techniques de data science pour <strong>évaluer la satisfaction client</strong>, en analysant les commentaires des clients.
  # Créer trois colonnes
  col1, col2, col3= st.columns([100,500, 100])

   # Placer l'image dans la colonne du milieu
  with col2:
   
    st.image("contexte.png",width=600)

########################  PAGE 1 #########################################

if page == pages[1] : 
   
  st.markdown("""
<h2 style='color: #1f487e;font-size: 30px; text-align: center;'>Objectif </h2>
<p style='text-align: center;'></p>
<hr style="border: 2px solid #1f487e;">
<p>
L'objectif de ce projet est de <strong>développer un modèle de machine learning capable d'analyser et de prédire la satisfaction client</strong> à partir des commentaires des utilisateurs.       

Nous allons analyser les données textuelles des avis et les informations clés issues des commentaires afin
<div style="margin-left: 50px;">              
  ★ <strong>d’identifier des tendances et des sentiments, qu'ils soient positifs ou négatifs</strong>

  ★ <strong>de mesurer la satisfaction client sur une échelle de satisfaction allant de 1 à 5. </strong>
</div>  
Cette échelle est largement utilisée en raison de sa simplicité d'interprétation.
<br>
<br>
Grâce à nos analyses, nous espérons fournir des recommandations concrètes aux entreprises pour améliorer la satisfaction de leurs clients.
<br>
<br>             
Pour ce faire, nous allons mettre en place une méthodologie qui inclut la collecte des données, le prétraitement des commentaires, l'application d'algorithmes de machine learning, et enfin une démonstration avec notre modèle.
<p>
</p>
""", unsafe_allow_html=True)

  st.image("image_ordi2.png")

########################  PAGE 2 #########################################

if page == pages[2] : 

  st.markdown("""
<h2 style='color: #1f487e;font-size: 30px; text-align: center;'>Création de la Base de données</h2>
<p style='text-align: center;'></p>
<hr style="border: 2px solid #1f487e;">
<p>
Nous avons recueilli des informations sur le site Trustpilot, une plateforme dédiée aux avis sur les entreprises.     

Ainsi, nous avons constitué un ensemble de données composé de 33 563 avis concernant trois sociétés de la catégorie 'ordinateur et téléphone', qui sont les suivantes : 
<p>
</p>
""", unsafe_allow_html=True)
  st.write("")
  st.write("")
  st.image("3 entreprises.png")
  st.write("")
  st.write("")
  st.write("La constitution de notre jeu de données a été effectuée à l'aide de la méthode de web scraping avec BeautifulSoup.")

  st.markdown("""
**Aperçu des premières lignes de notre jeu de données scrapé**
""")
            
  st.dataframe(df_scraping.head())
  st.write(df_scraping.shape)

########################  PAGE 3 #########################################


if page == pages[3] : 
  
   st.markdown("""
  <h2 style='color: #1f487e;font-size: 30px;text-align: center;'>Exploration du jeu de données</h2>
  <p style='text-align: center;'></p>
  <hr style="border: 2px solid #1f487e;">
  <p>
  Dans notre jeu de données comprenant 33 563 avis et 11 variables, nous avons identifié 40 doublons et 3 206 valeurs manquantes.
  <p>
  </p>
  """, unsafe_allow_html=True) 
   
   st.write("")
  
   # Mot à survoler
   entreprise = "Nom de l'entreprise 📊"
   page = "Page"
   noteG="Note globale"
   nombre_avis="Nombre total avis"
   client="Nom client"
   pays='Pays'
   date='Date de publication 📊'
   nombre_avis_c="Nombre d'avis du client 📊"
   note="Note client 📊"
   titre="Titre commentaire"
   commentaire="Commentaire 📊"


  # Fonction pour créer un graphique pour entreprise
   def create_plot():
    materiel = (df_scraping['Nom entreprise'] == "Materiel.net").sum()
    Rebuy = (df_scraping['Nom entreprise'] == "Rebuy").sum()
    Recommerce = (df_scraping['Nom entreprise'] == "Recommerce.com").sum()

   # Création des listes de labels et de tailles
    labels = ['Materiel.net', 'Rebuy', 'Recommerce.com']
    sizes = [materiel, Rebuy, Recommerce]
    colors = ['#ff9999', '#66b3ff', '#99ff99']  # Couleurs pour chaque secteur
    explode = (0.1, 0, 0)  # Mettre en avant Materiel.net

  # Création du diagramme circulaire
    plt.figure(figsize=(10, 6))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
          autopct='%1.1f%%', shadow=True, startangle=140 , textprops={'fontsize': 16})
    plt.axis('equal')  
    plt.title('Répartition des entreprises',fontsize=20)
    #st.pyplot(plt)
    plt.grid()
    plt.tight_layout()


   st.markdown(
    """
    <style>
    .expander {
        background-color: #e0f7fa; /* Couleur bleu pâle */
        padding: 10px;
        border-radius: 5px;
    }
    .stExpander {
        background-color: transparent; /* Assurez-vous que l'expander n'a pas de fond par défaut */
    }
    </style>
    """,
    unsafe_allow_html=True)

# Créer l'expander avec le nom d'entreprise
   with st.expander(entreprise, expanded=False):
    # Ajouter une classe pour le style
      st.markdown('<div class="expander">', unsafe_allow_html=True)

      col13, col23, col33 = st.columns([60, 100, 1])

      with col23:  # Colonne centrale pour le graphique
        create_plot()
        st.pyplot(plt)

      with col13:  # Colonne gauche pour le texte
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write("Materiel.net ➡️ 26 433")
        st.write("Rebuy ➡️ 4 550")
        st.write("Recommerce.com ➡️ 2 580")

      with col33:  # Colonne droite vide ou pour tout autre contenu
        pass  # Vous pouvez ajouter d'autres contenus ici si nécessaire
    
    # Fermer la div pour le style
      st.markdown('</div>', unsafe_allow_html=True)


  #Créer l'expander page
   with st.expander(page, expanded=False):
    # Ajouter une classe pour le style
      st.markdown('<div class="expander">', unsafe_allow_html=True)
      st.write("URL du commentaire ")

  #Créer l'expander note globale
   with st.expander(noteG, expanded=False):
    # Ajouter une classe pour le style
      st.markdown('<div class="expander">', unsafe_allow_html=True)
      st.write("Note globale par entreprise")

  #Créer l'expander nombre_avis total
   with st.expander(nombre_avis, expanded=False):
    # Ajouter une classe pour le style
      st.markdown('<div class="expander">', unsafe_allow_html=True)
      st.write("Nombre total d'avis par entreprise")
  
  #Créer l'expander nom client
   with st.expander(client, expanded=False):
    # Ajouter une classe pour le style
      st.markdown('<div class="expander">', unsafe_allow_html=True)

  # Créer l'expander pays
   with st.expander(pays, expanded=False):
    # Ajouter une classe pour le style
      st.markdown('<div class="expander">', unsafe_allow_html=True)
      st.write("Pays où le commentaire a été rédigé")
      st.write("92 % des commentaires proviennent de la France")

   # Créer l'expander date de publication
   with st.expander(date, expanded=False):
    # Ajouter une classe pour le style
      st.markdown('<div class="expander">', unsafe_allow_html=True) 
      col15, col25, col35 = st.columns([60, 100, 1])
    # Ajouter une classe pour le style

      with col25:  # Colonne centrale pour le graphique
        # Nombre d'avis par année et par note
        df_scraping['Date de publication'] = pd.to_datetime(df_scraping['Date de publication'], errors='coerce')
        df_scraping['année'] = df_scraping['Date de publication'].dt.year
        an = df_scraping['année'].value_counts().sort_index()

        publication_counts = df_scraping.groupby(['année', 'Note client']).size().reset_index(name='count')
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=publication_counts, x='année', y='count', hue='Note client', marker='o',palette="Set1")
        plt.title("Évolution nombre d'avis selon la note attribuée par année")
        plt.xlabel("Année")
        plt.ylabel("Nombre d'avis")
        plt.xticks(rotation=45)  
        plt.grid()  
        plt.legend(title='Note client')  
        st.pyplot(plt)

      with col15:  # Colonne gauche pour le texte
        st.write(" ")
        st.write(" ")
        st.write("Période d'analyse : 2012 - 2024")
        st.write("📌Une concentration d'avis en 2020")
        st.write("📌L’évolution des notes suit la même tendance au fil des années")

      with col35:  # Colonne droite vide ou pour tout autre contenu
        pass  # Vous pouvez ajouter d'autres contenus ici si nécessaire
    

  #Créer l'expander Nombre d'avis du client
   with st.expander(nombre_avis_c , expanded=False):
      st.markdown('<div class="expander">', unsafe_allow_html=True)
      col14, col24, col34 = st.columns([60, 100, 1])
    # Ajouter une classe pour le style

      with col24:  # Colonne centrale pour le graphique
        
        plt.figure(figsize=(10, 5))
        plt.boxplot(df_scraping["Nombre d'avis du client"])
        #st.write(df_scraping.boxplot('Nombre total avis'))
        plt.title("Nombre d'avis du client")
        plt.ylabel('Nombre avis')
        plt.ylim(0, 130)
        st.pyplot(plt)

      with col14:  # Colonne gauche pour le texte
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write("📌 Cette variable affiche une grande amplitude, allant de 1 à 123 avis.")
      
      with col34:  # Colonne droite vide ou pour tout autre contenu
        pass  # Vous pouvez ajouter d'autres contenus ici si nécessaire
    
    # Fermer la div pour le style
      st.markdown('</div>', unsafe_allow_html=True)

  #Créer l'expander note client
   with st.expander(note , expanded=False):
      st.markdown('<div class="expander">', unsafe_allow_html=True)
      col16, col26, col36 = st.columns([60, 100, 1])
    # Ajouter une classe pour le style

      with col26:  # Colonne centrale pour le graphique
        
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(data=df_scraping, x='Note client', palette='viridis')
        plt.title('Répartition des notes clients')
        plt.xlabel('Note client')
        plt.ylabel('Nombre de clients')
        plt.xticks(rotation=0)
     # Supprime le contour du graphique
        sns.despine()
    # Déplace la légende à l'extérieur
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Affiche le nombre au-dessus des barres
        for p in ax.patches:
           ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom')
        plt.tight_layout()  
        st.pyplot(plt)

      with col16:  # Colonne gauche pour le texte
        st.write(" ")
        st.write("Les notes vont de 1 à 5 où « 5 » indique une satisfaction élevée.")
        st.write("")    
        st.write("📌Le jeu de données présente un déséquilibre, avec une dominance d’avis positifs, ce qui pourrait affecter les résultats du modèle.")

      with col36:  # Colonne droite vide ou pour tout autre contenu
        pass  # Vous pouvez ajouter d'autres contenus ici si nécessaire
    
    # Fermer la div pour le style
      st.markdown('</div>', unsafe_allow_html=True)


  # Créer l'expander titre commentaire
   with st.expander(titre, expanded=False):
    # Ajouter une classe pour le style
      st.markdown('<div class="expander">', unsafe_allow_html=True)
      

# Créer l'expander commentaire
   with st.expander(commentaire, expanded=False):
    # Ajouter une classe pour le style
      st.markdown('<div class="expander">', unsafe_allow_html=True)
      col17, col27, col37 = st.columns([60, 50, 1])

      with col27:  # Colonne centrale pour le graphique
        st.image("taillecommentaire.png")
        
      with col17:  # Colonne gauche pour le texte
        st.write(" ")
        st.write("La longueur des commentaires varie 1 à 1 000 mots par commentaire.")
        st.write(" ")
        st.write("📌Cette variabilité peut avoir un impact sur la qualité et la pertinence des informations extraites des commentaires.")
        st.write("")

      with col37:  # Colonne droite vide ou pour tout autre contenu
        pass  # Vous pouvez ajouter d'autres contenus ici si nécessaire

      # Fermer la div pour le style
      st.markdown('</div>', unsafe_allow_html=True)
############################ PAGE 4 ###############################################


if page == pages[4]:
    st.markdown("""
    <h2 style='color: #1f487e;font-size: 30px;text-align: center;'>Feature Engineering</h2>
    <p style='text-align: center;'></p>
    <hr style="border: 2px solid #1f487e;">
    <p>
    Pour améliorer le modèle, certaines colonnes ont été supprimées et des caractéristiques (features) ont été créées à partir des données existantes.
    <br>
    Nous avons effectué un prétraitement text mining sur les données textuelles, incluant la normalisation des caractères, la tokenisation, le Stop Words, la lemmatisation et la vectorisation.    
    <br>
    </p>
    """, unsafe_allow_html=True)
   
    



# Fonction pour créer un encadré avec un fond de couleur
    def styled_text(text, background_color):
       return f"""
    <div style="background-color: {background_color}; padding: 10px; border-radius: 5px; border: 1px solid #ccc; text-align: center; width: 100%; margin: 0;">
        <span style="font-weight: bold; font-size: 16px; color: navy;">
            {text}
        </span>
    </div>
    """

# texte à encadrer
    texte_a_encadrer = "Variables"

# Couleur de fond
    couleur_fond = "#f0f8ff"  # Exemple de couleur (Aliceblue)

# Afficher le texte encadré et stylisé
    st.markdown(styled_text(texte_a_encadrer, couleur_fond), unsafe_allow_html=True)

    st.write(" ")
#Ajout du choix pour masquer ou afficher le tableau des colonnes supprimées
    show_table = st.checkbox("❌ Variables supprimées", value=False)
    
    if show_table:
        Var_supprimees = {
            " ": list(range(1, 7)),  
            "Feature": [
                "Nom entreprise", "Page", "Note globale", "Nom du client", "Pays", "Date de publication"
                
            ]
        }
        df_var_supprimees = pd.DataFrame(Var_supprimees)

        # Affichage du tableau des colonnes supprimées
        st.table(df_var_supprimees.set_index(" "))



# Ajout du choix pour masquer ou afficher le tableau des colonnes supprimées
    st.write(" ")
    show_table = st.checkbox("✅ Variables retenues", value=False)

    if show_table:
        Var_retenues = {
            " ": list(range(1, 6)),  
            "Feature": [
                "Nombre total avis", "Nombre d'avis du client", "Note client", "Titre commentaire", "Commentaire"      
            ]
        }
        df_var_retenues = pd.DataFrame(Var_retenues)

        # Affichage du tableau des colonnes supprimées
        st.table(df_var_retenues.set_index(" "))


    # Ajout du choix pour les nouvelles varaibles
    st.write(" ")
    show_Var_crees = st.checkbox("🆕 Création de nouvelles variables")
    if show_Var_crees:
         # Mot à survoler
        Avis="Catégorie Nombre_Avis"
        mot="Catégorie Longueur_mot"
        phrase="Catégorie Nombre_phrases"
        point="Catégorie Point_exclamation"
        emoticon="Catégorie Emoticone"
        sentiment="Catégorie Sentiment"
        negation="Catégorie Négation"
        sentiment2="Catégorie Sentiment_dl"

         
    #Créer l'expander avis
        with st.expander(Avis, expanded=False):
      # Ajouter une classe pour le style
         st.markdown('<div class="expander">', unsafe_allow_html=True)
         col51, col52, col53 = st.columns([10, 30, 10])

        with col52:  # Colonne centrale pour le graphique
         st.image("image_nbre_Avis.png")
        
        with col51:  # Colonne gauche pour le texte
         st.write(" ")
         

        with col53:  # Colonne droite vide ou pour tout autre contenu
         pass  # Vous pouvez ajouter d'autres contenus ici si nécessaire
      
      # # Fermer la div pour le style
      #   st.markdown('</div>', unsafe_allow_html=True)


      #Créer l'expander mot
        with st.expander(mot, expanded=False):
      # Ajouter une classe pour le style
         st.markdown('<div class="expander">', unsafe_allow_html=True)
         col61, col62, col63 = st.columns([10, 30, 10])

        with col62:  # Colonne centrale pour le graphique
         st.image("image_class_longueur_mot.png")
        
        with col61:  # Colonne gauche pour le texte
         st.write(" ")
         st.write("")
         

        with col63:  # Colonne droite vide ou pour tout autre contenu
         pass  # Vous pouvez ajouter d'autres contenus ici si nécessaire
      
      #Créer l'expander phrase
        with st.expander(phrase, expanded=False):
      # Ajouter une classe pour le style
         st.markdown('<div class="expander">', unsafe_allow_html=True)
         st.write ("Nombre de phrases par commentaire")
         
         

      #Créer l'expander point
        with st.expander(point, expanded=False):
      # Ajouter une classe pour le style
         st.markdown('<div class="expander">', unsafe_allow_html=True)
         col71, col72, col73 = st.columns([10, 30, 10])

        with col72:  # Colonne centrale pour le graphique
         st.image("image_class_pt_exclam.png")
        
        with col71:  # Colonne gauche pour le texte
         st.write(" ")
         

        with col73:  # Colonne droite vide ou pour tout autre contenu
         pass  # Vous pouvez ajouter d'autres contenus ici si nécessaire
      
      #Créer l'expander emoticon
        with st.expander(emoticon, expanded=False):
      # Ajouter une classe pour le style
         st.markdown('<div class="expander">', unsafe_allow_html=True)
         col81, col82, col83 = st.columns([10, 30, 10])

        with col82:  # Colonne centrale pour le graphique
         st.image("image_Emoticone.png")
        
        with col81:  # Colonne gauche pour le texte
         st.write(" ")
         
        with col83:  # Colonne droite vide ou pour tout autre contenu
         pass  # Vous pouvez ajouter d'autres contenus ici si nécessaire
      
        #Créer l'expander sentiment
        with st.expander(sentiment, expanded=False):
      # Ajouter une classe pour le style
         st.markdown('<div class="expander">', unsafe_allow_html=True)
         col91, col92, col93 = st.columns([50, 70, 1])

        with col92:  # Colonne centrale pour le graphique
         st.image("image_class_sentiment.png")
        
        with col91:  # Colonne gauche pour le texte
         
         st.write("**Score polarité : -1 à 1**")
      
         st.write("Nous appliquons la fonction TxtBlob-fr pour évaluer le sentiment exprimé dans le commentaire")
         st.write("")

        with col93:  # Colonne droite vide ou pour tout autre contenu
         pass  # Vous pouvez ajouter d'autres contenus ici si nécessaire


         #Créer l'expander negation
        with st.expander(negation, expanded=False):
      # Ajouter une classe pour le style
         st.markdown('<div class="expander">', unsafe_allow_html=True)
         st.write ("Cette variable **repère la négation présente dans les titres des commentaires**, qui sont plus courts et plus clairs. À la suite de l'analyse des erreurs, nous avons observé que **la négation et le mot 'non'** n'étaient pas pris en compte lors de la tokenisation. Or, cette notion est nécessaire pour prédire une note de satisfaction.")
         

         #Créer l'expander sentiment2
        with st.expander(sentiment2, expanded=False):
      # Ajouter une classe pour le style
         st.markdown('<div class="expander">', unsafe_allow_html=True)
        
         st.write ("Cette variable provient du **modèle BERT (nlptown/bert-base-multilingual-uncased-sentiment)**, qui évalue les sentiments exprimés dans les commentaires. Ce modèle attribue une **note de sentiment sur une échelle de 1 à 5**, en considérant à la fois le contenu et le titre de chaque commentaire.")
    st.write(" ")
    st.write(" ")

    st.markdown("""
    Après avoir analysé les caractéristiques de notre jeu de données et s'être assuré que les variables soient corrélées à la variable cible. Nous avons préparé les données pour les intégrer dans notre modèle de prédiction des notes.
      <br>
      """, unsafe_allow_html=True)
    
     # texte à encadrer
    texte_a_encadrer2 = "Préparation et Traitement des Données"

   # Couleur de fond
    couleur_fond = "#f0f8ff"  # Exemple de couleur (Aliceblue)

   # Afficher le texte encadré et stylisé
    st.markdown(styled_text(texte_a_encadrer2, couleur_fond), unsafe_allow_html=True)

    st.write(" ")
 
    st.markdown("""
    <div style="margin-left: 30px;">     
    ✂️ <strong> Séparation du jeu de données</strong> 
    </div> 
      """, unsafe_allow_html=True)
    
    

    #Ajout du choix pour masquer ou afficher le tableau des colonnes supprimées
    # Créez une ligne avec des colonnes
    col1, col2 = st.columns([1, 6])  # Ajustez la largeur des colonnes si nécessaire

    with col2:
    # Ajoutez la case à cocher ici
      show_table = st.checkbox("🔍", value=False)

# Ajoutez le reste de votre code ici
    if show_table:
        st.image("separation.png")
    
   
    st.markdown("""
      <div style="margin-left: 30px;">        
      🔄 <strong> Normalisation des données numériques </strong><br><br>
          
      📝 <strong> Encodage des variables catégorielles et textuelles </strong><br>     
      ⚖️ <strong> Rééquilibrage des données : </strong> <br>   
       </div> 
                                                    
    <div style="margin-left: 50px;">              
        Rééquilibrage de la variable cible sur l’ensemble d’entrainement afin d'éviter tout biais. <br>
        Utilisation de la méthode de suréchantillonnage des classes minoritaires avec la technique SMOTE (Synthetic Minority Over-sampling Technique).
    </div>
""", unsafe_allow_html=True)
    st.write(" ")
    st.write(" ")
        


##################### PAGE 5 #############################################################


if page == pages[5]: 
    # Titre principal
    st.markdown("""
        <h2 style='color: #1f487e; font-size: 30px; text-align: center;'> Data Visualisation</h2>
        <p style='text-align: center;'></p>
        <hr style="border: 2px solid #1f487e;">
    """, unsafe_allow_html=True)


    # 📈 Répartition des differentes notes

    st.markdown("""
        <h2 style='color: #1f487e; font-size: 26px; text-align: left;'>🟦 Répartition des notes</h2>
        
        Cette visualisation montre la distribution des notes exprimées par les utilisateurs
        <br>       
    """, unsafe_allow_html=True)

    #st.subheader("📈 Répartition des notes")
    #st.write("Cette visualisation montre la distribution des notes exprimées par les utilisateurs. ")
    st.write(" ")
    st.image("repartition_note.png", caption="Distribution des notes clients", use_container_width=True)

    # 📌 📊Répartition des sentiments
    st.markdown("""
        <h2 style='color: #1f487e; font-size: 26px; text-align: left;'>🟦 Répartition des sentiments </h2>
        
        Cette visualisation montre la distribution des sentiments exprimés dans les commentaires
        <br>       
    """, unsafe_allow_html=True)

    st.write(" ")
    #st.subheader("📌 Répartition des sentiments ")
    #st.write("Cette visualisation montre la distribution des sentiments exprimés dans les commentaires. ")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("🔹 **Repartition des sentiments dans les commentaires**")
        st.image("sentiment.png", caption="Distribution des sentiments", use_container_width=True)
    with col2:
        st.write("🔹 **Répartition des sentiments dans les commentaires selon les notes**")
        st.image("sentiment2.png", caption="Répartition des sentiments dans les commentaires selon les notes", use_container_width=True)

   # 📌 Répartition des tailles de commentaire
    st.markdown("""
        <h2 style='color: #1f487e; font-size: 26px; text-align: left;'>🟦 Répartition de la classe taille de commentaire </h2>
        
        Cette visualisation montre la répartition des notes par classe de taille de commentaire
        <br>       
    """, unsafe_allow_html=True)

    st.write(" ")

    #st.subheader("📌 Répartition de la classe taille de commentaire ")
    #st.write("Cette visualisation montre la répartition des notes par classe de taille de commentaire. ")
    col1, col2 = st.columns(2)
    with col1:
        st.write("🔹Repartition taille du commentaire ")
        st.image("taille_comm2.png", caption="Repartition taille du commentaire", use_container_width=True)
    with col2:
        st.write("🔹 Repartition des notes clients par classe de taille")
        st.image("taille_comm.png", caption="Repartition des notes clients par classe de taille du commentaire", use_container_width=True)

   # 📌 Répartition des points d'exclamation
    st.markdown("""
        <h2 style='color: #1f487e; font-size: 26px; text-align: left;'>🟦 Répartition de la classe nombre de point d'exclamation </h2>
        
       Cette visualisation montre la répartition des notes par classe de nombre d'exclamation dans le commentaire
        <br>       
    """, unsafe_allow_html=True)

    st.write(" ")

    #st.subheader("📌 Répartition de la classe nombre de point d'exclamation      ")
    #st.write("Cette visualisation montre la répartition des notes par classe de nombre d'exclamation dans le commentaire. ")
    col1, col2 = st.columns(2)
    with col1:
        st.write("🔹Repartition nombre de point d'exclamation  ")
        st.image("pt_exclamation.png", caption="Repartition nombre de point d'exclamation ", use_container_width=True)
    with col2:
        st.write("🔹 Repartition des notes clients par classe ")
        st.image("pt_exclamation2.png", caption="Repartition des notes clients par classe de nombre de point d'exclamation", use_container_width=True)

    # 📌 Répartition des emoticones

    st.markdown("""
        <h2 style='color: #1f487e; font-size: 26px; text-align: left;'>🟦 Répartition de la classe d'emoticones</h2>
        
       Cette visualisation montre la répartition des notes par classe d'émoticones
        <br>       
    """, unsafe_allow_html=True)

    st.write(" ")
    #st.subheader("📌 Répartition de la classe d'emoticones   ")
    #st.write("Cette visualisation montre la répartition des notes par classe d'émoticones. ")
    st.write("🔹 😊 Positif : 😃😍👍✨💚 ")
    st.write("🔹 😡 Négatif : 😠😢👎⚠️💔 ")
    #st.write("🔹  😐 Neutre : 😐🤔🧐🔍🔄 ")

    col1, col2 = st.columns(2)
    with col1:
        st.write("🔹Repartition de type d'emoticones ")
        st.image("emoticone.png", caption="Repartition de type d'emoticones dans les commentaires  ", use_container_width=True)
    with col2:
        st.write("🔹 Repartition des types d'emoticones par note ")
        st.image("emoticone2.png", caption="Repartition des types d'emoticones par note ", use_container_width=True)

   
    # 📊 Heatmap des variables quantitatives

    st.markdown("""
        <h2 style='color: #1f487e; font-size: 26px; text-align: left;'>🟦 Heatmap des variables quantitatives</h2>
        
       Cette heatmap permet d'analyser les corrélations entre les différentes variables numériques du dataset
        <br>       
    """, unsafe_allow_html=True)

    st.write(" ")

    #st.subheader("📊 Heatmap des variables quantitatives")
    #st.write("Cette heatmap permet d'analyser les corrélations entre les différentes variables numériques du dataset.")
    st.image("heatmap.png", caption="Matrice de corrélation", use_container_width=True)


    # 🌍 WordClouds des commentaires
    st.markdown("""
        <h2 style='color: #1f487e; font-size: 26px; text-align: left;'>🌍 WordCloud des commentaires"</h2>
        
       Cette heatmap permet d'analyser les corrélations entre les différentes variables numériques du dataset
        <br>       
    """, unsafe_allow_html=True)

    st.write(" ")


    #st.subheader("🌍 WordCloud des commentaires")
    col1, col2 = st.columns(2)
    with col1:
        st.write("🔹 **Titres des commentaires**")
        st.image("wordcloud.png", caption="Mots-clés des titres", use_container_width=True)
    with col2:
        st.write("🔹 **Commentaires négatifs (note 1 et 2)**")
        st.image("wordcloud_negatif.png", caption="Mots-clés des avis négatifs", use_container_width=True)


    col3, col4 = st.columns(2)
    with col3:
        st.write("🔹 **Commentaires neutres (note 3)**")
        st.image("wordcloud_note3.png", caption="Mots-clés des avis neutres",use_container_width=True)
    with col4:
        st.write("🔹 **Commentaires positifs (note 4 et 5)**")
        st.image("wordcloud_note45.png", caption="Mots-clés des avis positifs", use_container_width=True)




   







########################  PAGE 6 #########################################

if page == pages[6] : 
 
####################################### - PREPARATION A LA MODELISATION1- #########################################################
###################################################################################################################################

  tabs = st.tabs(["Modélisation","Modèles MultiClasses", "Modèles Binaire", "Modèle Deep Learning"])

  # Contenu de l'onglet 1
  with tabs[0]:
  
   st.markdown("""
   <h2 style='color: #1f487e;font-size: 30px;text-align: center;'>Modélisation </h2>
   <p style='text-align: center;'></p>
   <hr style="border: 2px solid #1f487e;">
   <p>
   Nous cherchons à identifier le modèle de classification le plus adapté pour prédire une note en fonction du contenu des commentaires.
   Nous avons testé plusieurs modèles de classification avec l'intégration des données NLP, notamment RandomForestClassifier, GradientBoostingClassifier, Support Vector Machines, LightGBM Classifier et la régression logistique.
  
   Pour poursuivre notre analyse, nous sélectionnons les trois modèles les plus performants : 
   
   <strong> le RandomForestClassifier, le LightGBM Classifier et la régression logistique.</strong>
  
   Nous commençons par analyser des modèles multiclasses, suivis de modèles binaires, avant de passer à un modèle de deep learning.
    <br>
    <br>
    <br>
                     
   <p>
   </p>
   """, unsafe_allow_html=True) 
    #st.image("engineering-processing-automation-machine-gears-64.webp")
  
    # Créer trois colonnes
   col1, col2, col3, col4, col5= st.columns([1, 3,3,2, 1])

   # Placer l'image dans la colonne du milieu
   with col3:
    st.image("engineering-processing-automation-machine-gears-64.webp")


  # Contenu de l'onglet 2
  with tabs[1]:
     st.markdown("""
   <h2 style='color: #1f487e;font-size: 24px;'>Modélisation muti-classes</h2>
   <p>
   Ce modèle est conçu pour évaluer et anticiper la satisfaction des clients en se basant sur les retours des utilisateurs.
   Il doit prédire une note de satisfaction comprise entre 1 et 5 en utilisant les données textuelles des avis et les nouvelles variables créées.
   <p>
   </p>
   """, unsafe_allow_html=True) 
     
      
    # Séparation l'ensemble d'entraînement, de validation et de test
     X = df.drop(['Note_client',"Commentaire","Titre_commentaire","Nombre_avis_client","nbre_mots",'sentimentfr', 'Nombre_Émoticônes'],axis=1)
     y = df['Note_client']

    #Séparation des données en ensembles d'entraînement(80%), de validation(10%) et de test(10%)
     X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
     X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

     # Encodage indépendamment des 3 ensembles
     X_numerique_train= X_train[["nbre_phrases","nbneg"]]
     X_numerique_test= X_test[["nbre_phrases","nbneg"]]
     X_numerique_val= X_val[["nbre_phrases","nbneg"]]

    # Encodage la variable cible avec labelEncoder (pour des variables ordinales)
    #le = LabelEncoder()
     y_train= le.fit_transform(y_train)+1
     y_test = le.transform(y_test)+1
     y_val= le.transform(y_val)+1

     # Encodage des autres variables categorielles
     X_categorielle_train=X_train[['class_longueur_mot',"class_pt_exclam","emoticones","class_nbavis","sentiment_dl","class_sentiment"]]  
     X_categorielle_encoded_train = pd.get_dummies(X_categorielle_train, drop_first=True)
     X_categorielle_encoded_train=X_categorielle_encoded_train.astype(int)

     X_categorielle_test=X_test[['class_longueur_mot',"class_pt_exclam","emoticones","class_nbavis","sentiment_dl","class_sentiment"]]
     X_categorielle_encoded_test = pd.get_dummies(X_categorielle_test, drop_first=True)
     X_categorielle_encoded_test=X_categorielle_encoded_test.astype(int)

     X_categorielle_val=X_val[['class_longueur_mot',"class_pt_exclam","emoticones","class_nbavis","sentiment_dl","class_sentiment"]]
     X_categorielle_encoded_val = pd.get_dummies(X_categorielle_val, drop_first=True)
     X_categorielle_encoded_val=X_categorielle_encoded_val.astype(int)

    ## Vectorisation sur les données textuelles
     X_texte_train = X_train['Lemmes'] + ' ' + X_train['Lemmes_titre_commentaire']  
     X_text_vectorized_train = vectorizer.fit_transform(X_texte_train)

    # Pour l'ensemble de test
     X_texte_test = X_test['Lemmes'] + ' ' + X_test['Lemmes_titre_commentaire']  
     X_text_vectorized_test = vectorizer.transform(X_texte_test)  

    # Pour l'ensemble de val
     X_texte_val = X_val['Lemmes'] + ' ' + X_val['Lemmes_titre_commentaire']  
     X_text_vectorized_val = vectorizer.transform(X_texte_val)  

    # Transforme  en matrice creuse pour gagner du temps de calcul
     X_categorielle_encoded_train_sparse = csr_matrix(X_categorielle_encoded_train)
     X_numerique_train_sparse = csr_matrix(X_numerique_train)
     X_text_vectorized_train_sparse = csr_matrix(X_text_vectorized_train)

     X_numerique_val_sparse = csr_matrix(X_numerique_val)
     X_categorielle_encoded_val_sparse = csr_matrix(X_categorielle_encoded_val)
     X_text_vectorized_val_sparse = csr_matrix(X_text_vectorized_val)

     X_numerique_test_sparse = csr_matrix(X_numerique_test)
     X_categorielle_encoded_test_sparse = csr_matrix(X_categorielle_encoded_test)
     X_text_vectorized_test_sparse = csr_matrix(X_text_vectorized_test)

     # Combine les caractéristiques textuelles et numériques pour l'entraînement
     X_train_combined = hstack([X_text_vectorized_train_sparse, X_categorielle_encoded_train_sparse, X_numerique_train_sparse])

     #  pour les ensembles de validation et de test

     X_val_combined = hstack([X_text_vectorized_val_sparse,X_categorielle_encoded_val_sparse, X_numerique_val_sparse])
     X_test_combined = hstack([X_text_vectorized_test_sparse,X_categorielle_encoded_test_sparse, X_numerique_test_sparse])

    # Normalisation des 3 ensembles
     scaler = StandardScaler(with_mean=False)  # with_mean=False car TF-IDF peut avoir des valeurs nulles
     X_train_scaled = scaler.fit_transform(X_train_combined)
     X_val_scaled = scaler.transform(X_val_combined)
     X_test_scaled = scaler.transform(X_test_combined)
  
# -----Modelisation multiclasse----
  
     reg = joblib.load("regression_logistique1")
     rfc = joblib.load("RandomForestClassifier1")
     gbm = joblib.load("GBM1")

     def prediction(option1, X_test_scaled, y_test, rfc, gbm, reg):
         # Vérifiez si l'option est vide
       if option1 == " ":
         return "Aucun modèle sélectionné.", None, None

       if option1 == 'Random Forest Classifier':
          y_pred = rfc.predict(X_test_scaled)
       elif option1 == 'LightGBM Classifier':
          y_pred = gbm.predict(X_test_scaled)
       elif option1 == 'Logistic Regression':
          y_pred = reg.predict(X_test_scaled)
       else:
         raise ValueError(f"Modèle non reconnu : {option1}")

       classR = classification_report(y_test, y_pred)
       accuracy = accuracy_score(y_test, y_pred)
       cmsm = pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite'])

       return classR, accuracy, cmsm

     # Appel de la fonction et affichage des résultats
     option1 = st.selectbox("Choisissez le modèle multi-classes", 
                        [" ", 'Random Forest Classifier', 'LightGBM Classifier', 'Logistic Regression'], key="selectbox_multiclasse")

   # Initialisation des variables
     classR = accuracy = cmsm = None
  
   # Convertit un rapport de classification en DataFrame pour plus de visibilité.
     def report_to_df(report):
       report_lines = report.split('\n')
       report_data = {}
       classes = []

       for line in report_lines[2:-3]:  # Ignorer les lignes de titre et de moyenne
          if line.strip():  # Ignorer les lignes vides
             parts = line.split()
             if len(parts) >= 5:  # 5 classes
                class_name = parts[0]
                classes.append(class_name)
                report_data[class_name] = {
                    'precision': round(float(parts[1]), 2),
                    'recall': round(float(parts[2]), 2),
                    'f1-score': round(float(parts[3]), 2),
                    'support': int(parts[4])
                }
             else:
                print(f"Ligne ignorée (pas assez d'éléments) : {line}")

    # DataFrame à partir des données
       report_df = pd.DataFrame.from_dict(report_data, orient='index')

    # Ajouter les moyennes
       report_df.loc['macro avg'] = report_df.mean().round(2)
       report_df.loc['weighted avg'] = report_df.apply(
        lambda x: (x * report_df['support']).sum() / report_df['support'].sum(), axis= 0).round(2)
       return report_df

     #Appel de la fonction seulement si un modèle est sélectionné
     if option1 != " ":
      classR, accuracy, cmsm = prediction(option1, X_test_scaled, y_test, rfc, gbm, reg)
      display = st.radio("Choix d'affichage", ( "Classification Report", 'Matrice de confusion'),key='radio1')
      #display = st.radio("Choix d'affichage", ('Accuracy', "Classification Report", 'Matrice de confusion'),key='radio1')

      # if display == 'Accuracy':
      #    if accuracy is not None:
      #      st.markdown(f"<h4 style='color: #2ddff3;'>Accuracy: {accuracy:.3f}</h4>", unsafe_allow_html=True)
      #    else:
      #      st.write("Aucune précision disponible.")

      if display == "Matrice de confusion":
         if cmsm is not None:
           st.markdown("<h4 style='color: #2ddff3'>Matrice de confusion</h4>", unsafe_allow_html=True)
           st.dataframe(cmsm)
         else:
          st.write("Aucune matrice de confusion disponible.")

      elif display == 'Classification Report':
        if classR is not None:
          st.markdown("<h4 style='color: #2ddff3;'>Classification Report</h4>", unsafe_allow_html=True)  # Titre pour le rapport de classification
          report_df = report_to_df(classR)  # Convertir le rapport de classification en DataFrame
          st.dataframe(report_df.style.format(
            {
                'precision': '{:.2f}',
                'recall': '{:.2f}',
                'f1-score': '{:.2f}',
                'support': '{:.0f}'
            }
        ))
      else:
        st.write("Aucun rapport de classification disponible.")
     else:
      #st.write("Veuillez sélectionner un modèle.")
      st.write(" ")


 
########################################################################################################################################


  # Contenu de l'onglet 3
  with tabs[2]:
    st.markdown("""
    <h2 style='color: #1f487e;font-size: 24px;'>Modélisation binaire</h2>
    <p>
    Nous transformons la variable cible en un modèle binaire, où la valeur 1 correspond aux notes 1 et 2, et la valeur 2 aux notes 3, 4 et 5.
              
    Cette répartition est un choix basé sur l'analyse des nuages de mots, qui a montré que les notes 3, 4 et 5 sont particulièrement difficiles à différencier sur les mots utilisés.
                
    Ce modèle va nous aider à identifier des tendances positives ou négatives.
    <p>
    </p>
    """, unsafe_allow_html=True) 


   #Classification de la variable cible en variable binaire
    df_binaire["Note_client"] = df_binaire["Note_client"].replace(to_replace=[1, 2, 3, 4, 5], value=[1, 1, 2, 2, 2])


####################################### - PREPARATION A LA MODELISATION 2- #########################################################
###################################################################################################################################


    # Séparation l'ensemble d'entraînement, de validation et de test
    X = df_binaire.drop(['Note_client',"Commentaire","Titre_commentaire","Nombre_avis_client","nbre_mots",'sentimentfr', 'Nombre_Émoticônes'],axis=1)
    y = df_binaire['Note_client']


    #Séparation des données en ensembles d'entraînement(80%), de validation(10%) et de test(10%)
    X_train2, X_temp2, y_train2, y_temp2 = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_val2, X_test2, y_val2, y_test2 = train_test_split(X_temp2, y_temp2, test_size=0.5, random_state=42, stratify=y_temp2)

    # Encodage indépendamment des 3 ensembles
    X_numerique_train2= X_train2[["nbre_phrases","nbneg"]]
    X_numerique_test2= X_test2[["nbre_phrases","nbneg"]]
    X_numerique_val2= X_val2[["nbre_phrases","nbneg"]]

    # Encodage la variable cible avec labelEncoder (pour des variables ordinales)
    y_train2= le2.fit_transform(y_train2)+1
    y_test2 = le2.transform(y_test2)+1
    y_val2= le2.transform(y_val2)+1

    # Encodage des autres variables categorielles
    X_categorielle_train2=X_train2[['class_longueur_mot',"class_pt_exclam","emoticones","class_nbavis","sentiment_dl","class_sentiment"]]  
    X_categorielle_encoded_train2 = pd.get_dummies(X_categorielle_train2, drop_first=True)
    X_categorielle_encoded_train2=X_categorielle_encoded_train2.astype(int)

    X_categorielle_test2=X_test2[['class_longueur_mot',"class_pt_exclam","emoticones","class_nbavis","sentiment_dl","class_sentiment"]]
    X_categorielle_encoded_test2 = pd.get_dummies(X_categorielle_test2, drop_first=True)
    X_categorielle_encoded_test2=X_categorielle_encoded_test2.astype(int)

    X_categorielle_val2=X_val2[['class_longueur_mot',"class_pt_exclam","emoticones","class_nbavis","sentiment_dl","class_sentiment"]]
    X_categorielle_encoded_val2 = pd.get_dummies(X_categorielle_val2, drop_first=True)
    X_categorielle_encoded_val2=X_categorielle_encoded_val2.astype(int)

    ## Vectorisation sur les données textuelles

    # Pour l'ensemble d'entraînement
    #vectorizer2 = TfidfVectorizer()
    X_texte_train2 = X_train2['Lemmes'] + ' ' + X_train2['Lemmes_titre_commentaire'] 
    X_text_vectorized_train2 = vectorizer2.fit_transform(X_texte_train2)

   # Pour l'ensemble de test
    X_texte_test2 = X_test2['Lemmes'] + ' ' + X_test2['Lemmes_titre_commentaire']  
    X_text_vectorized_test2 = vectorizer2.transform(X_texte_test2)  

  # Pour l'ensemble de val
    X_texte_val2 = X_val2['Lemmes'] + ' ' + X_val2['Lemmes_titre_commentaire']  
    X_text_vectorized_val2 = vectorizer2.transform(X_texte_val2)  


   # Transforme  en matrice creuse pour gagner du temps de calcul
    X_categorielle_encoded_train_sparse2 = csr_matrix(X_categorielle_encoded_train2)
    X_numerique_train_sparse2 = csr_matrix(X_numerique_train2)
    X_text_vectorized_train_sparse2 = csr_matrix(X_text_vectorized_train2)

    X_numerique_val_sparse2 = csr_matrix(X_numerique_val2)
    X_categorielle_encoded_val_sparse2 = csr_matrix(X_categorielle_encoded_val2)
    X_text_vectorized_val_sparse2 = csr_matrix(X_text_vectorized_val2)

    X_numerique_test_sparse2 = csr_matrix(X_numerique_test2)
    X_categorielle_encoded_test_sparse2 = csr_matrix(X_categorielle_encoded_test2)
    X_text_vectorized_test_sparse2 = csr_matrix(X_text_vectorized_test2)

    # Combine les caractéristiques textuelles et numériques pour l'entraînement
    X_train_combined2 = hstack([X_text_vectorized_train_sparse2, X_categorielle_encoded_train_sparse2, X_numerique_train_sparse2])

    #  pour les ensembles de validation et de test

    X_val_combined2= hstack([X_text_vectorized_val_sparse2,X_categorielle_encoded_val_sparse2, X_numerique_val_sparse2])
    X_test_combined2 = hstack([X_text_vectorized_test_sparse2,X_categorielle_encoded_test_sparse2, X_numerique_test_sparse2])

    # Normalisation des 3 ensembles
    scaler2 = StandardScaler(with_mean=False)  # with_mean=False car TF-IDF peut avoir des valeurs nulles
    X_train_scaled2 = scaler2.fit_transform(X_train_combined2)
    X_val_scaled2 = scaler2.transform(X_val_combined2)
    X_test_scaled2 = scaler2.transform(X_test_combined2)

# -----Modelisation BINAIRE----

    reg2 = joblib.load("regression_logistique_BI")
    rfc2 = joblib.load("RandomForestClassifier_BI")
    gbm2 = joblib.load("GBM_BI")

    def prediction(option2, X_test_scaled2, y_test2, rfc2, gbm2, reg2):
      # Vérifiez si l'option est vide
      if option2 == " ":
          return "Aucun modèle sélectionné.", None, None
      if option2 == 'Random Forest Classifier':
          y_pred2 = rfc2.predict(X_test_scaled2)
      elif option2 == 'LightGBM Classifier':
          y_pred2 = gbm2.predict(X_test_scaled2)
      elif option2 == 'Logistic Regression':
          y_pred2 = reg2.predict(X_test_scaled2)
      else:
          raise ValueError(f"Modèle non reconnu : {option2}")

      classR2 = classification_report(y_test2, y_pred2)
      accuracy2 = accuracy_score(y_test2, y_pred2)
      cmsm2 = pd.crosstab(y_test2, y_pred2, rownames=['Classe réelle'], colnames=['Classe prédite'])

      return classR2, accuracy2, cmsm2

  # Appel de la fonction et affichage des résultats
    option2 = st.selectbox("Choisissez le modèle binaire", 
                        [" ", 'Random Forest Classifier', 'LightGBM Classifier', 'Logistic Regression'],key="selectbox_binaire")

  # Initialisation des variables
    classR2 = accuracy2 = cmsm2 = None

   #Appel de la fonction seulement si un modèle est sélectionné
    if option2 != " ":
        classR2, accuracy2, cmsm2 = prediction(option2, X_test_scaled2, y_test2, rfc2, gbm2, reg2)
        display2 = st.radio("Choix d'affichage", ( "Classification Report", 'Matrice de confusion'),key='radio2')
        #display2 = st.radio("Choix d'affichage", ('Accuracy', "Classification Report", 'Matrice de confusion'),key='radio2')
        # if display2 == 'Accuracy':
        #   if accuracy2 is not None:
        #     st.markdown(f"<h4 style='color: #2ddff3;'>Accuracy: {accuracy2:.3f}</h4>", unsafe_allow_html=True)
        #   else:
        #    st.write("Aucune précision disponible.")

        if display2 == "Matrice de confusion":
          if cmsm2 is not None:
            st.markdown("<h4 style='color: #2ddff3'>Matrice de confusion</h4>", unsafe_allow_html=True)
            st.dataframe(cmsm2)
          else:
            st.write("Aucune matrice de confusion disponible.")

        elif display2 == 'Classification Report':
          if classR2 is not None:
            st.markdown("<h4 style='color: #2ddff3;'>Classification Report</h4>", unsafe_allow_html=True)  # Titre pour le rapport de classification
            report_df = report_to_df(classR2)  # Convertir le rapport de classification en DataFrame
            st.dataframe(report_df.style.format(
            {
                'precision': '{:.2f}',
                'recall': '{:.2f}',
                'f1-score': '{:.2f}',
                'support': '{:.0f}'
            }
        ))
        else:
          st.write("Aucun rapport de classification disponible.")
    else:
      st.write(" ")


 # Contenu de l'onglet 4
  with tabs[3]:

    st.markdown("""
    <h2 style='color: #1f487e;font-size: 24px;'>Modélisation Deep Learning</h2>
    <p>
    Les réseaux de neurones récurrents LSTM (Long Short-Term Memory) sont actuellement l’un des types d’apprentissage profond les plus intéressants. 
    <p>
    </p>
    """, unsafe_allow_html=True) 
    st.write(" ")
    st.write(" ")
    st.write(" ")


    choix3 = [" ",'Long Short-Term Memory']
    option3 = st.selectbox('Choix du modèle Deep Learning', choix3)
    if option3 == 'Long Short-Term Memory':
      st.image("Resultat_lstm.png")

      
###################################  PAGE 7  ################################################

if page == pages[7] :
  st.markdown("""        
  <h2 style='color: #1f487e;font-size: 30px;text-align: center;'>Démo avec Certideal</h2>
   <p>
              
  """, unsafe_allow_html=True) 

  # Créer trois colonnes
  col1, col2, col3 = st.columns([10, 4, 10])
    
    # Placer l'image dans la colonne du milieu
  with col2:
        st.image("Truspilot1.png")

  st.markdown("""
    <hr style="border: 2px solid #1f487e;">
    <p style='text-align: center;'></p>
    """, unsafe_allow_html=True)

  st.markdown("""
  <P>
  <p>
   Nous vous proposons une démonstration de notre modèle de classification avec le classificateur LightGBM, le plus performant. 
   
   Cette démonstration est réalisée sur un ensemble de données que nous avons collecté sur le site Trustpilot concernant l'entreprise Certideal, du même secteur que les trois entreprises utilisées pour l'entraînement.
   Nous disposons d'un ensemble de plus de 9 000 avis à tester avec les modèles multiclasse et binaire. 
  <p>
  </p>
  """, unsafe_allow_html=True) 
   

  reg = joblib.load("regression_logistique1")
  rfc = joblib.load("RandomForestClassifier1")
  gbm = joblib.load("GBM1")

  reg2 = joblib.load("regression_logistique_BI")
  rfc2 = joblib.load("RandomForestClassifier_BI")
  gbm2 = joblib.load("GBM_BI")

   #Chargement des objets pour multi classe
  
  scaler1 = joblib.load("C:/Users/magal/Documents/ProjetStreamlit/scaler1")
  le1 = joblib.load("C:/Users/magal/Documents/ProjetStreamlit/label_encoder1")
  vectorizer1 = joblib.load("C:/Users/magal/Documents/ProjetStreamlit/tfidf_vectorizer1")

  #Chargement des objets pour modele binaire
  
  scaler2 = joblib.load("C:/Users/magal/Documents/ProjetStreamlit/scaler2")
  le2 = joblib.load("C:/Users/magal/Documents/ProjetStreamlit/label_encoder2")
  vectorizer2 = joblib.load("C:/Users/magal/Documents/ProjetStreamlit/tfidf_vectorizer2")

### Sélection du modèle pour la démo
  st.markdown("<h1 style='color: #1f487e;font-size: 24px;'>Analyse des Commentaires</h1>", unsafe_allow_html=True)
  option_demo = st.selectbox("Séléctionnez le modèle pour la démo", 
                        [" ", 'Modèle muticlasses', 'Modèle binaire'], key="selectbox_demo")
  

################### Preprocessing base certideal scrapée pour la démo ##########################
############################### Modele multi classe ###########################################


    
  if option_demo == 'Modèle muticlasses':
  # Prétraitement et affichage des résultats
    Xd = df_demo.drop(['Note_client', "Commentaire", "Titre_commentaire", "Nombre_avis_client", "nbre_mots", 'sentimentfr', 'Nombre_Émoticônes'], axis=1)
    yd = df_demo['Note_client']

    # Encodage
    X_numerique_df = Xd[["nbre_phrases", "nbneg"]]

    # Encodage la variable cible avec LabelEncoder
    y_df = le1.transform(yd) + 1

    # Encodage des autres variables catégorielles
    X_categorielle_df = Xd[['class_longueur_mot', "class_pt_exclam", "emoticones", "class_nbavis", "sentiment_dl", "class_sentiment"]]
    X_categorielle_encoded_df = pd.get_dummies(X_categorielle_df, drop_first=True)
    X_categorielle_encoded_df = X_categorielle_encoded_df.astype(int)

    # Vectorisation sur les données textuelles
    X_texte_df = Xd['Lemmes'] + ' ' + Xd['Lemmes_titre_commentaire']  
    X_text_vectorized_df = vectorizer1.transform(X_texte_df)

    # Transforme en matrice creuse
    X_numerique_df_sparse = csr_matrix(X_numerique_df)
    X_categorielle_encoded_df_sparse = csr_matrix(X_categorielle_encoded_df)
    X_text_vectorized_df_sparse = csr_matrix(X_text_vectorized_df)

    X_df_combined = hstack([X_text_vectorized_df_sparse, X_categorielle_encoded_df_sparse, X_numerique_df_sparse])

    # Normalisation
    X_df_scaled = scaler1.transform(X_df_combined)

    # Affichage du DataFrame df_demo
    #st.markdown("<h1 style='color: #1f487e;font-size: 24px;'>Analyse des Commentaires</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #1f487e;font-size: 20px;'>Données des Commentaires</h3>", unsafe_allow_html=True)
  
    #filtre 
   # Filtre sur la note
    note_filter = st.slider("Sélectionnez une note", min_value=1, max_value=5, value=(1, 5), step=1)

  # Application du filtre sur le DataFrame
    df_filtered = df_demo[(df_demo['Note_client'] >= note_filter[0]) & (df_demo['Note_client'] <= note_filter[1])]

   # Affichage du DataFrame filtré
    df_display = df_filtered[['Note_client', 'Titre_commentaire', 'Commentaire']]
    st.dataframe(df_display)

    #df pour conserver l'index
    df_display0 = df_demo[['Note_client', 'Titre_commentaire', 'Commentaire']]

    # Choix de la ligne à analyser
    index_selection = st.number_input("Sélectionnez l'index de la ligne à analyser", min_value=0, max_value=len(df_display0)-1, step=1)

 
  # Prédiction (remplacez cette partie par votre logique de prédiction)
    prediction = gbm.predict(X_df_scaled)  
    note_predite = le1.inverse_transform(prediction - 1)  

    # Affichage de la note prédite
    st.markdown(
        f"<h4 style='color: #2ddff3; text-align: center;'>La note prédite pour le commentaire est : {note_predite[index_selection]}</h4>",
        unsafe_allow_html=True)
    


    # # Affichage du DataFrame
    # df_display = df_demo[['Note_client', 'Titre_commentaire', 'Commentaire']]
    # st.dataframe(df_display)

    # # Choix de la ligne à analyser
    # index_selection = st.number_input("Sélectionnez l'index de la ligne à analyser", min_value=0, max_value=len(df_display)-1, step=1, key="index_selection_1")

    # # Prédiction
    # prediction = gbm.predict(X_df_scaled)  
    # note_predite = le1.inverse_transform(prediction - 1)  

    # # Affichage de la note prédite
    # st.markdown(
    # f"<h4 style='color: #2ddff3; text-align: center;'>La note prédite pour le commentaire est : {note_predite[index_selection]}</h4>",
    # unsafe_allow_html=True)

    # Option pour visualiser le graphe
    show_graph = st.checkbox("Afficher la répartition des notes prédites vs Notes réelles")

    if show_graph:
        # Extraction des notes réelles et prédites
        notes_reelles = df_demo['Note_client']
        
       #  les histogrammes
        plt.hist(notes_reelles, bins=10, alpha=0.5, label='Notes Réelles', color='blue' )  #,edgecolor='black')
        plt.hist(note_predite, bins=10, alpha=0.5, label='Notes Prédites', color='orange') 
 
        plt.title("Comparaison des Notes Prédites et Réelles")
        plt.xlabel("Notes")
        plt.ylabel("Fréquence")
        ticks = range(1,6)  
        plt.xticks(ticks, ticks)
        #plt.xticks([])
        #plt.gca().set_frame_on(False)
        # Changer la couleur des spines
        ax = plt.gca()
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        plt.legend()
        st.pyplot(plt)

#   # Création d'un DataFrame pour afficher les résultats pour la verif
 
#     resultats = pd.DataFrame({
#       'Note Prédite': note_predite,
#       'Note Client': df_demo['Note_client'],
#       'Titre Commentaire': df_demo['Titre_commentaire'],
#       'Commentaire': df_demo['Commentaire']
      
#    })

# #   Affichage des résultats
#     st.write("Résultats de la Prédiction :")
#     st.dataframe(resultats)

  elif option_demo == 'Modèle binaire':

    ################## Preprocessing base certideal scrapée pour la démo ##########################
    ############################### Modele binaire ###########################################

    #Classification de la variable cible en variable binaire
    df_demo_binaire["Note_client"] = df_demo_binaire["Note_client"].replace(to_replace=[1, 2, 3, 4, 5], value=[1, 1, 2, 2, 2])

    # Séparation
    Xdbi = df_demo_binaire.drop(['Note_client', "Commentaire", "Titre_commentaire", "Nombre_avis_client", "nbre_mots", 'sentimentfr', 'Nombre_Émoticônes'], axis=1)
    ydbi = df_demo_binaire['Note_client'] 

    # Encodage 
    X_numerique_df_bi = Xdbi[["nbre_phrases", "nbneg"]]

    # Encodage la variable cible avec LabelEncoder 
    y_df_bi = le2.transform(ydbi) + 1

    # Encodage des autres variables catégorielles
    X_categorielle_df_bi = Xdbi[['class_longueur_mot', "class_pt_exclam", "emoticones", "class_nbavis", "sentiment_dl", "class_sentiment"]]
    X_categorielle_encoded_df_bi = pd.get_dummies(X_categorielle_df_bi, drop_first=True)
    X_categorielle_encoded_df_bi = X_categorielle_encoded_df_bi.astype(int)

    ## Vectorisation sur les données textuelles
    X_texte_df_bi = Xdbi['Lemmes'] + ' ' + Xdbi['Lemmes_titre_commentaire']  
    X_text_vectorized_df_bi = vectorizer2.transform(X_texte_df_bi)  

    # Transforme en matrice creuse pour gagner du temps de calcul
    X_numerique_df_sparse_bi = csr_matrix(X_numerique_df_bi)
    X_categorielle_encoded_df_sparse_bi = csr_matrix(X_categorielle_encoded_df_bi)
    X_text_vectorized_df_sparse_bi = csr_matrix(X_text_vectorized_df_bi)

    X_df_combined_bi = hstack([X_text_vectorized_df_sparse_bi, X_categorielle_encoded_df_sparse_bi, X_numerique_df_sparse_bi])

    # Normalisation
    X_df_scaled_bi = scaler2.transform(X_df_combined_bi)

   # Affichage du DataFrame df_demo
    st.markdown("<h3 style='color: #1f487e;font-size: 20px;'>Données des Commentaires</h3>", unsafe_allow_html=True)
    df_display_bi = df_demo_binaire[['Note_client', 'Titre_commentaire','Commentaire']]
    #st.dataframe(df_display_bi.style.highlight_min(axis=0))
    st.dataframe(df_display_bi)

   # Choix de la ligne à analyser
    index_selection_bi = st.number_input("Sélectionnez l'index de la ligne à analyser", min_value=0, max_value=len(df_display_bi)-1, step=1,key="index_selection_bi") 

   # Prédiction
    prediction_bi = gbm2.predict(X_df_scaled_bi)  
    note_predite_bi = le2.inverse_transform(prediction_bi - 1)  

    # Affichage de la note prédite
    #st.write(f"La note prédite pour le commentaire est : **{note_predite[index_selection]}**")
    st.markdown (f"<h4 style='color: #2ddff3; text-align: center;'>La note prédite pour le commentaire est : {note_predite_bi[index_selection_bi]}</h4>",
    unsafe_allow_html=True)

    # Option pour visualiser le graphe
    show_graph2 = st.checkbox("Afficher la répartition des notes prédites vs Notes réelles")

    if show_graph2:
        # Extraction des notes réelles et prédites
        notes_reelles = df_demo_binaire['Note_client']
        
       #  les histogrammes
        plt.hist(notes_reelles, bins=3, alpha=0.5, label='Notes Réelles', color='blue')
        plt.hist(note_predite_bi, bins=3, alpha=0.5, label='Notes Prédites', color='orange') 
 
        plt.title("Comparaison des Notes Prédites et Réelles")
        plt.xlabel("Notes")
        plt.ylabel("Fréquence")
        ticks = range(1,3)  
        plt.xticks(ticks, ticks)
        #plt.xticks([])
        #plt.gca().set_frame_on(False)
        # Changer la couleur des spines
        ax = plt.gca()
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        #plt.xticks([])
        plt.legend()
        st.pyplot(plt)







#################### PAGE 8 #######################################################

if page == pages[8] : 
#   st.markdown("""
# <h2 style='color: #1d3461;'>Projet Supply chain Satisfaction client</h2>
# <p>
# </p>
# """, unsafe_allow_html=True)
   # Créer trois colonnes
  col1, col2, col3= st.columns([1, 5, 6])

   # Placer l'image dans la colonne du milieu
  with col2:
   st.image("PRESENTATION2.png", width=500)

