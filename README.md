# Analyse thématique automatisée des réponses ouvertes (santé mentale)

Ce dépôt contient l’ensemble du code source développé dans le cadre d’un stage
de 6 mois portant sur l’analyse thématique automatisée de réponses à des
questions ouvertes en santé mentale (psychologie clinique).

L’objectif général est d’identifier, de structurer et de visualiser des
**thématiques latentes** présentes dans les verbatims de trois échantillons :

- **E2 : personnes concernées**
- **E3 : proches**
- **E4 : professionnel·les de santé**

Ce projet repose sur une chaîne complète :

- **prétraitement linguistique avancé** (spaCy, nettoyage, fusion de colonnes),
- **vectorisation sémantique** (SentenceTransformers),
- **réduction de dimension** (UMAP),
- **clustering hiérarchique** (HDBSCAN),
- **modélisation thématique** (BERTopic),
- **analyses descriptives**,
- **visualisations** (wordclouds, barplots).

Les données ne sont pas fournies dans ce dépôt pour des raisons de confidentialité (RGPD).  
Pour reproduire les analyses, les fichiers sources doivent être placés dans `data/raw/`.



##  Structure du dépôt

La structure réelle du projet est la suivante :

data/
├── processed/                
├── raw/                     
notebooks/
├── 01_descriptif/
│   └── Analyse descriptive E2.ipynb
│
├── 02_pretraitement/
│   ├── Prétraitement E2.ipynb
│   ├── Prétraitement E3.ipynb
│   └── Prétraitement E4.ipynb
│
├── 03_bertopic/
│   ├── BERTopic E2.ipynb
│   ├── BERTopic E3.ipynb
│   └── BERTopic E4.ipynb
│
└── 04_visualisation/
└── Wordcloud.E2.ipynb

README.md
requirements.txt


##  1. Prétraitement

Les notebooks de prétraitement produisent les bases nettoyées utilisées dans la suite du pipeline.

### 02_pretraitement/
- **Prétraitement E2.ipynb** : nettoyage base E2, regroupement des psychothérapies, activité professionnelle, classification des diagnostics.
- **Prétraitement E3.ipynb** : regroupement proche/répondant, psychothérapies du proche, catégorisation du diagnostic.
- **Prétraitement E4.ipynb** : type de structure, activité professionnelle, tranches d’âge accompagnées, diagnostic majoritaire.

Chaque notebook construit une fonction `load_and_clean()`, documentée et commentée.



##  2. Analyse descriptive

### 01_descriptif/
- **Analyse descriptive E2.ipynb** :  
  analyse des distributions (âge, genre, diagnostics), répartition des catégories, graphiques.  
  Sert de base au cadrage statistique avant modélisation.



##  3. Modélisation thématique (BERTopic)

### 03_bertopic/
Notebooks dédiés pour chaque échantillon :

- **BERTopic E2.ipynb**  
- **BERTopic E3.ipynb**  
- **BERTopic E4.ipynb**

Chacun applique un pipeline complet :

1. Lemmatisation française (spaCy `fr_core_news_md`)
2. Nettoyage sémantique (stopwords, ponctuation, POS = NOUN/ADJ)
3. Embeddings (`distiluse-base-multilingual-cased-v1`)
4. Réduction de dimension (UMAP)
5. Clustering (HDBSCAN)
6. Modélisation des topics (BERTopic)
7. Construction d’un tableau final :
   - numéro du topic  
   - “Name of the topic”  
   - mots-clés (10 best words)  
   - exemples de verbatims  
   - prévalence (%)  
   - diagnostic `Trouble_Catégorisé` lié  
8. Export des tableaux en `.xlsx`



##  4. Visualisations – WordClouds

### 04_visualisation/
- **Wordcloud.E2.ipynb** :  
  Wordclouds exploratoires pour les catégories “pensées”, “émotions”, “souffrances”, etc.

Les wordclouds sont descriptifs et ne remplacent pas la modélisation thématique.



## ⛓ Installation de l’environnement

Créer un environnement Python :

```bash
python -m venv env
source env/bin/activate      # macOS / Linux
env\Scripts\activate         # Windows

Installer les dépendances :

pip install -r requirements.txt

Le fichier requirements.txt inclut notamment :
	•	pandas, numpy
	•	scikit-learn
	•	matplotlib, seaborn
	•	wordcloud
	•	umap-learn
	•	hdbscan
	•	bertopic
	•	sentence-transformers
	•	spaCy + modèle français
	•	openpyxl



Reproductibilité du pipeline complet
	1.	Prétraitement :
Exécuter les notebooks de 02_pretraitement/ pour générer les bases nettoyées dans data/processed/.
	2.	Analyse descriptive :
Utiliser le notebook de 01_descriptif/.
	3.	Modélisation des topics :
Lancer les notebooks de 03_bertopic/ pour produire les tableaux thématiques.
	4.	Visualisations :
Wordclouds et autres figures dans 04_visualisation/.



Limites méthodologiques
	•	Les wordclouds ne représentent pas des thématiques → uniquement des fréquences.
	•	BERTopic est sensible aux paramètres UMAP/HDBSCAN.
	•	La lemmatisation française reste imparfaite.
	•	La validation clinique est indispensable pour interpréter les topics.
	•	Les métriques de performance (cohérence C_v, silhouette, Davies–Bouldin)
	  	ont été calculées pour comparer quelques configurations de BERTopic, mais
	  	**aucune optimisation exhaustive** de ces indices n’a été menée faute de temps.
	  	Les résultats sont donc à lire comme une évaluation exploratoire, complétée
	  	par une validation clinique des topics.



Contact

Travail réalisé dans le cadre d’un stage de Master 1
Management de l’Intelligence Artificielle en Santé (Centrale Lille).

Pour toute question ou collaboration scientifique : merci de me contacter.

