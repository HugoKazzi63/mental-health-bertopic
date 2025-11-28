
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import normalize
from hdbscan import HDBSCAN
from umap import UMAP
import pandas as pd
import numpy as np
import string
import spacy

def comparer_modeles_bertopic(df, colonnes, n_topics_list, min_topic_size_list, langue_modele='distiluse-base-multilingual-cased-v1'):

	"""
    Compare plusieurs modèles BERTopic en faisant varier deux hyperparamètres :
    - le nombre de topics cibles (nr_topics),
    - la taille minimale de topic (min_topic_size / min_cluster_size).

    Pour chaque combinaison, la fonction calcule :
    - la cohérence thématique C_v,
    - le Silhouette Score,
    - l'indice de Davies–Bouldin,
    - le nombre de clusters,
    - le pourcentage de documents non classés (label = -1).

    Paramètres
    ----------
    df : pandas.DataFrame
        DataFrame contenant les réponses textuelles.
    colonnes : list of str
        Noms des colonnes textuelles à fusionner (une ligne = un document).
    n_topics_list : list of int
        Liste des valeurs de `nr_topics` à tester.
    min_topic_size_list : list of int
        Liste des valeurs de `min_topic_size` / `min_cluster_size` à tester.
    langue_modele : str, optionnel
        Nom du modèle SentenceTransformer pour les embeddings.

    Retourne
    --------
    pd.DataFrame
        Tableau récapitulatif avec une ligne par combinaison d'hyperparamètres
        et les métriques associées.
    """
    nlp = spacy.load("fr_core_news_md")
    results = []

    def nettoyer_texte(txt):
    	"""
        Nettoie et lemmatise un texte en français :
        - suppression de la ponctuation,
        - passage en minuscules,
        - suppression des stopwords,
        - sélection des noms et adjectifs uniquement.
        """
        txt = txt.translate(str.maketrans('', '', string.punctuation))
        doc = nlp(txt.lower())
        return ' '.join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.lemma_ != '-PRON-' and token.pos_ in ['NOUN', 'ADJ']])
    
    textes = df[colonnes].dropna(how='all').astype(str).apply(lambda row: ' '.join(row), axis=1)
    textes_nettoyes = textes.apply(nettoyer_texte)
    textes_nettoyes = textes_nettoyes.drop_duplicates()
    textes_nettoyes = textes_nettoyes[textes_nettoyes.str.split().str.len() >= 3]
    textes_nettoyes = textes_nettoyes[~textes_nettoyes.str.fullmatch(r"\d+", na=False)]

    model = SentenceTransformer(langue_modele)
    embeddings = model.encode(textes_nettoyes.tolist(), show_progress_bar=True)

    stopwords_perso = [
        "je", "ne", "pas", "plus", "que", "dans", "de", "des", "les", "pour",
        "c'est", "il", "elle", "ils", "elles", "sur", "avec", "le", "la", "et", "est",
        "en", "au", "aux", "à", "du", "un", "une", "mes", "tes", "ma", "ta", "on",
        'jai', 'nan', 'cest', 'ner', 'quil', 'quils', "faire", "sentir", "mettre", "voir", "arriver",
        "penser", "savoir", "devoir", "pouvoir", "vouloir"
    ]
    vectorizer_model = CountVectorizer(stop_words=stopwords_perso)

    for n_topics in n_topics_list:
        for min_topic_size in min_topic_size_list:
            umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')
            hdbscan_model = HDBSCAN(min_cluster_size=min_topic_size, min_samples=3, metric='euclidean')

            topic_model = BERTopic(
                vectorizer_model=vectorizer_model,
                embedding_model=model,
                hdbscan_model=hdbscan_model,
                umap_model=umap_model,
                min_topic_size=min_topic_size,
                nr_topics=n_topics,
                language="french"
            )

            topics, probs = topic_model.fit_transform(textes_nettoyes.tolist(), embeddings)

            def coherence_bertopic():
                topic_words = [[word for word, _ in topic_model.get_topic(topic)] for topic in topic_model.get_topics().keys() if topic != -1]
                tokenized_texts = [text.split() for text in textes_nettoyes]
                dictionary = Dictionary(tokenized_texts)
                corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
                model_coh = CoherenceModel(topics=topic_words, texts=tokenized_texts, dictionary=dictionary, coherence='c_v')
                return model_coh.get_coherence()

            def clustering_scores():
                mask = np.array(topics) != -1
                if sum(mask) < 2 or len(set(np.array(topics)[mask])) < 2:
                    return None, None
                embeddings_filtered = normalize(np.array(embeddings)[mask])
                labels_filtered = np.array(topics)[mask]
                silhouette = silhouette_score(embeddings_filtered, labels_filtered)
                db_index = davies_bouldin_score(embeddings_filtered, labels_filtered)
                return silhouette, db_index

            try:
                coherence = coherence_bertopic()
                silhouette, db_index = clustering_scores()
                n_clusters = len(set(topics)) - (1 if -1 in topics else 0)
                pct_non_classés = (topics.count(-1) / len(topics)) * 100
            except:
                coherence, silhouette, db_index, n_clusters, pct_non_classés = None, None, None, None, None

            results.append({
                "n_topics": n_topics,
                "min_topic_size": min_topic_size,
                "coherence": coherence,
                "silhouette": silhouette,
                "davies_bouldin": db_index,
                "n_clusters": n_clusters,
                "pct_non_classés": pct_non_classés
            })

    return pd.DataFrame(results)
