# Système de Retrieval Augmented Generation (RAG) pour Documents Scientifiques

## Vue d'ensemble

Ce projet implémente un système de RAG (Retrieval Augmented Generation) spécialisé pour l'analyse de documents scientifiques historiques au format XML-TEI. Le système permet d'interroger un corpus de documents en français et de générer des réponses précises et sourcées en utilisant des modèles de langage de pointe.

## Architecture du Pipeline

L'architecture actuelle est principalement de type **RAG Naïf** avec des éléments de **Retrieve-and-Rerank**. Voici les composants principaux :

### 1. Traitement des Documents

- **Chargement des documents** : Les fichiers XML-TEI sont chargés depuis les emplacements par défaut ou via téléchargement utilisateur
- **Parsing XML-TEI** : Extraction du texte et des métadonnées (titre, date, personnes mentionnées)
- **Découpage en fragments** : Utilisation de `RecursiveCharacterTextSplitter` avec une taille de fragment de 2500 caractères et un chevauchement de 800 caractères

### 2. Création des Embeddings

Le système propose deux options pour les embeddings :

- **Traitement en temps réel** : Utilisation du modèle "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
- **Embeddings pré-calculés** : Option d'utiliser des embeddings déjà générés avec le modèle "instruct-e5"

### 3. Mécanisme de Récupération (Retrieval)

- **Base de données vectorielle** : Utilisation de FAISS pour stocker et récupérer les fragments de documents
- **Retriever MMR** : Implémentation de Maximum Marginal Relevance pour équilibrer pertinence et diversité
  ```python
  retriever = vectordb.as_retriever(
      search_type="mmr", 
      search_kwargs={'k': 5, 'fetch_k': 10}
  )
  ```

### 4. Génération de Réponses

- **Modèles de langage supportés** :
  
Via HuggingFace :

  - Llama 3 (Meta-Llama-3-8B-Instruct)
  - Mistral (Mistral-7B-Instruct-v0.2)
  - Phi (Phi-4-mini)
    
Via OpenRouter : 

  - Llama 4 Maverick
    
- **Framework de prompting COSTAR** :

## Fonctionnalités Principales

1. **Interface utilisateur Streamlit** avec configuration dans la barre latérale
2. **Options de traitement flexibles** :
   - Utilisation d'embeddings pré-calculés
   - Traitement en temps réel des documents
3. **Personnalisation du prompt** via le cadre COSTAR
4. **Visualisation des sources** utilisées pour générer la réponse et vérifier si on y peut faire confience
5. **Support multilingue** optimisé pour les documents scientifiques en français
6. **Gestion des erreurs OCR** avec niveaux de confiance

## Utilisation

1. Configurer les clés API dans la barre latérale (Hugging Face, OpenRouter)
2. Choisir entre embeddings pré-calculés ou traitement en temps réel
3. Sélectionner un modèle LLM
4. Télécharger des documents XML-TEI ou charger les embeddings
5. Traiter les documents ou charger les embeddings
6. Poser des questions dans l'interface de chat

## Structure du Système

Le système est organisé autour des fonctions principales suivantes :
- `load_documents` : Chargement des documents XML-TEI
- `parse_xmltei_document` : Extraction du contenu et des métadonnées
- `split_documents` : Découpage en fragments pour le traitement
- `embeddings_on_local_vectordb` : Création des embeddings et de la base vectorielle
- `load_precomputed_embeddings` : Chargement des embeddings pré-calculés
- `query_llm` : Interrogation du modèle de langage avec la requête utilisateur
- `process_documents` : Orchestration du processus de traitement


## 🔄 Format des fichiers XML-TEI supportés

L'application est conçue pour traiter des documents XML-TEI avec les balises suivantes :
- `<tei:titleStmt>/<tei:title>` pour le titre du document
- `<tei:sourceDesc>/<tei:p>/<tei:date>` pour la date
- `<tei:p>` pour les paragraphes de contenu
- `<tei:persName>` pour les noms de personnes mentionnées

## 📄 Licence

Ce projet est sous une licence open source MIT. 

## 🤝 Contributions

Le projet est préparé par [Mikhail Biriuchinskii](https://www.linkedin.com/in/mikhail-biriuchinskii/), ingénieur en Traitement Automatique des Langues, équipe ObTIC, Sorbonne Université.

Pour découvrir d'autres projets de l'équipe ObTIC ainsi que les formations proposées, consultez le site : https://obtic.sorbonne-universite.fr/

