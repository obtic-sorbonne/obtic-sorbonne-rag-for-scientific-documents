# Système de Retrieval Augmented Generation (RAG) pour Documents Scientifiques

## Vue d'ensemble

Ce projet implémente un système de RAG (Retrieval Augmented Generation) spécialisé pour l'analyse de documents scientifiques historiques au format XML-TEI. Le système permet d'interroger un corpus de documents en français et de générer des réponses précises et sourcées en utilisant des modèles de langage de pointe.

## Architecture du Pipeline

L'architecture actuelle est principalement de type **RAG Naïf** avec des éléments de **Retrieve-and-Rerank**. Voici les composants principaux :

### 1. Traitement des Documents

-   **Chargement des documents** : Les fichiers XML-TEI sont chargés depuis les emplacements par défaut (`./`, `data/`) ou via téléchargement utilisateur (sauvegardés dans `data/uploaded/`).
-   **Parsing XML-TEI** : Extraction du texte et des métadonnées (titre, date, année, personnes mentionnées, et contenu textuel complet).
-   **Découpage en fragments** : Utilisation de `RecursiveCharacterTextSplitter` avec une taille de fragment de 2500 caractères et un chevauchement de 800 caractères.

### 2. Création des Embeddings

Le système propose deux options pour les embeddings :

-   **Traitement en temps réel** : Utilisation du modèle "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2". Les embeddings et l'index FAISS sont sauvegardés localement dans `vector_store/`.
-   **Embeddings pré-calculés** : Option d'utiliser des embeddings déjà générés et stockés dans `embeddings/`. Le modèle d'embedding utilisé pour ces fichiers est indiqué dans leurs métadonnées (`embeddings/document_metadata.pkl`). Si le nom du modèle n'est pas trouvé dans les métadonnées, "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" est utilisé par défaut.

### 3. Mécanisme de Récupération (Retrieval)

-   **Base de données vectorielle** : Utilisation de FAISS pour stocker et récupérer les fragments de documents.
-   **Retriever MMR** : Implémentation de Maximum Marginal Relevance pour équilibrer pertinence et diversité.
    ```python
    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 3, 'fetch_k': 20} # Mise à jour des valeurs k et fetch_k
    )
    ```

### 4. Génération de Réponses

-   **Modèles de langage supportés** (sélectionnables via l'interface utilisateur) :

    Via HuggingFace (nécessite une clé API Hugging Face) :
    -   Zephyr (HuggingFaceH4/zephyr-7b-beta) - *Sélectionné via l'option "Zephyr"*
    -   Mistral (mistralai/Mistral-7B-Instruct-v0.3) - *Sélectionné via l'option "Mistral"*
    -   Phi (microsoft/Phi-3-mini-4k-instruct) - *Sélectionné via l'option "Phi"*

    Via OpenRouter (nécessite une clé API OpenRouter) :
    -   Llama 4 Maverick (meta-llama/llama-4-maverick:free) - *Sélectionné via l'option "Llama"*

-   **Structure du Prompt** :
    -   Un **Prompt Système fixe** (non modifiable par l'utilisateur) instruit le modèle sur son rôle et l'importance du sourçage.
        ```
        Tu es un agent RAG chargé de générer des réponses en t'appuyant exclusivement sur les informations fournies dans les documents de référence.

        IMPORTANT: Pour chaque information ou affirmation dans ta réponse, tu DOIS indiquer explicitement le numéro de la source (Source 1, Source 2, etc.) dont provient cette information.
        ```
    -   Un **Prompt de Requête Utilisateur par défaut (COSTAR)** qui peut être personnalisé dans l'interface.
    -   Des **Instructions Additionnelles** sont dynamiquement ajoutées pour le référencement des sources et le contexte documentaire.

## Fonctionnalités Principales

1.  **Interface utilisateur Streamlit** avec configuration dans la barre latérale.
2.  **Options de traitement flexibles** :
    -   Utilisation d'embeddings pré-calculés (chargés depuis `embeddings/`).
    -   Traitement en temps réel des documents (sauvegarde dans `vector_store/`).
    -   Option pour utiliser uniquement les fichiers téléchargés par l'utilisateur.
3.  **Personnalisation du prompt de requête** via le cadre COSTAR dans la barre latérale.
4.  **Visualisation des sources** utilisées pour générer la réponse (titre, date, fichier, personnes mentionnées, extrait du contenu) pour vérifier la fiabilité.
5.  **Support multilingue** optimisé pour les documents scientifiques en français (notamment via le modèle d'embedding `paraphrase-multilingual-MiniLM-L12-v2`).
6.  **Gestion des erreurs OCR** (via instructions dans le prompt COSTAR) avec demande de niveaux de confiance pour les informations extraites.
7.  **Affichage d'informations sur les modèles LLM** sélectionnables et leurs caractéristiques.
8.  **Gestion et affichage des fichiers XML téléchargés** par l'utilisateur.

## Utilisation

1.  Configurer les clés API dans la barre latérale (Hugging Face, et OpenRouter si le modèle Llama 4 est utilisé).
2.  Choisir entre l'utilisation d'embeddings pré-calculés ou le traitement en temps réel des documents.
    -   Si traitement en temps réel, spécifier si seuls les documents téléchargés doivent être utilisés.
3.  Sélectionner un modèle LLM parmi les options proposées.
4.  Si traitement en temps réel :
    -   Télécharger des documents XML-TEI via l'interface (optionnel si utilisation du corpus par défaut).
    -   Cliquer sur "Traiter les documents".
5.  Si embeddings pré-calculés :
    -   S'assurer que le dossier `embeddings/` contient les fichiers `faiss_index/` (avec `index.faiss` et `index.pkl`) et `document_metadata.pkl`.
    -   Cliquer sur "Charger embeddings pré-calculés".
6.  Poser des questions dans l'interface de chat.
7.  Optionnellement, modifier le prompt de requête COSTAR dans la barre latérale.

## Structure du Système

Le système est organisé autour des fonctions principales suivantes :
-   `parse_xmltei_document` : Extraction du contenu et des métadonnées des fichiers XML-TEI.
-   `load_documents` : Chargement des documents XML-TEI depuis le disque ou les fichiers téléchargés.
-   `split_documents` : Découpage des documents en fragments.
-   `embeddings_on_local_vectordb` : Création des embeddings en temps réel et de la base vectorielle FAISS (sauvegardée dans `LOCAL_VECTOR_STORE_DIR`).
-   `load_precomputed_embeddings` : Chargement des embeddings pré-calculés depuis `EMBEDDINGS_DIR`.
-   `query_llm` : Interrogation du modèle de langage avec la requête utilisateur, le contexte récupéré et la gestion des différents modèles LLM.
-   `process_documents` : Orchestration du processus de traitement des documents (chargement, découpage, création d'embeddings).
-   `input_fields` : Configuration de la barre latérale Streamlit (clés API, sélection de modèle, options de traitement, upload de fichiers, configuration du prompt).
-   `boot` : Fonction principale de l'application Streamlit, initialise l'interface et gère le flux de l'application.

## 🔄 Format des fichiers XML-TEI supportés

L'application est conçue pour traiter des documents XML-TEI avec les balises suivantes :
-   `<tei:titleStmt>/<tei:title>` pour le titre du document.
-   `<tei:sourceDesc>/<tei:p>/<tei:date>` pour la date. Si cette balise n'est pas trouvée, le système tente subsidiairement d'extraire la date depuis `<tei:sourceDesc>/<tei:p>`.
-   `<tei:p>` pour les paragraphes de contenu.
-   `<tei:persName>` pour les noms de personnes mentionnées.
La fonction `extract_year` tente d'extraire l'année (format AAAA) à partir de la chaîne de date.

## 📄 Licence

Ce projet est sous une licence open source MIT.

## 🤝 Contributions

Le projet est préparé par [Mikhail Biriuchinskii](https://www.linkedin.com/in/mikhail-biriuchinskii/), ingénieur en Traitement Automatique des Langues, équipe ObTIC, Sorbonne Université.

Pour découvrir d'autres projets de l'équipe ObTIC ainsi que les formations proposées, consultez le site : https://obtic.sorbonne-universite.fr/
