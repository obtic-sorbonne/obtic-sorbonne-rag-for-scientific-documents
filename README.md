# SimpleRAG - Application de Retrieval Augmented Generation

Cette application Streamlit implémente un système de Retrieval Augmented Generation (RAG) permettant d'interroger des documents scientifiques au format XML-TEI. L'application utilise au choix plusieurs LLMs via l'API Hugging Face ou GPT-3.5 via l'API OpenAI pour générer des réponses précises à partir de votre corpus de documents.

## 🌟 Fonctionnalités

- **Interface conversationnelle** pour poser des questions sur vos documents
- **Support de multiples LLMs** : choix entre Llama 3, GPT-3.5, Mistral Small 24B et Phi-4-mini
- **Traitement de corpus personnalisé** via l'upload de fichiers XML-TEI
- **Affichage des sources** pour chaque réponse avec métadonnées détaillées
- **Personnalisation avancée** du prompt système pour ajuster les réponses
- **Visualisation des extraits** de texte pertinents pour chaque réponse

## 📋 Prérequis pour le déploiement

- Compte Streamlit (même gratuit)
- Compte Hugging Face (pour l'API key)
- Compte OpenAI (optionnel, pour utiliser GPT-3.5 avec l'API key)

## 🚀 Lancement de l'application

L'application actuelle est exécutée directement via le service Streamlit, qui prend en entrée le répertoire GitHub et construit l'application sur leur infrastructure cloud, la rendant immédiatement utilisable. Pour cela, il est nécessaire de disposer d'un compte Streamlit et de créer un projet. Les instructions sur leur site sont claires et faciles à suivre.

## 📊 Structure du projet

```
simple-rag/
├── app.py              # Application Streamlit principale
├── requirements.txt    # Dépendances Python
├── README.md           # Documentation (ce fichier)
├── .gitignore          # Fichiers ignorés par Git
└── data/               # Répertoire pour les documents à traiter par défaut
```

## 📝 Guide d'utilisation

### Configurer l'application
1. Dans la barre latérale, entrez votre clé API Hugging Face (obligatoire)
2. Si vous souhaitez utiliser GPT-3.5, entrez également votre clé API OpenAI
3. Choisissez le modèle LLM à utiliser parmi les 4 options disponibles

### Ajouter des documents
1. Téléchargez vos fichiers XML-TEI via le sélecteur de fichiers dans la barre latérale
2. Cochez "Utiliser uniquement les fichiers téléchargés" si vous ne voulez pas utiliser le corpus par défaut
3. Cliquez sur "Traiter les documents" pour indexer votre corpus (cela peut prendre un peu du temps)

### Interroger votre corpus
1. Saisissez votre question dans le champ de texte en bas de l'écran
2. Consultez la réponse générée et les sources utilisées
3. Cliquez sur les sources pour voir les extraits exacts utilisés pour la réponse

### Personnaliser les réponses
Pour ajuster le style ou le comportement des réponses, utilisez l'option "Options avancées" pour modifier le prompt système.

## 🧠 Spécifications techniques

### LLMs utilisés
- **Llama 3** : Meta-Llama-3-8B-Instruct via l'API Hugging Face
- **GPT-3.5** : gpt-3.5-turbo via l'API OpenAI
- **Mistral Small** : Mistral-Small-24B-Instruct-2501 via l'API Hugging Face  
- **Phi-4-mini** : Phi-4-mini-instruct via l'API Hugging Face
- **Température** : 0.4 
- **Tokens maximum** : 512
- **Top_p** : 0.95 (permet une diversité contrôlée dans les réponses)

### Traitement des documents
- **Technique de chunking** : [RecursiveCharacterTextSplitter](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/recursive_text_splitter/) de Langchain
- **Taille des chunks** : 1000 caractères
- **Chevauchement** : 100 caractères (assure une continuité entre les chunks)
- **Extraction des métadonnées** : titre, date, personnes mentionnées
- **Organisation** : métadonnées en en-tête pour contextualiser les chunks

### Embeddings et recherche
- **Modèle d'embedding** : [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) (optimisé pour le français)
- **Base de données vectorielle** : FAISS (rapide et efficace pour la recherche de similarité)
- **Configuration du retriever** : k=3 (récupère les 3 documents les plus pertinents)

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
