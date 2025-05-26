# 🧠 Système RAG pour Documents Scientifiques (XML-TEI)

Un système **Retrieval Augmented Generation (RAG)** conçu pour interroger des documents scientifiques historiques en **français**, au format **XML-TEI**, et générer des réponses **sourcées** à l’aide de modèles de langage avancés.

---

## 🚀 Fonctionnalités Clés

- 📂 Prise en charge des fichiers **XML-TEI**
- 🔍 Recherche vectorielle avec **FAISS** et **MMR**
- 🧠 Génération de texte avec des modèles **LLM** sélectionnables
- 🧾 Réponses **sourcées** avec citations automatiques
- 🌍 Support **multilingue**, optimisé pour le français
- 🖼️ Interface **Streamlit** simple et interactive

---

## ⚙️ Installation (Pas à pas)

> **Prérequis** : Python 3.12.10

### 1. Installer Python 3.12.10

#### Sous Linux/macOS :

```bash
# Utiliser pyenv (recommandé)
curl https://pyenv.run | bash

# Ajouter pyenv à votre shell
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# Installer Python
pyenv install 3.12.10
pyenv local 3.12.10
```

#### Sous Windows :

Téléchargez Python 3.12.10 depuis : https://www.python.org/downloads/release/python-31210/

---

### 2. Créer un environnement virtuel

```bash
python -m venv .venv
source .venv/bin/activate   # Sous Windows : .venv\Scripts\activate
```

---

### 3. Installer les dépendances

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 4. Lancer l’application

```bash
streamlit run app.py
```

L’interface s’ouvrira automatiquement dans votre navigateur.

---

## 🧬 Vue d'ensemble du Pipeline

### 1. Traitement des Documents

- Chargement des fichiers `.xml` depuis `./`, `data/`, ou `data/uploaded/`
- Parsing XML-TEI : extraction du **titre**, **date**, **année**, **noms propres**, **contenu**
- Fragmentation : `RecursiveCharacterTextSplitter` (2500 caractères, chevauchement 800)

### 2. Embeddings

- **En temps réel** : via `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Ou pré-calculés** : chargés depuis `embeddings/` avec `document_metadata.pkl`

### 3. Recherche Vectorielle

- Index FAISS local
- Retrieve-and-Rerank avec **MMR** :
```python
retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={'k': 3, 'fetch_k': 20}
)
```

### 4. Génération de Réponse

- Modèles disponibles :

| Source        | Modèle                                      | Option |
|---------------|---------------------------------------------|--------|
| HuggingFace   | `HuggingFaceH4/zephyr-7b-beta`              | Zephyr |
|               | `mistralai/Mistral-7B-Instruct-v0.3`        | Mistral |
|               | `microsoft/Phi-3-mini-4k-instruct`          | Phi    |
| OpenRouter    | `meta-llama/llama-4-maverick:free`          | Llama  |

- **Prompt système** intégré (non modifiable) :
```text
Tu es un agent RAG chargé de générer des réponses en t'appuyant exclusivement sur les informations fournies dans les documents de référence.
IMPORTANT: Pour chaque information ou affirmation dans ta réponse, tu DOIS indiquer explicitement le numéro de la source (Source 1, Source 2, etc.).
```

---

## 🖥️ Interface Utilisateur

1. Ajouter vos **clés API** dans la barre latérale
2. Sélectionner un **modèle de LLM**
3. Télécharger des documents XML-TEI (ou utiliser le corpus par défaut)
4. Choisir entre :
   - Génération d’embeddings en temps réel
   - Utilisation d’embeddings pré-calculés
5. Poser une question dans le champ de requête
6. Visualiser les sources utilisées dans la réponse

---

## 🗂️ Structure du Projet

Fonctions principales :

- `parse_xmltei_document()` → parsing des fichiers XML
- `load_documents()` → chargement local ou upload
- `split_documents()` → découpage en fragments
- `embeddings_on_local_vectordb()` → embeddings + index FAISS
- `load_precomputed_embeddings()` → chargement `embeddings/`
- `query_llm()` → envoi à un LLM + gestion des modèles
- `process_documents()` → traitement complet
- `input_fields()` → configuration Streamlit
- `boot()` → fonction principale Streamlit

---

## 📄 Format TEI supporté

Les balises suivantes sont nécessaires :

- `<tei:titleStmt>/<tei:title>` → Titre
- `<tei:sourceDesc>/<tei:p>/<tei:date>` → Date
- `<tei:p>` → Contenu principal
- `<tei:persName>` → Noms propres
- `extract_year()` → Extrait l’année (format AAAA)

---

## 📜 Licence

Ce projet est distribué sous licence MIT.

---

## 🤝 Auteur

Développé par [Mikhail Biriuchinskii](https://www.linkedin.com/in/mikhail-biriuchinskii/), ingénieur TAL, équipe ObTIC, Sorbonne Université.

➡️ Plus d'infos : https://obtic.sorbonne-universite.fr/
