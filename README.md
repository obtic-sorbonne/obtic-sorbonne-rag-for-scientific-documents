# Système RAG pour Documents Scientifiques (XML-TEI)

Un système **Retrieval Augmented Generation (RAG)** conçu pour interroger des documents scientifiques historiques en **français**, au format **XML-TEI**, et générer des réponses **sourcées** à l'aide de modèles de langage.

✨ **Nouveauté** : Support complet des modèles **Ollama** locaux, incluant **DeepSeek-R1** !

---

## Fonctionnalités clés

- Prise en charge des fichiers **XML-TEI**
- Recherche vectorielle avec **FAISS** (similarité cosinus ou **MMR** – Maximal Marginal Relevance)
- **Modèles multiples** : cloud (HuggingFace, OpenRouter) + **local (Ollama)**
- **Support Ollama** : DeepSeek-R1, Llama, Mistral, et autres
- Génération de réponses **sourcées** avec citations automatiques
- Support **multilingue**, optimisé pour le français
- Chargement rapide grâce aux **embeddings pré-calculés**
- Interface utilisateur **Streamlit** moderne et intuitive

---

## Modes de déploiement

### 1. Démo en ligne (API uniquement)

Accès direct sans installation : [**Démo Streamlit Cloud**](https://langchainragdemo-clqunhvtdazahepkgvopen.streamlit.app/)

### 2. Exécution locale avec Docker + Ollama

Support complet des modèles locaux et cloud.

---

## Installation

### Option 1 : Déploiement rapide avec Docker + Ollama

#### Prérequis

- Docker et Docker Compose
- 8 GB de RAM minimum pour DeepSeek-R1

#### Étapes

```bash
git clone https://github.com/votre-repo/langchain_rag_demo.git
cd langchain_rag_demo

# Configuration Streamlit
mkdir -p .streamlit
echo "[server]
enableStaticServing = true" > .streamlit/config.toml

# Clés API
echo "[hf_api_key = \"YOUR_HF_API_KEY\" 
openrouter_api_key = \"YOUR_OPENROUTER_API_KEY\"]" > .streamlit/secrets.toml

# Lancer le script
chmod +x start-rag.sh
./start-rag.sh
```

Le script configure automatiquement :

- Le service Ollama avec DeepSeek-R1
- L'application Streamlit
- Le réseau Docker pour la communication entre services

**Accès local** : http://localhost:8502

---

### Option 2 : Installation manuelle

#### 1. Installer Python 3.12

```bash
# Avec pyenv (recommandé)
curl https://pyenv.run | bash
pyenv install 3.12.10
pyenv local 3.12.10
```

#### 2. Environnement virtuel

```bash
python -m venv .venv
source .venv/bin/activate   # Windows : .venv\Scripts\activate
```

#### 3. Dépendances

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Clés API Streamlit

```bash
mkdir -p .streamlit
echo "[server]
enableStaticServing = true" > .streamlit/config.toml

echo "[hf_api_key = \"YOUR_HF_API_KEY\"
openrouter_api_key = \"YOUR_OPENROUTER_API_KEY\"]" > .streamlit/secrets.toml
```

#### 5. Lancement de l'application

```bash
streamlit run app.py
```

---

## Architecture du pipeline

### 1. Traitement des documents

- **Sources** : `./data/`, fichiers uploadés ou corpus prédéfini
- **Parsing XML-TEI** : extraction du titre, date, auteurs et contenu
- **Fragmentation** : `RecursiveCharacterTextSplitter` (2500 caractères, chevauchement de 800)

### 2. Génération des embeddings

#### Mode rapide

Utiliser le script `generate_embeddings_pre_computed.py` pour créer des embeddings avec [intfloat/multilingual-e5-large-instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct), stockés dans `/embeddings/`.

1. Cocher **Utiliser les embeddings pré-calculés**
2. Cliquer sur **Charger les embeddings pré-calculés**

#### Mode personnalisé

Utilise [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2).

**Deux modes :**

1. **Traitement du corpus complet** dans `/data/`
   - Cliquer sur **Traiter le corpus par défaut**

2. **Traitement de fichiers uploadés uniquement**
   - Cocher **Utiliser uniquement les fichiers téléchargés**
   - Sélectionner les fichiers via **Browse files**
   - Cliquer sur **Traiter les documents**

### 3. Recherche vectorielle

**Stratégies disponibles :**
- **Cosine Similarity** : précision maximale
- **MMR** : équilibre pertinence/diversité

### 4. Modèles de langage

#### Locaux (Ollama)

| Modèle | Taille | Performance | Cas d'usage |
|--------|--------|-------------|-------------|
| **DeepSeek-R1** | 1.5B–7B | ⭐⭐⭐⭐⭐ | Raisonnement avancé, analyse fine |
| Llama 3.2 | 3B–70B | ⭐⭐⭐⭐ | Usage général, conversations |
| Mistral | 7B | ⭐⭐⭐ | Rapide, efficace |

#### Cloud (API HuggingFace / OpenRouter)

| Service | Modèle | Avantages |
|---------|--------|-----------|
| **OpenRouter** | Llama 4 Maverick | Dernière génération, gratuit |
| | Gemma-3n-e4b | Contexte étendu (32K), multilingue |
| | Qwen3-32B | Logique avancée, 131K tokens |
| **HuggingFace** | Zephyr-7B | Bonne précision factuelle |
| | Mistral-7B | Spécialisé science et extraction d'infos |

---

## Structure du projet

```
langchain_rag_demo/
├── .streamlit/
│   ├── config.toml              # Config serveur Streamlit
│   └── secrets.toml             # Clés API
├── data/                         # Documents XML-TEI
├── embeddings/                   # Index FAISS + métadonnées
├── static/                       # Fichiers statiques (logo, etc.)
├── tmp/                          # Fichiers temporaires
├── vector_store/                 # Stockage alternatif FAISS
├── app.py                        # Application principale
├── ollama_utils.py               # Utilitaires Ollama
├── generate_embeddings_pre_computed.py
├── requirements.txt              # Dépendances Python
├── Dockerfile
├── docker-compose.yml
├── start-rag.sh                  # Script de démarrage
└── README.md
```

---

## Configuration XML-TEI

### Balises requises

```xml
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <titleStmt>
      <title>Titre du document</title>
    </titleStmt>
    <sourceDesc>
      <p><date when="1995">1995</date></p>
    </sourceDesc>
  </teiHeader>
  <text>
    <body>
      <p>Contenu principal…</p>
    </body>
  </text>
</TEI>
```

### Métadonnées extraites

- **Titre** : `<tei:title>`
- **Date** : `<tei:date>` (année extraite automatiquement)
- **Contenu** : `<tei:p>`

---

## Prompt Engineering

### Framework COSTAR intégré

Optimisation structurée des requêtes et des réponses :

- **C**ontexte : Documents scientifiques XML-TEI
- **O**bjectif : Réponses factuelles basées sur les sources
- **S**tyle : Markdown structuré avec citations
- **T**on : Académique, formel et précis
- **A**udience : Chercheurs, historien·nes
- **R**éponse : Citations obligatoires, niveau de confiance

---

## Contribution

1. Fork du projet
2. Création d'une branche :
   ```bash
   git checkout -b feature/nouvelle-fonctionnalite
   ```
3. Commit :
   ```bash
   git commit -m 'Ajout nouvelle fonctionnalité'
   ```
4. Push :
   ```bash
   git push origin feature/nouvelle-fonctionnalite
   ```
5. Ouvrir une Pull Request

---

## Licence

Distribué sous licence **MIT**.

---

## Auteur & Équipe

**Développé par** [Mikhail Biriuchinskii](https://www.linkedin.com/in/mikhail-biriuchinskii/)  
Ingénieur TAL • Équipe **ObTIC**, Sorbonne Université

🔗 https://obtic.sorbonne-universite.fr/

⭐ **Si ce projet vous est utile, pensez à lui attribuer une étoile !**