import os
import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
import pickle

import streamlit as st
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

# Defining paths
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" # Using mirror for Hugging Face Hub

TMP_DIR = Path(__file__).resolve().parent.joinpath('tmp')
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath('vector_store')
EMBEDDINGS_DIR = Path(__file__).resolve().parent.joinpath('embeddings')

TMP_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True) # Ensure embeddings directory exists

# Define namespaces for XML-tei
NAMESPACES = {
    'tei': 'http://www.tei-c.org/ns/1.0'
}

st.set_page_config(page_title="RAG Démonstration", page_icon="🤖", layout="wide")
st.title("Retrieval Augmented Generation")
if os.path.exists("static/sfp_logo.png"):
    st.image("static/sfp_logo.png", width=100)
st.markdown("#### Projet préparé par l'équipe ObTIC.")

# Fixed system prompt - not modifiable by users
SYSTEM_PROMPT = """Tu es un agent RAG chargé de générer des réponses en t'appuyant exclusivement sur les informations fournies dans les documents de référence.

IMPORTANT: Pour chaque information ou affirmation dans ta réponse, tu DOIS indiquer explicitement le numéro de la source (Source 1, Source 2, etc.) dont provient cette information."""

# Default query prompt - can be modified by users
DEFAULT_QUERY_PROMPT = """Voici la requête de l'utilisateur :
{query}

# Instructions COSTAR pour traiter cette requête :

[C] **Corpus** : Documents scientifiques historiques en français, au format XML-TEI. Corpus vectorisé disponible. Présence fréquente d'erreurs OCR, notamment sur les chiffres. Entrée = question + documents pertinents.

[O] **Objectif** : Fournir des réponses factuelles et précises, exclusivement basées sur les documents fournis. L'extraction doit être claire, structurée, et signaler toute erreur OCR détectée. Ne rien inventer.

[S] **Style** : Clair et structuré. Utiliser le Markdown pour marquer la hiérarchie. Séparer les faits établis des incertitudes. Citer les documents avec exactitude.

[T] **Ton** : Professionnel et académique. Précis, neutre, et transparent quant aux limites des réponses.

[A] **Audience** : Chercheurs et historien·ne·s, en quête d'informations fiables, vérifiables et bien sourcées.

[R] **Règles de restitution** :
- Titres en **gras** - Informations citées textuellement depuis les documents
- Pour chaque information importante, indiquer explicitement le numéro de la source (ex: Source 1, Source 2, etc.)
- En l'absence d'information : écrire _"Les documents fournis ne contiennent pas cette information."_
- Chaque information doit comporter un **niveau de confiance** : Élevé / Moyen / Faible
- Chiffres présentés de manière claire et lisible
- Mettre en **gras** les informations importantes
- 4-5 phrases maximum

⚠️ **Attention aux chiffres** : les erreurs OCR sont fréquentes. Vérifier la cohérence à partir du contexte. Être prudent sur les séparateurs utilisés (espaces, virgules, points)."""

def extract_year(date_str):
    """Extract year from a date string."""
    year_match = re.search(r'\b(19\d{2}|20\d{2})\b', date_str)
    if year_match:
        return int(year_match.group(1))
    return None

def parse_xmltei_document(file_path):
    """Parse an XML-TEI document and extract text content with metadata."""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Extract document metadata for context
        title = root.find('.//tei:titleStmt/tei:title', NAMESPACES)
        title_text = title.text if title is not None else "Unknown Title"

        # Extract publication date
        date = root.find('.//tei:sourceDesc/tei:p/tei:date', NAMESPACES)
        if date is None:
            date = root.find('.//tei:sourceDesc/tei:p', NAMESPACES)
        date_text = date.text if date is not None else "Unknown Date"

        # Extract year
        year = extract_year(date_text)

        # Get all paragraphs
        paragraphs = root.findall('.//tei:p', NAMESPACES)

        # Also get all persName elements to find scientists/authors
        person_names = root.findall('.//tei:persName', NAMESPACES)
        person_text = []
        for person in person_names:
            name = ''.join(person.itertext()).strip()
            if name:
                person_text.append(name)

        # Create document header with metadata
        header = f"Document: {title_text} | Date: {date_text}\n\n"

        # Extract paragraph text
        all_paragraphs = []
        for para in paragraphs:
            para_text = ''.join(para.itertext()).strip()
            if para_text:
                all_paragraphs.append(para_text)

        # Combine header with paragraphs
        full_text = header + "\n".join(all_paragraphs)

        if person_text:
            person_section = "\n\nPersonnes mentionnées: " + ", ".join(person_text)
            full_text += person_section

        return {
            "title": title_text,
            "date": date_text,
            "year": year,
            "text": full_text,
            "persons": person_text
        }

    except Exception as e:
        st.error(f"Error parsing XML file {file_path}: {str(e)}")
        return None

def load_documents(use_uploaded_only=False):
    """Load XML-TEI documents"""
    documents = []
    document_dates = {}
    xml_files = []

    if use_uploaded_only:
        if "uploaded_files" in st.session_state and st.session_state.uploaded_files:
            for file_path in st.session_state.uploaded_files:
                if os.path.exists(file_path) and (file_path.endswith(".xml") or file_path.endswith(".xmltei")):
                    xml_files.append(file_path)
    else:
        for path in [".", "data", "data/uploaded"]: # Also check 'data/uploaded' for continuity
            if os.path.exists(path):
                for file in os.listdir(path):
                    if file.endswith(".xml") or file.endswith(".xmltei"):
                        file_path = os.path.join(path, file)
                        xml_files.append(file_path)

    if not xml_files:
        st.error("No XML files found. Please upload XML files or use the default corpus.")
        return documents, document_dates

    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Process files with progress updates
    for i, file_path in enumerate(xml_files):
        progress = (i) / len(xml_files)
        progress_bar.progress(progress)
        status_text.text(f"Traitement du fichier {i+1}/{len(xml_files)}: {os.path.basename(file_path)}")

        doc_data = parse_xmltei_document(file_path)

        if doc_data:
            doc = Document(
                page_content=doc_data["text"],
                metadata={
                    "source": file_path,
                    "title": doc_data["title"],
                    "date": doc_data["date"],
                    "year": doc_data["year"],
                    "persons": doc_data["persons"]
                }
            )
            documents.append(doc)

            if doc_data["year"]:
                document_dates[file_path] = doc_data["year"]

    progress_bar.progress(1.0)
    status_text.text(f"Traitement terminé! {len(documents)} documents analysés.")

    return documents, document_dates

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=800)
    texts = text_splitter.split_documents(documents)
    return texts

def get_embedding_model(hf_api_key):
    """Initializes and returns the HuggingFaceEmbeddings model."""
    if not hf_api_key:
        st.error("Hugging Face API key is required to initialize embedding model.")
        return None
    
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_api_key
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"} # Using CPU, change to "cuda" if GPU is available
        )
        return embeddings
    except Exception as e:
        st.error(f"Error initializing HuggingFaceEmbeddings model: {str(e)}")
        st.warning("Please ensure your Hugging Face API key is correct and the model can be loaded.")
        return None

def load_precomputed_embeddings(hf_api_key):
    """Load precomputed embeddings from the embeddings directory."""
    embeddings_path = EMBEDDINGS_DIR / "faiss_index"
    metadata_path = EMBEDDINGS_DIR / "document_metadata.pkl"
    
    if not embeddings_path.exists():
        st.error(f"Pre-computed embeddings folder not found at {embeddings_path}")
        return None
        
    if not (embeddings_path / "index.faiss").exists():
        st.error(f"FAISS index file not found at {embeddings_path}/index.faiss")
        return None
        
    if not (embeddings_path / "index.pkl").exists():
        st.error(f"Index pickle file not found at {embeddings_path}/index.pkl")
        return None
    
    embedding_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # Default
    
    if metadata_path.exists():
        try:
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
                st.success(f"Loaded pre-computed embeddings with {metadata['chunk_count']} chunks from {metadata['document_count']} documents")
                
                if 'model_name' in metadata:
                    embedding_model_name = metadata['model_name']
                    st.info(f"Embedding model used for pre-computed embeddings: {embedding_model_name}")
                else:
                    st.warning("Model information not found in metadata, using default embedding model.")
        except Exception as e:
            st.warning(f"Error loading metadata: {str(e)}")
            st.warning("Using default embedding model.")
    else:
        st.warning("Metadata file not found for pre-computed embeddings. Using default embedding model.")
    
    embeddings = get_embedding_model(hf_api_key) # Use the new helper function
    if not embeddings:
        return None

    try:
        st.info(f"Loading FAISS index with model: {embedding_model_name}")
        vectordb = FAISS.load_local(
            embeddings_path.as_posix(), 
            embeddings,
            allow_dangerous_deserialization=True # Be cautious with this in production
        )
        
        retriever = vectordb.as_retriever(
            search_type="mmr", 
            search_kwargs={'k': 5, 'fetch_k': 10}
        )
        
        st.success("FAISS index loaded successfully!")
        return retriever
        
    except Exception as e:
        st.error(f"Error loading FAISS index: {str(e)}")
        st.error("Unable to load pre-computed embeddings. You'll need to process documents instead.")
        return None

def embeddings_on_local_vectordb(texts, hf_api_key, document_count):
    """Create embeddings and store in a local vector database using FAISS."""
    if not hf_api_key:
        st.error("Hugging Face API key is required for creating embeddings.")
        return None

    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_api_key
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    embeddings = get_embedding_model(hf_api_key)
    if not embeddings:
        return None

    try:
        # Create vector store in batches to prevent potential memory issues or timeouts
        st.info("Creating vector store from documents (this may take time)...")
        batch_size = 100 # Adjust batch size based on your system's memory
        if len(texts) > batch_size:
            batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
            vectordb = FAISS.from_documents(batches[0], embeddings)
            for i, batch in enumerate(batches[1:], 1):
                st.info(f"Adding batch {i+1}/{len(batches)} to vector store...")
                vectordb.add_documents(batch)
        else:
            vectordb = FAISS.from_documents(texts, embeddings)
        
        LOCAL_VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
        vectordb.save_local(LOCAL_VECTOR_STORE_DIR.as_posix())
        
        # Save metadata including model name and chunk count
        metadata_path = LOCAL_VECTOR_STORE_DIR / "document_metadata.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump({
                "model_name": model_name,
                "chunk_count": len(texts),
                "document_count": document_count
            }, f)
            
        st.success(f"Embeddings created and saved locally at {LOCAL_VECTOR_STORE_DIR}")

        retriever = vectordb.as_retriever(
            search_type="mmr", 
            search_kwargs={'k': 5, 'fetch_k': 10}
        )
        
        return retriever
        
    except Exception as e:
        st.error(f"Error creating embeddings or saving FAISS index: {str(e)}")
        st.exception(e) # Show full traceback for debugging
        return None

def query_llm(retriever, query, hf_api_key, openrouter_api_key=None, model_choice="llama"):
    """Query the LLM using one of the supported models."""
    
    progress_container = st.empty()
    progress_container.info("Recherche des documents pertinents...")
    progress_bar = st.progress(0)
    
    try:
        relevant_docs = retriever.invoke(query)
        
        # Create a source mapping to include in the prompt
        source_mapping = []
        for i, doc in enumerate(relevant_docs):
            doc_title = doc.metadata.get('title', 'Document sans titre')
            doc_date = doc.metadata.get('date', 'Date inconnue')
            source_mapping.append(f"Source {i+1}: {doc_title} | {doc_date}")
            
        source_references = "\n".join(source_mapping)
        
        # Format the query using the template from session state
        base_query_template = st.session_state.query_prompt
        formatted_query = base_query_template.format(query=query)
        
        # Add explicit instruction to reference source numbers
        additional_instructions = f"""

INSTRUCTIONS IMPORTANTES:
- Pour CHAQUE fait ou information mentionné dans ta réponse, indique EXPLICITEMENT le numéro de la source correspondante (ex: Source 1, Source 3)
- Cite les sources même pour les informations de confiance élevée
- Fais référence aux sources numérotées ci-dessous dans chaque section de ta réponse

SOURCES DISPONIBLES:
{source_references}
"""
        
        # Complete query with source references
        complete_query = formatted_query + additional_instructions
        
        # Initialize LLM based on model choice
        llm = None
        if model_choice == "openrouter":
            if not openrouter_api_key:
                st.error("OpenRouter API key is required to use Llama 4 Maverick model.")
                return None, None
                
            llm = ChatOpenAI(
                temperature=0.4,
                model_name="meta-llama/llama-4-maverick:free",
                openai_api_key=openrouter_api_key,
                max_tokens=2000, # Adjust max_tokens for OpenRouter as well
                openai_api_base="https://openrouter.ai/api/v1",
                default_headers={
                    "HTTP-Referer": "https://your-streamlit-app.com", # Replace with your actual app URL if deployed
                    "X-Title": "RAG Démonstration Streamlit"
                }
            )
        elif model_choice == "mistral":
            if not hf_api_key:
                st.error("Hugging Face API key is required to use Mistral model.")
                return None, None
                
            llm = HuggingFaceHub(
                repo_id="mistralai/Mistral-7B-Instruct-v0.2",
                huggingfacehub_api_token=hf_api_key,
                model_kwargs={
                    "temperature": 0.4,
                    "max_new_tokens": 1000,
                    "top_p": 0.95
                }
            )
        elif model_choice == "phi":
            if not hf_api_key:
                st.error("Hugging Face API key is required to use Phi model.")
                return None, None
                
            llm = HuggingFaceHub(
                repo_id="microsoft/Phi-4-mini-instruct",
                huggingfacehub_api_token=hf_api_key,
                model_kwargs={
                    "temperature": 0.4,
                    "max_new_tokens": 1000,
                    "top_p": 0.95
                }
            )
        else: # Default to Llama
            if not hf_api_key:
                st.error("Hugging Face API key is required to use Llama model.")
                return None, None
                
            llm = HuggingFaceHub(
                repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
                huggingfacehub_api_token=hf_api_key,
                model_kwargs={
                    "temperature": 0.4,
                    "max_new_tokens": 2000, # This can be quite high, consider reducing if OOM
                    "top_p": 0.95
                }
            )
        
        if llm is None: # Should not happen if API keys are validated
            st.error("LLM model could not be initialized.")
            return None, None

        progress_bar.progress(0.3)
        progress_container.info("Création de la chaîne de traitement...")
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            verbose=True
        )
        
        progress_bar.progress(0.5)
        progress_container.info("Génération de la réponse avec le modèle " + model_choice.upper() + "...")
        
        # Use invoke instead of __call__ (fixes deprecation warning)
        result = qa_chain.invoke({"query": complete_query})
        
        progress_bar.progress(0.9)
        progress_container.info("Finalisation et mise en forme de la réponse...")
        
        answer = result["result"]
        source_docs = result["source_documents"]
        
        # Update message history
        if "messages" in st.session_state:
            st.session_state.messages.append((query, answer))
        
        progress_bar.progress(1.0)
        progress_container.empty()
        
        return answer, source_docs
        
    except Exception as e:
        progress_container.error(f"Erreur pendant la génération: {str(e)}")
        st.exception(e)  # This will show the full traceback
        return None, None

def process_documents(hf_api_key, use_uploaded_only):
    if not hf_api_key:
        st.warning("Please provide the Hugging Face API key. It's required for embeddings.")
        return None
        
    try:
        status_container = st.empty()
        status_container.info("Chargement des documents...")
        
        documents, document_dates = load_documents(use_uploaded_only)
        if not documents:
            st.error("No documents found to process. Please upload files or ensure default corpus exists.")
            return None
        
        status_container.info("Découpage des documents en fragments...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=800)
        texts = text_splitter.split_documents(documents)
        
        status_container.info(f"Création des embeddings pour {len(texts)} fragments (cela peut prendre plusieurs minutes)...")
        progress_bar = st.progress(0)
        
        # Ensure embeddings are initialized correctly before proceeding
        embeddings = get_embedding_model(hf_api_key)
        if not embeddings:
            return None # Return if embedding model failed to initialize

        progress_bar.progress(0.2)
        
        retriever = embeddings_on_local_vectordb(texts, hf_api_key, len(documents)) # Pass document_count
        
        progress_bar.progress(0.8)
        status_container.info("Finalisation...")
        
        progress_bar.progress(1.0)
        if retriever:
            status_container.success(f"Traitement terminé! {len(texts)} fragments créés à partir de {len(documents)} documents et stockés.")
        else:
            status_container.error("Le traitement des documents a échoué.")
        
        return retriever
        
    except Exception as e:
        st.error(f"Une erreur s'est produite lors du traitement des documents: {e}")
        st.exception(e)  # Show full traceback for debugging
        return None

def input_fields():
    """Set up the input fields in the sidebar with improved responsive layout."""
    with st.sidebar:
        st.markdown("""
        <style>
        .stSelectbox, .stRadio > div, .stExpander, [data-testid="stFileUploader"] {
            max-width: 100%;
            overflow-x: hidden;
        }
        .stCheckbox label p {
            font-size: 14px;
            margin-bottom: 0;
            white-space: normal;
        }
        div.row-widget.stRadio > div {
            flex-direction: column;
            margin-top: -10px;
        }
        div.row-widget.stRadio > div label {
            margin: 0;
            padding: 2px 0;
        }
        .stExpander {
            font-size: 14px;
        }
        .stExpander details summary p {
            margin-bottom: 0;
        }
        .stExpander details summary::marker {
            margin-right: 5px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.title("Configuration")
        
        # Hugging Face API Key
        # Prioritize Streamlit secrets, then environment variable, then input field
        hf_key_from_secrets = os.environ.get("HF_API_KEY") or st.secrets.get("hf_api_key")
        if hf_key_from_secrets:
            st.session_state.hf_api_key = hf_key_from_secrets
            st.success("Hugging Face API Key loaded from secrets/environment.")
        else:
            st.session_state.hf_api_key = st.text_input(
                "Hugging Face API Key (for embeddings & HF models)", 
                type="password", 
                key="hf_key",
                help="Required for embedding model and Hugging Face Hub LLMs (Llama 3, Mistral, Phi). You can set it in .streamlit/secrets.toml or as an environment variable."
            )
        
        # OpenRouter API Key
        openrouter_key_from_secrets = os.environ.get("OPENROUTER_API_KEY") or st.secrets.get("openrouter_api_key")
        if openrouter_key_from_secrets:
            st.session_state.openrouter_api_key = openrouter_key_from_secrets
            st.success("OpenRouter API Key loaded from secrets/environment.")
        else:
            st.session_state.openrouter_api_key = st.text_input(
                "OpenRouter API Key (for Llama 4 Maverick)", 
                type="password", 
                key="openrouter_key",
                help="Required for Llama 4 Maverick model. You can set it in .streamlit/secrets.toml or as an environment variable."
            )
            
        # Add option to use pre-computed embeddings
        embeddings_path = EMBEDDINGS_DIR / "faiss_index"
        embeddings_available = embeddings_path.exists() and (embeddings_path / "index.faiss").exists() and (embeddings_path / "index.pkl").exists()
        
        st.session_state.use_precomputed = st.checkbox(
            "Utiliser embeddings pré-calculés (si disponibles)",
            value=embeddings_available,
            disabled=not embeddings_available,
            key="use_precomputed_cb"
        )
        
        if embeddings_available and st.session_state.use_precomputed:
            metadata_path = EMBEDDINGS_DIR / "document_metadata.pkl"
            if metadata_path.exists():
                try:
                    with open(metadata_path, "rb") as f:
                        metadata = pickle.load(f)
                        st.info(f"Modèle d'embeddings: {metadata.get('model_name', 'Unknown')}\n"
                                f"Fragments: {metadata.get('chunk_count', 'Unknown')}\n"
                                f"Documents: {metadata.get('document_count', 'Unknown')}")
                except:
                    st.warning("Could not load metadata for pre-computed embeddings.")
            
        st.markdown("---")
            
        # Model selection
        st.session_state.model_choice = st.radio(
            "Modèle LLM",
            ["llama", "mistral", "phi", "openrouter"],
            format_func=lambda x: {
                "llama": "Llama 3 (via HF)",
                "mistral": "Mistral 7B (via HF)",
                "phi": "Phi-4-mini (via HF)",
                "openrouter": "Llama 4 Maverick (via OpenRouter)"
            }[x],
            horizontal=False,
            key="model_choice_radio"
        )

        # Model information
        with st.expander("Infos modèle", expanded=False):
            if st.session_state.model_choice == "llama":
                st.markdown("""
                **Meta-Llama-3-8B-Instruct** (via Hugging Face Hub)
                
                * Bonne compréhension des instructions
                * Fort en synthèse de documents longs
                * Précision factuelle solide
                """)
            elif st.session_state.model_choice == "mistral":
                st.markdown("""
                **Mistral-7B-Instruct-v0.2** (via Hugging Face Hub)
                
                * Raisonnement sur documents scientifiques
                * Bonne extraction d'informations
                * Réponses structurées en français
                """)
            elif st.session_state.model_choice == "phi":
                st.markdown("""
                **Phi-4-mini-instruct** (via Hugging Face Hub)
                
                * Rapide pour traitement RAG léger
                * Bon ratio performance/taille
                * Précision sur citations textuelles
                """)
            elif st.session_state.model_choice == "openrouter":
                st.markdown("""
                **Llama 4 Maverick** (via OpenRouter.ai)
                
                * Dernière génération de Llama
                * Performances supérieures
                * Excellente compréhension du français
                * Nécessite une clé API OpenRouter
                """)
        
        # Prompt configuration
        with st.expander("Configuration du prompt (COSTAR)", expanded=False):
            if "query_prompt" not in st.session_state:
                st.session_state.query_prompt = DEFAULT_QUERY_PROMPT
            
            st.markdown("##### Framework COSTAR")
            st.markdown("*Méthodologie structurée pour des réponses précises*")
            
            st.info("""
            **COSTAR** est un framework de prompting structuré:
            - **C**ontexte: Le cadre de l'analyse
            - **O**bjectif: But précis de la requête
            - **S**tyle: Format et structure
            - **T**on: Registre de langage
            - **A**udience: Destinataires de la réponse
            - **R**éponse: Format attendu
            """)
            
            st.markdown("##### Prompt de requête")
            st.session_state.query_prompt = st.text_area(
                "Prompt de requête pour le LLM (modifiez avec prudence)", # Descriptive label
                value=st.session_state.query_prompt,
                height=300,
                key="query_prompt_area",
                help="Ce prompt guide le LLM pour générer des réponses en suivant la méthodologie COSTAR."
            )
            
            if st.button("Réinitialiser le prompt", key="reset_prompt_btn"):
                st.session_state.query_prompt = DEFAULT_QUERY_PROMPT
                st.rerun()
            
        # Initialize uploaded_files in session state if not present
        if "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = []

        st.markdown("### Fichiers XML du corpus")
        
        # File uploader
        uploaded_files = st.file_uploader("Télécharger des fichiers XML/XML-TEI", 
                                          type=["xml", "xmltei"], 
                                          accept_multiple_files=True,
                                          key="file_uploader",
                                          help="Téléchargez vos propres documents XML-TEI pour le RAG.")
        
        # Process uploaded files and store them in session state
        if uploaded_files:
            new_files_saved = []
            os.makedirs("data/uploaded", exist_ok=True)
            
            for uploaded_file in uploaded_files:
                file_path = os.path.join("data/uploaded", uploaded_file.name)
                # Only save if not already present to avoid duplicates
                if file_path not in st.session_state.uploaded_files:
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    new_files_saved.append(file_path)
            
            for file_path in new_files_saved:
                if file_path not in st.session_state.uploaded_files:
                    st.session_state.uploaded_files.append(file_path)
            
            if len(new_files_saved) > 0:
                st.success(f"{len(new_files_saved)} nouveau(x) fichier(s) sauvegardé(s) dans data/uploaded.")
        
        # Display checkbox for using only uploaded files
        st.session_state.use_uploaded_only = st.checkbox(
            "Utiliser uniquement les fichiers téléchargés (désactive le corpus par défaut)",
            value=bool(st.session_state.uploaded_files), # Default to True if files uploaded
            key="use_uploaded_only_cb",
            help="Si coché, seuls les fichiers que vous avez téléchargés seront utilisés pour le RAG."
        )
        
        if st.session_state.use_uploaded_only and not st.session_state.uploaded_files:
            st.warning("Aucun fichier téléchargé. Si vous cochez cette option, le corpus sera vide.")
            
        # Display the list of uploaded files
        if st.session_state.uploaded_files:
            total_files = len(st.session_state.uploaded_files)
            with st.expander(f"Fichiers actuellement téléchargés ({total_files})", expanded=False):
                file_list_html = "<div style='max-height: 150px; overflow-y: auto;'>"
                for file_path in st.session_state.uploaded_files:
                    file_name = os.path.basename(file_path)
                    file_list_html += f"<div style='padding: 2px 0; font-size: 13px;'>✓ {file_name}</div>"
                file_list_html += "</div>"
                st.markdown(file_list_html, unsafe_allow_html=True)
                
                if st.button("Effacer tous les fichiers téléchargés", key="clear_files"):
                    # Remove files from disk
                    for file_path in st.session_state.uploaded_files:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    st.session_state.uploaded_files = []
                    st.rerun()

def boot():
    """Main function to run the application."""
    # Initialize query prompt if not present
    if "query_prompt" not in st.session_state:
        st.session_state.query_prompt = DEFAULT_QUERY_PROMPT
    
    # Setup input fields (sidebar)
    input_fields()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    
    # Add buttons for different processing methods
    col1, col2 = st.columns(2)

    # Button for pre-computed embeddings
    with col1:
        # Check if pre-computed embeddings directory exists and contains FAISS files
        embeddings_path = EMBEDDINGS_DIR / "faiss_index"
        embeddings_exist = embeddings_path.exists() and \
                           (embeddings_path / "index.faiss").exists() and \
                           (embeddings_path / "index.pkl").exists()

        if embeddings_exist:
            if st.button("Charger embeddings pré-calculés", use_container_width=True, disabled=not st.session_state.hf_api_key):
                if not st.session_state.hf_api_key:
                    st.error("Veuillez fournir la clé API Hugging Face pour charger les embeddings.")
                else:
                    with st.spinner("Chargement des embeddings pré-calculés..."):
                        st.session_state.retriever = load_precomputed_embeddings(st.session_state.hf_api_key)
        else:
            st.info("Aucun embedding pré-calculé trouvé. Veuillez traiter les documents.")

    # Button for processing documents
    with col2:
        # Always show "Traiter les documents" button
        if st.button("Traiter les documents du corpus", use_container_width=True, disabled=not st.session_state.hf_api_key):
            if not st.session_state.hf_api_key:
                st.error("Veuillez fournir la clé API Hugging Face pour créer les embeddings.")
            else:
                st.session_state.retriever = process_documents(
                    st.session_state.hf_api_key,   
                    st.session_state.use_uploaded_only
                )

    # Display chat history
    for message_query, message_answer in st.session_state.messages:
        with st.chat_message('human'):
            st.write(message_query)
        with st.chat_message('ai'):
            st.markdown(message_answer)
    
    # Chat input
    if query := st.chat_input("Posez votre question..."):
        if not st.session_state.retriever:
            st.error("Veuillez d'abord charger les embeddings ou traiter les documents avant de poser une question.")
            return
        
        # Check API key for selected model
        if st.session_state.model_choice != "openrouter" and not st.session_state.hf_api_key:
            st.error(f"La clé API Hugging Face est requise pour utiliser le modèle {st.session_state.model_choice.upper()}.")
            return
        if st.session_state.model_choice == "openrouter" and not st.session_state.openrouter_api_key:
            st.error(f"La clé API OpenRouter est requise pour utiliser le modèle Llama 4 Maverick.")
            return

        with st.chat_message("human"):
            st.write(query)
        
        with st.spinner("Génération de la réponse..."):
            try:
                answer, source_docs = query_llm(
                    st.session_state.retriever,   
                    query,   
                    st.session_state.hf_api_key,
                    st.session_state.openrouter_api_key,   
                    st.session_state.model_choice
                )
                
                # Display the answer with markdown support
                if answer:
                    response_container = st.chat_message("ai")
                    response_container.markdown(answer)
                    
                    if source_docs:
                        response_container.markdown("---")
                        response_container.markdown("**Sources:**")
                        
                        # Create an expander for each source
                        for i, doc in enumerate(source_docs):
                            # Prepare document info
                            doc_title = doc.metadata.get('title', 'Document sans titre')
                            doc_date = doc.metadata.get('date', 'Date inconnue')
                            doc_file = os.path.basename(doc.metadata.get('source', 'Fichier inconnu')) # Show only filename
                            
                            # Use expander as a button-like interface
                            with response_container.expander(f"📄 Source {i+1}: {doc_title}", expanded=False):
                                st.markdown(f"**Date:** {doc_date}")
                                st.markdown(f"**Fichier:** {doc_file}")
                                
                                # Show persons if available
                                if doc.metadata.get('persons'):
                                    persons = doc.metadata.get('persons')
                                    if isinstance(persons, list) and persons:
                                        st.markdown("**Personnes mentionnées:**")
                                        st.markdown(", ".join(persons))
                                
                                # Show content - fixed empty label warning
                                st.markdown("**Extrait:**")
                                content = doc.page_content
                                # Clean up content if needed
                                if content.startswith(f"Document: {doc_title}"):
                                    content = content.replace(f"Document: {doc_title} | Date: {doc_date}\n\n", "")
                                
                                st.text_area(f"Content of Source {i+1}", value=content, height=150, disabled=True, label_visibility="collapsed")
                else:
                    st.chat_message("ai").error("Désolé, je n'ai pas pu générer de réponse. Veuillez vérifier votre configuration et réessayer.")
                        
            except Exception as e:
                st.error(f"Une erreur inattendue s'est produite lors de la génération de la réponse: {e}")
                st.exception(e) # Show full traceback for debugging

if __name__ == '__main__':
    boot()
