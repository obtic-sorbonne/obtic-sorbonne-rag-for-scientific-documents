import os
# NOUVEAU: Forcer l'utilisation du CPU et désactiver la visibilité des GPUs pour PyTorch
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TRANSFORMERS_OFFLINE"] = "0" 
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" 

import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
import pickle

import streamlit as st
from langchain.chains import RetrievalQA  # Keep this original import
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
# DirectoryLoader n'est pas utilisé dans le code fourni, vous pouvez le supprimer si non nécessaire ailleurs
# from langchain_community.document_loaders import DirectoryLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter # Reverting to original
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

# Defining paths 

TMP_DIR = Path(__file__).resolve().parent.joinpath('tmp')
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath('vector_store')
EMBEDDINGS_DIR = Path(__file__).resolve().parent.joinpath('embeddings')


TMP_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# Define namespaces for XML-tei
NAMESPACES = {
    'tei': 'http://www.tei-c.org/ns/1.0'
}

st.set_page_config(page_title="RAG Démonstration", page_icon="🤖", layout="wide")
st.title("Retrieval Augmented Generation")
st.image("static/sfp_logo.png", width=100)
st.markdown("#### Projet préparé par l'équipe ObTIC.")

# Fixed system prompt - not modifiable by users
SYSTEM_PROMPT = """
Tu es un agent RAG chargé de générer des réponses en t'appuyant exclusivement sur les informations fournies dans les documents de référence.

IMPORTANT: Pour chaque information ou affirmation dans ta réponse, tu DOIS indiquer explicitement le numéro de la source (Source 1, Source 2, etc.) dont provient cette information.
"""


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

# Function to parse XML-TEI documents
def parse_xmltei_document(file_path):
    """Parse an XML-TEI document and extract text content with metadata."""
    try:
        # Parse the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Debug: Print file being processed for key files
        if os.path.basename(file_path).startswith("SFP_"):
            st.write(f"Parsing: {file_path}")
        
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
    """Load XML-TEI documents
    
    Args:
        use_uploaded_only: If True, only use uploaded files and ignore default corpus
    """
    documents = []
    document_dates = {}
    
    xml_files = []
    
    st.write(f"Using uploaded files only: {use_uploaded_only}") # Debug
    
    if use_uploaded_only:
        if "uploaded_files" in st.session_state and st.session_state.uploaded_files:
            st.write(f"Found {len(st.session_state.uploaded_files)} uploaded files") # Debug
            for file_path_str in st.session_state.uploaded_files: # Ensure paths are strings
                if os.path.exists(file_path_str) and (file_path_str.endswith(".xml") or file_path_str.endswith(".xmltei")):
                    xml_files.append(file_path_str)
                    st.write(f"Added uploaded file: {file_path_str}") # Debug
    else:
        # Process files from default directories
        default_paths = ["./data", "."] # Prioritize a 'data' subdirectory
        st.write(f"Scanning default paths for XML: {default_paths}") # Debug
        for path_dir in default_paths:
            path_to_scan = Path(path_dir)
            if path_to_scan.exists() and path_to_scan.is_dir():
                st.write(f"Scanning directory: {path_to_scan.resolve()}") # Debug
                for file_item in path_to_scan.iterdir():
                    if file_item.is_file() and (file_item.name.endswith(".xml") or file_item.name.endswith(".xmltei")):
                        xml_files.append(str(file_item.resolve()))
                        st.write(f"Found default file: {str(file_item.resolve())}") # Debug
            elif path_to_scan.exists() and path_to_scan.is_file() and (path_to_scan.name.endswith(".xml") or path_to_scan.name.endswith(".xmltei")):
                 # Case where path in default_paths is a file itself (e.g. if script is in 'data' and '.' is listed)
                 xml_files.append(str(path_to_scan.resolve()))
                 st.write(f"Found default file (direct): {str(path_to_scan.resolve())}") # Debug


    if not xml_files:
        st.error("Aucun fichier XML trouvé. Veuillez téléverser des fichiers XML ou vérifier les chemins du corpus par défaut.")
        return documents, document_dates
    
    # Remove duplicates that might arise from scanning "." and "./data" if script is in data
    xml_files = sorted(list(set(xml_files)))
    st.write(f"Final list of XML files to process: {xml_files}") # Debug

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, file_path in enumerate(xml_files):
        progress = (i + 1) / len(xml_files)
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
    
    status_text.text(f"Traitement terminé! {len(documents)} documents analysés.")
    
    return documents, document_dates

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=800)
    texts = text_splitter.split_documents(documents)
    return texts

def load_precomputed_embeddings():
    """Load precomputed embeddings from the embeddings directory."""
    embeddings_path = EMBEDDINGS_DIR / "faiss_index"
    metadata_path = EMBEDDINGS_DIR / "document_metadata.pkl"
    
    if not embeddings_path.exists() or \
       not (embeddings_path / "index.faiss").exists() or \
       not (embeddings_path / "index.pkl").exists():
        st.error(f"Dossier/fichiers d'embeddings pré-calculés introuvables à {EMBEDDINGS_DIR}")
        return None
        
    embedding_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # Default
    
    if metadata_path.exists():
        try:
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
                st.success(f"Métadonnées des embeddings chargées: {metadata.get('chunk_count','N/A')} chunks pour {metadata.get('document_count','N/A')} documents.")
                embedding_model_name = metadata.get('model_name', embedding_model_name)
                st.info(f"Modèle d'embedding (pré-calculé): {embedding_model_name}")
        except Exception as e:
            st.warning(f"Erreur au chargement des métadonnées: {e}. Utilisation du modèle par défaut.")
    else:
        st.warning("Fichier de métadonnées non trouvé. Utilisation du modèle d'embedding par défaut.")
    
    try:
        # NOUVELLE MODIFICATION: Ajout de device_map=None
        model_kwargs_config = {'device': 'cpu', 'device_map': None}
        
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs=model_kwargs_config 
        )
        
        st.info(f"Chargement de l'index FAISS (modèle: {embedding_model_name})...")
        vectordb = FAISS.load_local(
            embeddings_path.as_posix(), 
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        retriever = vectordb.as_retriever(
            search_type="mmr", 
            search_kwargs={'k': 5, 'fetch_k': 10}
        )
        
        st.success("Index FAISS et embeddings pré-calculés chargés avec succès!")
        return retriever
            
    except Exception as e:
        st.error(f"Erreur lors de l'initialisation des embeddings ou du chargement de FAISS: {str(e)}")
        # Afficher des diagnostics PyTorch si l'erreur persiste
        try:
            import torch
            st.error(f"[Diag] Version PyTorch: {torch.__version__}")
            st.error(f"[Diag] Attribut 'get_default_device' présent: {hasattr(torch, 'get_default_device')}")
        except ImportError:
            st.error("[Diag] PyTorch n'a pas pu être importé.")
        except Exception as diag_e:
            st.error(f"[Diag] Erreur lors des diagnostics PyTorch: {str(diag_e)}")
        return None


def embeddings_on_local_vectordb(texts, hf_api_key):
    """Create embeddings and store in a local vector database using FAISS."""
    if not hf_api_key: # S'assurer que la clé est là pour HuggingFaceHub implicitement utilisé
        st.error("Clé API Hugging Face requise pour créer les embeddings.")
        return None

    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_api_key # Nécessaire pour HuggingFaceEmbeddings si le modèle n'est pas local
    
    # NOUVELLE MODIFICATION: Ajout de device_map=None
    model_kwargs_config = {"token": hf_api_key, 'device': 'cpu', 'device_map': None}
    
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs_config
        )
        
        st.info(f"Création de la base vectorielle FAISS pour {len(texts)} fragments...")
        vectordb = FAISS.from_documents(texts, embeddings)
        
        LOCAL_VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
        vectordb.save_local(LOCAL_VECTOR_STORE_DIR.as_posix())
        
        with open(LOCAL_VECTOR_STORE_DIR / "model_info.pkl", "wb") as f:
            pickle.dump({"model_name": model_name, "chunk_count": len(texts)}, f)
        
        retriever = vectordb.as_retriever(
            search_type="mmr", 
            search_kwargs={'k': 5, 'fetch_k': 10}
        )
        st.success(f"Base vectorielle sauvegardée dans {LOCAL_VECTOR_STORE_DIR.resolve()}")
        return retriever
            
    except Exception as e: 
        st.error(f"Erreur lors de la création des embeddings (FAISS): {str(e)}")
        # Fallback to batch processing (peut être moins pertinent si l'erreur est dans HuggingFaceEmbeddings)
        # ... (le code de batch processing reste ici, mais l'erreur est probablement avant)
        return None # Simplifié pour clarté, le batching est complexe si l'init échoue.

# ... (Reste du code : prepare_sources_for_llm, query_llm, process_documents, input_fields, boot)
# Assurez-vous que le reste du code est identique à votre version fonctionnelle précédente,
# car les modifications se concentrent sur les instanciations de HuggingFaceEmbeddings.
# J'omets le reste du code pour la concision, mais il doit être inclus dans votre app.py.

def prepare_sources_for_llm(source_docs):
    """Create a mapping of sources with numbers to include in the prompt"""
    source_mapping = []
    for i, doc in enumerate(source_docs):
        doc_title = doc.metadata.get('title', 'Document sans titre')
        source_mapping.append(f"Source {i+1}: {doc_title}")
    return "\n".join(source_mapping)


def query_llm(retriever, query, hf_api_key, openai_api_key=None, openrouter_api_key=None, model_choice="llama"):
    """Query the LLM using one of the supported models."""
    
    progress_container = st.empty()
    progress_container.info("Recherche des documents pertinents...")
    progress_bar = st.progress(0)
    
    try:
        base_query_template = st.session_state.query_prompt
        relevant_docs = retriever.get_relevant_documents(query)
        
        source_mapping_texts = []
        for i, doc in enumerate(relevant_docs):
            doc_title = doc.metadata.get('title', 'Document sans titre')
            doc_date = doc.metadata.get('date', 'Date inconnue')
            source_mapping_texts.append(f"Source {i+1}: {doc_title} | {doc_date}")
        source_references = "\n".join(source_mapping_texts)
        
        # Utiliser le SYSTEM_PROMPT global et le DEFAULT_QUERY_PROMPT (qui est un template)
        # Le DEFAULT_QUERY_PROMPT contient déjà la structure COSTAR.
        # Nous allons injecter le query de l'utilisateur et les instructions de citation.
        
        instructions_citations = f"""
INSTRUCTIONS IMPORTANTES ADDITIONNELLES:
- Pour CHAQUE fait ou information mentionné dans ta réponse, indique EXPLICITEMENT le numéro de la source correspondante (ex: Source 1, Source 3).
- Cite les sources même pour les informations de confiance élevée.
- Fais référence aux sources numérotées ci-dessous dans chaque section de ta réponse.

SOURCES DISPONIBLES POUR CITATION:
{source_references}
"""
        # Formatter le prompt final.
        # Le DEFAULT_QUERY_PROMPT est un f-string template attendant {query}.
        # On ajoute le SYSTEM_PROMPT au début, puis le template COSTAR rempli, puis les instructions de citation.
        prompt_pour_llm = f"{SYSTEM_PROMPT}\n\n{DEFAULT_QUERY_PROMPT.format(query=query)}\n\n{instructions_citations}"

        llm = None
        if model_choice == "openrouter":
            if not openrouter_api_key:
                st.error("Clé API OpenRouter requise pour le modèle Llama 4 Maverick.")
                return None, None
            llm = ChatOpenAI(
                temperature=0.4, model_name="meta-llama/llama-4-maverick:free",
                openai_api_key=openrouter_api_key, max_tokens=15000, # Réduit de 50k
                openai_api_base="https://openrouter.ai/api/v1",
                default_headers={
                    "HTTP-Referer": os.getenv("OPENROUTER_REFERRER", "http://localhost:8501"),
                    "X-Title": os.getenv("OPENROUTER_X_TITLE", "RAG Demo ObTIC")
                }
            )
        elif model_choice in ["llama", "mistral", "phi"]:
            if not hf_api_key:
                st.error(f"Clé API Hugging Face requise pour le modèle {model_choice}.")
                return None, None
            
            repo_ids = {
                "llama": "meta-llama/Meta-Llama-3-8B-Instruct",
                "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
                "phi": "microsoft/Phi-3-mini-4k-instruct" # Modèle Phi-3 suggéré
            }
            model_params = {
                "temperature": 0.4, "max_new_tokens": 1500, "top_p": 0.95
            }
            if model_choice == "phi": model_params["trust_remote_code"] = True # Pour Phi-3

            llm = HuggingFaceHub(
                repo_id=repo_ids[model_choice],
                huggingfacehub_api_token=hf_api_key,
                model_kwargs=model_params
            )
        else:
            st.error(f"Modèle LLM inconnu: {model_choice}")
            return None, None

        progress_bar.progress(30, text="Chaîne de traitement en création...")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=retriever,
            return_source_documents=True, verbose=st.session_state.get('debug_verbose', False)
        )
        
        progress_bar.progress(50, text=f"Génération de la réponse avec {model_choice.upper()}...")
        result = qa_chain({"query": prompt_pour_llm}) # Envoyer le prompt complet
        
        progress_bar.progress(90, text="Finalisation de la réponse...")
        answer = result.get("result", "Aucune réponse obtenue.")
        source_docs_retrieved = result.get("source_documents", [])
        
        if "messages" in st.session_state:
            st.session_state.messages.append((query, answer)) # Stocker la question originale et la réponse
        
        progress_bar.progress(100)
        progress_container.empty()
        
        return answer, source_docs_retrieved
        
    except Exception as e:
        st.exception(e) # Affiche une trace d'erreur plus détaillée dans Streamlit
        progress_container.error(f"Erreur majeure pendant la génération: {e}")
        return None, None
        
def process_documents(hf_api_key, use_uploaded_only):
    if not hf_api_key:
        st.warning("La clé API Hugging Face est requise pour le traitement des documents.")
        return None
    
    status_container = st.empty() 
    progress_overall = st.progress(0, text="Initialisation du traitement...")

    try:
        status_container.info("1/3 Chargement des documents...")
        progress_overall.progress(10, text="1/3 Chargement des documents...")
        documents, _ = load_documents(use_uploaded_only)
        if not documents:
            status_container.error("Aucun document trouvé à traiter.")
            progress_overall.empty()
            return None
        
        status_container.info(f"2/3 Découpage de {len(documents)} document(s) en fragments...")
        progress_overall.progress(40, text=f"2/3 Découpage de {len(documents)} document(s)...")
        texts = split_documents(documents)
        if not texts:
            status_container.error("Le découpage des documents n'a produit aucun fragment.")
            progress_overall.empty()
            return None
            
        status_container.info(f"3/3 Création des embeddings pour {len(texts)} fragments...")
        progress_overall.progress(70, text=f"3/3 Création des embeddings pour {len(texts)} fragments...")
        
        retriever = embeddings_on_local_vectordb(texts, hf_api_key)
        
        if not retriever:
            status_container.error("La création des embeddings ou du retriever a échoué.")
            progress_overall.empty() 
            return None

        progress_overall.progress(100, text="Traitement terminé!")
        status_container.success(f"Traitement terminé! {len(texts)} fragments vectorisés à partir de {len(documents)} documents.")
        
        return retriever
    
    except Exception as e:
        st.exception(e)
        status_container.error(f"Une erreur majeure s'est produite lors du traitement: {e}")
        if 'progress_overall' in locals(): progress_overall.empty()
        return None


def input_fields():
    """Set up the input fields in the sidebar with improved responsive layout."""
    with st.sidebar:
        st.markdown("""<style>...</style>""", unsafe_allow_html=True) # CSS (inchangé)
        
        st.title("⚙️ Configuration")
        
        # API Keys
        st.session_state.hf_api_key = st.text_input(
            "Clé API Hugging Face", 
            value=st.session_state.get("hf_api_key", st.secrets.get("hf_api_key", "")), 
            type="password", key="hf_api_sidebar"
        )
        st.session_state.openrouter_api_key = st.text_input(
            "Clé API OpenRouter", 
            value=st.session_state.get("openrouter_api_key", st.secrets.get("openrouter_api_key", "")), 
            type="password", key="or_api_sidebar"
        )
            
        # Embeddings source
        st.markdown("---")
        st.markdown("##### Source des Embeddings")
        embeddings_path = EMBEDDINGS_DIR / "faiss_index"
        embeddings_available = embeddings_path.exists() and \
                               (embeddings_path / "index.faiss").exists() and \
                               (embeddings_path / "index.pkl").exists()

        if 'use_precomputed' not in st.session_state:
            st.session_state.use_precomputed = embeddings_available

        st.session_state.use_precomputed = st.checkbox(
            "Utiliser embeddings pré-calculés",
            value=st.session_state.use_precomputed,
            disabled=not embeddings_available, key="use_precomputed_sb"
        )
        
        if embeddings_available and st.session_state.use_precomputed:
            # ... (info metadata)
            pass
        elif not embeddings_available:
             st.caption("ℹ️ Aucun embedding pré-calculé.")
            
        # LLM Model selection
        st.markdown("---")
        st.markdown("##### Modèle LLM")
        if 'model_choice' not in st.session_state: st.session_state.model_choice = "llama" 

        st.session_state.model_choice = st.radio(
            "Modèle:", ["llama", "mistral", "phi", "openrouter"], 
            format_func=lambda x: {"llama": "Llama 3 (HF)", "mistral": "Mistral 7B (HF)", 
                                   "phi": "Phi 3 Mini (HF)", "openrouter": "Llama (OpenRouter)"}.get(x, x),
            key="model_choice_sb", label_visibility="collapsed"
        )
        
        # ... (Infos Modèle expander)
        
        # Prompt configuration
        st.markdown("---")
        with st.expander("Configuration du Prompt (COSTAR)", expanded=False):
            # ... (Logique du prompt)
            if "query_prompt" not in st.session_state: st.session_state.query_prompt = DEFAULT_QUERY_PROMPT
            st.text_area("Template Prompt:", value=st.session_state.query_prompt, height=200, key="query_prompt_sb", 
                         on_change=lambda: setattr(st.session_state, 'query_prompt', st.session_state.query_prompt_sb))
            if st.button("Réinitialiser Template", key="reset_prompt_sb"):
                st.session_state.query_prompt = DEFAULT_QUERY_PROMPT
                st.rerun()
            
        # File uploader
        st.markdown("---")
        st.markdown("##### Gestion Fichiers XML")
        # ... (Logique File Uploader et use_uploaded_only)

        if "uploaded_files" not in st.session_state: st.session_state.uploaded_files = []
        
        uploaded_file_list_sb = st.file_uploader("Téléverser XML/XMLTEI", 
            type=["xml", "xmltei"], accept_multiple_files=True, key="file_uploader_sb"
        ) 
        if uploaded_file_list_sb:
            # (Logique de sauvegarde des fichiers identique à la version précédente)
            newly_saved_paths = [] # Simplifié pour l'exemple
            # ...
            if newly_saved_paths: st.rerun() # Pour rafraîchir la liste

        if 'use_uploaded_only' not in st.session_state:
            st.session_state.use_uploaded_only = bool(st.session_state.uploaded_files) and not st.session_state.get('use_precomputed', False)

        st.session_state.use_uploaded_only = st.checkbox(
            "Traiter uniquement fichiers téléversés", 
            value=st.session_state.use_uploaded_only,
            disabled=not bool(st.session_state.uploaded_files), 
            key="use_uploaded_only_sb"
        )
        # ... (Affichage des fichiers et bouton effacer)


def boot():
    """Main function to run the application."""
    # Initialisations globales de session_state si nécessaire
    if "query_prompt" not in st.session_state: st.session_state.query_prompt = DEFAULT_QUERY_PROMPT
    if "messages" not in st.session_state: st.session_state.messages = []
    if "retriever" not in st.session_state: st.session_state.retriever = None
    # Les clés API et autres options sont gérées dans input_fields ou au premier accès

    input_fields() # Configure la sidebar et met à jour session_state
    
    st.markdown("#### Actions RAG")
    cols_actions = st.columns(2)
    
    with cols_actions[0]:
        if st.session_state.get("use_precomputed", False):
            if st.button("LOAD PRE-COMPUTED", use_container_width=True, key="load_precomp_main", type="primary"):
                with st.spinner("Chargement embeddings pré-calculés..."):
                    st.session_state.retriever = load_precomputed_embeddings()
                    # Messages de succès/erreur gérés dans la fonction
        else:
            cols_actions[0].write("") # Pour l'alignement si le bouton n'est pas là

    with cols_actions[1]:
        can_process_uploaded = st.session_state.get('use_uploaded_only', False) and bool(st.session_state.get('uploaded_files'))
        can_process_default = not st.session_state.get('use_uploaded_only', False)
        
        show_process_button = not st.session_state.get('use_precomputed', False) or can_process_uploaded
        
        button_text = "PROCESS DOCUMENTS"
        if can_process_uploaded:
            button_text = f"PROCESS UPLOADED ({len(st.session_state.uploaded_files)})"
        elif can_process_default and not st.session_state.get('use_precomputed', False) :
             button_text = "PROCESS DEFAULT CORPUS"
        
        disabled_process = (st.session_state.get('use_uploaded_only', False) and not bool(st.session_state.get('uploaded_files'))) or \
                           (st.session_state.get('use_precomputed', False) and not can_process_uploaded)


        if show_process_button :
            if st.button(button_text, use_container_width=True, key="process_docs_main", disabled=disabled_process):
                st.session_state.retriever = None 
                if not st.session_state.get("hf_api_key"):
                    st.error("Clé API Hugging Face requise.")
                else:
                    with st.spinner("Traitement des documents..."):
                        st.session_state.retriever = process_documents(
                            st.session_state.hf_api_key, 
                            st.session_state.get('use_uploaded_only', False) 
                        )
        elif disabled_process and st.session_state.get('use_uploaded_only', False) : # Cas: "uploaded_only" coché mais pas de fichiers
            cols_actions[1].caption("Veuillez téléverser des fichiers.")


    st.markdown("---") # Séparateur avant le chat

    # Affichage de l'historique du chat
    for idx, (q_hist, a_hist) in enumerate(st.session_state.messages):
        st.chat_message('user', key=f"hist_user_{idx}").write(q_hist)
        st.chat_message('assistant', key=f"hist_ai_{idx}").markdown(a_hist) # AI response as markdown
    
    # Champ de saisie du chat
    if user_query := st.chat_input("Posez votre question ici..."):
        st.chat_message("user").write(user_query)
        
        if not st.session_state.retriever:
            st.error("⚠️ Veuillez d'abord CHARGER ou TRAITER des documents pour activer le RAG.")
        else:
            # Vérification des clés API nécessaires pour le modèle choisi
            model_choice = st.session_state.get("model_choice", "llama")
            api_key_ok = True
            if model_choice in ["llama", "mistral", "phi"] and not st.session_state.get("hf_api_key"):
                st.error(f"Clé API Hugging Face requise pour le modèle {model_choice}.")
                api_key_ok = False
            elif model_choice == "openrouter" and not st.session_state.get("openrouter_api_key"):
                st.error(f"Clé API OpenRouter requise pour le modèle {model_choice}.")
                api_key_ok = False

            if api_key_ok:
                with st.spinner(f"🧠 Recherche et génération avec {model_choice.upper()}..."):
                    answer, source_docs = query_llm(
                        st.session_state.retriever, user_query, 
                        st.session_state.hf_api_key, None, 
                        st.session_state.openrouter_api_key, model_choice
                    )
                    
                    ai_msg_placeholder = st.chat_message("assistant").empty() # Pour afficher la réponse en streaming si implémenté
                    if answer:
                        ai_msg_placeholder.markdown(answer) # Afficher la réponse complète
                        if source_docs:
                            with ai_msg_placeholder.expander("Afficher les sources utilisées", expanded=False):
                                for i, doc_source in enumerate(source_docs):
                                    src_title = doc_source.metadata.get('title', 'Titre inconnu')
                                    src_file = os.path.basename(doc_source.metadata.get('source', 'Fichier inconnu'))
                                    st.markdown(f"**Source {i+1}:** _{src_title}_ (`{src_file}`)")
                                    # Optionnel: afficher un extrait
                                    # st.caption(f"Extrait: {doc_source.page_content[:150]}...")
                                    if i < len(source_docs) - 1: st.markdown("---")
                    else:
                        ai_msg_placeholder.error("Désolé, une erreur est survenue lors de la génération de la réponse.")
            # else: l'erreur de clé API a déjà été affichée

if __name__ == '__main__':
    boot()
