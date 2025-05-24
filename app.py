import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" 
os.environ["TRANSFORMERS_OFFLINE"] = "0" 
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" 

import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
import pickle

import streamlit as st
# Ensure torch is imported so we can patch it if needed,
# but do it carefully to not hide other import errors.
try:
    import torch
except ImportError:
    st.error("ERREUR CRITIQUE : PyTorch n'a pas pu être importé. Veuillez vérifier votre installation.")
    # Potentially exit or prevent further execution if torch is critical from the start
    # For now, we'll let downstream errors occur if torch is not found by libraries.

from langchain.chains import RetrievalQA 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

# ... (le reste de vos constantes et configurations globales) ...
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

def _apply_torch_patch():
    """Applies a monkey-patch to torch if get_default_device is missing."""
    try:
        import torch
        if not hasattr(torch, 'get_default_device'):
            st.warning(
                f"Patch appliqué à PyTorch (version {getattr(torch, '__version__', 'inconnue')}): "
                f"l'attribut 'get_default_device' est manquant. "
                f"CECI EST UN CONTOURNEMENT, veuillez réparer votre installation PyTorch."
            )
            def _dummy_get_default_device():
                # os.environ["CUDA_VISIBLE_DEVICES"] = "" devrait rendre torch.cuda.is_available() False
                if torch.cuda.is_available():
                    return 'cuda'
                return 'cpu'
            torch.get_default_device = _dummy_get_default_device
    except ImportError:
        # torch n'est pas importable, l'erreur se produira ailleurs de toute façon.
        st.error("Échec de l'import de PyTorch pour appliquer le patch.")
    except Exception as e:
        st.error(f"Erreur lors de l'application du patch à PyTorch: {e}")

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
    
    # Debug message to verify the value of use_uploaded_only
    st.write(f"Using uploaded files only: {use_uploaded_only}")
    
    if use_uploaded_only:
        # Only process uploaded files when the flag is True
        if "uploaded_files" in st.session_state and st.session_state.uploaded_files:
            st.write(f"Found {len(st.session_state.uploaded_files)} uploaded files")
            for file_path in st.session_state.uploaded_files:
                if os.path.exists(file_path) and (file_path.endswith(".xml") or file_path.endswith(".xmltei")):
                    xml_files.append(file_path)
                    st.write(f"Added uploaded file: {file_path}")
    else:
        # Process files from default directories
        for path in [".", "data"]: # Consider "data" relative to script, or absolute
            if os.path.exists(path):
                for file in os.listdir(path):
                    if file.endswith(".xml") or file.endswith(".xmltei"):
                        file_path = os.path.join(path, file)
                        xml_files.append(file_path)
    
    if not xml_files:
        st.error("No XML files found. Please upload XML files or ensure the 'data' directory is correctly populated if not using uploaded files.")
        return documents, document_dates
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Process files with progress updates
    for i, file_path in enumerate(xml_files):
        # Update progress bar and status
        progress = (i +1) / len(xml_files) 
        progress_bar.progress(progress)
        status_text.text(f"Traitement du fichier {i+1}/{len(xml_files)}: {os.path.basename(file_path)}")
        
        doc_data = parse_xmltei_document(file_path)
        
        if doc_data:
            # Create a Document object with metadata
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
            
            # Store year information
            if doc_data["year"]:
                document_dates[file_path] = doc_data["year"]
    
    # Complete the progress bar
    # progress_bar.progress(1.0) # Already at 1.0 if loop completes
    status_text.text(f"Traitement terminé! {len(documents)} documents analysés.")
    
    return documents, document_dates

def split_documents(documents):
    # Increased chunk size to 2500 and overlap to 800 for better context
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=800)
    texts = text_splitter.split_documents(documents)
    
    return texts

def load_precomputed_embeddings():
    """Load precomputed embeddings from the embeddings directory."""
    _apply_torch_patch() # Tentative de patch avant d'utiliser HuggingFaceEmbeddings

    embeddings_path = EMBEDDINGS_DIR / "faiss_index"
    metadata_path = EMBEDDINGS_DIR / "document_metadata.pkl"
    
    if not embeddings_path.exists() or \
       not (embeddings_path / "index.faiss").exists() or \
       not (embeddings_path / "index.pkl").exists():
        st.error(f"Dossier/fichiers d'embeddings pré-calculés introuvables ou incomplets dans {EMBEDDINGS_DIR}.")
        return None
        
    embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" 
    
    if metadata_path.exists():
        try:
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
                st.success(f"Métadonnées chargées: {metadata.get('chunk_count', 'N/A')} chunks de {metadata.get('document_count', 'N/A')} documents.")
                embedding_model = metadata.get('model_name', embedding_model)
                st.info(f"Modèle d'embedding (pré-calculé): {embedding_model}")
        except Exception as e:
            st.warning(f"Erreur au chargement des métadonnées: {e}. Utilisation du modèle par défaut.")
    else:
        st.warning("Fichier de métadonnées non trouvé. Utilisation du modèle d'embedding par défaut.")
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'} 
        )
        
        st.info(f"Chargement de l'index FAISS avec le modèle: {embedding_model}")
        vectordb = FAISS.load_local(
            embeddings_path.as_posix(), 
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        retriever = vectordb.as_retriever(
            search_type="mmr", 
            search_kwargs={'k': 5, 'fetch_k': 10}
        )
        
        st.success("Index FAISS chargé avec succès!")
        return retriever
            
    except Exception as e:
        st.error(f"Erreur lors de l'initialisation des embeddings ou du chargement de FAISS: {str(e)}")
        # Les diagnostics sont maintenant dans _apply_torch_patch ou seront ré-affichés si l'erreur est post-patch
        return None


def embeddings_on_local_vectordb(texts, hf_api_key):
    """Create embeddings and store in a local vector database using FAISS."""
    _apply_torch_patch() # Tentative de patch avant d'utiliser HuggingFaceEmbeddings

    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_api_key
    
    _model_kwargs = {"token": hf_api_key, "device": "cpu"} 
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=_model_kwargs
        )
        
        vectordb = FAISS.from_documents(texts, embeddings)
        
        LOCAL_VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
        vectordb.save_local(LOCAL_VECTOR_STORE_DIR.as_posix())
        
        with open(LOCAL_VECTOR_STORE_DIR / "model_info.pkl", "wb") as f:
            pickle.dump({"model_name": model_name, "chunk_count": len(texts)}, f)
        
        retriever = vectordb.as_retriever(
            search_type="mmr", 
            search_kwargs={'k': 5, 'fetch_k': 10}
        )
        return retriever
            
    except Exception as e: 
        st.error(f"Erreur lors de la création des embeddings (FAISS): {str(e)}")
        # Batch processing fallback - this might be problematic if HuggingFaceEmbeddings itself failed
        st.info("Tentative de traitement par lots...")
        try:
            # Ensure embeddings object is valid if we reach here
            if 'embeddings' not in locals() or not isinstance(embeddings, HuggingFaceEmbeddings):
                 embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=_model_kwargs)

            batch_size = 50 # Adjust as needed
            if not texts: return None # No texts to process
            
            # Initialize FAISS with the first batch
            first_batch = texts[:batch_size]
            remaining_texts = texts[batch_size:]
            if not first_batch: return None

            vectordb = FAISS.from_documents(first_batch, embeddings)

            # Add remaining documents in batches
            for i in range(0, len(remaining_texts), batch_size):
                batch = remaining_texts[i:i + batch_size]
                st.info(f"Traitement du lot {i//batch_size + 2}...")
                if batch: vectordb.add_documents(batch)
            
            vectordb.save_local(LOCAL_VECTOR_STORE_DIR.as_posix())
            with open(LOCAL_VECTOR_STORE_DIR / "model_info.pkl", "wb") as f:
                 pickle.dump({"model_name": model_name, "chunk_count": len(texts)}, f)
            retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'fetch_k': 10})
            st.success("Traitement par lots terminé avec succès.")
            return retriever
        except Exception as batch_e:
            st.error(f"Erreur lors du traitement par lots: {str(batch_e)}")
            return None

# ... (le reste de votre code : query_llm, process_documents, input_fields, boot) ...
# Assurez-vous que les fonctions query_llm, process_documents, input_fields et boot 
# sont bien présentes et inchangées par rapport à la version précédente que vous aviez, 
# car les modifications se concentrent sur _apply_torch_patch, load_precomputed_embeddings, 
# et embeddings_on_local_vectordb.

# Example of how the rest of the file would continue:

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
        # Construct COSTAR-based prompt
        base_query_template = st.session_state.query_prompt
        
        # First, retrieve relevant documents
        relevant_docs = retriever.get_relevant_documents(query)
        
        # Create a source mapping to include in the prompt
        source_mapping = []
        for i, doc in enumerate(relevant_docs):
            doc_title = doc.metadata.get('title', 'Document sans titre')
            doc_date = doc.metadata.get('date', 'Date inconnue')
            source_mapping.append(f"Source {i+1}: {doc_title} | {doc_date}")
        
        source_references = "\n".join(source_mapping)
        
        # Update system prompt to emphasize source citations
        enhanced_system_prompt = SYSTEM_PROMPT # Using the global SYSTEM_PROMPT
        
        # Enhance the query with COSTAR components and source references
        # COSTAR components are now directly in DEFAULT_QUERY_PROMPT, 
        # which is used as base_query_template.
        # The template itself should handle {query} and other COSTAR parts.
        
        # Add explicit instruction to reference source numbers
        additional_instructions = """
        INSTRUCTIONS IMPORTANTES: 
        - Pour CHAQUE fait ou information mentionné dans ta réponse, indique EXPLICITEMENT le numéro de la source correspondante (ex: Source 1, Source 3) 
        - Cite les sources même pour les informations de confiance élevée
        - Fais référence aux sources numérotées ci-dessous dans chaque section de ta réponse
        
        SOURCES DISPONIBLES:
        {}
        """.format(source_references)
        
        # For OpenAI model
        if model_choice == "openrouter":
            if not openrouter_api_key:
                st.error("OpenRouter API key is required to use Llama 4 Maverick model")
                return None, None
                
            # Use ChatOpenAI with OpenRouter base URL
            llm = ChatOpenAI(
                temperature=0.4,
                model_name="meta-llama/llama-4-maverick:free", # Ensure this model is correct / still free
                openai_api_key=openrouter_api_key,
                max_tokens=15000, 
                openai_api_base="https://openrouter.ai/api/v1",
                # For OpenRouter, system prompt is often passed in messages
                # model_kwargs might not directly support "messages" for ChatOpenAI in this way.
                # Instead, construct messages for the invoke call if needed, or rely on system prompt capabilities of the model.
                # For now, let's assume enhanced_system_prompt will be part of the chain's prompt.
                default_headers={ 
                    "HTTP-Referer": os.getenv("OPENROUTER_REFERRER", "http://localhost:8501"), 
                    "X-Title": os.getenv("OPENROUTER_X_TITLE", "RAG Demo ObTIC") 
                }
            )
        elif model_choice == "mistral":
            if not hf_api_key:
                st.error("Hugging Face API key is required to use Mistral model")
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
        elif model_choice == "phi": # Assuming Phi-3
            if not hf_api_key:
                st.error("Hugging Face API key is required to use Phi model")
                return None, None
                
            llm = HuggingFaceHub(
                repo_id="microsoft/Phi-3-mini-4k-instruct", 
                huggingfacehub_api_token=hf_api_key,
                model_kwargs={
                    "temperature": 0.4,
                    "max_new_tokens": 1000,
                    "top_p": 0.95,
                    "trust_remote_code": True # Phi-3 might need this
                }
            )
        else: # Default Llama model
            if not hf_api_key: 
                st.error("Hugging Face API key is required to use Llama model")
                return None, None

            llm = HuggingFaceHub(
                repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
                huggingfacehub_api_token=hf_api_key,
                model_kwargs={
                    "temperature": 0.4,
                    "max_new_tokens": 2000,
                    "top_p": 0.95
                }
            )
        
        progress_bar.progress(0.3)
        progress_container.info("Création de la chaîne de traitement...")
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff", # "stuff" chain type might not use system prompts in the same way as ChatModels.
                               # The prompt formatting becomes crucial.
            retriever=retriever,
            return_source_documents=True,
            verbose=st.session_state.get('debug_verbose', False) # Make verbosity configurable
        )
        
        progress_bar.progress(0.5)
        progress_container.info("Génération de la réponse avec le modèle " + model_choice.upper() + "...")
        
        # Construct the final query string for the "stuff" chain
        # The DEFAULT_QUERY_PROMPT is a template. We fill {query}.
        # The SYSTEM_PROMPT and additional_instructions about sources need to be part of this.
        
        # For "stuff" chain, the context (retrieved docs) will be stuffed into the LLM prompt.
        # The query to qa_chain should be the user's question, potentially augmented.
        # Let's ensure the LLM gets all necessary instructions.
        # A common way is to format the prompt for the LLM within the chain,
        # or ensure the final query passed to qa_chain contains all instructions.

        # The `RetrievalQA` chain of type "stuff" will construct a prompt that includes
        # the retrieved documents and the input query. We need to ensure our
        # instructions (system prompt, COSTAR, source citing) are part of the prompt
        # that the LLM finally sees.
        # The `DEFAULT_QUERY_PROMPT` is designed to take the user's {query}.
        # We can prepend the system prompt and append source instructions to the user's query itself.
        
        final_query_for_llm = f"{enhanced_system_prompt}\n\n{base_query_template.format(query=query)}\n\n{additional_instructions}"
        
        result = qa_chain({"query": final_query_for_llm}) # Pass the fully formed prompt query
        
        progress_bar.progress(0.9)
        progress_container.info("Finalisation et mise en forme de la réponse...")
        
        answer = result["result"]
        source_docs = result["source_documents"]
        
        if "messages" in st.session_state:
            st.session_state.messages.append((query, answer)) # Store original query and final answer
        
        progress_bar.progress(1.0)
        progress_container.empty()
        
        return answer, source_docs
        
    except Exception as e:
        st.exception(e) # Provides a more detailed traceback in Streamlit for debugging
        progress_container.error(f"Erreur pendant la génération: {str(e)}")
        return None, None
        
def process_documents(hf_api_key, use_uploaded_only):
    if not hf_api_key: 
        st.warning("La clé API Hugging Face est requise pour le traitement des documents.")
        return None
    
    status_container = st.empty() # For messages like "Chargement..."
    progress_overall = st.progress(0, text="Initialisation du traitement...")

    try:
        status_container.info("1/3 Chargement des documents...")
        progress_overall.progress(10, text="1/3 Chargement des documents...")
        documents, _ = load_documents(use_uploaded_only) # document_dates not used here
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
            progress_overall.empty() # Clear progress bar on definitive failure
            return None

        progress_overall.progress(100, text="Traitement terminé!")
        status_container.success(f"Traitement terminé! {len(texts)} fragments vectorisés à partir de {len(documents)} documents.")
        
        return retriever
    
    except Exception as e:
        st.exception(e)
        status_container.error(f"Une erreur majeure s'est produite lors du traitement: {e}")
        progress_overall.empty()
        return None
    finally:
        # Ensure status messages are cleared if needed, or progress bar finalized
        # status_container.empty() # Or let success/error message persist
        pass


def input_fields():
    """Set up the input fields in the sidebar with improved responsive layout."""
    with st.sidebar:
        # Apply custom CSS to make sidebar elements more compact and responsive
        st.markdown("""
        <style>
        .stSelectbox, .stRadio > div, .stExpander, [data-testid="stFileUploader"] {
            max-width: 100%; /* Ensure elements fit within sidebar */
            overflow-x: hidden;
        }
        .stCheckbox label p { /* Target checkbox label paragraph */
            font-size: 14px;
            margin-bottom: 0; /* Reduce bottom margin */
            white-space: normal; /* Allow text wrapping */
        }
        div.row-widget.stRadio > div { /* Target radio button container */
            flex-direction: column; /* Stack radio buttons vertically */
            margin-top: -10px; /* Adjust top margin */
        }
        div.row-widget.stRadio > div label { /* Target individual radio labels */
            margin: 0; /* Remove default margins */
            padding: 2px 0; /* Add small padding */
        }
        .stExpander {
            font-size: 14px;
        }
        .stExpander details summary p {
            margin-bottom: 0;
        }
        .stExpander details summary::marker { /* Style the expander arrow */
            margin-right: 5px; 
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.title("⚙️ Configuration")
        
        # API Keys
        st.session_state.hf_api_key = st.text_input(
            "Clé API Hugging Face", 
            value=st.session_state.get("hf_api_key", st.secrets.get("hf_api_key", "")), 
            type="password", 
            key="hf_api_input_sidebar"
        )
        st.session_state.openrouter_api_key = st.text_input(
            "Clé API OpenRouter", 
            value=st.session_state.get("openrouter_api_key", st.secrets.get("openrouter_api_key", "")), 
            type="password", 
            key="openrouter_api_input_sidebar"
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
            disabled=not embeddings_available,
            key="use_precomputed_checkbox_sidebar",
            help="Si des embeddings pré-calculés sont disponibles, cochez pour les utiliser. Sinon, les documents seront traités à la demande."
        )
        
        if embeddings_available and st.session_state.use_precomputed:
            # Display info about precomputed embeddings
            metadata_path = EMBEDDINGS_DIR / "document_metadata.pkl"
            if metadata_path.exists():
                try:
                    with open(metadata_path, "rb") as f:
                        metadata = pickle.load(f)
                        model_name_display = metadata.get('model_name', 'Inconnu')
                        st.caption(f"ℹ️ Modèle (pré-calculé): `{model_name_display}`")
                except Exception: pass # Silently ignore if metadata fails here
        elif not embeddings_available:
             st.caption("ℹ️ Aucun embedding pré-calculé détecté.")
            
        # LLM Model selection
        st.markdown("---")
        st.markdown("##### Modèle LLM")
        if 'model_choice' not in st.session_state:
            st.session_state.model_choice = "llama" 

        st.session_state.model_choice = st.radio(
            "Choisir le modèle de langage:",
            ["llama", "mistral", "phi", "openrouter"], 
            format_func=lambda x: {
                "llama": "Llama 3 (via HF)",
                "mistral": "Mistral 7B (via HF)",
                "phi": "Phi 3 Mini (via HF)", # Updated model name
                "openrouter": "Llama (via OpenRouter)" 
            }.get(x, x.capitalize()),
            horizontal=False, 
            key="model_choice_radio_sidebar",
            label_visibility="collapsed"
        )
        
        with st.expander("Infos sur les modèles", expanded=False):
            model_info_text = {
                "llama": "**Meta-Llama-3-8B-Instruct (via HuggingFaceHub)**\n\n* Compréhension avancée, synthèse de documents longs.\n* Nécessite une clé API Hugging Face.",
                "mistral": "**Mistral-7B-Instruct-v0.2 (via HuggingFaceHub)**\n\n* Bonnes capacités de raisonnement, extraction d'info.\n* Nécessite une clé API Hugging Face.",
                "phi": "**Microsoft Phi-3-mini-4k-instruct (via HuggingFaceHub)**\n\n* Léger et rapide, bon pour RAG simple.\n* Nécessite une clé API Hugging Face et `trust_remote_code=True`.",
                "openrouter": "**Modèle Llama (gratuit ou payant via OpenRouter)**\n\n* Haute performance, excellente compréhension du français.\n* Nécessite une clé API OpenRouter."
            }
            st.markdown(model_info_text.get(st.session_state.model_choice, "Information non disponible."))
        
        # Prompt configuration
        st.markdown("---")
        with st.expander("Configuration du Prompt (COSTAR)", expanded=False):
            if "query_prompt" not in st.session_state:
                st.session_state.query_prompt = DEFAULT_QUERY_PROMPT
            
            st.markdown("###### Framework COSTAR")
            st.caption("*Méthodologie structurée pour des réponses précises et contextuelles.*")
            # COSTAR explanation can be simplified or removed if space is an issue
            
            st.markdown("###### Template du Prompt Utilisateur (modifiable)")
            st.session_state.query_prompt = st.text_area(
                "Template du prompt", 
                value=st.session_state.query_prompt,
                height=250, # Reduced height
                key="query_prompt_area_sidebar",
                label_visibility="collapsed",
                help="Modifiez ce template pour changer la manière dont la requête de l'utilisateur est formatée pour le LLM. `{query}` sera remplacé par la question de l'utilisateur."
            )
            
            if st.button("Réinitialiser le template", key="reset_prompt_btn_sidebar", use_container_width=True):
                st.session_state.query_prompt = DEFAULT_QUERY_PROMPT
                st.rerun()
            
        # File uploader section
        st.markdown("---")
        st.markdown("##### Gestion des Fichiers XML")
        if "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = []
        
        uploaded_file_list = st.file_uploader("Télécharger fichiers XML/XMLTEI", 
                                            type=["xml", "xmltei"], 
                                            accept_multiple_files=True,
                                            key="file_uploader_widget_sidebar",
                                            help="Téléchargez vos propres documents XML-TEI. Ils seront stockés temporairement.")
        
        if uploaded_file_list: # This block handles new uploads
            # ... (same upload handling logic as before, ensure it's robust)
            newly_saved_paths = []
            upload_dir = TMP_DIR / "uploaded_xml" 
            upload_dir.mkdir(parents=True, exist_ok=True)
            
            for uploaded_file_obj in uploaded_file_list:
                safe_filename = "".join(c if c.isalnum() or c in ('.', '_', '-') else '_' for c in uploaded_file_obj.name)
                file_path = upload_dir / safe_filename
                try:
                    with open(file_path, "wb") as f: f.write(uploaded_file_obj.getbuffer())
                    newly_saved_paths.append(file_path.as_posix())
                except Exception as e: st.warning(f"Erreur sauvegarde {uploaded_file_obj.name}: {e}")
            
            current_files_set = set(st.session_state.uploaded_files)
            added_count = 0
            for fp_str in newly_saved_paths:
                if fp_str not in current_files_set:
                    st.session_state.uploaded_files.append(fp_str)
                    current_files_set.add(fp_str)
                    added_count +=1
            if added_count > 0:
                st.success(f"{added_count} nouveau(x) fichier(s) ajouté(s).")
                # st.rerun() # Rerun might be too disruptive here, let user trigger processing

        # Initialize use_uploaded_only if not in session_state
        if 'use_uploaded_only' not in st.session_state:
            st.session_state.use_uploaded_only = bool(st.session_state.uploaded_files) and not st.session_state.get('use_precomputed', False)

        st.session_state.use_uploaded_only = st.checkbox(
            "Traiter uniquement les fichiers téléchargés", 
            value=st.session_state.use_uploaded_only,
            disabled=not bool(st.session_state.uploaded_files), 
            key="use_uploaded_only_checkbox_sidebar",
            help="Si coché, seuls les fichiers que vous avez téléchargés seront traités. Sinon, le corpus par défaut sera utilisé (sauf si 'Utiliser embeddings pré-calculés' est actif)."
        )
        
        if st.session_state.use_uploaded_only and not st.session_state.uploaded_files:
            st.caption("⚠️ Aucun fichier téléchargé à traiter.")
        
        if st.session_state.uploaded_files:
            total_files = len(st.session_state.uploaded_files)
            with st.expander(f"Fichiers téléchargés ({total_files})", expanded=False):
                # ... (same display logic as before)
                file_list_html = "<div style='max-height: 100px; overflow-y: auto; font-size: 12px;'>"
                for file_path_str in st.session_state.uploaded_files:
                    file_list_html += f"<div style='padding: 1px 0;'>✓ {os.path.basename(file_path_str)}</div>"
                file_list_html += "</div>"
                st.markdown(file_list_html, unsafe_allow_html=True)
                
                if st.button("Effacer la liste", key="clear_uploaded_files_btn_sidebar", use_container_width=True):
                    st.session_state.uploaded_files = []
                    st.session_state.use_uploaded_only = False 
                    st.rerun()

def boot():
    """Main function to run the application."""
    # Initialize session state variables first
    if "query_prompt" not in st.session_state: st.session_state.query_prompt = DEFAULT_QUERY_PROMPT
    if "messages" not in st.session_state: st.session_state.messages = []
    if "retriever" not in st.session_state: st.session_state.retriever = None
    if "hf_api_key" not in st.session_state: st.session_state.hf_api_key = "" # Init for input_fields
    if "openrouter_api_key" not in st.session_state: st.session_state.openrouter_api_key = "" # Init for input_fields
    # Other session state vars like 'use_precomputed', 'model_choice', 'uploaded_files', 'use_uploaded_only'
    # are initialized within input_fields() if not present.

    input_fields() # Setup sidebar, which initializes/updates session_state variables
    
    # Main area for action buttons
    st.markdown("#### Actions RAG")
    cols_actions = st.columns(2)
    
    with cols_actions[0]:
        if st.session_state.get("use_precomputed", False):
            if st.button("LOAD PRE-COMPUTED", use_container_width=True, key="load_precomputed_main_btn_v2", type="primary"):
                with st.spinner("Chargement des embeddings pré-calculés..."):
                    st.session_state.retriever = load_precomputed_embeddings()
                    if st.session_state.retriever: st.success("✅ Embeddings pré-calculés chargés.")
                    # Error is handled in load_precomputed_embeddings
        else:
            cols_actions[0].caption(" ") # Placeholder to keep layout consistent if button not shown

    with cols_actions[1]:
        button_text_process = "PROCESS DOCUMENTS"
        can_process_default = not st.session_state.get('use_uploaded_only', False)
        can_process_uploaded = st.session_state.get('use_uploaded_only', False) and bool(st.session_state.get('uploaded_files'))
        
        # Determine button text based on what will be processed
        if st.session_state.get('use_precomputed', False) and not can_process_uploaded:
             # If using precomputed and not specifically processing uploaded files, this button might be less relevant or disabled
             pass # No button or a disabled one
        else:
            if can_process_uploaded:
                button_text_process = f"PROCESS UPLOADED ({len(st.session_state.uploaded_files)})"
            elif can_process_default:
                button_text_process = "PROCESS DEFAULT CORPUS"
            
            # Disable button if "uploaded_only" is checked but no files are present
            disable_process_button = (st.session_state.get('use_uploaded_only', False) and not bool(st.session_state.get('uploaded_files')))
            # Also disable if using precomputed and not specifically targeting uploaded files
            if st.session_state.get('use_precomputed', False) and not can_process_uploaded :
                 disable_process_button = True


            if st.button(button_text_process, use_container_width=True, key="process_documents_main_btn_v2", disabled=disable_process_button):
                st.session_state.retriever = None # Clear previous retriever
                if not st.session_state.get("hf_api_key"):
                    st.error("Clé API Hugging Face requise pour le traitement.")
                else:
                    with st.spinner("Traitement des documents en cours..."):
                        st.session_state.retriever = process_documents(
                            st.session_state.hf_api_key, 
                            st.session_state.get('use_uploaded_only', False) 
                        )
                        # process_documents handles its own success/error messages now
    st.markdown("---")

    # Display chat history
    for idx, (q, a) in enumerate(st.session_state.messages):
        st.chat_message('user', key=f"user_msg_hist_{idx}").write(q)
        st.chat_message('assistant', key=f"ai_msg_hist_{idx}").markdown(a)
    
    # Chat input
    if prompt := st.chat_input("Posez votre question ici..."):
        st.chat_message("user").write(prompt)
        
        if not st.session_state.retriever:
            st.error("⚠️ Veuillez d'abord charger/traiter des documents pour activer le RAG.")
        else:
            # API key checks based on model choice
            api_key_ok = True
            model_choice = st.session_state.get("model_choice", "llama")
            if model_choice in ["llama", "mistral", "phi"] and not st.session_state.get("hf_api_key"):
                st.error(f"Clé API Hugging Face requise pour {model_choice}.")
                api_key_ok = False
            elif model_choice == "openrouter" and not st.session_state.get("openrouter_api_key"):
                st.error(f"Clé API OpenRouter requise pour le modèle {model_choice}.")
                api_key_ok = False

            if api_key_ok:
                with st.spinner(f"🧠 Recherche et génération avec {model_choice}..."):
                    answer, source_docs = query_llm(
                        st.session_state.retriever, 
                        prompt, 
                        st.session_state.hf_api_key,
                        None, # openai_api_key
                        st.session_state.openrouter_api_key, 
                        model_choice
                    )
                    
                    ai_message_placeholder = st.chat_message("assistant").empty()
                    if answer:
                        ai_message_placeholder.markdown(answer)
                        if source_docs:
                            with ai_message_placeholder.expander("Voir les sources utilisées", expanded=False):
                                for i, doc in enumerate(source_docs):
                                    doc_title = doc.metadata.get('title', 'Document sans titre')
                                    doc_file = os.path.basename(doc.metadata.get('source', 'Fichier inconnu'))
                                    st.markdown(f"**Source {i+1}:** *{doc_title}* (`{doc_file}`)")
                                    # Optionally show a snippet:
                                    # st.caption(doc.page_content[:200] + "...")
                                    st.markdown("---")
                    else:
                        ai_message_placeholder.error("Désolé, je n'ai pas pu générer de réponse.")
            # else: API key error already handled by st.error

if __name__ == '__main__':
    _apply_torch_patch() # Appliquer le patch une fois au démarrage globalement si possible,
                         # ou s'assurer qu'il est appelé avant chaque utilisation critique de torch.
                         # Les appels dans les fonctions sont plus ciblés.
                         # Pour un script Streamlit, le flux d'exécution peut être complexe.
                         # Le plus sûr est de l'appeler dans les fonctions juste avant l'instanciation.
    boot()
