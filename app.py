import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" # NOUVEAU: Tenter de forcer l'utilisation du CPU globalement pour torch
os.environ["TRANSFORMERS_OFFLINE"] = "0" # Assurez-vous que le mode hors ligne est désactivé
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" # Utiliser le miroir HF

import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
import pickle

import streamlit as st
from langchain.chains import RetrievalQA # Keep this original import
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
# from langchain_community.document_loaders import DirectoryLoader # Not used in the provided code snippet relevant to the fix, can be kept if used elsewhere
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
        progress = (i) / len(xml_files) # progress = (i + 1) for 1-based progress display
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
    progress_bar.progress(1.0)
    status_text.text(f"Traitement terminé! {len(documents)} documents analysés.")
    
    return documents, document_dates

def split_documents(documents):
    # Increased chunk size to 2500 and overlap to 800 for better context
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=800)
    texts = text_splitter.split_documents(documents)
    
    return texts

def load_precomputed_embeddings():
    """Load precomputed embeddings from the embeddings directory."""
    embeddings_path = EMBEDDINGS_DIR / "faiss_index"
    metadata_path = EMBEDDINGS_DIR / "document_metadata.pkl"
    
    # First check if paths exist
    if not embeddings_path.exists():
        st.error(f"Pre-computed embeddings folder not found at {embeddings_path}")
        return None
        
    if not (embeddings_path / "index.faiss").exists():
        st.error(f"FAISS index file not found at {embeddings_path}/index.faiss")
        return None
        
    if not (embeddings_path / "index.pkl").exists():
        st.error(f"Index pickle file not found at {embeddings_path}/index.pkl")
        return None
    
    # Load metadata to get model information
    embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # Default model
    
    if metadata_path.exists():
        try:
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
                st.success(f"Loaded pre-computed embeddings with {metadata['chunk_count']} chunks from {metadata['document_count']} documents")
                
                # Get the model name from metadata if available
                if 'model_name' in metadata:
                    embedding_model = metadata['model_name']
                    st.info(f"Embedding model: {embedding_model}")
                else:
                    st.warning("Model information not found in metadata, using default model")
        except Exception as e:
            st.warning(f"Error loading metadata: {str(e)}")
            st.warning("Using default embedding model")
    else:
        st.warning("Metadata file not found. Using default embedding model.")
    
    try:
        # Initialize the embeddings model using the model from metadata
        # from langchain_community.embeddings import HuggingFaceEmbeddings # Already imported globally
        
        # Use the same model that created the embeddings
        # MODIFICATION: Added model_kwargs={'device': 'cpu'}
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'} 
        )
        
        # Try to load the FAISS index
        try:
            # from langchain_community.vectorstores import FAISS # Already imported globally
            
            # Load with allow_dangerous_deserialization
            st.info(f"Loading FAISS index with model: {embedding_model}")
            vectordb = FAISS.load_local(
                embeddings_path.as_posix(), 
                embeddings,
                allow_dangerous_deserialization=True
            )
            
            # Create retriever
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
    
    except Exception as e:
        st.error(f"Error in embeddings initialization: {str(e)}") # This is where the original error was caught
        # Diagnostic print for PyTorch state if error persists
        try:
            import torch
            st.error(f"[Diag] PyTorch version: {torch.__version__}")
            st.error(f"[Diag] Has 'get_default_device': {hasattr(torch, 'get_default_device')}")
        except ImportError:
            st.error("[Diag] PyTorch could not be imported.")
        except Exception as diag_e:
            st.error(f"[Diag] Error during PyTorch diagnostics: {str(diag_e)}")
        return None


def embeddings_on_local_vectordb(texts, hf_api_key):
    """Create embeddings and store in a local vector database using FAISS.
    This function always uses the paraphrase-multilingual-MiniLM-L12-v2 model for real-time embedding.
    """
    # import os # Already imported globally
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_api_key
    
    # MODIFICATION: Added 'device': 'cpu' to model_kwargs
    _model_kwargs = {"token": hf_api_key, "device": "cpu"} 
    
    # Always use this model for real-time processing
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=_model_kwargs # Use the modified kwargs
        )
        
        # Create vector database
        vectordb = FAISS.from_documents(texts, embeddings)
        
        # Make sure directory exists
        LOCAL_VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save vector database
        vectordb.save_local(LOCAL_VECTOR_STORE_DIR.as_posix())
        
        # Also save model information
        with open(LOCAL_VECTOR_STORE_DIR / "model_info.pkl", "wb") as f:
            pickle.dump({
                "model_name": model_name,
                "chunk_count": len(texts)
            }, f)
        
        # Create retriever
        retriever = vectordb.as_retriever(
            search_type="mmr", 
            search_kwargs={'k': 5, 'fetch_k': 10}
        )
        
        return retriever
            
    except Exception as e: # Catching error during HuggingFaceEmbeddings init or FAISS processing
        st.error(f"Error creating embeddings: {str(e)}")
        # Diagnostic print for PyTorch state if error persists
        try:
            import torch
            st.error(f"[Diag] PyTorch version: {torch.__version__}")
            st.error(f"[Diag] Has 'get_default_device': {hasattr(torch, 'get_default_device')}")
        except ImportError:
            st.error("[Diag] PyTorch could not be imported.")
        except Exception as diag_e:
            st.error(f"[Diag] Error during PyTorch diagnostics: {str(diag_e)}")

        # Try batching approach if regular approach fails (though if HFEmbeddings fails, this won't be reached)
        # This batching logic might be better placed if FAISS.from_documents is the failing part, not HFEmbeddings
        # For now, keeping it as is, but the error above is more likely from HFEmbeddings.
        try:
            st.info("Trying batch processing approach (if previous error was not from embedding model init)...")
            
            # Re-initialize embeddings here if the first attempt failed before FAISS.
            # However, if HuggingFaceEmbeddings itself fails, this code path might not make sense.
            # Assuming 'embeddings' object might exist if only FAISS.from_documents failed.
            if 'embeddings' not in locals(): # If embeddings object was not created
                 embeddings = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs=_model_kwargs
                )

            batch_size = 50
            batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
            
            if not batches:
                st.error("No text batches to process.")
                return None

            vectordb = FAISS.from_documents(batches[0], embeddings)
            
            for i, batch in enumerate(batches[1:], 1):
                st.info(f"Processing batch {i+1}/{len(batches)}...")
                vectordb.add_documents(batch)
            
            vectordb.save_local(LOCAL_VECTOR_STORE_DIR.as_posix())
            
            with open(LOCAL_VECTOR_STORE_DIR / "model_info.pkl", "wb") as f:
                pickle.dump({
                    "model_name": model_name,
                    "chunk_count": len(texts)
                }, f)
            
            retriever = vectordb.as_retriever(
                search_type="mmr", 
                search_kwargs={'k': 5, 'fetch_k': 10}
            )
            
            return retriever
            
        except Exception as batch_e:
            st.error(f"Error with batch processing: {str(batch_e)}")
            return None

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
        enhanced_system_prompt = """
        Tu es un agent RAG chargé de générer des réponses en t'appuyant exclusivement sur les informations fournies dans les documents de référence.
        
        IMPORTANT: Pour chaque information ou affirmation dans ta réponse, tu DOIS indiquer explicitement le numéro de la source (Source 1, Source 2, etc.) dont provient cette information.
        """
        
        # Enhance the query with COSTAR components and source references
        costar_query = {
            "query": query,
            "context": "Analyse des documents scientifiques historiques en français.",
            "objective": f"Réponds précisément à la question: {query}",
            "style": "Factuel, précis et structuré avec formatage markdown.",
            "tone": "Académique et objectif.",
            "audience": "Chercheurs et historiens travaillant sur des documents scientifiques.",
            "response_format": "Structure en sections avec citations exactes, niveau de confiance et numéro de source explicite."
        }
        
        # Format the query using the template
        query_prompt_template = base_query_template
        
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
                max_tokens=15000, # Reduced from 50000 as it's very large
                openai_api_base="https://openrouter.ai/api/v1",
                model_kwargs={
                    "messages": [
                        {"role": "system", "content": enhanced_system_prompt}
                    ]
                },
                default_headers={ # Referer might be needed by OpenRouter
                    "HTTP-Referer": os.getenv("OPENROUTER_REFERRER", "http://localhost:8501"), # Example
                    "X-Title": os.getenv("OPENROUTER_X_TITLE", "RAG Demo ObTIC") # Example
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
                    "max_new_tokens": 1000, # max_length is deprecated, use max_new_tokens
                    "top_p": 0.95
                }
            )
        elif model_choice == "phi":
            if not hf_api_key:
                st.error("Hugging Face API key is required to use Phi model")
                return None, None
                
            llm = HuggingFaceHub(
                repo_id="microsoft/Phi-3-mini-4k-instruct", # Changed to Phi-3 as Phi-4-mini-instruct might not be on HF Hub directly
                huggingfacehub_api_token=hf_api_key,
                model_kwargs={
                    "temperature": 0.4,
                    "max_new_tokens": 1000,
                    "top_p": 0.95
                    # "trust_remote_code": True # May be needed for some models like Phi-3
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
        
        # Update progress
        progress_bar.progress(0.3)
        progress_container.info("Création de la chaîne de traitement...")
        
        # Use the original import
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            verbose=True # Prints to console, not Streamlit UI
        )
        
        # Update progress
        progress_bar.progress(0.5)
        progress_container.info("Génération de la réponse avec le modèle " + model_choice.upper() + "...")
        
        # Use the COSTAR-enhanced query template
        enh_query = query_prompt_template.format(
            query=query,
            context=costar_query["context"],
            objective=costar_query["objective"],
            style=costar_query["style"],
            tone=costar_query["tone"],
            audience=costar_query["audience"],
            response_format=costar_query["response_format"]
        )
        
        # Add the source references and additional instructions
        enh_query = enh_query + "\n\n" + additional_instructions
        
        # Generate response
        result = qa_chain({"query": enh_query})
        
        # Update progress
        progress_bar.progress(0.9)
        progress_container.info("Finalisation et mise en forme de la réponse...")
        
        answer = result["result"]
        source_docs = result["source_documents"]
        
        # Update message history
        if "messages" in st.session_state:
            st.session_state.messages.append((query, answer))
        
        # Complete progress
        progress_bar.progress(1.0)
        progress_container.empty()
        
        return answer, source_docs
        
    except Exception as e:
        progress_container.error(f"Erreur pendant la génération: {str(e)}")
        return None, None
        
def process_documents(hf_api_key, use_uploaded_only):
    if not hf_api_key: # Ensure API key for HuggingFaceEmbeddings is present
        st.warning("Please provide the Hugging Face API key for processing documents.")
        return None
    
    try:
        # Create main status container
        status_container = st.empty()
        status_container.info("Chargement des documents...")
        
        documents, document_dates = load_documents(use_uploaded_only)
        if not documents:
            # load_documents already shows an error if no files found
            return None
        
        # Split into chunks with progress indication
        status_container.info("Découpage des documents en fragments...")
        texts = split_documents(documents) # Using the dedicated split_documents function
        
        if not texts:
            st.error("Le découpage des documents n'a produit aucun fragment.")
            return None
            
        # Create embeddings with progress indication
        status_container.info(f"Création des embeddings pour {len(texts)} fragments (cela peut prendre plusieurs minutes)...")
        progress_bar_embed = st.progress(0) # Separate progress bar for embedding step
        
        # Update manually with approximate progress values
        progress_bar_embed.progress(0.1) # Start progress
        
        # Create embeddings
        retriever = embeddings_on_local_vectordb(texts, hf_api_key) # This function now has its own error handling & diagnostics
        
        if not retriever: # Check if retriever creation failed
            status_container.error("La création des embeddings ou du retriever a échoué.")
            progress_bar_embed.progress(1.0) # Finalize progress bar on error
            return None

        # Update progress
        progress_bar_embed.progress(0.9) # Nearing completion
        status_container.info("Finalisation...")
        
        # Complete progress
        progress_bar_embed.progress(1.0)
        status_container.success(f"Traitement terminé! {len(texts)} fragments créés à partir de {len(documents)} documents.")
        
        return retriever
    
    except Exception as e:
        st.error(f"Une erreur s'est produite lors du traitement des documents: {e}")
        if 'progress_bar_embed' in locals(): progress_bar_embed.progress(1.0) # Ensure progress bar is finalized
        if 'status_container' in locals(): status_container.empty()
        return None


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
        
        st.title("Configuration")
        
        # Hugging Face API Key
        # Initialize from secrets if available, else use text_input
        st.session_state.hf_api_key = st.secrets.get("hf_api_key", "")
        if not st.session_state.hf_api_key:
            st.session_state.hf_api_key = st.text_input("Hugging Face API Key", type="password", key="hf_api_input")
        
        # Open Router API Key
        st.session_state.openrouter_api_key = st.secrets.get("openrouter_api_key", "")
        if not st.session_state.openrouter_api_key:
            st.session_state.openrouter_api_key = st.text_input("OpenRouter API Key (Llama 4)", type="password", key="openrouter_api_input")
            
        # Add option to use pre-computed embeddings
        embeddings_path = EMBEDDINGS_DIR / "faiss_index"
        embeddings_available = embeddings_path.exists() and \
                               (embeddings_path / "index.faiss").exists() and \
                               (embeddings_path / "index.pkl").exists()

        # Initialize use_precomputed if not in session_state
        if 'use_precomputed' not in st.session_state:
            st.session_state.use_precomputed = embeddings_available

        st.session_state.use_precomputed = st.checkbox(
            "Utiliser embeddings pré-calculés",
            value=st.session_state.use_precomputed, # Use session state value
            disabled=not embeddings_available,
            key="use_precomputed_checkbox"
        )
        
        if embeddings_available and st.session_state.use_precomputed:
            metadata_path = EMBEDDINGS_DIR / "document_metadata.pkl"
            if metadata_path.exists():
                try:
                    with open(metadata_path, "rb") as f:
                        metadata = pickle.load(f)
                        model_name_display = metadata.get('model_name', 'Inconnu')
                        st.info(f"Modèle (pré-calculé): {model_name_display}")
                except Exception as e:
                    st.warning(f"Impossible de lire les métadonnées des embeddings: {e}")
            else:
                st.warning("Fichier de métadonnées pour embeddings pré-calculés non trouvé.")
            st.markdown("---")
            
        # Model selection
        # Initialize model_choice if not in session_state
        if 'model_choice' not in st.session_state:
            st.session_state.model_choice = "llama" # Default model

        st.session_state.model_choice = st.radio(
            "Modèle LLM",
            ["llama", "mistral", "phi", "openrouter"], 
            format_func=lambda x: {
                "llama": "Llama 3 (HF)",
                "mistral": "Mistral 7B (HF)",
                "phi": "Phi 3 Mini (HF)",
                "openrouter": "Llama (OpenRouter)" # Clarified label
            }.get(x, x.capitalize()), # Use .get for safety
            horizontal=False, 
            key="model_choice_radio"
        )
        
        # Model information expander
        with st.expander("Infos modèle", expanded=False):
            model_info_text = {
                "llama": "**Meta-Llama-3-8B-Instruct (via HuggingFaceHub)**\n\n* Bonne compréhension des instructions\n* Fort en synthèse de documents longs\n* Précision factuelle solide",
                "mistral": "**Mistral-7B-Instruct-v0.2 (via HuggingFaceHub)**\n\n* Raisonnement sur documents scientifiques\n* Bonne extraction d'informations\n* Réponses structurées en français",
                "phi": "**Microsoft Phi-3-mini-4k-instruct (via HuggingFaceHub)**\n\n* Rapide pour traitement RAG léger\n* Bon ratio performance/taille\n* Précision sur citations textuelles",
                "openrouter": "**Meta-Llama (Modèle Maverick Gratuit via OpenRouter)**\n\n* Haute performance (varie selon le modèle exact)\n* Excellente compréhension du français\n* Nécessite clé OpenRouter"
            }
            st.markdown(model_info_text.get(st.session_state.model_choice, "Information non disponible."))
        
        # Prompt configuration expander
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
            
            st.markdown("##### Prompt de requête (modifiable)")
            st.session_state.query_prompt = st.text_area(
                "Query prompt", 
                value=st.session_state.query_prompt,
                height=300,
                key="query_prompt_area",
                label_visibility="collapsed"
            )
            
            if st.button("Réinitialiser le prompt", key="reset_prompt_btn"):
                st.session_state.query_prompt = DEFAULT_QUERY_PROMPT
                st.rerun() # st.experimental_rerun is deprecated
            
        # File uploader section
        if "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = []

        st.markdown("---") # Separator
        st.markdown("### Fichiers XML") 
        
        uploaded_file_list = st.file_uploader("Télécharger des fichiers XML/XMLTEI", 
                                            type=["xml", "xmltei"], 
                                            accept_multiple_files=True,
                                            key="file_uploader_widget") 
        
        if uploaded_file_list:
            newly_saved_paths = []
            upload_dir = TMP_DIR / "uploaded_xml" # Save to a specific subfolder in tmp
            upload_dir.mkdir(parents=True, exist_ok=True)
            
            for uploaded_file_obj in uploaded_file_list:
                # Sanitize filename (optional, but good practice)
                safe_filename = "".join(c if c.isalnum() or c in ('.', '_', '-') else '_' for c in uploaded_file_obj.name)
                file_path = upload_dir / safe_filename
                try:
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file_obj.getbuffer())
                    newly_saved_paths.append(file_path.as_posix())
                except Exception as e:
                    st.warning(f"Erreur lors de la sauvegarde du fichier {uploaded_file_obj.name}: {e}")
            
            # Add new unique file paths to session state
            # Convert to set for efficient duplicate checking
            current_files_set = set(st.session_state.uploaded_files)
            added_count = 0
            for fp_str in newly_saved_paths:
                if fp_str not in current_files_set:
                    st.session_state.uploaded_files.append(fp_str)
                    current_files_set.add(fp_str) # Keep set updated
                    added_count +=1
            
            if added_count > 0:
                st.success(f"{added_count} nouveau(x) fichier(s) ajouté(s).")
                st.rerun() # Rerun to update UI elements like the checkbox or file list count

        # Initialize use_uploaded_only if not in session_state
        if 'use_uploaded_only' not in st.session_state:
             # Default to True if files are uploaded and precomputed is not selected, else False
            st.session_state.use_uploaded_only = bool(st.session_state.uploaded_files) and not st.session_state.get('use_precomputed', False)

        st.session_state.use_uploaded_only = st.checkbox(
            "Traiter uniquement les fichiers téléchargés", 
            value=st.session_state.use_uploaded_only, # Use session state value
            disabled=not bool(st.session_state.uploaded_files), # Disable if no files uploaded
            key="use_uploaded_only_checkbox"
        )
        
        if st.session_state.use_uploaded_only and not st.session_state.uploaded_files:
            st.warning("Aucun fichier n'a été téléchargé pour être traité.")
        
        if st.session_state.uploaded_files:
            total_files = len(st.session_state.uploaded_files)
            with st.expander(f"Fichiers téléchargés ({total_files})", expanded=False):
                file_list_html = "<div style='max-height: 150px; overflow-y: auto; font-size: 13px;'>"
                for file_path_str in st.session_state.uploaded_files:
                    file_name = os.path.basename(file_path_str)
                    # Display a small part of the path if names are not unique (optional)
                    # parent_dir = Path(file_path_str).parent.name
                    # display_name = f"{parent_dir}/{file_name}" if parent_dir != "uploaded_xml" else file_name
                    file_list_html += f"<div style='padding: 1px 0;'>✓ {file_name}</div>"
                file_list_html += "</div>"
                st.markdown(file_list_html, unsafe_allow_html=True)
                
                if st.button("Effacer tous les fichiers téléchargés", key="clear_uploaded_files_btn"):
                    # Optionally, delete files from TMP_DIR/uploaded_xml
                    # for fp_str in st.session_state.uploaded_files:
                    #     try: Path(fp_str).unlink(missing_ok=True)
                    #     except OSError: pass # Ignore errors if file is locked etc.
                    st.session_state.uploaded_files = []
                    st.session_state.use_uploaded_only = False # Reset this flag too
                    st.rerun()
def boot():
    """Main function to run the application."""
    # Initialize query prompt if not present
    if "query_prompt" not in st.session_state:
        st.session_state.query_prompt = DEFAULT_QUERY_PROMPT
    
    # Setup input fields in the sidebar
    input_fields() # This will initialize session_state vars like hf_api_key, etc.
    
    # Initialize other session state variables if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    
    # Main area for buttons and chat
    # Control button visibility and actions
    
    action_buttons_placeholder = st.empty() # Placeholder for buttons
    with action_buttons_placeholder.container():
        cols = st.columns(2)
        with cols[0]:
            if st.session_state.get("use_precomputed", False): # Check if 'use_precomputed' exists
                if st.button("Charger embeddings pré-calculés", use_container_width=True, key="load_precomputed_main_btn"):
                    with st.spinner("Chargement des embeddings pré-calculés..."):
                        st.session_state.retriever = load_precomputed_embeddings()
                        if st.session_state.retriever:
                            st.success("Embeddings pré-calculés chargés.")
                        else:
                            st.error("Échec du chargement des embeddings pré-calculés.")
        
        with cols[1]: # Use the second column or adjust layout as needed
            # Show "Traiter les documents" if:
            # 1. Precomputed is NOT selected OR
            # 2. Precomputed IS selected BUT "process uploaded only" is ALSO selected (implies overriding precomputed)
            #    AND there are uploaded files.
            # More simply: if not using precomputed, OR if using uploaded_only with files.
            process_button_visible = False
            if not st.session_state.get("use_precomputed", False):
                process_button_visible = True
            elif st.session_state.get("use_uploaded_only", False) and st.session_state.get("uploaded_files"):
                process_button_visible = True

            # Also, always allow processing if there are uploaded files and 'use_uploaded_only' is true,
            # regardless of 'use_precomputed' state (as it implies a specific user action).
            if st.session_state.get("use_uploaded_only", False) and st.session_state.get("uploaded_files"):
                 process_button_visible = True
            elif not st.session_state.get("use_precomputed", False) and ( # If not using precomputed, process if default corpus or uploaded files
                not st.session_state.get("use_uploaded_only", False) or # Default corpus
                (st.session_state.get("use_uploaded_only", False) and st.session_state.get("uploaded_files")) # Uploaded files only
            ):
                 process_button_visible = True


            if process_button_visible:
                 # Determine what to process based on use_uploaded_only
                button_text = "Traiter les documents"
                if st.session_state.get('use_uploaded_only', False) and st.session_state.get('uploaded_files'):
                    button_text = f"Traiter {len(st.session_state.uploaded_files)} fichier(s) téléchargé(s)"
                elif not st.session_state.get('use_uploaded_only', False):
                     button_text = "Traiter le corpus par défaut"


                if st.button(button_text, use_container_width=True, key="process_documents_main_btn",
                             disabled=(st.session_state.get('use_uploaded_only', False) and not st.session_state.get('uploaded_files'))): # Disable if "uploaded only" but no files
                    
                    # Clear any previous retriever if we are reprocessing
                    st.session_state.retriever = None 
                    
                    hf_key_present = bool(st.session_state.get("hf_api_key"))
                    if not hf_key_present:
                        st.error("La clé API Hugging Face est requise pour le traitement des documents.")
                    else:
                        with st.spinner("Traitement des documents en cours..."):
                            st.session_state.retriever = process_documents(
                                st.session_state.hf_api_key, 
                                st.session_state.get('use_uploaded_only', False) 
                            )
                            if st.session_state.retriever:
                                st.success("Traitement des documents terminé.")
                            else:
                                st.error("Échec du traitement des documents.")
            elif not st.session_state.get("use_precomputed", False) and \
                 st.session_state.get('use_uploaded_only', False) and \
                 not st.session_state.get('uploaded_files'):
                 cols[1].warning("Veuillez télécharger des fichiers pour les traiter.")


    # Display chat history
    for message_idx, message_content in enumerate(st.session_state.messages):
        st.chat_message('human', key=f"human_msg_{message_idx}").write(message_content[0])
        st.chat_message('ai', key=f"ai_msg_{message_idx}").markdown(message_content[1]) # Use markdown for AI responses
    
    # Chat input
    if query := st.chat_input("Posez votre question..."):
        if not st.session_state.retriever:
            st.error("Veuillez d'abord charger les embeddings ou traiter les documents pour activer le RAG.")
        else:
            st.chat_message("human").write(query)
            
            # Check for necessary API keys based on model choice
            model_ok = True
            if st.session_state.model_choice in ["llama", "mistral", "phi"] and not st.session_state.get("hf_api_key"):
                st.error(f"La clé API Hugging Face est requise pour utiliser le modèle {st.session_state.model_choice}.")
                model_ok = False
            elif st.session_state.model_choice == "openrouter" and not st.session_state.get("openrouter_api_key"):
                st.error("La clé API OpenRouter est requise pour utiliser le modèle Llama via OpenRouter.")
                model_ok = False

            if model_ok:
                with st.spinner("Génération de la réponse..."):
                    try:
                        answer, source_docs = query_llm(
                            st.session_state.retriever, 
                            query, 
                            st.session_state.hf_api_key, # Always pass, HuggingFaceHub might need it
                            None, # openai_api_key (not used for selected models)
                            st.session_state.openrouter_api_key, 
                            st.session_state.model_choice
                        )
                        
                        if answer is not None: 
                            response_container = st.chat_message("ai")
                            response_container.markdown(answer) # Ensure AI responses are rendered as Markdown
                            
                            if source_docs:
                                response_container.markdown("---")
                                response_container.markdown("**Sources:**")
                                
                                for i, doc in enumerate(source_docs):
                                    doc_title = doc.metadata.get('title', 'Document sans titre')
                                    doc_date = doc.metadata.get('date', 'Date inconnue')
                                    doc_file = os.path.basename(doc.metadata.get('source', 'Fichier inconnu')) # Just filename
                                    
                                    expander_title = f"📄 Source {i+1}: {doc_title} ({doc_file})"
                                    with response_container.expander(expander_title, expanded=False):
                                        st.markdown(f"**Date:** {doc_date}")
                                        # st.markdown(f"**Fichier:** {doc_file}") # Already in title
                                        
                                        persons_metadata = doc.metadata.get('persons')
                                        if persons_metadata and isinstance(persons_metadata, list) and persons_metadata:
                                            st.markdown("**Personnes mentionnées:** " + ", ".join(persons_metadata))
                                        
                                        st.markdown("**Extrait:**")
                                        content = doc.page_content
                                        header_to_remove = f"Document: {doc_title} | Date: {doc_date}\n\n"
                                        if content.startswith(header_to_remove):
                                             content = content.replace(header_to_remove, "", 1)
                                        
                                        st.text_area("", value=content, height=150, disabled=True, 
                                                     key=f"source_content_{st.session_state.messages[-1][0]}_{i}") # Unique key
                        else:
                            st.chat_message("ai").error("La génération de la réponse a échoué ou n'a rien retourné.")
                    
                    except Exception as e:
                        st.error(f"Erreur lors de la génération de la réponse: {e}")
            # else: API key error already shown by st.error

if __name__ == '__main__':
    boot()
