import os
import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
import pickle

import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings # Sticking to this due to constraints
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain_community.document_loaders import DirectoryLoader # Included as per original script
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

# Defining paths
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

TMP_DIR = Path(__file__).resolve().parent.joinpath('tmp')
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath('vector_store')
EMBEDDINGS_DIR = Path(__file__).resolve().parent.joinpath('embeddings')

TMP_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True) # Ensure embeddings dir also exists

# Define namespaces for XML-tei
NAMESPACES = {
    'tei': 'http://www.tei-c.org/ns/1.0'
}

st.set_page_config(page_title="RAG Démonstration", page_icon="🤖", layout="wide")
st.title("Retrieval Augmented Generation")
st.image("static/sfp_logo.png", width=100) # Make sure this path is correct or remove if not needed
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
- Titres en **gras**
- Informations citées textuellement depuis les documents
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

        if os.path.basename(file_path).startswith("SFP_"):
            st.write(f"Parsing: {file_path}")

        title = root.find('.//tei:titleStmt/tei:title', NAMESPACES)
        title_text = title.text if title is not None else "Unknown Title"

        date = root.find('.//tei:sourceDesc/tei:p/tei:date', NAMESPACES)
        if date is None:
            date = root.find('.//tei:sourceDesc/tei:p', NAMESPACES)
        date_text = date.text if date is not None else "Unknown Date"

        year = extract_year(date_text)

        paragraphs = root.findall('.//tei:p', NAMESPACES)
        person_names = root.findall('.//tei:persName', NAMESPACES)
        person_text = []
        for person in person_names:
            name = ''.join(person.itertext()).strip()
            if name:
                person_text.append(name)

        header = f"Document: {title_text} | Date: {date_text}\n\n"
        all_paragraphs = []
        for para in paragraphs:
            para_text = ''.join(para.itertext()).strip()
            if para_text:
                all_paragraphs.append(para_text)

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

    st.write(f"Using uploaded files only: {use_uploaded_only}")

    if use_uploaded_only:
        if "uploaded_files" in st.session_state and st.session_state.uploaded_files:
            st.write(f"Found {len(st.session_state.uploaded_files)} uploaded files")
            for file_path in st.session_state.uploaded_files:
                # Ensure file_path is a string path, not UploadedFile object for os.path.exists
                # Assuming file_path in session_state is already a string path to saved file
                if os.path.exists(file_path) and (file_path.endswith(".xml") or file_path.endswith(".xmltei")):
                    xml_files.append(file_path)
                    st.write(f"Added uploaded file: {file_path}")
    else:
        # Process files from default directories (current and data)
        for path_dir in [".", "data"]: # Ensure "data" directory is correctly structured
            if os.path.exists(path_dir) and os.path.isdir(path_dir):
                for file in os.listdir(path_dir):
                    if file.endswith(".xml") or file.endswith(".xmltei"):
                        file_path = os.path.join(path_dir, file)
                        xml_files.append(file_path)
            elif not os.path.isdir(path_dir) and os.path.exists(path_dir): # if path_dir is a file itself
                 if path_dir.endswith(".xml") or path_dir.endswith(".xmltei"):
                        xml_files.append(path_dir)


    if not xml_files:
        st.error("No XML files found. Please upload XML files or ensure they are in the 'data' directory or current directory.")
        return documents, document_dates

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, file_path in enumerate(xml_files):
        progress = (i + 1) / len(xml_files) # Corrected progress calculation
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

    progress_bar.progress(1.0) # Ensure it completes
    status_text.text(f"Traitement terminé! {len(documents)} documents analysés.")
    return documents, document_dates

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=800)
    texts = text_splitter.split_documents(documents)
    return texts

def load_precomputed_embeddings():
    """Load precomputed embeddings from the embeddings directory.
    MODIFIED to attempt forcing CPU usage for HuggingFaceEmbeddings.
    """
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

    embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    if metadata_path.exists():
        try:
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
                st.success(f"Loaded pre-computed embeddings metadata: {metadata.get('chunk_count', 'N/A')} chunks from {metadata.get('document_count', 'N/A')} documents")
                if 'model_name' in metadata:
                    embedding_model = metadata['model_name']
                    st.info(f"Embedding model from metadata: {embedding_model}")
                else:
                    st.warning("Model information not found in metadata, using default model")
        except Exception as e:
            st.warning(f"Error loading metadata: {str(e)}")
            st.warning(f"Using default embedding model: {embedding_model}")
    else:
        st.warning(f"Metadata file not found. Using default embedding model: {embedding_model}")

    try:
        # WORKAROUND for "torch.get_default_device" error with PyTorch >= 2.0
        # by attempting to force sentence-transformers to use the CPU.
        # This is necessary because new packages cannot be installed.
        st.info(f"Attempting to initialize embeddings on CPU for model: {embedding_model}")
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'}  # <--- MODIFICATION HERE
        )
        st.info("Embeddings object initialized on CPU.")

        try:
            st.info(f"Loading FAISS index with model: {embedding_model}")
            vectordb = FAISS.load_local(
                embeddings_path.as_posix(),
                embeddings,
                allow_dangerous_deserialization=True
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
    except Exception as e:
        st.error(f"Error in embeddings initialization (even with CPU forcing): {str(e)}")
        st.error("The 'torch.get_default_device' error might persist if the installed sentence-transformers version doesn't respect 'device=cpu' for this specific check.")
        return None

def embeddings_on_local_vectordb(texts, hf_api_key):
    """Create embeddings and store in a local vector database using FAISS.
    MODIFIED to attempt forcing CPU usage for HuggingFaceEmbeddings.
    """
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_api_key

    # WORKAROUND for "torch.get_default_device" error with PyTorch >= 2.0
    # by attempting to force sentence-transformers to use the CPU.
    model_kwargs = {"token": hf_api_key, 'device': 'cpu'}  # <--- MODIFICATION HERE
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    st.info(f"Attempting to initialize embeddings on CPU for model: {model_name}")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs
        )
        st.info("Embeddings object initialized on CPU for real-time processing.")
    except Exception as e_init:
        st.error(f"Failed to initialize HuggingFaceEmbeddings on CPU: {str(e_init)}")
        return None

    try:
        st.info(f"Creating FAISS vector store from {len(texts)} texts...")
        vectordb = FAISS.from_documents(texts, embeddings)
        LOCAL_VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
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
        st.success(f"Vector store created and saved successfully with {len(texts)} chunks.")
        return retriever
    except Exception as e:
        st.error(f"Error creating embeddings during FAISS.from_documents (even with CPU forcing): {str(e)}")
        st.info("The 'torch.get_default_device' error might persist here too.")

        # Fallback to batch processing
        st.info("Trying batch processing approach (still on CPU)...")
        try:
            batch_size = 50
            if not texts:
                st.error("No texts to process for embeddings.")
                return None
            if not isinstance(texts[0], Document):
                 st.error("Input 'texts' for batch processing is not in the expected format (list of LangChain Document objects).")
                 return None

            batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
            
            st.info(f"Processing batch 1/{len(batches)}...")
            vectordb = FAISS.from_documents(batches[0], embeddings) # Initialize with first batch
            
            for i, batch in enumerate(batches[1:], 1): # Start from the second batch (index 1)
                st.info(f"Processing batch {i+1}/{len(batches)}...") # i is 0-indexed for batches[1:]
                if batch:
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
            st.success(f"Vector store created via batch processing with {len(texts)} chunks.")
            return retriever
        except Exception as batch_e:
            st.error(f"Error with batch processing (on CPU): {str(batch_e)}")
            return None

def prepare_sources_for_llm(source_docs):
    source_mapping = []
    for i, doc in enumerate(source_docs):
        doc_title = doc.metadata.get('title', 'Document sans titre')
        source_mapping.append(f"Source {i+1}: {doc_title}")
    return "\n".join(source_mapping)

def query_llm(retriever, query, hf_api_key, openai_api_key=None, openrouter_api_key=None, model_choice="llama"):
    progress_container = st.empty()
    progress_container.info("Recherche des documents pertinents...")
    progress_bar = st.progress(0)

    try:
        base_query_template = st.session_state.query_prompt
        relevant_docs = retriever.get_relevant_documents(query)
        source_mapping = []
        for i, doc in enumerate(relevant_docs):
            doc_title = doc.metadata.get('title', 'Document sans titre')
            doc_date = doc.metadata.get('date', 'Date inconnue')
            source_mapping.append(f"Source {i+1}: {doc_title} | {doc_date}")
        source_references = "\n".join(source_mapping)

        enhanced_system_prompt = SYSTEM_PROMPT # Using the global one, can be customized if needed here
        
        costar_query = {
            "query": query,
            "context": "Analyse des documents scientifiques historiques en français.",
            "objective": f"Réponds précisément à la question: {query}",
            "style": "Factuel, précis et structuré avec formatage markdown.",
            "tone": "Académique et objectif.",
            "audience": "Chercheurs et historiens travaillant sur des documents scientifiques.",
            "response_format": "Structure en sections avec citations exactes, niveau de confiance et numéro de source explicite."
        }
        
        query_prompt_template = base_query_template
        additional_instructions = f"""
INSTRUCTIONS IMPORTANTES:
- Pour CHAQUE fait ou information mentionné dans ta réponse, indique EXPLICITEMENT le numéro de la source correspondante (ex: Source 1, Source 3)
- Cite les sources même pour les informations de confiance élevée
- Fais référence aux sources numérotées ci-dessous dans chaque section de ta réponse

SOURCES DISPONIBLES:
{source_references}
"""
        llm = None # Initialize llm
        if model_choice == "openrouter":
            if not openrouter_api_key:
                st.error("OpenRouter API key is required to use Llama 4 Maverick model")
                return None, None
            llm = ChatOpenAI(
                temperature=0.4,
                model_name="meta-llama/llama-4-maverick:free", # Check if this model is still free/available
                openai_api_key=openrouter_api_key,
                max_tokens=1500, # Reduced from 50000, as that's usually too high for response
                openai_api_base="https://openrouter.ai/api/v1",
                model_kwargs={"messages": [{"role": "system", "content": enhanced_system_prompt}]},
                default_headers={"HTTP-Referer": "https://your-streamlit-app.com"} # Replace with actual URL if needed
            )
        elif model_choice == "mistral":
            if not hf_api_key:
                st.error("Hugging Face API key is required to use Mistral model")
                return None, None
            llm = HuggingFaceHub(
                repo_id="mistralai/Mistral-7B-Instruct-v0.2",
                huggingfacehub_api_token=hf_api_key,
                model_kwargs={"temperature": 0.4, "max_new_tokens": 1000, "top_p": 0.95}
            )
        elif model_choice == "phi":
            if not hf_api_key:
                st.error("Hugging Face API key is required to use Phi model")
                return None, None
            llm = HuggingFaceHub(
                repo_id="microsoft/Phi-3-mini-4k-instruct", # Phi-4-mini-instruct was a typo, Phi-3 is more common
                huggingfacehub_api_token=hf_api_key,
                model_kwargs={"temperature": 0.4, "max_new_tokens": 1000, "top_p": 0.95}
            )
        else: # Default Llama
            if not hf_api_key:
                st.error("Hugging Face API key is required to use Llama model") # Added check for default
                return None, None
            llm = HuggingFaceHub(
                repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
                huggingfacehub_api_token=hf_api_key,
                model_kwargs={"temperature": 0.4, "max_new_tokens": 2000, "top_p": 0.95}
            )
        
        progress_bar.progress(0.3)
        progress_container.info("Création de la chaîne de traitement...")
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            verbose=True # Consider setting to False for cleaner logs in production
        )
        
        progress_bar.progress(0.5)
        progress_container.info(f"Génération de la réponse avec le modèle {model_choice.upper()}...")
        
        enh_query = query_prompt_template.format(
            query=query,
            context=costar_query["context"],
            objective=costar_query["objective"],
            style=costar_query["style"],
            tone=costar_query["tone"],
            audience=costar_query["audience"],
            response_format=costar_query["response_format"]
        )
        enh_query = enh_query + "\n\n" + additional_instructions
        
        result = qa_chain({"query": enh_query})
        
        progress_bar.progress(0.9)
        progress_container.info("Finalisation et mise en forme de la réponse...")
        
        answer = result["result"]
        source_docs = result["source_documents"]
        
        if "messages" in st.session_state:
            st.session_state.messages.append((query, answer))
            
        progress_bar.progress(1.0)
        progress_container.empty()
        return answer, source_docs
    except Exception as e:
        progress_container.error(f"Erreur pendant la génération: {str(e)}")
        # import traceback # For more detailed debugging if needed
        # traceback.print_exc()
        return None, None

def process_documents(hf_api_key, use_uploaded_only):
    if not hf_api_key:
        st.warning("Please provide the Hugging Face API key.")
        return None
    
    retriever = None # Initialize retriever
    try:
        status_container = st.empty()
        status_container.info("Chargement des documents...")
        
        documents, document_dates = load_documents(use_uploaded_only)
        if not documents:
            st.error("No documents found to process.")
            status_container.empty() # Clear status
            return None
            
        status_container.info("Découpage des documents en fragments...")
        texts = split_documents(documents) # Using the separate split_documents function
        if not texts:
            st.error("Failed to split documents into texts.")
            status_container.empty()
            return None

        status_container.info("Création des embeddings (cela peut prendre plusieurs minutes)...")
        # Embeddings creation progress is handled within embeddings_on_local_vectordb
        
        retriever = embeddings_on_local_vectordb(texts, hf_api_key)
        
        if retriever:
            status_container.success(f"Traitement terminé! {len(texts)} fragments créés à partir de {len(documents)} documents.")
        else:
            status_container.error("Failed to create embeddings and retriever.")
            
        return retriever
    except Exception as e:
        st.error(f"Une erreur s'est produite lors du traitement des documents: {e}")
        if 'status_container' in locals(): status_container.empty() # Clear status on error
        return None


def input_fields():
    with st.sidebar:
        st.markdown("""
        <style>
        .stSelectbox, .stRadio > div, .stExpander, [data-testid="stFileUploader"] {
            max-width: 100%; overflow-x: hidden;
        }
        .stCheckbox label p { font-size: 14px; margin-bottom: 0; white-space: normal; }
        div.row-widget.stRadio > div { flex-direction: column; margin-top: -10px; }
        div.row-widget.stRadio > div label { margin: 0; padding: 2px 0; }
        .stExpander { font-size: 14px; }
        .stExpander details summary p { margin-bottom: 0; }
        .stExpander details summary::marker { margin-right: 5px; }
        </style>
        """, unsafe_allow_html=True)
        
        st.title("Configuration")
        
        if "hf_api_key" not in st.session_state: st.session_state.hf_api_key = ""
        if "openrouter_api_key" not in st.session_state: st.session_state.openrouter_api_key = ""

        st.session_state.hf_api_key = st.text_input(
            "Hugging Face API Key", 
            type="password", 
            value=st.secrets.get("hf_api_key", st.session_state.hf_api_key)
        )
        st.session_state.openrouter_api_key = st.text_input(
            "OpenRouter API Key (Llama 4)", 
            type="password",
            value=st.secrets.get("openrouter_api_key", st.session_state.openrouter_api_key)
        )
        
        embeddings_path = EMBEDDINGS_DIR / "faiss_index"
        embeddings_available = embeddings_path.exists() and (embeddings_path / "index.faiss").exists()

        if "use_precomputed" not in st.session_state: st.session_state.use_precomputed = embeddings_available

        st.session_state.use_precomputed = st.checkbox(
            "Utiliser embeddings pré-calculés",
            value=st.session_state.use_precomputed,
            disabled=not embeddings_available
        )
        
        if embeddings_available and st.session_state.use_precomputed:
            metadata_path = EMBEDDINGS_DIR / "document_metadata.pkl"
            if metadata_path.exists():
                try:
                    with open(metadata_path, "rb") as f:
                        metadata = pickle.load(f)
                        st.info(f"Modèle (pré-calculé): {metadata.get('model_name', 'Unknown')}")
                except Exception: pass # Silently ignore if metadata loading fails
        st.markdown("---")
        
        if "model_choice" not in st.session_state: st.session_state.model_choice = "llama"
        st.session_state.model_choice = st.radio(
            "Modèle LLM",
            ["llama", "mistral", "phi", "openrouter"],
            format_func=lambda x: {
                "llama": "Llama 3", "mistral": "Mistral 7B",
                "phi": "Phi-3-mini", "openrouter": "Llama 4 Maverick (OpenRouter)"
            }[x],
            horizontal=False # Easier to read vertically
        )

        with st.expander("Infos modèle", expanded=False):
            model_info = {
                "llama": "**Meta-Llama-3-8B**\n\n* Bonne compréhension\n* Fort en synthèse\n* Précision factuelle solide",
                "mistral": "**Mistral-7B-Instruct**\n\n* Raisonnement documents scientifiques\n* Bonne extraction\n* Réponses structurées",
                "phi": "**Phi-3-mini-4k-instruct**\n\n* Rapide pour RAG léger\n* Bon ratio performance/taille\n* Précision sur citations",
                "openrouter": "**Llama 4 Maverick (via OpenRouter)**\n\n* Modèle performant\n* Potentiellement coûteux\n* Nécessite clé OpenRouter"
            }
            st.markdown(model_info.get(st.session_state.model_choice, "Info non disponible"))
            
        with st.expander("Configuration du prompt (COSTAR)", expanded=False):
            if "query_prompt" not in st.session_state:
                st.session_state.query_prompt = DEFAULT_QUERY_PROMPT
            
            st.markdown("##### Framework COSTAR")
            st.info("""
            **COSTAR** est un framework de prompting structuré:
            - **C**ontexte, **O**bjectif, **S**tyle, **T**on, **A**udience, **R**éponse.
            """)
            st.markdown("##### Prompt de requête (modifiable)")
            st.session_state.query_prompt = st.text_area(
                "Query prompt editor", # Unique key for text_area
                value=st.session_state.query_prompt, height=300,
                key="query_prompt_area", label_visibility="collapsed"
            )
            if st.button("Réinitialiser le prompt"):
                st.session_state.query_prompt = DEFAULT_QUERY_PROMPT
                st.rerun() # Use st.rerun instead of experimental_rerun
        
        if "uploaded_files" not in st.session_state: st.session_state.uploaded_files = []
        if "use_uploaded_only" not in st.session_state: st.session_state.use_uploaded_only = False

        st.markdown("### Fichiers XML")
        
        # Ensure "data/uploaded" directory exists for uploads
        uploaded_dir = Path("data/uploaded")
        uploaded_dir.mkdir(parents=True, exist_ok=True)

        uploaded_file_objects = st.file_uploader(
            "Télécharger des fichiers XML/XMLTEI",
            type=["xml", "xmltei"], accept_multiple_files=True,
            label_visibility="collapsed" # Use "visible" or "hidden" instead of "collapsed" if preferred
        )
        
        if uploaded_file_objects:
            new_files_paths = []
            for uploaded_file_obj in uploaded_file_objects:
                file_path = uploaded_dir / uploaded_file_obj.name
                try:
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file_obj.getbuffer())
                    if str(file_path) not in st.session_state.uploaded_files: # Store as string
                        new_files_paths.append(str(file_path))
                except Exception as e:
                    st.error(f"Erreur lors de la sauvegarde du fichier {uploaded_file_obj.name}: {e}")

            if new_files_paths:
                st.session_state.uploaded_files.extend(new_files_paths)
                st.success(f"{len(new_files_paths)} nouveau(x) fichier(s) sauvegardé(s) et ajouté(s) à la liste.")
                # Automatically check "use uploaded only" if new files are uploaded
                st.session_state.use_uploaded_only = True


        st.session_state.use_uploaded_only = st.checkbox(
            "Utiliser uniquement fichiers téléchargés",
            value=st.session_state.use_uploaded_only or bool(st.session_state.uploaded_files) # Default to true if files exist
        )
        
        if st.session_state.use_uploaded_only and not st.session_state.uploaded_files:
            st.warning("Aucun fichier téléchargé n'est listé, mais l'option 'Utiliser uniquement fichiers téléchargés' est cochée.")
        
        if st.session_state.uploaded_files:
            total_files = len(st.session_state.uploaded_files)
            with st.expander(f"Fichiers téléchargés ({total_files})", expanded=False):
                file_list_html = "<div style='max-height: 150px; overflow-y: auto;'>"
                for file_path_str in st.session_state.uploaded_files:
                    file_name = os.path.basename(file_path_str)
                    file_list_html += f"<div style='padding: 2px 0; font-size: 13px;'>✓ {file_name}</div>"
                file_list_html += "</div>"
                st.markdown(file_list_html, unsafe_allow_html=True)
                
                if st.button("Effacer tous les fichiers téléchargés", key="clear_files"):
                    # Optionally delete files from disk too
                    # for fp_str in st.session_state.uploaded_files:
                    #     try: os.remove(fp_str)
                    #     except OSError: pass
                    st.session_state.uploaded_files = []
                    st.session_state.use_uploaded_only = False # Reset this flag
                    st.rerun()

def boot():
    if "query_prompt" not in st.session_state: st.session_state.query_prompt = DEFAULT_QUERY_PROMPT
    if "messages" not in st.session_state: st.session_state.messages = []
    if "retriever" not in st.session_state: st.session_state.retriever = None
    
    input_fields() # Call sidebar setup
    
    # Button layout: Use columns for better organization if many buttons
    # For now, sequential buttons are fine.
    
    # Button for pre-computed embeddings
    # Check if precomputed embeddings are available and user wants to use them
    embeddings_path = EMBEDDINGS_DIR / "faiss_index"
    embeddings_available = embeddings_path.exists() and (embeddings_path / "index.faiss").exists()

    if st.session_state.get("use_precomputed", False) and embeddings_available:
        if st.button("Charger embeddings pré-calculés", use_container_width=True, key="load_precomputed_btn"):
            with st.spinner("Chargement des embeddings pré-calculés..."):
                st.session_state.retriever = load_precomputed_embeddings()
                if st.session_state.retriever:
                    st.success("Embeddings pré-calculés chargés.")
                else:
                    st.error("Échec du chargement des embeddings pré-calculés.")
    elif not embeddings_available and st.session_state.get("use_precomputed", False):
         st.warning("Embeddings pré-calculés non trouvés. Veuillez décocher l'option ou traiter les documents.")


    # Button for processing documents (always show if not using precomputed, or if files uploaded and precomputed not yet loaded)
    # Logic: show if (not using precomputed) OR (using precomputed BUT no retriever loaded yet AND (files uploaded OR not using uploaded only))
    show_process_button = not st.session_state.get("use_precomputed", False) or \
                          (st.session_state.get("use_precomputed", False) and not st.session_state.retriever and \
                           (st.session_state.get("uploaded_files") or not st.session_state.get("use_uploaded_only", False)))

    if show_process_button:
        process_button_text = "Traiter les documents"
        if st.session_state.get("use_uploaded_only", False) and st.session_state.get("uploaded_files"):
            process_button_text = f"Traiter les {len(st.session_state.uploaded_files)} fichier(s) téléchargé(s)"
        elif not st.session_state.get("use_uploaded_only", False):
            process_button_text = "Traiter le corpus par défaut"
        
        # Disable button if "use_uploaded_only" is true but no files are uploaded
        disable_process_btn = st.session_state.get("use_uploaded_only", False) and not st.session_state.get("uploaded_files")

        if st.button(process_button_text, use_container_width=True, key="process_docs_btn", disabled=disable_process_btn):
            if not st.session_state.get("hf_api_key"):
                 st.error("Veuillez fournir une clé API Hugging Face dans la barre latérale.")
            else:
                with st.spinner("Traitement des documents en cours..."):
                    st.session_state.retriever = process_documents(
                        st.session_state.hf_api_key,
                        st.session_state.get("use_uploaded_only", False)
                    )
                    if st.session_state.retriever:
                        st.success("Documents traités et retriever prêt.")
                    else:
                        st.error("Échec du traitement des documents.")

    for message_query, message_answer in st.session_state.messages: # Unpack tuple
        st.chat_message('human').write(message_query)
        st.chat_message('ai').markdown(message_answer) # Use markdown for AI response
        
    if query := st.chat_input("Posez votre question..."):
        if not st.session_state.retriever:
            st.error("Veuillez d'abord charger les embeddings pré-calculés ou traiter les documents.")
        else:
            st.chat_message("human").write(query)
            
            # Check API key requirements based on model choice
            model_ok = True
            if st.session_state.model_choice == "openrouter" and not st.session_state.get("openrouter_api_key"):
                st.error("La clé API OpenRouter est requise pour le modèle Llama 4 Maverick.")
                model_ok = False
            elif st.session_state.model_choice in ["llama", "mistral", "phi"] and not st.session_state.get("hf_api_key"):
                st.error(f"La clé API Hugging Face est requise pour le modèle {st.session_state.model_choice.capitalize()}.")
                model_ok = False

            if model_ok:
                with st.spinner("Génération de la réponse..."):
                    answer, source_docs = query_llm(
                        st.session_state.retriever,
                        query,
                        st.session_state.hf_api_key,
                        None, # openai_api_key explicitly None
                        st.session_state.openrouter_api_key,
                        st.session_state.model_choice
                    )
                    
                    if answer:
                        response_container = st.chat_message("ai")
                        response_container.markdown(answer)
                        if source_docs:
                            response_container.markdown("---")
                            response_container.markdown("**Sources:**")
                            for i, doc in enumerate(source_docs):
                                doc_title = doc.metadata.get('title', 'Document sans titre')
                                doc_date = doc.metadata.get('date', 'Date inconnue')
                                doc_file = os.path.basename(doc.metadata.get('source', 'Fichier inconnu'))
                                
                                with response_container.expander(f"📄 Source {i+1}: {doc_title}", expanded=False):
                                    st.markdown(f"**Date:** {doc_date}")
                                    st.markdown(f"**Fichier:** {doc_file}")
                                    if doc.metadata.get('persons'):
                                        persons = doc.metadata.get('persons')
                                        if isinstance(persons, list) and persons:
                                            st.markdown("**Personnes mentionnées:**")
                                            st.markdown(", ".join(persons))
                                    
                                    st.markdown("**Extrait:**")
                                    content = doc.page_content
                                    # Basic cleanup of header if present in content (optional)
                                    # header_in_content = f"Document: {doc_title} | Date: {doc_date}\n\n"
                                    # if content.startswith(header_in_content):
                                    #    content = content.replace(header_in_content, "", 1)
                                    st.text_area(f"source_content_{i}", value=content, height=150, disabled=True, key=f"src_exp_{i}")
                    else:
                        st.error("N'a pas pu générer de réponse.")
            # else: error message already shown for missing API key

if __name__ == '__main__':
    boot()
