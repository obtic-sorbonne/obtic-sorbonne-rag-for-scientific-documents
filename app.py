import os
import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
import pickle

import streamlit as st
from langchain_core.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

# Defining paths 

os.environ["TRANSFORMERS_OFFLINE"] = "0"  # Make sure offline mode is disabled
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # Use HF mirror

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
- En l'absence d'information : écrire _"Les documents fournis ne contiennent pas cette information."_  
- Chaque information doit comporter un **niveau de confiance** : Élevé / Moyen / Faible  
- Chiffres présentés de manière claire et lisible  
- Mettre en **gras** les informations importantes

⚠️ **Attention aux chiffres** : les erreurs OCR sont fréquentes (ex : "71 (11" peut signifier "71 011"). Vérifier la cohérence à partir du contexte. Être prudent sur les séparateurs utilisés (espaces, virgules, points)."""

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
    """Default execution - using files from ./data/ 
    
    Args:
        use_uploaded_only: If True, only use uploaded files and ignore default corpus
    """
    documents = []
    document_dates = {}
    
    xml_files = []
    
    if use_uploaded_only:
        if "uploaded_files" in st.session_state and st.session_state.uploaded_files:
            for file_path in st.session_state.uploaded_files:
                if os.path.exists(file_path) and (file_path.endswith(".xml") or file_path.endswith(".xmltei")):
                    xml_files.append(file_path)
    else:
        for path in [".", "data"]:
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
        # Update progress bar and status
        progress = (i) / len(xml_files)
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
    # Increased chunk size to 5000 and overlap to 700 for better context
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
    
    # Load metadata for display
    if metadata_path.exists():
        try:
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
                st.success(f"Loaded pre-computed embeddings with {metadata['chunk_count']} chunks from {metadata['document_count']} documents")
                st.info(f"Embedding model: {metadata.get('model_name', 'Unknown')}")
        except Exception as e:
            st.warning(f"Error loading metadata: {str(e)}")
    
    try:
        # Initialize the embeddings model
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        
        # Try to load the FAISS index
        try:
            # Import from langchain_community instead
            from langchain_community.vectorstores import FAISS as LC_FAISS
            
            # Load the FAISS index directly without pickle.load which is causing issues
            vectordb = LC_FAISS.load_local(
                embeddings_path.as_posix(), 
                embeddings,
                allow_dangerous_deserialization=True
            )
            
            retriever = vectordb.as_retriever(
                search_type="mmr", 
                search_kwargs={'k': 5, 'fetch_k': 10}
            )
            return retriever
            
        except Exception as e:
            st.error(f"Error loading FAISS index: {str(e)}")
            
            # If the first method fails, try the manual approach
            try:
                # Load the index pickle manually
                with open(embeddings_path / "index.pkl", "rb") as f:
                    stored_data = pickle.load(f)
                
                # Create a retriever from the loaded data
                vectordb = LC_FAISS(
                    embedding=embeddings,
                    index=None,
                    docstore=stored_data["docstore"],
                    index_to_docstore_id=stored_data["index_to_docstore_id"]
                )
                
                # Load the FAISS index file separately
                import faiss
                vectordb.index = faiss.read_index(str(embeddings_path / "index.faiss"))
                
                retriever = vectordb.as_retriever(
                    search_type="mmr", 
                    search_kwargs={'k': 5, 'fetch_k': 10}
                )
                return retriever
            except Exception as e2:
                st.error(f"Error with fallback loading method: {str(e2)}")
                st.error("Unable to load pre-computed embeddings. You'll need to process documents instead.")
                st.info("Alternatively, check that your embeddings were properly committed to GitHub and are in the correct format.")
                return None
    
    except Exception as e:
        st.error(f"Error in embeddings initialization: {str(e)}")
        return None

def embeddings_on_local_vectordb(texts, hf_api_key):
    """Create embeddings and store in a local vector database using FAISS."""
    import os
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_api_key
    
    model_kwargs = {"token": hf_api_key}
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs=model_kwargs
    )
    
    # Create and save the vector database
    vectordb = FAISS.from_documents(texts, embeddings)
    
    # Make sure the directory exists
    LOCAL_VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save with explicit disabling of pickle protocol version
    vectordb.save_local(LOCAL_VECTOR_STORE_DIR.as_posix())
    
    # Create retriever with MMR search
    retriever = vectordb.as_retriever(
        search_type="mmr", 
        search_kwargs={'k': 5, 'fetch_k': 10}
    )
    
    return retriever

def query_llm(retriever, query, hf_api_key, openai_api_key=None, openrouter_api_key=None, model_choice="llama"):
    """Query the LLM using one of the supported models."""
    
    progress_container = st.empty()
    progress_container.info("Recherche des documents pertinents...")
    progress_bar = st.progress(0)
    
    try:
        # Construct COSTAR-based prompt
        base_query_template = st.session_state.query_prompt
        
        # Enhance the query with COSTAR components
        costar_query = {
            "query": query,
            "context": "Analyse des documents scientifiques historiques en français.",
            "objective": f"Réponds précisément à la question: {query}",
            "style": "Factuel, précis et structuré avec formatage markdown.",
            "tone": "Académique et objectif.",
            "audience": "Chercheurs et historiens travaillant sur des documents scientifiques.",
            "response_format": "Structure en sections avec citations exactes et niveau de confiance."
        }
        
        # Format the query using the template
        query_prompt_template = base_query_template
        
        # For OpenAI model
        if model_choice == "openrouter":
            if not openrouter_api_key:
                st.error("OpenRouter API key is required to use Llama 4 Maverick model")
                return None, None
                
            # Use ChatOpenAI with OpenRouter base URL
            from langchain_openai import ChatOpenAI
            
            llm = ChatOpenAI(
                temperature=0.4,
                model_name="meta-llama/llama-4-maverick:free",
                openai_api_key=openrouter_api_key,
                max_tokens=50000,
                openai_api_base="https://openrouter.ai/api/v1",
                model_kwargs={
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT}
                    ]
                },
                default_headers={
                    "HTTP-Referer": "https://your-streamlit-app.com" 
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
        elif model_choice == "phi":
            if not hf_api_key:
                st.error("Hugging Face API key is required to use Phi model")
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
        else:
            # Default Llama model
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
        
        # Updated import for RetrievalQA
        from langchain.chains import RetrievalQA
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            verbose=True
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
    if not hf_api_key:
        st.warning("Please provide the Hugging Face API key.")
        return None
    
    try:
        # Create main status container
        status_container = st.empty()
        status_container.info("Chargement des documents...")
        
        documents, document_dates = load_documents(use_uploaded_only)
        if not documents:
            st.error("No documents found to process.")
            return None
        
        # Split into chunks with progress indication
        status_container.info("Découpage des documents en fragments...")
        # Updated chunking parameters to match split_documents function
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=800)
        texts = text_splitter.split_documents(documents)
        
        # Create embeddings with progress indication
        status_container.info("Création des embeddings (cela peut prendre plusieurs minutes)...")
        progress_bar = st.progress(0)
        
        # Update manually with approximate progress values
        progress_bar.progress(0.2)
        
        # Create embeddings
        retriever = embeddings_on_local_vectordb(texts, hf_api_key)
        
        # Update progress
        progress_bar.progress(0.8)
        status_container.info("Finalisation...")
        
        # Complete progress
        progress_bar.progress(1.0)
        status_container.success(f"Traitement terminé! {len(texts)} fragments créés à partir de {len(documents)} documents.")
        
        return retriever
    
    except Exception as e:
        st.error(f"Une erreur s'est produite: {e}")
        return None


def input_fields():
    """Set up the input fields in the sidebar with improved responsive layout."""
    with st.sidebar:
        # Apply custom CSS to make sidebar elements more compact and responsive
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
        if "hf_api_key" in st.secrets:
            st.session_state.hf_api_key = st.secrets.hf_api_key
        else:
            st.session_state.hf_api_key = st.text_input("Hugging Face API Key", type="password")
        
        # OpenAI API Key - commented out but preserved
        # # OpenAI API Key
        # if "openai_api_key" in st.secrets:
        #     st.session_state.openai_api_key = st.secrets.openai_api_key
        # else:
        #     st.session_state.openai_api_key = st.text_input("OpenAI API Key (GPT-3.5)", type="password")
            
        # Open Router 
        if "openrouter_api_key" in st.secrets:
            st.session_state.openrouter_api_key = st.secrets.openrouter_api_key
        else:
            st.session_state.openrouter_api_key = st.text_input("OpenRouter API Key (Llama 4)", type="password")
            
        # Add option to use pre-computed embeddings
        embeddings_path = EMBEDDINGS_DIR / "faiss_index"
        embeddings_available = embeddings_path.exists()
        
        st.session_state.use_precomputed = st.checkbox(
            "Utiliser embeddings pré-calculés",
            value=embeddings_available,
            disabled=not embeddings_available
        )
        
        if embeddings_available and st.session_state.use_precomputed:
            metadata_path = EMBEDDINGS_DIR / "document_metadata.pkl"
            if metadata_path.exists():
                try:
                    with open(metadata_path, "rb") as f:
                        metadata = pickle.load(f)
                        st.info(f"Modèle: {metadata.get('model_name', 'Unknown')}")
                except:
                    pass
            
            st.markdown("---")
            
        # Model selection - Modified to remove GPT option
        st.session_state.model_choice = st.radio(
            "Modèle LLM",
            ["llama", "mistral", "phi", "openrouter"],  # "gpt" removed
            format_func=lambda x: {
                "llama": "Llama 3",
                # "gpt": "GPT-3.5",  # commented out
                "mistral": "Mistral 7B",
                "phi": "Phi-4-mini",
                "openrouter": "Llama 4 Maverick"
            }[x],
            horizontal=False
        )

        
        # Model information with clean markdown formatting
        with st.expander("Infos modèle", expanded=False):
            if st.session_state.model_choice == "llama":
                st.markdown("""
                **Meta-Llama-3-8B**
                
                * Bonne compréhension des instructions
                * Fort en synthèse de documents longs
                * Précision factuelle solide
                """)
            # GPT model info - commented out but preserved
            # elif st.session_state.model_choice == "gpt":
            #     st.markdown("""
            #     **GPT-3.5-Turbo**
            #     
            #     * Excellent en analyse contextuelle
            #     * Fort en résumé et reformulation
            #     * Bonnes capacités multilingues
            #     """)
            elif st.session_state.model_choice == "mistral":
                st.markdown("""
                **Mistral-7B-Instruct**
                
                * Raisonnement sur documents scientifiques
                * Bonne extraction d'informations
                * Réponses structurées en français
                """)
            elif st.session_state.model_choice == "phi":
                st.markdown("""
                **Phi-4-mini**
                
                * Rapide pour traitement RAG léger
                * Bon ratio performance/taille
                * Précision sur citations textuelles
                """)
            elif st.session_state.model_choice == "openrouter":
                st.markdown("""
                **Llama 4 Maverick**
                
                * Dernière génération de Llama
                * Performances supérieures
                * Excellente compréhension du français
                """)
        
        # Prompt configuration in expander - only query prompt is customizable
        with st.expander("Configuration du prompt (COSTAR)", expanded=False):
            # Initialize query prompt if not present
            if "query_prompt" not in st.session_state:
                st.session_state.query_prompt = DEFAULT_QUERY_PROMPT
            
            st.markdown("##### Framework COSTAR")
            st.markdown("*Méthodologie structurée pour des réponses précises*")
            
            # Explain COSTAR
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
                "Query prompt",
                value=st.session_state.query_prompt,
                height=300,
                key="query_prompt_area",
                label_visibility="collapsed"
            )
            
            # Add button to reset prompt to default
            if st.button("Réinitialiser le prompt"):
                st.session_state.query_prompt = DEFAULT_QUERY_PROMPT
                st.experimental_rerun()
            
        # Initialize uploaded_files in session state if not present
        if "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = []

        st.markdown("### Fichiers XML")  # Section header
        
        # File uploader with clear label
        uploaded_files = st.file_uploader("Télécharger", 
                                        type=["xml", "xmltei"], 
                                        accept_multiple_files=True,
                                        label_visibility="collapsed")  # Hide redundant label
        
        # Process uploaded files and store them in session state
        if uploaded_files:
            # Clear existing files first
            new_files = []
            
            # Create the upload directory if it doesn't exist
            os.makedirs("data/uploaded", exist_ok=True)
            
            # Process each file silently without success messages here
            for uploaded_file in uploaded_files:
                file_path = os.path.join("data/uploaded", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                new_files.append(file_path)
            
            # Add new files to existing list
            for file_path in new_files:
                if file_path not in st.session_state.uploaded_files:
                    st.session_state.uploaded_files.append(file_path)
            
            # Show a single success message instead of multiple ones
            if len(new_files) > 0:
                st.success(f"{len(new_files)} fichier(s) sauvegardé(s)")
        
        # Display checkbox for using only uploaded files - with compact styling
        st.session_state.use_uploaded_only = st.checkbox(
            "Utiliser uniquement fichiers téléchargés",  # Shortened label
            value=bool(st.session_state.uploaded_files)
        )
        
        # Warning if checkbox is checked but no files are uploaded
        if st.session_state.use_uploaded_only and not st.session_state.uploaded_files:
            st.warning("Aucun fichier téléchargé")
        
        # Display the list of uploaded files in a more compact way
        if st.session_state.uploaded_files:
            total_files = len(st.session_state.uploaded_files)
            with st.expander(f"Fichiers ({total_files})", expanded=False):
                # Create a scrollable area for the files with fixed height
                file_list_html = "<div style='max-height: 150px; overflow-y: auto;'>"
                for file_path in st.session_state.uploaded_files:
                    file_name = os.path.basename(file_path)
                    file_list_html += f"<div style='padding: 2px 0; font-size: 13px;'>✓ {file_name}</div>"
                file_list_html += "</div>"
                st.markdown(file_list_html, unsafe_allow_html=True)
                
                # Add a clear button
                if st.button("Effacer tous", key="clear_files"):
                    st.session_state.uploaded_files = []
                    st.experimental_rerun()
def boot():
    """Main function to run the application."""
    # Initialize query prompt if not present
    if "query_prompt" not in st.session_state:
        st.session_state.query_prompt = DEFAULT_QUERY_PROMPT
    
    # Setup input fields
    input_fields()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    
    # Add buttons for different processing methods
    col1, col2 = st.columns(2)
    
    # Button for pre-computed embeddings
    if st.session_state.use_precomputed:
        with col1:
            if st.button("Charger embeddings pré-calculés", use_container_width=True):
                with st.spinner("Chargement des embeddings pré-calculés..."):
                    st.session_state.retriever = load_precomputed_embeddings()
    
    # Button for processing documents
    if not st.session_state.use_precomputed:
        with col1:
            if st.button("Traiter les documents", use_container_width=True):
                st.session_state.retriever = process_documents(
                    st.session_state.hf_api_key, 
                    st.session_state.use_uploaded_only
                )
    
    # Display chat history
    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('ai').write(message[1])
    
    # Chat input
    if query := st.chat_input("Posez votre question..."):
        if not st.session_state.retriever:
            st.error("Veuillez d'abord charger les embeddings ou traiter les documents.")
            return
        
        st.chat_message("human").write(query)
        
        with st.spinner("Génération de la réponse..."):
            try:
                # Check model requirements - GPT check commented out
                # if st.session_state.model_choice == "gpt" and not st.session_state.openai_api_key:
                #     st.error("La clé API OpenAI est requise pour utiliser le modèle GPT-3.5.")
                #     return
                
                # For backward compatibility, still pass openai_api_key even though it's not used
                answer, source_docs = query_llm(
                    st.session_state.retriever, 
                    query, 
                    st.session_state.hf_api_key,
                    None,  # openai_api_key set to None
                    st.session_state.openrouter_api_key, 
                    st.session_state.model_choice
                )
                
                # Display the answer with markdown support
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
                        doc_year = doc.metadata.get('year', '')
                        doc_file = doc.metadata.get('source', 'Fichier inconnu')
                        
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
                            
                            # Show content
                            st.markdown("**Extrait:**")
                            content = doc.page_content
                            # Clean up content if needed
                            if content.startswith(f"Document: {doc_title}"):
                                content = content.replace(f"Document: {doc_title} | Date: {doc_date}\n\n", "")
                            
                            st.text_area("", value=content, height=150, disabled=True)
            
            except Exception as e:
                st.error(f"Error generating response: {e}")

if __name__ == '__main__':
    boot()
