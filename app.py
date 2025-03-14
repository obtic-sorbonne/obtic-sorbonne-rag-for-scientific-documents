import streamlit as st
import os
import pickle
import gdown
import torch
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
import platform

# Set page configuration
st.set_page_config(
    page_title="RAG Démonstration",
    page_icon="🤖",
    layout="wide"
)

# Path configurations
VECTORSTORE_PATH = "vectorstore.pkl"
BACKUP_URL = "https://drive.google.com/uc?id=YOUR_GOOGLE_DRIVE_FILE_ID"  # Replace with your actual Google Drive file ID

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

# Function to load the pre-trained vectorstore
@st.cache_resource
def load_vectorstore(path):
    """Load a pre-trained vectorstore from disk with CUDA handling"""
    # Method 1: Direct loading of the pickle file with CUDA handling
    try:
        if os.path.exists(path):
            st.info("Loading vectorstore from disk...")
            # Map CUDA tensors to CPU if CUDA is not available
            if torch.cuda.is_available():
                with open(path, "rb") as f:
                    vectorstore = pickle.load(f)
            else:
                with open(path, "rb") as f:
                    # Use map_location to move tensors to CPU
                    vectorstore = pickle.load(f, map_location=torch.device('cpu'))
            st.success("Vectorstore loaded successfully!")
            return vectorstore
    except Exception as e:
        st.warning(f"Error loading vectorstore directly: {str(e)}")
        
        try:
            # Try again with a more robust approach
            st.info("Trying alternative loading method...")
            
            # Custom unpickler to handle CUDA tensors
            class CPUUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if module == 'torch.storage' and name == '_load_from_bytes':
                        return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
                    else:
                        return super().find_class(module, name)
            
            import io
            with open(path, 'rb') as f:
                vectorstore = CPUUnpickler(f).load()
                
            st.success("Vectorstore loaded successfully with alternative method!")
            return vectorstore
        except Exception as e2:
            st.warning(f"Error with alternative loading method: {str(e2)}")
    
    # Method 2: Try to download from Google Drive
    try:
        st.info("Attempting to download vectorstore from backup...")
        if "YOUR_GOOGLE_DRIVE_FILE_ID" not in BACKUP_URL:
            output = gdown.download(BACKUP_URL, "backup_vectorstore.pkl", quiet=False)
            if output:
                # Use CPU loading for downloaded file
                with open("backup_vectorstore.pkl", "rb") as f:
                    vectorstore = pickle.load(f, map_location=torch.device('cpu'))
                st.success("Vectorstore downloaded and loaded successfully!")
                return vectorstore
        else:
            st.warning("No backup URL configured. Skipping download.")
    except Exception as e:
        st.warning(f"Error downloading vectorstore: {str(e)}")
    
    # Method 3: Create a basic vectorstore in-memory if all else fails
    try:
        st.info("Attempting to create a basic in-memory vectorstore...")
        from langchain_community.vectorstores import Chroma
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.document_loaders import TextLoader
        
        # Check if there's a sample document we can use
        metadata_path = os.path.splitext(VECTORSTORE_PATH)[0] + "_metadata.txt"
        if os.path.exists(metadata_path):
            # Create a simple document from the metadata
            with open(metadata_path, "r") as f:
                text = f.read()
                
            from langchain.docstore.document import Document
            from langchain.text_splitter import CharacterTextSplitter
            
            # Create a document and split it
            doc = Document(page_content=text, metadata={"source": "metadata"})
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            docs = text_splitter.split_documents([doc])
            
            # Create embeddings and store
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
            vectorstore = Chroma.from_documents(
                docs, 
                embeddings,
                collection_name="demo_collection"
            )
            
            st.success("Created basic in-memory vectorstore!")
            return vectorstore
    except Exception as e:
        st.warning(f"Error creating basic vectorstore: {str(e)}")
    
    # All methods failed
    st.error("Could not load or create a vectorstore through any method.")
    return None

# Function to setup LLM with Hugging Face API
@st.cache_resource
def get_llm(model_name, api_key):
    """Setup LLM using Hugging Face API"""
    if not api_key:
        return None
        
    return HuggingFaceEndpoint(
        repo_id=model_name,
        huggingfacehub_api_token=api_key,
        max_length=512,
        temperature=0.7,
        top_p=0.95
    )

# Function to setup QA chain
def setup_qa_chain(vectorstore, model_name, api_key, k_value=3):
    """Setup the QA chain with the vectorstore and LLM"""
    # Get LLM
    llm = get_llm(model_name, api_key)
    if not llm:
        return None
    
    # Define custom prompt template
    template = """Tu es un assistant spécialisé qui répond aux questions basées uniquement sur les informations fournies.
    
    Contexte:
    {context}
    
    Question: {question}
    
    Réponds de manière précise et informative en utilisant uniquement les informations du contexte. 
    Si l'information ne se trouve pas dans le contexte, dis simplement que tu ne disposes pas de cette information.
    Réponse:"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": k_value})
    
    # Create QA chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return chain

# Display system info for debugging
if st.sidebar.checkbox("Show debug info", False):
    st.sidebar.write(f"Platform: {platform.system()}")
    st.sidebar.write(f"Python version: {platform.python_version()}")
    st.sidebar.write(f"CUDA available: {torch.cuda.is_available()}")
    if os.path.exists(VECTORSTORE_PATH):
        st.sidebar.write(f"Vectorstore file exists, size: {os.path.getsize(VECTORSTORE_PATH)/1024/1024:.2f} MB")
    else:
        st.sidebar.write("Vectorstore file does not exist!")

# Main UI
st.title("🤖 Démonstrateur de RAG")
st.markdown("""
Cette application démontre la fonctionnalité de Retrieval Augmented Generation (RAG) avec une base de connaissances pré-entraînée.
Posez simplement vos questions sur le contenu de la base de connaissances.
""")

# Sidebar for configuration
with st.sidebar:
    st.title("Configuration")
    
    # API key input
    hf_api_key = st.text_input("Hugging Face API Key", type="password")
    if not hf_api_key:
        st.warning("Veuillez entrer votre clé API Hugging Face pour utiliser les modèles.")
        st.info("Vous pouvez obtenir une clé API gratuite sur [huggingface.co](https://huggingface.co/settings/tokens)")
    
    # Model selection
    llm_model = st.selectbox(
        "Modèle LLM",
        ["mistralai/Mistral-7B-Instruct-v0.1", "meta-llama/Llama-2-7b-chat-hf", "bigscience/bloom"],
        index=0
    )
    
    # Top-k retrieval
    k_value = st.slider("Nombre de chunks à récupérer", 1, 5, 3)
    
    # Initialize system button
    if st.button("Initialiser le système"):
        if not hf_api_key:
            st.error("Veuillez entrer votre clé API Hugging Face pour continuer.")
        else:
            with st.spinner("Chargement de la base de connaissances..."):
                # Load the vectorstore
                vectorstore = load_vectorstore(VECTORSTORE_PATH)
                
                if vectorstore:
                    st.session_state.vectorstore = vectorstore
                    
                    # Setup QA chain
                    with st.spinner("Configuration du modèle de génération..."):
                        qa_chain = setup_qa_chain(vectorstore, llm_model, hf_api_key, k_value)
                        
                        if qa_chain:
                            st.session_state.qa_chain = qa_chain
                            st.success("Système initialisé avec succès!")
                        else:
                            st.error("Erreur lors de la configuration du modèle de génération.")
                else:
                    st.error("Erreur lors du chargement de la base de connaissances.")
    
    # Display metadata if available
    metadata_path = os.path.splitext(VECTORSTORE_PATH)[0] + "_metadata.txt"
    if os.path.exists(metadata_path):
        with st.expander("Informations sur la base de connaissances"):
            with open(metadata_path, "r") as f:
                st.text(f.read())

# Chat interface
if st.session_state.vectorstore is not None and st.session_state.qa_chain is not None:
    # Display existing messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input
    if prompt := st.chat_input("Posez votre question..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Use the QA chain to get response
            try:
                with st.spinner("Génération de la réponse..."):
                    result = st.session_state.qa_chain({"query": prompt})
                    answer = result["result"]
                    source_docs = result.get("source_documents", [])
                
                # Display the answer
                message_placeholder.markdown(answer)
                
                # Display source documents
                if source_docs:
                    st.markdown("---")
                    st.markdown("**Sources:**")
                    for i, doc in enumerate(source_docs):
                        with st.expander(f"Source {i+1}"):
                            st.markdown(doc.page_content)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                error_msg = f"Erreur lors de la génération de la réponse: {str(e)}"
                message_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
else:
    st.info("Veuillez initialiser le système en utilisant le bouton dans la barre latérale.")

# Footer
st.markdown("---")
st.markdown("Démonstration RAG - Développé avec Langchain et Streamlit")

import streamlit as st
import os
import pickle
import gdown
import torch
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
import platform

# Set page configuration
st.set_page_config(
    page_title="RAG Démonstration",
    page_icon="🤖",
    layout="wide"
)

# Path configurations
VECTORSTORE_PATH = "vectorstore.pkl"
BACKUP_URL = "https://drive.google.com/uc?id=YOUR_GOOGLE_DRIVE_FILE_ID"  # Replace with your actual Google Drive file ID

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

# Function to load the pre-trained vectorstore
@st.cache_resource
def load_vectorstore(path):
    """Load a pre-trained vectorstore from disk with CUDA handling"""
    # Method 1: Direct loading of the pickle file with CUDA handling
    try:
        if os.path.exists(path):
            st.info("Loading vectorstore from disk...")
            # Map CUDA tensors to CPU if CUDA is not available
            if torch.cuda.is_available():
                with open(path, "rb") as f:
                    vectorstore = pickle.load(f)
            else:
                with open(path, "rb") as f:
                    # Use map_location to move tensors to CPU
                    vectorstore = pickle.load(f, map_location=torch.device('cpu'))
            st.success("Vectorstore loaded successfully!")
            return vectorstore
    except Exception as e:
        st.warning(f"Error loading vectorstore directly: {str(e)}")
        
        try:
            # Try again with a more robust approach
            st.info("Trying alternative loading method...")
            
            # Custom unpickler to handle CUDA tensors
            class CPUUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if module == 'torch.storage' and name == '_load_from_bytes':
                        return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
                    else:
                        return super().find_class(module, name)
            
            import io
            with open(path, 'rb') as f:
                vectorstore = CPUUnpickler(f).load()
                
            st.success("Vectorstore loaded successfully with alternative method!")
            return vectorstore
        except Exception as e2:
            st.warning(f"Error with alternative loading method: {str(e2)}")
    
    # Method 2: Try to download from Google Drive
    try:
        st.info("Attempting to download vectorstore from backup...")
        if "YOUR_GOOGLE_DRIVE_FILE_ID" not in BACKUP_URL:
            output = gdown.download(BACKUP_URL, "backup_vectorstore.pkl", quiet=False)
            if output:
                # Use CPU loading for downloaded file
                with open("backup_vectorstore.pkl", "rb") as f:
                    vectorstore = pickle.load(f, map_location=torch.device('cpu'))
                st.success("Vectorstore downloaded and loaded successfully!")
                return vectorstore
        else:
            st.warning("No backup URL configured. Skipping download.")
    except Exception as e:
        st.warning(f"Error downloading vectorstore: {str(e)}")
    
    # Method 3: Create a basic vectorstore in-memory if all else fails
    try:
        st.info("Attempting to create a basic in-memory vectorstore...")
        from langchain_community.vectorstores import Chroma
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.document_loaders import TextLoader
        
        # Check if there's a sample document we can use
        metadata_path = os.path.splitext(VECTORSTORE_PATH)[0] + "_metadata.txt"
        if os.path.exists(metadata_path):
            # Create a simple document from the metadata
            with open(metadata_path, "r") as f:
                text = f.read()
                
            from langchain.docstore.document import Document
            from langchain.text_splitter import CharacterTextSplitter
            
            # Create a document and split it
            doc = Document(page_content=text, metadata={"source": "metadata"})
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            docs = text_splitter.split_documents([doc])
            
            # Create embeddings and store
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
            vectorstore = Chroma.from_documents(
                docs, 
                embeddings,
                collection_name="demo_collection"
            )
            
            st.success("Created basic in-memory vectorstore!")
            return vectorstore
    except Exception as e:
        st.warning(f"Error creating basic vectorstore: {str(e)}")
    
    # All methods failed
    st.error("Could not load or create a vectorstore through any method.")
    return None

# Function to setup LLM with Hugging Face API
@st.cache_resource
def get_llm(model_name, api_key):
    """Setup LLM using Hugging Face API"""
    if not api_key:
        return None
        
    return HuggingFaceEndpoint(
        repo_id=model_name,
        huggingfacehub_api_token=api_key,
        max_length=512,
        temperature=0.7,
        top_p=0.95
    )

# Function to setup QA chain
def setup_qa_chain(vectorstore, model_name, api_key, k_value=3):
    """Setup the QA chain with the vectorstore and LLM"""
    # Get LLM
    llm = get_llm(model_name, api_key)
    if not llm:
        return None
    
    # Define custom prompt template
    template = """Tu es un assistant spécialisé qui répond aux questions basées uniquement sur les informations fournies.
    
    Contexte:
    {context}
    
    Question: {question}
    
    Réponds de manière précise et informative en utilisant uniquement les informations du contexte. 
    Si l'information ne se trouve pas dans le contexte, dis simplement que tu ne disposes pas de cette information.
    Réponse:"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": k_value})
    
    # Create QA chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return chain

# Display system info for debugging
if st.sidebar.checkbox("Show debug info", False):
    st.sidebar.write(f"Platform: {platform.system()}")
    st.sidebar.write(f"Python version: {platform.python_version()}")
    st.sidebar.write(f"CUDA available: {torch.cuda.is_available()}")
    if os.path.exists(VECTORSTORE_PATH):
        st.sidebar.write(f"Vectorstore file exists, size: {os.path.getsize(VECTORSTORE_PATH)/1024/1024:.2f} MB")
    else:
        st.sidebar.write("Vectorstore file does not exist!")

# Main UI
st.title("🤖 Démonstrateur de RAG")
st.markdown("""
Cette application démontre la fonctionnalité de Retrieval Augmented Generation (RAG) avec une base de connaissances pré-entraînée.
Posez simplement vos questions sur le contenu de la base de connaissances.
""")

# Sidebar for configuration
with st.sidebar:
    st.title("Configuration")
    
    # API key input
    hf_api_key = st.text_input("Hugging Face API Key", type="password")
    if not hf_api_key:
        st.warning("Veuillez entrer votre clé API Hugging Face pour utiliser les modèles.")
        st.info("Vous pouvez obtenir une clé API gratuite sur [huggingface.co](https://huggingface.co/settings/tokens)")
    
    # Model selection
    llm_model = st.selectbox(
        "Modèle LLM",
        ["mistralai/Mistral-7B-Instruct-v0.1", "meta-llama/Llama-2-7b-chat-hf", "bigscience/bloom"],
        index=0
    )
    
    # Top-k retrieval
    k_value = st.slider("Nombre de chunks à récupérer", 1, 5, 3)
    
    # Initialize system button
    if st.button("Initialiser le système"):
        if not hf_api_key:
            st.error("Veuillez entrer votre clé API Hugging Face pour continuer.")
        else:
            with st.spinner("Chargement de la base de connaissances..."):
                # Load the vectorstore
                vectorstore = load_vectorstore(VECTORSTORE_PATH)
                
                if vectorstore:
                    st.session_state.vectorstore = vectorstore
                    
                    # Setup QA chain
                    with st.spinner("Configuration du modèle de génération..."):
                        qa_chain = setup_qa_chain(vectorstore, llm_model, hf_api_key, k_value)
                        
                        if qa_chain:
                            st.session_state.qa_chain = qa_chain
                            st.success("Système initialisé avec succès!")
                        else:
                            st.error("Erreur lors de la configuration du modèle de génération.")
                else:
                    st.error("Erreur lors du chargement de la base de connaissances.")
    
    # Display metadata if available
    metadata_path = os.path.splitext(VECTORSTORE_PATH)[0] + "_metadata.txt"
    if os.path.exists(metadata_path):
        with st.expander("Informations sur la base de connaissances"):
            with open(metadata_path, "r") as f:
                st.text(f.read())

# Chat interface
if st.session_state.vectorstore is not None and st.session_state.qa_chain is not None:
    # Display existing messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input
    if prompt := st.chat_input("Posez votre question..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Use the QA chain to get response
            try:
                with st.spinner("Génération de la réponse..."):
                    result = st.session_state.qa_chain({"query": prompt})
                    answer = result["result"]
                    source_docs = result.get("source_documents", [])
                
                # Display the answer
                message_placeholder.markdown(answer)
                
                # Display source documents
                if source_docs:
                    st.markdown("---")
                    st.markdown("**Sources:**")
                    for i, doc in enumerate(source_docs):
                        with st.expander(f"Source {i+1}"):
                            st.markdown(doc.page_content)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                error_msg = f"Erreur lors de la génération de la réponse: {str(e)}"
                message_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
else:
    st.info("Veuillez initialiser le système en utilisant le bouton dans la barre latérale.")

# Footer
st.markdown("---")
st.markdown("Démonstration RAG - Développé avec Langchain et Streamlit")