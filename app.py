import streamlit as st
import os
import pickle
import gdown
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS, Chroma
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
BACKUP_URL = "https://drive.google.com/file/d/1JuMct3RweKjzEVWVVSYo_afGUz6EQQWA/view?usp=sharing"

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

# If we can't load the existing vectorstore, we'll try to create a new one from the content
def convert_faiss_to_chroma(faiss_store_path):
    """Convert a FAISS vectorstore to Chroma which doesn't use FAISS"""
    try:
        # Load the FAISS store
        with open(faiss_store_path, "rb") as f:
            faiss_store = pickle.load(f)
        
        # Extract the documents and their embeddings
        docs = []
        for doc_id in faiss_store.docstore._dict:
            docs.append(faiss_store.docstore._dict[doc_id])
        
        # Create a new embedding function
        embedding_function = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        # Create a Chroma vectorstore from the documents
        chroma_store = Chroma.from_documents(
            docs, 
            embedding_function,
            collection_name="converted_from_faiss"
        )
        
        return chroma_store
    except Exception as e:
        st.error(f"Error converting vectorstore: {str(e)}")
        return None

# Function to load the pre-trained vectorstore
@st.cache_resource
def load_vectorstore(path):
    """Load a pre-trained vectorstore from disk"""
    # Try different methods to load the vectorstore
    
    # Method 1: Direct loading of the pickle file
    try:
        if os.path.exists(path):
            st.info("Loading vectorstore from disk...")
            with open(path, "rb") as f:
                vectorstore = pickle.load(f)
            st.success("Vectorstore loaded successfully!")
            return vectorstore
    except Exception as e:
        st.warning(f"Error loading vectorstore directly: {str(e)}")
    
    # Method 2: Try to download from Google Drive
    try:
        st.info("Attempting to download vectorstore from backup...")
        if "YOUR_GOOGLE_DRIVE_FILE_ID" not in BACKUP_URL:
            output = gdown.download(BACKUP_URL, path, quiet=False)
            if output:
                with open(path, "rb") as f:
                    vectorstore = pickle.load(f)
                st.success("Vectorstore downloaded and loaded successfully!")
                return vectorstore
        else:
            st.warning("No backup URL configured. Skipping download.")
    except Exception as e:
        st.warning(f"Error downloading vectorstore: {str(e)}")
    
    # Method 3: Try to convert FAISS to another format
    try:
        st.info("Attempting to convert vectorstore format...")
        converted_store = convert_faiss_to_chroma(path)
        if converted_store:
            st.success("Vectorstore converted successfully!")
            return converted_store
    except Exception as e:
        st.warning(f"Error converting vectorstore: {str(e)}")
    
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