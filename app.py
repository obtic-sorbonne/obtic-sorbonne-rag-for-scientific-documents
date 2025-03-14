import streamlit as st
import os
import pickle
import platform
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Set page configuration
st.set_page_config(
    page_title="RAG Démonstration",
    page_icon="🤖",
    layout="wide"
)

# Path to your pre-trained vectorstore
VECTORSTORE_PATH = "vectorstore.pkl"

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
    """Load a pre-trained vectorstore from disk"""
    try:
        # Check if the file exists
        if not os.path.exists(path):
            st.error(f"Vector store file not found: {path}")
            st.info("Please make sure the file exists. If you're using Git LFS, verify it was pulled correctly.")
            return None
            
        with open(path, "rb") as f:
            vectorstore = pickle.load(f)
        return vectorstore
    except Exception as e:
        st.error(f"Error loading vectorstore: {str(e)}")
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