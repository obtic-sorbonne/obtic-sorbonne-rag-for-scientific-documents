import streamlit as st
import os
import torch

# Set page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="RAG Démonstration",
    page_icon="🤖",
    layout="wide"
)

# Path configurations
VECTORSTORE_PATH = "vectorstore.pkl"
METADATA_PATH = "vectorstore_metadata.txt"

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

# Function to create a new vectorstore from metadata content
def create_chroma_vectorstore(text_content):
    """Create a Chroma vectorstore from text content"""
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.docstore.document import Document
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    
    # Create a sample document
    doc = Document(page_content=text_content, metadata={"source": "Metadata"})
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents([doc])
    
    # Create embedding function
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # Create vectorstore
    vectorstore = Chroma.from_documents(
        docs, 
        embedding_function,
        collection_name="demo_collection"
    )
    
    return vectorstore

# Function to setup LLM with Hugging Face API
@st.cache_resource
def get_llm(model_name, api_key):
    """Setup LLM using Hugging Face API"""
    from langchain_community.llms import HuggingFaceEndpoint
    
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
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    
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

# Main UI
st.title("🤖 Démonstrateur de RAG")
st.markdown("""
Cette application démontre la fonctionnalité de Retrieval Augmented Generation (RAG) avec une base de connaissances pré-entraînée.
Posez simplement vos questions sur le contenu de la base de connaissances.
""")

# Display debug info
if st.sidebar.checkbox("Show debug info", False):
    st.sidebar.write(f"CUDA available: {torch.cuda.is_available()}")
    
    if os.path.exists(VECTORSTORE_PATH):
        st.sidebar.write(f"Vectorstore exists: {os.path.getsize(VECTORSTORE_PATH)/1024/1024:.2f} MB")
    else:
        st.sidebar.write("Vectorstore does not exist!")
        
    if os.path.exists(METADATA_PATH):
        st.sidebar.write(f"Metadata exists: {os.path.getsize(METADATA_PATH)/1024:.1f} KB")
    else:
        st.sidebar.write("Metadata does not exist!")

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
            # Create vectorstore from metadata
            with st.spinner("Création de la base de connaissances..."):
                metadata_content = "Bienvenue dans la démonstration de RAG. Ceci est un texte d'exemple pour la base de connaissances."
                
                # Try to load metadata content if available
                if os.path.exists(METADATA_PATH):
                    with open(METADATA_PATH, "r") as f:
                        metadata_content = f.read()
                
                try:
                    vectorstore = create_chroma_vectorstore(metadata_content)
                    st.session_state.vectorstore = vectorstore
                    st.success("Base de connaissances créée avec succès!")
                    
                    # Setup QA chain
                    with st.spinner("Configuration du modèle de génération..."):
                        qa_chain = setup_qa_chain(vectorstore, llm_model, hf_api_key, k_value)
                        
                        if qa_chain:
                            st.session_state.qa_chain = qa_chain
                            st.success("Système initialisé avec succès!")
                        else:
                            st.error("Erreur lors de la configuration du modèle de génération.")
                except Exception as e:
                    st.error(f"Erreur lors de la création de la base de connaissances: {str(e)}")
    
    # Display metadata if available
    if os.path.exists(METADATA_PATH):
        with st.expander("Informations sur la base de connaissances"):
            with open(METADATA_PATH, "r") as f:
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