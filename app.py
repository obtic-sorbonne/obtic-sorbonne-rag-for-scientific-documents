import streamlit as st
import os
import xml.etree.ElementTree as ET
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import numpy as np
import pickle
from typing import List, Dict, Any
from langchain.llms.huggingface_hub import HuggingFaceHub

# Set page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="RAG Démonstration",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

# Define namespaces for XML-TEI documents
NAMESPACES = {
    'tei': 'http://www.tei-c.org/ns/1.0'
}

# Simple vector store implementation
class SimpleVectorStore:
    def __init__(self, embedding_function):
        self.embedding_function = embedding_function
        self.documents = []
        self.embeddings = []
        
    def add_documents(self, documents):
        texts = [doc.page_content for doc in documents]
        embeddings = self.embedding_function.embed_documents(texts)
        
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
        
    def similarity_search_with_score(self, query, k=4):
        query_embedding = self.embedding_function.embed_query(query)
        
        if not self.embeddings:
            return []
        
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            similarities.append((i, similarity))
        
        # Sort by similarity score (higher is better)
        sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
        
        # Return top k results
        results = []
        for i, score in sorted_similarities[:k]:
            results.append((self.documents[i], score))
        
        return results
    
    def _cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump((self.documents, self.embeddings), f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            self.documents, self.embeddings = pickle.load(f)
            
    def as_retriever(self, search_kwargs=None):
        return SimpleRetriever(self, search_kwargs)

class SimpleRetriever:
    def __init__(self, vectorstore, search_kwargs=None):
        self.vectorstore = vectorstore
        self.search_kwargs = search_kwargs or {"k": 4}
        
    def get_relevant_documents(self, query):
        results = self.vectorstore.similarity_search_with_score(
            query, 
            k=self.search_kwargs.get("k", 4)
        )
        return [doc for doc, _ in results]

# Function to parse XML-TEI documents
def parse_xmltei_document(file_path):
    """Parse an XML-TEI document and extract text content with metadata."""
    try:
        # Parse the XML file
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
        
        # Get all paragraphs
        paragraphs = root.findall('.//tei:p', NAMESPACES)
        
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
        
        return {
            "title": title_text,
            "date": date_text,
            "text": full_text,
            "paragraphs": all_paragraphs
        }
        
    except Exception as e:
        st.error(f"Error parsing XML file {file_path}: {str(e)}")
        return None

# Function to create documents from XML files
def create_documents_from_xml_files():
    """Process XML files in the data directory and create documents."""
    # Check for XML files in the current directory and data directory
    xml_files = []
    for path in [".", "data"]:
        if os.path.exists(path):
            for file in os.listdir(path):
                if file.endswith(".xml") or file.endswith(".xmltei"):
                    file_path = os.path.join(path, file)
                    xml_files.append(file_path)
    
    if not xml_files:
        st.error("No XML files found. Please upload XML files to the 'data' directory.")
        return []
    
    # Parse each XML file and create documents
    documents = []
    for file_path in xml_files:
        st.info(f"Processing {file_path}...")
        doc_data = parse_xmltei_document(file_path)
        
        if doc_data:
            # Create a Document object with metadata
            doc = Document(
                page_content=doc_data["text"],
                metadata={
                    "source": file_path,
                    "title": doc_data["title"],
                    "date": doc_data["date"]
                }
            )
            documents.append(doc)
    
    return documents

# Function to create vectorstore from documents
def create_vectorstore(documents, chunk_size=1000, chunk_overlap=100):
    """Create a vectorstore from the provided documents."""
    if not documents:
        st.error("No documents to process.")
        return None
    
    # Split documents into chunks
    st.info(f"Splitting {len(documents)} documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    st.success(f"Created {len(chunks)} chunks.")
    
    # Create embedding function
    st.info("Initializing embedding model...")
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # Create our simple vectorstore
    st.info("Creating vector database...")
    vectorstore = SimpleVectorStore(embedding_function)
    vectorstore.add_documents(chunks)
    
    # Save the vectorstore to a file for persistence
    os.makedirs("vectorstore", exist_ok=True)
    vectorstore.save("vectorstore/index.pkl")
    
    return vectorstore

# Function to setup LLM with Hugging Face API
@st.cache_resource
def get_llm(model_name, api_key):
    """Setup LLM using Hugging Face API"""
    if not api_key:
        return None
    
    # Using HuggingFaceHub instead of HuggingFaceEndpoint
    return HuggingFaceHub(
        huggingfacehub_api_token=api_key,
        repo_id=model_name,
        model_kwargs={"temperature": 0.7, "max_length": 512, "top_p": 0.95}
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

# Main UI
st.title("🤖 Démonstrateur de RAG")
st.markdown("""
Cette application démontre la fonctionnalité de Retrieval Augmented Generation (RAG) avec des documents XML-TEI.
Posez simplement vos questions sur le contenu des documents.
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
    
    # Chunking parameters
    st.subheader("Paramètres de découpage")
    chunk_size = st.slider("Taille des chunks", 500, 2000, 1000)
    chunk_overlap = st.slider("Chevauchement", 0, 200, 100)
    
    # Top-k retrieval
    k_value = st.slider("Nombre de chunks à récupérer", 1, 5, 3)
    
    # Initialize system button
    if st.button("Initialiser le système"):
        if not hf_api_key:
            st.error("Veuillez entrer votre clé API Hugging Face pour continuer.")
        else:
            # Create vectorstore from XML files
            with st.spinner("Création de la base de connaissances..."):
                # Process XML files and create documents
                documents = create_documents_from_xml_files()
                
                if documents:
                    # Create vectorstore from documents
                    vectorstore = create_vectorstore(documents, chunk_size, chunk_overlap)
                    
                    if vectorstore:
                        st.session_state.vectorstore = vectorstore
                        st.success(f"Base de connaissances créée avec {len(documents)} documents!")
                        
                        # Setup QA chain
                        with st.spinner("Configuration du modèle de génération..."):
                            qa_chain = setup_qa_chain(vectorstore, llm_model, hf_api_key, k_value)
                            
                            if qa_chain:
                                st.session_state.qa_chain = qa_chain
                                st.success("Système initialisé avec succès!")
                            else:
                                st.error("Erreur lors de la configuration du modèle de génération.")
                    else:
                        st.error("Erreur lors de la création de la base de connaissances.")
                else:
                    st.error("Aucun document trouvé pour créer la base de connaissances.")

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
                            st.markdown(f"**Document:** {doc.metadata.get('title', 'Unknown')}")
                            st.markdown(f"**Date:** {doc.metadata.get('date', 'Unknown')}")
                            st.markdown(f"**Fichier:** {doc.metadata.get('source', 'Unknown')}")
                            st.markdown("**Extrait:**")
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