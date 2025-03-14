import streamlit as st
import os
import xml.etree.ElementTree as ET
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np
import pickle

# Set page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="RAG Démonstration",
    page_icon="🤖",
    layout="wide"
)

# Define namespaces for XML-TEI documents
NAMESPACES = {
    'tei': 'http://www.tei-c.org/ns/1.0'
}

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = []
if 'embedding_function' not in st.session_state:
    st.session_state.embedding_function = None
if 'is_ready' not in st.session_state:
    st.session_state.is_ready = False

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

def initialize_system(api_key, chunk_size=1000, chunk_overlap=100):
    """Initialize the RAG system by processing documents and creating embeddings."""
    # Process XML files and create documents
    documents = create_documents_from_xml_files()
    
    if not documents:
        st.error("No documents found to process.")
        return False
    
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
    
    # Create embeddings for all chunks
    st.info("Creating vector embeddings...")
    texts = [doc.page_content for doc in chunks]
    embeddings = embedding_function.embed_documents(texts)
    
    # Store in session state
    st.session_state.documents = documents
    st.session_state.chunks = chunks
    st.session_state.embeddings = embeddings
    st.session_state.embedding_function = embedding_function
    st.session_state.is_ready = True
    
    return True

def search_similar_documents(query, k=3):
    """Search for documents similar to the query."""
    if not st.session_state.is_ready:
        st.error("System not initialized.")
        return []
    
    # Get query embedding
    query_embedding = st.session_state.embedding_function.embed_query(query)
    
    # Calculate similarity with all chunks
    similarities = []
    for i, doc_embedding in enumerate(st.session_state.embeddings):
        similarity = np.dot(query_embedding, doc_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
        )
        similarities.append((i, similarity))
    
    # Sort by similarity (higher is better)
    sorted_results = sorted(similarities, key=lambda x: x[1], reverse=True)
    
    # Return top k results
    top_results = []
    for i, score in sorted_results[:k]:
        top_results.append((st.session_state.chunks[i], score))
    
    return top_results

def generate_answer(query, context):
    """Generate an answer based on the retrieved context."""
    # Simple template for the answer
    answer = f"""
Sur la base des informations trouvées dans les documents, voici une réponse à votre question :

"{query}"

Les documents pertinents indiquent que :
"""
    
    # Add context from each document
    for i, (doc, _) in enumerate(context):
        answer += f"\n- {doc.page_content[:500]}...\n"
        
    answer += "\n\nCeci est un résumé des informations disponibles dans les documents fournis."
    
    return answer

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
            # Initialize system
            with st.spinner("Initialisation du système..."):
                success = initialize_system(hf_api_key, chunk_size, chunk_overlap)
                if success:
                    st.success("Système initialisé avec succès!")
                else:
                    st.error("Erreur lors de l'initialisation du système.")

# Chat interface
if st.session_state.is_ready:
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
            
            try:
                with st.spinner("Recherche de documents pertinents..."):
                    # Search for similar documents
                    results = search_similar_documents(prompt, k=k_value)
                    
                    # Generate answer
                    answer = generate_answer(prompt, results)
                
                # Display the answer
                message_placeholder.markdown(answer)
                
                # Display source documents
                if results:
                    st.markdown("---")
                    st.markdown("**Sources:**")
                    for i, (doc, score) in enumerate(results):
                        with st.expander(f"Source {i+1} (Similarité: {score:.4f})"):
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
st.markdown("Démonstration RAG - Version simplifiée")