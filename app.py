import streamlit as st
import os
import xml.etree.ElementTree as ET
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import requests
import json

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
    """Initialize the RAG system by processing documents."""
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
    
    # Store in session state
    st.session_state.documents = documents
    st.session_state.chunks = chunks
    st.session_state.is_ready = True
    
    return True

def search_documents(query, k=3):
    """Search for documents relevant to the query."""
    if not st.session_state.is_ready:
        st.error("System not initialized.")
        return []
    
    # Simple hybrid search - combine keyword and relevance
    query_terms = set(query.lower().split())
    results = []
    
    for chunk in st.session_state.chunks:
        content = chunk.page_content.lower()
        content_words = set(content.split())
        
        # Calculate term overlap
        term_overlap = len(query_terms.intersection(content_words))
        if term_overlap > 0:
            # Simple relevance score based on term overlap and density
            relevance = term_overlap / max(1, len(query_terms))
            results.append((chunk, relevance))
    
    # Sort by relevance score
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    
    # Return top k results or all if fewer
    return sorted_results[:k]

def query_llm_api(api_key, query, context):
    """Query LLM API with the context and question."""
    # Prepare the context from retrieved documents
    context_text = "\n\n".join([doc.page_content for doc, _ in context])
    
    # Construct the prompt
    prompt = f"""Tu es un assistant spécialisé qui répond aux questions basées uniquement sur les informations fournies.
    
Contexte:
{context_text}

Question: {query}

Réponds de manière précise et informative en utilisant uniquement les informations du contexte. 
Si l'information ne se trouve pas dans le contexte, dis simplement que tu ne disposes pas de cette information."""

    try:
        # Send request to Hugging Face API
        API_URL = f"https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
        headers = {"Authorization": f"Bearer {api_key}"}
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.95,
                "do_sample": True
            }
        }
        
        response = requests.post(API_URL, headers=headers, json=payload)
        result = response.json()
        
        # Extract the generated text from response
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "").replace(prompt, "").strip()
        elif "generated_text" in result:
            return result["generated_text"].replace(prompt, "").strip()
        else:
            return "Erreur: Format de réponse inattendu de l'API."
    
    except Exception as e:
        return f"Erreur lors de la génération de la réponse: {str(e)}"

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
                    results = search_documents(prompt, k=k_value)
                
                if not results:
                    message_placeholder.warning("Aucun document pertinent trouvé pour répondre à votre question.")
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": "Aucun document pertinent trouvé pour répondre à votre question."
                    })
                else:
                    with st.spinner("Génération de la réponse..."):
                        # Generate answer using Hugging Face API
                        answer = query_llm_api(hf_api_key, prompt, results)
                    
                    # Display the answer
                    message_placeholder.markdown(answer)
                    
                    # Display source documents
                    st.markdown("---")
                    st.markdown("**Sources:**")
                    for i, (doc, score) in enumerate(results):
                        with st.expander(f"Source {i+1} (Pertinence: {score:.4f})"):
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
st.markdown("Démonstration RAG - Développé avec Streamlit")