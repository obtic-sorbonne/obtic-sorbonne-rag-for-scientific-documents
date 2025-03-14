import streamlit as st
import os
import xml.etree.ElementTree as ET
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Set page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="XML Document Explorer",
    page_icon="📄",
    layout="wide"
)

# Define namespaces for XML-TEI documents
NAMESPACES = {
    'tei': 'http://www.tei-c.org/ns/1.0'
}

# Initialize session state variables
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

def initialize_system(chunk_size=1000, chunk_overlap=100):
    """Initialize the system by processing documents and creating chunks."""
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

def search_documents(query):
    """Search for documents containing the query text (simple keyword search)."""
    if not st.session_state.is_ready:
        st.error("System not initialized.")
        return []
    
    query = query.lower()
    results = []
    
    for chunk in st.session_state.chunks:
        content = chunk.page_content.lower()
        if query in content:
            # Calculate a simple relevance score based on word frequency
            relevance = content.count(query) / len(content.split())
            results.append((chunk, relevance))
    
    # Sort by relevance score
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    
    # Take top 5 results or fewer if less available
    return sorted_results[:5]

# Main UI
st.title("📄 XML Document Explorer")
st.markdown("""
Cette application vous permet d'explorer le contenu de documents XML-TEI.
Vous pouvez effectuer une recherche simple dans les documents.
""")

# Sidebar for configuration
with st.sidebar:
    st.title("Configuration")
    
    # Chunking parameters
    st.subheader("Paramètres de découpage")
    chunk_size = st.slider("Taille des chunks", 500, 2000, 1000)
    chunk_overlap = st.slider("Chevauchement", 0, 200, 100)
    
    # Initialize system button
    if st.button("Initialiser le système"):
        # Initialize system
        with st.spinner("Initialisation du système..."):
            success = initialize_system(chunk_size, chunk_overlap)
            if success:
                st.success("Système initialisé avec succès!")
            else:
                st.error("Erreur lors de l'initialisation du système.")

# Search interface
if st.session_state.is_ready:
    st.subheader("Recherche dans les documents")
    search_query = st.text_input("Entrez votre terme de recherche:")
    
    if search_query:
        with st.spinner("Recherche..."):
            results = search_documents(search_query)
        
        if results:
            st.success(f"Trouvé {len(results)} résultats pertinents.")
            for i, (doc, score) in enumerate(results):
                with st.expander(f"Résultat {i+1} (Pertinence: {score:.4f})"):
                    st.markdown(f"**Document:** {doc.metadata.get('title', 'Unknown')}")
                    st.markdown(f"**Date:** {doc.metadata.get('date', 'Unknown')}")
                    st.markdown(f"**Fichier:** {doc.metadata.get('source', 'Unknown')}")
                    st.markdown("**Extrait:**")
                    st.markdown(doc.page_content)
        else:
            st.warning(f"Aucun résultat trouvé pour '{search_query}'.")
    
    # Show all documents
    st.subheader("Tous les documents")
    for i, doc in enumerate(st.session_state.documents):
        with st.expander(f"Document {i+1}: {doc.metadata.get('title', 'Unknown')}"):
            st.markdown(f"**Date:** {doc.metadata.get('date', 'Unknown')}")
            st.markdown(f"**Fichier:** {doc.metadata.get('source', 'Unknown')}")
            st.markdown("**Contenu:**")
            st.markdown(doc.page_content[:1000] + "..." if len(doc.page_content) > 1000 else doc.page_content)
else:
    st.info("Veuillez initialiser le système en utilisant le bouton dans la barre latérale.")

# Footer
st.markdown("---")
st.markdown("Explorer de documents XML - Version ultra simplifiée")