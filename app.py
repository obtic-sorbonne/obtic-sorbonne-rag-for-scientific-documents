import os
import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import streamlit as st
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document  
from langchain_community.vectorstores import FAISS

# Define paths
TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath('data', 'vector_store')

# Create directories if they don't exist
TMP_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# Define namespaces for XML-TEI documents
NAMESPACES = {
    'tei': 'http://www.tei-c.org/ns/1.0'
}

st.set_page_config(page_title="RAG Démonstration", page_icon="🤖", layout="wide")
st.title("Retrieval Augmented Generation avec Llama")

# Function to extract year from document date
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
        
        # Add person names as additional information
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

def load_documents():
    """Load XML documents from the current directory and data directory."""
    # Check for XML files in the current directory and data directory
    documents = []
    document_dates = {}
    
    xml_files = []
    for path in [".", "data"]:
        if os.path.exists(path):
            for file in os.listdir(path):
                if file.endswith(".xml") or file.endswith(".xmltei"):
                    file_path = os.path.join(path, file)
                    xml_files.append(file_path)
    
    if not xml_files:
        st.error("No XML files found. Please upload XML files to the 'data' directory.")
        return documents, document_dates
    
    # Parse each XML file and create documents
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
                    "date": doc_data["date"],
                    "year": doc_data["year"],
                    "persons": doc_data["persons"]
                }
            )
            documents.append(doc)
            
            # Store year information
            if doc_data["year"]:
                document_dates[file_path] = doc_data["year"]
    
    return documents, document_dates

def split_documents(documents):
    """Split documents into chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    
    return texts

def embeddings_on_local_vectordb(texts, hf_api_key):
    """Create embeddings and store in a local vector database using FAISS instead of Chroma."""
    import os
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_api_key
    
    model_kwargs = {"token": hf_api_key}
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs=model_kwargs
    )
    
    # Use FAISS instead of Chroma
    vectordb = FAISS.from_documents(texts, embeddings)
    
    # Save the index
    vectordb.save_local(LOCAL_VECTOR_STORE_DIR.as_posix())
    
    retriever = vectordb.as_retriever(search_kwargs={'k': 3})
    return retriever

def query_llm(retriever, query, hf_api_key):
    """Query the LLM using Hugging Face and LangChain."""
    # Add system message to instruct the model to respond in French
    llm = HuggingFaceEndpoint(
        endpoint_url="https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct",
        huggingfacehub_api_token=hf_api_key,
        task="text-generation",
        temperature=0.7,          
        max_new_tokens=512,       
        top_p=0.95,
        model_kwargs={
            "parameters": {
                "system": "Tu es un assistant IA français. Réponds toujours en français, quelle que soit la langue de la question."
            }
        }
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        verbose=True
    )
    
    # Modify the query to include language instruction if not already present
    enhanced_query = f"{query} Réponds en français."
    
    result = qa_chain({"query": enhanced_query})
    answer = result["result"]
    source_docs = result["source_documents"]
    
    # Update message history
    if "messages" in st.session_state:
        st.session_state.messages.append((query, answer))
    
    return answer, source_docs

def process_documents(hf_api_key):
    """Process documents and create the retriever."""
    if not hf_api_key:
        st.warning("Please provide the Hugging Face API key.")
        return None
    
    try:
        # Load documents
        documents, document_dates = load_documents()
        
        if not documents:
            st.error("No documents found to process.")
            return None
        
        # Split into chunks
        texts = split_documents(documents)
        st.success(f"Created {len(texts)} chunks from {len(documents)} documents.")
        
        # Create embeddings and retriever
        retriever = embeddings_on_local_vectordb(texts, hf_api_key)
        st.success("Embeddings created and stored in vector database.")
        
        return retriever
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

def input_fields():
    """Set up the input fields in the sidebar."""
    with st.sidebar:
        st.title("Configuration")
        
        if "hf_api_key" in st.secrets:
            st.session_state.hf_api_key = st.secrets.hf_api_key
        else:
            st.session_state.hf_api_key = st.text_input("Hugging Face API Key", type="password")
            
        # File uploader for XML files
        uploaded_files = st.file_uploader("Télécharger des fichiers XML", 
                                          type=["xml", "xmltei"], 
                                          accept_multiple_files=True)
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                # Save the uploaded file to the data directory
                os.makedirs("data", exist_ok=True)
                file_path = os.path.join("data", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"Fichier {uploaded_file.name} sauvegardé.")

def boot():
    """Main function to run the application."""
    # Setup input fields
    input_fields()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    
    # Submit documents button
    if st.button("Traiter les documents"):
        st.session_state.retriever = process_documents(st.session_state.hf_api_key)
    
    # Display chat history
    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('ai').write(message[1])
    
    # Chat input
    if query := st.chat_input("Posez votre question..."):
        if not st.session_state.retriever:
            st.error("Veuillez d'abord traiter les documents.")
            return
        
        st.chat_message("human").write(query)
        
        with st.spinner("Génération de la réponse..."):
            try:
                answer, source_docs = query_llm(
                    st.session_state.retriever, 
                    query, 
                    st.session_state.hf_api_key
                )
                
                # Display the answer
                response_container = st.chat_message("ai")
                response_container.write(answer)
                
                # Display source documents
                if source_docs:
                    response_container.markdown("---")
                    response_container.markdown("**Sources:**")
                    for i, doc in enumerate(source_docs):
                        with response_container.expander(f"Source {i+1}: {doc.metadata.get('title', 'Unknown')}"):
                            response_container.markdown(f"**Date:** {doc.metadata.get('date', 'Unknown')}")
                            response_container.markdown(f"**Fichier:** {doc.metadata.get('source', 'Unknown')}")
                            if doc.metadata.get('persons'):
                                response_container.markdown(f"**Personnes mentionnées:** {', '.join(doc.metadata.get('persons'))}")
            
            except Exception as e:
                st.error(f"Error generating response: {e}")

if __name__ == '__main__':
    boot()
