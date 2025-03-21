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
from langchain_openai import ChatOpenAI  # Added for OpenAI support

# Defining paths 
TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath('data', 'vector_store')

TMP_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# Define namespaces for XML-tei
NAMESPACES = {
    'tei': 'http://www.tei-c.org/ns/1.0'
}

st.set_page_config(page_title="RAG Démonstration", page_icon="🤖", layout="wide")
st.title("Retrieval Augmented Generation")
st.markdown("#### Projet préparé par l'équipe ObTIC.")

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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    
    return texts

def embeddings_on_local_vectordb(texts, hf_api_key):
    """Create embeddings and store in a local vector database using FAISS."""
    import os
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_api_key
    
    model_kwargs = {"token": hf_api_key}
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs=model_kwargs
    )
    
    vectordb = FAISS.from_documents(texts, embeddings)
    vectordb.save_local(LOCAL_VECTOR_STORE_DIR.as_posix()) # saving vectors during session
    retriever = vectordb.as_retriever(search_kwargs={'k': 3}) #top retrieval
    return retriever

def query_llm(retriever, query, hf_api_key, openai_api_key=None, model_choice="llama"):
    """Query the LLM using one of the supported models."""
    
    if model_choice == "gpt":
        if not openai_api_key:
            st.error("OpenAI API key is required to use GPT-3.5 model")
            return None, None
            
        llm = ChatOpenAI(
            temperature=0.4,
            model_name="gpt-3.5-turbo",
            openai_api_key=openai_api_key,
            max_tokens=512
        )
    elif model_choice == "mistral":
        if not hf_api_key:
            st.error("Hugging Face API key is required to use Mistral model")
            return None, None
            
        llm = HuggingFaceEndpoint(
            endpoint_url="https://api-inference.huggingface.co/models/mistralai/Mistral-Small-24B-Instruct-2501",
            huggingfacehub_api_token=hf_api_key,
            task="text-generation",
            temperature=0.4,
            max_new_tokens=512,
            top_p=0.95,
            model_kwargs={
                "parameters": {
                    "system": st.session_state.system_prompt
                }
            }
        )
    elif model_choice == "phi":
        if not hf_api_key:
            st.error("Hugging Face API key is required to use Phi model")
            return None, None
            
        llm = HuggingFaceEndpoint(
            endpoint_url="https://api-inference.huggingface.co/models/microsoft/Phi-4-mini-instruct",
            huggingfacehub_api_token=hf_api_key,
            task="text-generation",
            temperature=0.4,
            max_new_tokens=512,
            top_p=0.95,
            model_kwargs={
                "parameters": {
                    "system": st.session_state.system_prompt
                }
            }
        )
    else:  # Default to llama
        llm = HuggingFaceEndpoint(
            endpoint_url="https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct",
            huggingfacehub_api_token=hf_api_key,
            task="text-generation",
            temperature=0.4,
            max_new_tokens=512,
            top_p=0.95,
            model_kwargs={
                "parameters": {
                    "system": st.session_state.system_prompt
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
    
    enh_query = f"""
    {query}
    Important : 
    1. Présente ta réponse de façon claire et bien structurée.
    2. Utilise le formatage markdown pour mettre en évidence les points importants.
    3. Si tu cites des chiffres ou des statistiques, présente-les de manière structurée.
    4. Commence ta réponse par un court résumé de 1-2 phrases.
    5. Ajoute des titres et sous-titres si nécessaire pour organiser l'information.
    6. Utilise des listes à puces pour les énumérations.
    7. Réponds en français en utilisant un langage naturel et cohérent.
    """
    result = qa_chain({"query": enh_query})
    
    # Post-process to remove any notes that might still appear and format the answer
    answer = result["result"]
    if "Note:" in answer:
        answer = answer.split("Note:")[0].strip()
    if "Note :" in answer:
        answer = answer.split("Note :")[0].strip()
        
    # Apply additional formatting if needed
    if not any(marker in answer for marker in ["##", "**", "- ", "1. ", "_"]):
        # If the model didn't use markdown formatting, add some basic structure
        lines = answer.split("\n")
        if len(lines) > 2:
            # Add a summary header
            formatted_answer = f"## Résumé\n\n{lines[0]}\n\n## Détails\n\n" + "\n".join(lines[1:])
            answer = formatted_answer
        
    source_docs = result["source_documents"]
    
    # Update message history
    if "messages" in st.session_state:
        st.session_state.messages.append((query, answer))
    
    return answer, source_docs

def process_documents(hf_api_key, use_uploaded_only):
    if not hf_api_key:
        st.warning("Please provide the Hugging Face API key.")
        return None
    
    try:
        documents, document_dates = load_documents(use_uploaded_only)
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
        
        # Hugging Face API Key
        if "hf_api_key" in st.secrets:
            st.session_state.hf_api_key = st.secrets.hf_api_key
        else:
            st.session_state.hf_api_key = st.text_input("Hugging Face API Key", type="password")
        
        # OpenAI API Key
        if "openai_api_key" in st.secrets:
            st.session_state.openai_api_key = st.secrets.openai_api_key
        else:
            st.session_state.openai_api_key = st.text_input("OpenAI API Key (Pour GPT-3.5)", type="password")
            
        # Model selection radio button
        st.session_state.model_choice = st.radio(
            "Choisir un modèle LLM",
            ["llama", "gpt", "mistral", "phi"],
            format_func=lambda x: {
                "llama": "Llama 3",
                "gpt": "GPT-3.5",
                "mistral": "Mistral Small 24B",
                "phi": "Phi-4-mini"
            }[x]
        )
        
        # Affichage des informations sur le modèle sélectionné
        with st.expander("Informations sur le modèle", expanded=False):
            if st.session_state.model_choice == "llama":
                st.markdown("""
                ### Meta-Llama-3-8B-Instruct
                
                **Paramètres**: 8 milliards
                
                **Caractéristiques**:
                - Modèle d'instruction open source de Meta
                - Support multilingue
                - Fenêtre de contexte : 8K tokens
                - Bon équilibre entre qualité et performance
                """)
            elif st.session_state.model_choice == "gpt":
                st.markdown("""
                ### GPT-3.5-Turbo
                
                **Paramètres**: 175 milliards
                
                **Caractéristiques**:
                - Modèle propriétaire d'OpenAI
                - Excellent support multilingue
                - Fenêtre de contexte : 4K tokens (16K disponible)
                - Robuste pour de multiples tâches
                """)
            elif st.session_state.model_choice == "mistral":
                st.markdown("""
                ### Mistral-Small-24B-Instruct-2501
                
                **Paramètres**: 24 milliards
                
                **Caractéristiques**:
                - Modèle de Mistral AI sous licence Apache 2.0
                - Excellent support multilingue (10+ langues)
                - Fenêtre de contexte : 32K tokens
                - Capacités avancées de raisonnement et conversation
                - Optimisé pour les agents et fonction calling
                """)
            elif st.session_state.model_choice == "phi":
                st.markdown("""
                ### Phi-4-mini-instruct
                
                **Paramètres**: 3.8 milliards
                
                **Caractéristiques**:
                - Modèle léger de Microsoft sous licence MIT
                - Support pour 24 langues
                - Fenêtre de contexte : 128K tokens
                - Excellent en mathématiques et raisonnement logique
                - Optimisé pour des environnements avec ressources limitées
                """)
        
        # Add system prompt customization option
        with st.expander("Options avancées"):
            if "system_prompt" not in st.session_state:
                default_prompt = """Tu es un assistant IA français spécialisé dans l'analyse de documents scientifiques pour faire du RAG. 
                Instructions:
                1. Utilise uniquement les informations fournies dans le contexte ci-dessus pour répondre à la question.
                2. Si la réponse ne se trouve pas complètement dans le contexte, indique clairement les limites de ta réponse.
                3. Ne génère pas d'informations qui ne sont pas présentes dans le contexte.
                4. Cite les passages précis du contexte qui appuient ta réponse.
                5. Structure ta réponse de manière claire et concise.
                6. Si plusieurs interprétations sont possibles, présente les différentes perspectives.
                7. Si la question est ambiguë, demande des précisions.
                
                Réponds en français, dans un style professionnel et accessible."""
                st.session_state.system_prompt = default_prompt
            
            st.session_state.system_prompt = st.text_area(
                "Personnaliser l'instruction système (prompt)",
                value=st.session_state.system_prompt,
                height=200,
                key="system_prompt_textarea"  # Added unique key
            )
            
        # File uploader
        uploaded_files = st.file_uploader("Télécharger des fichiers XML", 
                                          type=["xml", "xmltei"], 
                                          accept_multiple_files=True)
        
        if "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = []
            
        if uploaded_files:
            st.session_state.uploaded_files = []
            
            for uploaded_file in uploaded_files:
                os.makedirs("data/uploaded", exist_ok=True)
                file_path = os.path.join("data/uploaded", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"Fichier {uploaded_file.name} sauvegardé.")
                st.session_state.uploaded_files.append(file_path)
        
        st.session_state.use_uploaded_only = st.checkbox(
            "Utiliser uniquement les fichiers téléchargés", 
            value=bool(st.session_state.uploaded_files)
        )
        
        if st.session_state.use_uploaded_only and not st.session_state.uploaded_files:
            st.warning("Aucun fichier téléchargé. Veuillez télécharger des fichiers ou utiliser le corpus par défaut.")

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
            st.error("Veuillez d'abord traiter les documents.")
            return
        
        st.chat_message("human").write(query)
        
        with st.spinner("Génération de la réponse..."):
            try:
                # Check model requirements
                if st.session_state.model_choice == "gpt" and not st.session_state.openai_api_key:
                    st.error("La clé API OpenAI est requise pour utiliser le modèle GPT-3.5.")
                    return
                
                answer, source_docs = query_llm(
                    st.session_state.retriever, 
                    query, 
                    st.session_state.hf_api_key,
                    st.session_state.openai_api_key,
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
