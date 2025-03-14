import streamlit as st
import os
import re
import xml.etree.ElementTree as ET
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import requests
import time
import numpy as np

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
if 'document_dates' not in st.session_state:
    st.session_state.document_dates = {}
if 'is_ready' not in st.session_state:
    st.session_state.is_ready = False

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
            "paragraphs": all_paragraphs,
            "persons": person_text
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
    document_dates = {}
    
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

def initialize_system(api_key, chunk_size=1000, chunk_overlap=100):
    """Initialize the RAG system by processing documents."""
    # Process XML files and create documents
    documents, document_dates = create_documents_from_xml_files()
    
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
    
    # Ensure metadata is preserved in chunks
    for chunk in chunks:
        if "year" not in chunk.metadata and "source" in chunk.metadata:
            source = chunk.metadata["source"]
            if source in document_dates:
                chunk.metadata["year"] = document_dates[source]
    
    st.success(f"Created {len(chunks)} chunks.")
    
    # Store in session state
    st.session_state.documents = documents
    st.session_state.chunks = chunks
    st.session_state.document_dates = document_dates
    st.session_state.is_ready = True
    
    return True

def search_documents(query, api_key, k=3, year_filter=None):
    """Search for documents by semantic similarity to the query with optional year filtering."""
    if not st.session_state.is_ready:
        st.error("System not initialized.")
        return []
    
    # Extract year ranges from query if present
    year_ranges = extract_year_ranges(query)
    
    # Filter chunks by year range if specified
    filtered_chunks = st.session_state.chunks
    if year_ranges:
        filtered_chunks = []
        for chunk in st.session_state.chunks:
            chunk_year = chunk.metadata.get("year")
            if chunk_year:
                for start_year, end_year in year_ranges:
                    if start_year <= chunk_year <= end_year:
                        filtered_chunks.append(chunk)
                        break
    
    # If no chunks after filtering, return empty
    if not filtered_chunks:
        return []
    
    # Extract content from filtered chunks
    chunk_texts = [chunk.page_content for chunk in filtered_chunks]
    
    try:
        # Use Embedding API to generate embeddings for query and chunks
        API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        headers = {"Authorization": f"Bearer {api_key}"}
        
        # Get query embedding
        response = requests.post(API_URL, 
                                headers=headers, 
                                json={"inputs": query})
        
        if response.status_code != 200:
            # Fall back to simple keyword matching if API fails
            return fallback_search(query, k, filtered_chunks)
            
        query_embedding = response.json()
        
        # Get embeddings for all chunks (in batches to avoid timeouts)
        batch_size = 10
        chunk_embeddings = []
        
        for i in range(0, len(chunk_texts), batch_size):
            batch = chunk_texts[i:i+batch_size]
            response = requests.post(API_URL, 
                                    headers=headers, 
                                    json={"inputs": batch})
            
            if response.status_code != 200:
                # Fall back if API fails
                return fallback_search(query, k, filtered_chunks)
                
            batch_embeddings = response.json()
            chunk_embeddings.extend(batch_embeddings)
            time.sleep(1)  # Avoid rate limits
        
        # Calculate similarity scores
        results = []
        for i, chunk_embedding in enumerate(chunk_embeddings):
            # Compute cosine similarity
            similarity = cosine_similarity(query_embedding, chunk_embedding)
            results.append((filtered_chunks[i], similarity))
        
        # Sort by similarity score (higher is better)
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
        
        # Return top k results
        return sorted_results[:k]
        
    except Exception as e:
        st.warning(f"Error in semantic search: {str(e)}. Using fallback search.")
        return fallback_search(query, k, filtered_chunks)

def extract_year_ranges(query):
    """Extract year ranges from a query string."""
    # Pattern to match "between YYYY and YYYY" or "YYYY-YYYY" or "de YYYY à YYYY"
    patterns = [
        r'entre\s+(\d{4})\s+et\s+(\d{4})',
        r'(\d{4})\s*[-–]\s*(\d{4})',
        r'de\s+(\d{4})\s+[àa]\s+(\d{4})'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        if matches:
            return [(int(start), int(end)) for start, end in matches]
    
    # Try to find individual years
    years = re.findall(r'\b(19\d{2}|20\d{2})\b', query)
    if len(years) == 1:
        year = int(years[0])
        return [(year, year)]
    
    return []

def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    v1, v2 = np.array(v1), np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def fallback_search(query, k=3, chunks=None):
    """Simple keyword-based search as fallback."""
    if chunks is None:
        chunks = st.session_state.chunks
        
    query_terms = set(query.lower().split())
    results = []
    
    for chunk in chunks:
        content = chunk.page_content.lower()
        content_words = set(content.split())
        
        # Calculate term overlap
        term_overlap = len(query_terms.intersection(content_words))
        if term_overlap > 0:
            # Simple relevance score based on term overlap
            relevance = term_overlap / max(1, len(query_terms)) 
            results.append((chunk, relevance))
    
    # Sort by relevance score
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    
    # Return top k results
    return sorted_results[:k]

def extract_keywords(text, n=5):
    """Extract top keywords from text, ignoring common words."""
    # Common French words to ignore
    common_words = set([
        "le", "la", "les", "un", "une", "des", "et", "ou", "de", "du", "a", "à", "au", "aux", 
        "ce", "cette", "ces", "est", "sont", "être", "avoir", "faire", "plus", "moins", "très",
        "sans", "avec", "pour", "par", "sur", "sous", "dans", "en", "vers", "qui", "que", "quoi",
        "dont", "où", "comment", "pourquoi", "quand", "si", "oui", "non"
    ])
    
    # Extract words, remove common words, and count frequencies
    words = re.findall(r'\b[a-zA-ZÀ-ÿ]{3,}\b', text.lower())
    word_counts = {}
    
    for word in words:
        if word not in common_words:
            word_counts[word] = word_counts.get(word, 0) + 1
    
    # Sort by frequency and return top n
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    return [word for word, count in sorted_words[:n]]

def enhance_prompt_based_on_query(query, context):
    """Enhance the prompt based on the query type."""
    context_text = "\n\n".join([doc.page_content for doc, _ in context])
    
    # Default prompt
    system_prompt = """Tu es un assistant spécialisé qui répond aux questions basées uniquement sur les informations fournies dans le contexte. Réponds de manière précise et informative en utilisant uniquement les informations du contexte. Si l'information ne se trouve pas dans le contexte, dis simplement que tu ne disposes pas de cette information."""
    
    # Detect specific query types
    if re.search(r'mot[s-]cl[ée]s|keywords', query.lower()):
        # For keyword extraction
        system_prompt += """
Pour les demandes d'extraction de mots-clés, identifie les termes techniques, scientifiques ou thématiques les plus importants qui apparaissent dans le texte. Présente une liste numérotée des mots-clés les plus pertinents, en excluant les mots courants. Indique pour chaque mot-clé le nombre d'occurrences si possible."""
    
    elif re.search(r'scientifique|auteur|chercheur|personne', query.lower()):
        # For people/scientist extraction
        system_prompt += """
Pour les demandes concernant des personnes, scientifiques ou auteurs, recherche attentivement les noms propres qui apparaissent dans le contexte. Présente une liste complète des noms de personnes mentionnées, avec leurs affiliations ou contributions si disponibles."""
    
    elif re.search(r'concept|sujet|th[èe]me|mentionn[ée]', query.lower()):
        # For concept/topic extraction
        system_prompt += """
Pour les demandes concernant des concepts ou thèmes, identifie les principales idées, théories ou sujets abordés dans les documents. Regroupe les concepts similaires et présente-les par ordre d'importance ou de fréquence."""
    
    elif re.search(r'date|ann[ée]e|p[ée]riode|entre|19\d{2}|20\d{2}', query.lower()):
        # For date-specific queries
        system_prompt += """
Pour les questions liées à des périodes spécifiques, concentre-toi sur les informations datées dans le contexte. Vérifie attentivement les dates mentionnées dans les documents et assure-toi que ta réponse se limite bien à la période demandée."""
        
        # Add information about available date ranges
        years = []
        for doc, _ in context:
            if "year" in doc.metadata and doc.metadata["year"]:
                years.append(doc.metadata["year"])
        
        if years:
            min_year, max_year = min(years), max(years)
            system_prompt += f"""
Note: Les documents fournis couvrent la période de {min_year} à {max_year}."""
    
    # Assemble the final prompt
    prompt = f"""<|system|>
{system_prompt}
</|system|>

<|user|>
Contexte:
{context_text}

Question: {query}
</|user|>

<|assistant|>"""

    return prompt

def generate_with_llama(api_key, query, context):
    """Generate a response using Meta-Llama-3-8B-Instruct with enhanced prompting."""
    # Create an enhanced prompt based on query type
    prompt = enhance_prompt_based_on_query(query, context)

    try:
        API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
        headers = {"Authorization": f"Bearer {api_key}"}
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.95,
                "return_full_text": False
            }
        }
        
        # Send the request
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            
            # Extract generated text
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "")
            elif isinstance(result, dict) and "generated_text" in result:
                return result["generated_text"]
            else:
                # Fallback to basic information extraction
                return generate_fallback_response(query, context)
        else:
            # API error, use fallback
            return generate_fallback_response(query, context)
    
    except Exception as e:
        return generate_fallback_response(query, context)

def generate_fallback_response(query, context):
    """Generate a basic response from context when API fails."""
    # Check what type of information is being requested
    query_lower = query.lower()
    
    # For keyword extraction
    if re.search(r'mot[s-]cl[ée]s|keywords', query_lower):
        all_text = " ".join([doc.page_content for doc, _ in context])
        keywords = extract_keywords(all_text, 10)
        response = "Voici les mots-clés extraits des documents :\n\n"
        for i, keyword in enumerate(keywords, 1):
            response += f"{i}. {keyword}\n"
        return response
    
    # For scientists/people
    elif re.search(r'scientifique|auteur|chercheur|personne', query_lower):
        all_persons = []
        for doc, _ in context:
            if "persons" in doc.metadata and doc.metadata["persons"]:
                all_persons.extend(doc.metadata["persons"])
        
        if all_persons:
            unique_persons = list(set(all_persons))
            response = "Voici les personnes mentionnées dans les documents :\n\n"
            for i, person in enumerate(unique_persons, 1):
                response += f"{i}. {person}\n"
            return response
        else:
            return "Je n'ai pas trouvé de noms de personnes dans les documents fournis."
    
    # For dates/years
    elif re.search(r'date|ann[ée]e|p[ée]riode|entre|19\d{2}|20\d{2}', query_lower):
        year_ranges = extract_year_ranges(query)
        if year_ranges:
            periods = [f"{start}-{end}" for start, end in year_ranges]
            period_text = ", ".join(periods)
            response = f"Informations extraites pour la période {period_text} :\n\n"
            
            # Extract content from documents in this period
            for doc, _ in context:
                doc_year = doc.metadata.get("year")
                doc_title = doc.metadata.get("title", "Document sans titre")
                response += f"- {doc_title} ({doc_year}) : "
                
                # Extract a short summary (first 200 chars)
                content = doc.page_content.replace(f"Document: {doc_title}", "")
                content = content.replace(f"Date: {doc.metadata.get('date', '')}", "").strip()
                summary = content[:200] + "..." if len(content) > 200 else content
                response += f"{summary}\n\n"
                
            return response
    
    # Default: simple summary of documents
    response = "Voici les informations extraites des documents :\n\n"
    for i, (doc, _) in enumerate(context, 1):
        title = doc.metadata.get("title", "Document sans titre")
        date = doc.metadata.get("date", "Date inconnue")
        response += f"{i}. {title} ({date})\n"
        
        # Extract first paragraph as summary
        content = doc.page_content.replace(f"Document: {title}", "")
        content = content.replace(f"Date: {date}", "").strip()
        paragraphs = content.split("\n\n")
        if paragraphs:
            summary = paragraphs[0][:200] + "..." if len(paragraphs[0]) > 200 else paragraphs[0]
            response += f"   {summary}\n\n"
    
    return response

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
                    # Search for similar documents using embeddings
                    results = search_documents(prompt, hf_api_key, k=k_value)
                
                if not results:
                    message_placeholder.warning("Aucun document pertinent trouvé pour répondre à votre question.")
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": "Aucun document pertinent trouvé pour répondre à votre question."
                    })
                else:
                    with st.spinner("Génération de la réponse avec Llama 3..."):
                        # Generate answer using Meta-Llama-3
                        answer = generate_with_llama(hf_api_key, prompt, results)
                    
                    # Display the answer
                    message_placeholder.markdown(answer)
                    
                    # Display source documents
                    st.markdown("---")
                    st.markdown("**Sources:**")
                    for i, (doc, score) in enumerate(results):
                        year_info = f" ({doc.metadata.get('year')})" if doc.metadata.get('year') else ""
                        with st.expander(f"Source {i+1}: {doc.metadata.get('title', 'Unknown')}{year_info} (Similarité: {score:.4f})"):
                            st.markdown(f"**Date:** {doc.metadata.get('date', 'Unknown')}")
                            st.markdown(f"**Fichier:** {doc.metadata.get('source', 'Unknown')}")
                            
                            # Display persons if available
                            if doc.metadata.get('persons'):
                                st.markdown(f"**Personnes mentionnées:** {', '.join(doc.metadata.get('persons'))}")
                            
                            st.markdown("**Extrait:**")
                            st.markdown(doc.page_content)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                error_msg = f"Erreur lors du traitement de la requête: {str(e)}"
                message_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
else:
    st.info("Veuillez initialiser le système en utilisant le bouton dans la barre latérale.")

st.markdown("---")
st.markdown("Démonstration RAG - Développé avec Streamlit")