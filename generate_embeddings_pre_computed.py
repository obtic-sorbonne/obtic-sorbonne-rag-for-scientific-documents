#!/usr/bin/env python3
"""
Script pour générer des embeddings pré-calculés avec intfloat/multilingual-e5-large-instruct
Utilise la même logique que l'application principale mais optimisé pour la génération batch.
"""

import os
import re
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
import argparse
import sys

# Imports pour les embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Configuration des chemins
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
EMBEDDINGS_DIR = SCRIPT_DIR / "embeddings"

# Créer les dossiers si nécessaire
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

# Configuration des modèles
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"
ALTERNATIVE_MODELS = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "intfloat/multilingual-e5-base",
    "intfloat/multilingual-e5-small"
]

# Espaces de noms XML-TEI
NAMESPACES = {
    'tei': 'http://www.tei-c.org/ns/1.0'
}

def extract_year(date_str):
    """Extract year from a date string."""
    if not date_str:
        return None
    year_match = re.search(r'\b(19\d{2}|20\d{2})\b', date_str)
    if year_match:
        return int(year_match.group(1))
    return None

def parse_xmltei_document(file_path):
    """Parse an XML-TEI document and extract text content with metadata."""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Extract document metadata
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
        
        # Extract person names
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
        print(f"❌ Error parsing XML file {file_path}: {str(e)}")
        return None

def load_documents_from_directory(data_dir):
    """Load all XML-TEI documents from the data directory."""
    documents = []
    document_dates = {}
    
    # Chercher les fichiers XML dans le dossier data
    xml_files = []
    if data_dir.exists():
        for file_path in data_dir.rglob("*.xml"):
            xml_files.append(file_path)
        for file_path in data_dir.rglob("*.xmltei"):
            xml_files.append(file_path)
    
    if not xml_files:
        print(f"❌ No XML files found in {data_dir}")
        return documents, document_dates
    
    print(f"📁 Found {len(xml_files)} XML files")
    
    # Process files with progress
    for i, file_path in enumerate(xml_files, 1):
        print(f"📄 Processing {i}/{len(xml_files)}: {file_path.name}")
        
        doc_data = parse_xmltei_document(file_path)
        
        if doc_data:
            doc = Document(
                page_content=doc_data["text"],
                metadata={
                    "source": str(file_path),
                    "title": doc_data["title"],
                    "date": doc_data["date"],
                    "year": doc_data["year"],
                    "persons": doc_data["persons"]
                }
            )
            documents.append(doc)
            
            if doc_data["year"]:
                document_dates[str(file_path)] = doc_data["year"]
        else:
            print(f"⚠️  Failed to parse {file_path.name}")
    
    print(f"✅ Successfully processed {len(documents)} documents")
    return documents, document_dates

def format_query_for_e5_instruct(query: str) -> str:
    """Format query for E5-large-instruct model (same as in app)."""
    task = "Given scientific documents about parasitology in French, retrieve relevant passages that answer the query"
    return f"Instruct: {task}\nQuery: {query}"

def split_documents(documents, chunk_size=2500, chunk_overlap=800):
    """Split documents into chunks."""
    print(f"🔪 Splitting documents into chunks (size={chunk_size}, overlap={chunk_overlap})")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,  
        separators=["\n\n", "\n", ". ", ".", " "]
    )
    
    texts = text_splitter.split_documents(documents)
    print(f"✅ Created {len(texts)} chunks from {len(documents)} documents")
    
    return texts

def create_embeddings(texts, model_name=DEFAULT_EMBEDDING_MODEL, device="cpu"):
    """Create embeddings using the specified model."""
    print(f"🧠 Creating embeddings with model: {model_name}")
    print(f"💻 Using device: {device}")
    
    try:
        # Configure embeddings
        model_kwargs = {"device": device}
        
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs
        )
        
        print(f"📊 Processing {len(texts)} text chunks...")
        
        # Create FAISS vector store
        try:
            vectordb = FAISS.from_documents(texts, embeddings)
            print("✅ FAISS index created successfully")
            
        except Exception as e:
            print(f"⚠️  Error with direct creation, trying batch approach: {str(e)}")
            
            # Batch processing for large datasets
            batch_size = 50
            batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
            
            print(f"🔄 Processing in {len(batches)} batches of {batch_size}")
            
            # Create initial vectordb with first batch
            vectordb = FAISS.from_documents(batches[0], embeddings)
            print(f"✅ Batch 1/{len(batches)} processed")
            
            # Add remaining batches
            for i, batch in enumerate(batches[1:], 2):
                print(f"🔄 Processing batch {i}/{len(batches)}...")
                vectordb.add_documents(batch)
            
            print("✅ All batches processed successfully")
        
        return vectordb, embeddings
        
    except Exception as e:
        print(f"❌ Error creating embeddings: {str(e)}")
        return None, None

def save_embeddings(vectordb, embeddings, texts, documents, document_dates, model_name):
    """Save the embeddings and metadata to disk."""
    print(f"💾 Saving embeddings to {EMBEDDINGS_DIR}")
    
    # Save FAISS index
    faiss_path = EMBEDDINGS_DIR / "faiss_index"
    vectordb.save_local(str(faiss_path))
    print(f"✅ FAISS index saved to {faiss_path}")
    
    # Save metadata
    metadata = {
        "document_count": len(documents),
        "chunk_count": len(texts),
        "model_name": model_name,
        "instruction_format": True,  # E5-instruct uses instruction format
        "document_dates": document_dates,
        "faiss_version": getattr(FAISS, '__version__', 'unknown'),
        "created_at": datetime.now().isoformat(),
        "chunk_size": 2500,
        "chunk_overlap": 800
    }
    
    metadata_path = EMBEDDINGS_DIR / "document_metadata.pkl"
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    print(f"✅ Metadata saved to {metadata_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("📊 GENERATION SUMMARY")
    print("="*50)
    print(f"Model: {model_name}")
    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(texts)}")
    print(f"Date range: {min(document_dates.values()) if document_dates else 'N/A'} - {max(document_dates.values()) if document_dates else 'N/A'}")
    print(f"Output: {EMBEDDINGS_DIR}")
    print("="*50)

def main():
    parser = argparse.ArgumentParser(description="Generate pre-computed embeddings for RAG system")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, 
                        help=f"Directory containing XML files (default: {DATA_DIR})")
    parser.add_argument("--output-dir", type=Path, default=EMBEDDINGS_DIR,
                        help=f"Output directory for embeddings (default: {EMBEDDINGS_DIR})")
    parser.add_argument("--model", type=str, default=DEFAULT_EMBEDDING_MODEL,
                        help=f"Embedding model to use (default: {DEFAULT_EMBEDDING_MODEL})")
    parser.add_argument("--chunk-size", type=int, default=2500,
                        help="Chunk size for text splitting (default: 2500)")
    parser.add_argument("--chunk-overlap", type=int, default=800,
                        help="Chunk overlap for text splitting (default: 800)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device to use for embeddings (default: cpu)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing embeddings")
    
    args = parser.parse_args()
    
    # Check if embeddings already exist
    if not args.force and (args.output_dir / "faiss_index").exists():
        response = input("⚠️  Embeddings already exist. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("❌ Aborted")
            return
    
    print("🚀 Starting embedding generation...")
    print(f"📁 Data directory: {args.data_dir}")
    print(f"💾 Output directory: {args.output_dir}")
    print(f"🧠 Model: {args.model}")
    print(f"🔪 Chunk size: {args.chunk_size} (overlap: {args.chunk_overlap})")
    print(f"💻 Device: {args.device}")
    print()
    
    # Update global paths
    global EMBEDDINGS_DIR
    EMBEDDINGS_DIR = args.output_dir
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load documents
    print("📚 Loading documents...")
    documents, document_dates = load_documents_from_directory(args.data_dir)
    
    if not documents:
        print("❌ No documents found. Exiting.")
        return
    
    # Split documents
    texts = split_documents(documents, args.chunk_size, args.chunk_overlap)
    
    # Create embeddings
    vectordb, embeddings = create_embeddings(texts, args.model, args.device)
    
    if vectordb is None:
        print("❌ Failed to create embeddings. Exiting.")
        return
    
    # Save embeddings
    save_embeddings(vectordb, embeddings, texts, documents, document_dates, args.model)
    
    print("\n🎉 Embedding generation completed successfully!")

if __name__ == "__main__":
    main()