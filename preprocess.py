import os
import pickle
import glob
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.vectorstores import FAISS
import argparse

def process_documents(folder_path, output_path, chunk_size=1000, chunk_overlap=100, embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    """
    Process documents in a folder, create embeddings, and save the vector store
    """
    print(f"Processing documents from {folder_path}")
    print(f"Using embedding model: {embedding_model}")
    print(f"Chunk size: {chunk_size}, Chunk overlap: {chunk_overlap}")
    
    try:
        # Load documents from the provided folder
        loader = DirectoryLoader(folder_path, glob="**/*.*", loader_cls=TextLoader)
        documents = loader.load()
        
        if not documents:
            print("No valid documents found.")
            return False
            
        print(f"Loaded {len(documents)} documents")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")
        
        # Load embeddings model
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        print("Embedding model loaded")
        
        # Create vector store
        vectorstore = FAISS.from_documents(chunks, embeddings)
        print("Vector store created")
        
        # Save vector store to disk
        with open(output_path, "wb") as f:
            pickle.dump(vectorstore, f)
        print(f"Vector store saved to {output_path}")
        
        # Create a metadata file
        metadata_path = os.path.splitext(output_path)[0] + "_metadata.txt"
        with open(metadata_path, "w") as f:
            f.write(f"Number of documents: {len(documents)}\n")
            f.write(f"Number of chunks: {len(chunks)}\n")
            f.write(f"Embedding model: {embedding_model}\n")
            f.write(f"Chunk size: {chunk_size}\n")
            f.write(f"Chunk overlap: {chunk_overlap}\n")
            f.write(f"Created from folder: {os.path.abspath(folder_path)}\n")
            
            # List all processed files
            f.write("\nProcessed files:\n")
            for doc in documents:
                f.write(f"- {doc.metadata.get('source', 'Unknown')}\n")
                
        return True
        
    except Exception as e:
        print(f"Error processing documents: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process documents and create embeddings')
    parser.add_argument('--folder', '-f', required=True, help='Path to folder containing documents')
    parser.add_argument('--output', '-o', default='vectorstore.pkl', help='Output path for the vector store')
    parser.add_argument('--chunk-size', '-cs', type=int, default=1000, help='Chunk size for splitting documents')
    parser.add_argument('--chunk-overlap', '-co', type=int, default=100, help='Chunk overlap for splitting documents')
    parser.add_argument('--embedding-model', '-em', default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 
                      help='Hugging Face embedding model to use')
    
    args = parser.parse_args()
    
    success = process_documents(
        args.folder, 
        args.output, 
        args.chunk_size, 
        args.chunk_overlap,
        args.embedding_model
    )
    
    if success:
        print("Document processing completed successfully!")
    else:
        print("Document processing failed.")
