import os
import pickle
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings

def regenerate_cpu_vectorstore(input_path, output_path):
    """
    Loads a vectorstore that was saved on CUDA and resaves it for CPU-only usage.
    
    Args:
        input_path (str): Path to the original vectorstore
        output_path (str): Path to save the CPU-friendly vectorstore
    """
    print(f"Loading vectorstore from {input_path}...")
    
    try:
        # Custom unpickler to handle CUDA tensors
        class CPUUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'torch.storage' and name == '_load_from_bytes':
                    return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
                else:
                    return super().find_class(module, name)
        
        # First try with regular loading + map_location
        try:
            with open(input_path, "rb") as f:
                vectorstore = pickle.load(f, map_location=torch.device('cpu'))
            print("Loaded using standard pickle with CPU mapping")
        except:
            # If that fails, try with the custom unpickler
            import io
            with open(input_path, 'rb') as f:
                vectorstore = CPUUnpickler(f).load()
            print("Loaded using custom CPU unpickler")
        
        # Save the vectorstore for CPU usage
        print(f"Saving CPU-friendly vectorstore to {output_path}...")
        with open(output_path, "wb") as f:
            pickle.dump(vectorstore, f)
        
        print("Success! The vectorstore has been converted for CPU usage.")
        return True
    
    except Exception as e:
        print(f"Error: {str(e)}")
        
        # If both loading methods fail, try to create a new vectorstore from scratch
        print("Attempting to rebuild the vectorstore from documents if available...")
        
        # Check if there's a metadata file with original document paths
        metadata_path = os.path.splitext(input_path)[0] + "_metadata.txt"
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    metadata = f.read()
                
                # Look for document paths in the metadata
                import re
                doc_paths = re.findall(r"Processed files:\n(.*?)(?:\n\n|\Z)", metadata, re.DOTALL)
                
                if doc_paths:
                    doc_list = doc_paths[0].split('\n')
                    doc_list = [d.strip('- ') for d in doc_list if d.strip()]
                    
                    if doc_list:
                        print(f"Found {len(doc_list)} document paths. Attempting to recreate vectorstore...")
                        
                        # Extract embedding model info
                        embedding_model = re.search(r"Embedding model: (.*?)\n", metadata)
                        if embedding_model:
                            model_name = embedding_model.group(1)
                        else:
                            model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                        
                        # Extract chunk size and overlap
                        chunk_size = re.search(r"Chunk size: (\d+)", metadata)
                        chunk_size = int(chunk_size.group(1)) if chunk_size else 1000
                        
                        chunk_overlap = re.search(r"Chunk overlap: (\d+)", metadata)
                        chunk_overlap = int(chunk_overlap.group(1)) if chunk_overlap else 0
                        
                        # Recreate vectorstore from documents
                        from langchain_community.document_loaders import TextLoader, DirectoryLoader
                        from langchain.text_splitter import RecursiveCharacterTextSplitter
                        from langchain_community.vectorstores import FAISS, Chroma
                        
                        # Use Chroma instead of FAISS to avoid CUDA issues
                        embedding_func = HuggingFaceEmbeddings(model_name=model_name)
                        
                        # Try to load documents
                        documents = []
                        for doc_path in doc_list:
                            if os.path.exists(doc_path):
                                try:
                                    loader = TextLoader(doc_path)
                                    documents.extend(loader.load())
                                except Exception as doc_err:
                                    print(f"Error loading {doc_path}: {str(doc_err)}")
                        
                        if documents:
                            # Split documents
                            text_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap
                            )
                            chunks = text_splitter.split_documents(documents)
                            
                            # Create vectorstore
                            vectorstore = Chroma.from_documents(chunks, embedding_func)
                            
                            # Save the vectorstore
                            with open(output_path, "wb") as f:
                                pickle.dump(vectorstore, f)
                            
                            print(f"Successfully created new vectorstore from {len(documents)} documents and {len(chunks)} chunks!")
                            return True
            except Exception as rebuild_err:
                print(f"Error rebuilding vectorstore: {str(rebuild_err)}")
        
        return False

if __name__ == "__main__":
    # Get paths from user
    input_path = input("Enter path to original vectorstore: ") or "vectorstore.pkl"
    output_path = input("Enter path for CPU-friendly vectorstore: ") or "vectorstore_cpu.pkl"
    
    # Convert the vectorstore
    success = regenerate_cpu_vectorstore(input_path, output_path)
    
    if success:
        print("\nUPDATE YOUR APP:")
        print(f"Change VECTORSTORE_PATH in your app.py to '{output_path}'")
    else:
        print("\nConversion failed. Try these alternatives:")
        print("1. Recreate the vectorstore locally on a CPU machine")
        print("2. Use a different vector database like Chroma in your app")
        print("3. Reprocess your original documents directly in the app")