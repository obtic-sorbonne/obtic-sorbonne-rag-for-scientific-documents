import requests
import os
import streamlit as st
import re

def get_ollama_host():
    """Get Ollama host from environment or default to localhost."""
    return os.environ.get('OLLAMA_HOST', 'http://localhost:11434')

def check_ollama_availability():
    """Check if Ollama is running locally or in container."""
    try:
        ollama_host = get_ollama_host()
        response = requests.get(f"{ollama_host}/api/tags", timeout=10)
        return response.status_code == 200
    except Exception as e:
        print(f"Ollama availability check failed: {e}")
        return False

def get_ollama_models():
    """Get list of available Ollama models."""
    try:
        ollama_host = get_ollama_host()
        response = requests.get(f"{ollama_host}/api/tags", timeout=10)
        if response.status_code == 200:
            data = response.json()
            models = []
            for model in data.get('models', []):
                # Extract model name without tag if it's 'latest'
                name = model['name']
                if name.endswith(':latest'):
                    name = name[:-7]  # Remove ':latest'
                models.append(name)
            return sorted(set(models))  # Remove duplicates and sort
        return []
    except Exception as e:
        print(f"Error getting Ollama models: {e}")
        return []

def query_ollama(model_name, prompt, temperature=0.7, max_tokens=2000):
    """Query Ollama model directly."""
    try:
        ollama_host = get_ollama_host()
        
        # Ensure model name has proper format
        if ':' not in model_name:
            model_name = f"{model_name}:latest"
        
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": 0.9,
                "repeat_penalty": 1.1
            }
        }
        
        print(f"Querying Ollama at {ollama_host} with model {model_name}")
        print(f"Prompt length: {len(prompt)} characters")
        
        response = requests.post(
            f"{ollama_host}/api/generate",
            json=payload,
            timeout=300  # 5 minutes timeout for larger models
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get('response', '')
            
            # Debug information
            print(f"Ollama response received: {len(answer)} characters")
            print(f"Model: {result.get('model', 'unknown')}")
            print(f"Done: {result.get('done', False)}")
            
            return answer.strip() if answer else None
        else:
            print(f"Ollama API error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print("Ollama request timed out")
        st.error("La requête vers Ollama a expiré. Le modèle pourrait être trop lent.")
        return None
    except Exception as e:
        print(f"Ollama query error: {e}")
        st.error(f"Erreur lors de la communication avec Ollama: {str(e)}")
        return None

def pull_ollama_model(model_name):
    """Pull a model from Ollama repository."""
    try:
        ollama_host = get_ollama_host()
        
        payload = {
            "name": model_name,
            "stream": False
        }
        
        response = requests.post(
            f"{ollama_host}/api/pull",
            json=payload,
            timeout=1800  # 30 minutes for model download
        )
        
        return response.status_code == 200
    except Exception as e:
        print(f"Error pulling Ollama model: {e}")
        return False

def get_model_info(model_name):
    """Get information about a specific model."""
    try:
        ollama_host = get_ollama_host()
        
        # Ensure model name has proper format
        if ':' not in model_name:
            model_name = f"{model_name}:latest"
            
        payload = {"name": model_name}
        
        response = requests.post(
            f"{ollama_host}/api/show",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"Error getting model info: {e}")
        return None
    
def extract_reasoning_and_response(text, model_name):
    """Extract reasoning and final response from DeepSeek-R1 output."""
    
    # Only process DeepSeek-R1 models
    if 'deepseek-r1' not in model_name.lower():
        return None, text
    
    # Pattern to match <think>...</think> blocks
    think_pattern = r'<think>(.*?)</think>'
    
    # Extract all thinking blocks
    think_matches = re.findall(think_pattern, text, re.DOTALL)
    
    # Remove thinking blocks from the main text
    cleaned_text = re.sub(think_pattern, '', text, flags=re.DOTALL)
    
    # Clean up extra whitespace
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text.strip())
    
    # Combine all thinking blocks
    reasoning = '\n\n'.join(think_matches) if think_matches else None
    
    return reasoning, cleaned_text

def display_deepseek_response(answer, model_name, response_container):
    """Display DeepSeek response with optional reasoning toggle."""
    
    reasoning, clean_answer = extract_reasoning_and_response(answer, model_name)
    
    # Display the clean answer
    response_container.markdown(clean_answer)
    
    # Add reasoning toggle if reasoning was found
    if reasoning and reasoning.strip():
        with response_container.expander("🧠 Voir le raisonnement (DeepSeek-R1)", expanded=False):
            st.markdown("**Processus de raisonnement du modèle :**")
            st.markdown("```")
            st.markdown(reasoning.strip())
            st.markdown("```")
    
    return clean_answer