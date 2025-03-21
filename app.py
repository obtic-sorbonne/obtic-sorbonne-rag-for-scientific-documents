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
from langchain_openai import ChatOpenAI

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

# Fixed system prompt - not modifiable by users
SYSTEM_PROMPT = """Tu es un assistant spécialisé pour l'analyse de documents scientifiques historiques en français.
CONTEXTE:
- Tu travailles avec un corpus de documents XML-TEI qui contiennent des informations scientifiques.
- Tu disposes d'une base de connaissances vectorielle qui permet de retrouver les passages pertinents.
- Tu reçois une question et plusieurs documents contenant potentiellement les informations pour y répondre.
- Certains documents sont OCRisés, donc contiennent du bruit. Il faut payer une attention particulière aux chiffres."""

# Default query prompt - can be modified by users
DEFAULT_QUERY_PROMPT = """Voici la question de l'utilisateur: 
{query}

Instructions : 
1. Recherche ATTENTIVEMENT toutes les informations pertinentes qui porte sur la question de l'utilisateur.
2. Si la question porte sur des chiffres ou des quantités, cherche explicitement ces données et présente-les de manière structurée.
3. Vérifie chaque document fourni avant de conclure à l'absence d'information.
4. Utilise le formatage markdown pour mettre en évidence les points importants.
5. Réponds en français, dans un style professionnel et accessible."""

def extract_year(date_str):
    """Extract year from a date string."""
    year_match = re.search(r'\b(19\d{2}|20\d{2})\b', date_str)
    if year_match:
