#!/bin/bash

# RAG System Startup Script
# This script will start the entire RAG system with Ollama and DeepSeek-R1

set -e

echo "🚀 Starting RAG System with Ollama and DeepSeek-R1..."
echo "=================================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data embeddings static tmp vector_store

# Stop any existing containers
echo "🛑 Stopping any existing containers..."
docker-compose down --remove-orphans 2>/dev/null || true

# Remove any existing volumes if requested
read -p "🗑️  Do you want to remove existing Ollama data (models will be re-downloaded)? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🗑️  Removing existing Ollama data..."
    docker volume rm $(docker volume ls -q | grep ollama) 2>/dev/null || true
fi

# Start the services
echo "🐳 Starting Docker services..."
echo "This will:"
echo "  1. Start Ollama service"
echo "  2. Download DeepSeek-R1 model (may take several minutes)"
echo "  3. Start the RAG application"
echo ""

# Use docker-compose or docker compose based on availability
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
else
    COMPOSE_CMD="docker compose"
fi

echo "📦 Building and starting services..."
$COMPOSE_CMD up --build -d ollama

echo "⏳ Waiting for Ollama to be ready..."
timeout=60
counter=0
while ! curl -f http://localhost:11434/api/tags &>/dev/null; do
    if [ $counter -eq $timeout ]; then
        echo "❌ Timeout waiting for Ollama to start"
        $COMPOSE_CMD logs ollama
        exit 1
    fi
    echo "   Waiting... ($counter/$timeout seconds)"
    sleep 2
    counter=$((counter + 2))
done

echo "✅ Ollama is ready!"

echo "📥 Downloading DeepSeek-R1 model..."
echo "   This may take 5-15 minutes depending on your internet connection..."
$COMPOSE_CMD up ollama-setup

echo "🏗️  Building and starting RAG application..."
$COMPOSE_CMD up --build -d rag-app

echo ""
echo "🎉 RAG System is starting up!"
echo "=================================================="
echo ""
echo "📊 Service Status:"
$COMPOSE_CMD ps

echo ""
echo "🌐 Access your application at:"
echo "   http://localhost:8501"
echo ""
echo "🔧 Useful commands:"
echo "   View logs:           $COMPOSE_CMD logs -f"
echo "   Stop services:       $COMPOSE_CMD down"
echo "   Restart:             $COMPOSE_CMD restart"
echo "   Update models:       $COMPOSE_CMD exec ollama ollama pull deepseek-r1:1.5b"
echo ""

# Wait for the application to be ready
echo "⏳ Waiting for RAG application to be ready..."
timeout=120
counter=0
while ! curl -f http://localhost:8501/_stcore/health &>/dev/null; do
    if [ $counter -eq $timeout ]; then
        echo "⚠️  Application may still be starting. Check logs if needed:"
        echo "   $COMPOSE_CMD logs rag-app"
        break
    fi
    if [ $((counter % 10)) -eq 0 ]; then
        echo "   Waiting... ($counter/$timeout seconds)"
    fi
    sleep 2
    counter=$((counter + 2))
done

if curl -f http://localhost:8501/_stcore/health &>/dev/null; then
    echo "✅ RAG Application is ready!"
    echo ""
    echo "🎯 Open your browser to: http://localhost:8501"
    echo "💡 The app includes DeepSeek-R1 model for local inference"
else
    echo "⚠️  Application might still be initializing..."
    echo "   Check the logs: $COMPOSE_CMD logs rag-app"
    echo "   Or try accessing: http://localhost:8501"
fi