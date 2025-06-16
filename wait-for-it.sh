#!/bin/sh

# Wait for Ollama server to be up
echo "Waiting for Ollama server to be ready..."
until curl -s http://ollama:11434/api/health > /dev/null; do
    echo "Ollama server not ready yet... waiting"
    sleep 2
done

# Wait for model to be available
echo "Checking if qwen3:8b model is available..."
until curl -s http://ollama:11434/api/tags | grep -q "qwen3:8b"; do
    echo "Model not ready yet... waiting"
    sleep 5
done

# Test if we can actually use the model
echo "Testing model availability..."
until curl -s -X POST http://ollama:11434/api/generate -d '{"model": "qwen3:8b", "prompt": "test"}' > /dev/null; do
    echo "Model not responding yet... waiting"
    sleep 5
done

echo "✅ Ollama server and model are ready!"
exec "$@" 