# NCreator - AI-Powered Content Generation System

NCreator is a RAG (Retrieval-Augmented Generation) system that combines vector search capabilities with LLM-based content generation. The system uses Weaviate for vector storage and search, Ollama for LLM inference, and includes a proxy service for orchestrating the components.

## System Architecture

The system consists of three main components:

1. **Weaviate Vector Database**
   - Handles vector storage and similarity search
   - Runs on port 8080
   - Supports both HTTP and gRPC protocols

2. **Ollama LLM Service**
   - Provides LLM inference capabilities
   - Runs on port 11434
   - GPU-accelerated for better performance

3. **Proxy Service**
   - Orchestrates communication between components
   - Handles API requests and responses
   - Runs on port 8000

## Project Structure

```
├── client/                 # Client-side components
│   ├── app/               # Main application code
│   ├── scripts/           # Processing scripts
│   └── texts/             # Text data management
├── server/                # Server-side components
│   ├── models/           # Model storage
│   └── weaviate-data/    # Vector database storage
```

## Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with appropriate drivers (for GPU acceleration)
- Python 3.9+

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/wayming/ncreator.git
   cd ncreator
   ```

2. Start the server components:
   ```bash
   cd server
   docker-compose up -d
   ```

3. Import data (if needed):
   ```bash
   cd client
   docker-compose run --rm importer
   ```

## Configuration

### Environment Variables

The proxy service can be configured using the following environment variables:

- `WEAVIATE_HTTP_HOST`: Weaviate HTTP endpoint host
- `WEAVIATE_HTTP_PORT`: Weaviate HTTP port (default: 8080)
- `WEAVIATE_GRPC_HOST`: Weaviate gRPC endpoint host
- `WEAVIATE_GRPC_PORT`: Weaviate gRPC port (default: 50051)
- `OLLAMA_BASE_URL`: Ollama service URL

## Development

The project uses a microservices architecture with Docker containers. Each service can be developed and tested independently.

### Adding New Models

Place new models in the `/models` directory and update the Ollama configuration as needed.

### Data Processing

The `client/scripts` directory contains tools for preprocessing and managing content before import into the vector database.

## License

This project is licensed under the MIT License - see the LICENSE file for details.