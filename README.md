# RAG PDF - PDF Question Answering System

This application implements a Retrieval-Augmented Generation (RAG) system that allows users to ask questions about the content of PDF documents. It uses a combination of vector embeddings, Pinecone for vector storage, and Ollama for text generation.

## How It Works

### 1. PDF Processing
- The application reads PDF files using PyPDF2
- Extracts text content from each page
- Splits the text into manageable chunks using LangChain's RecursiveCharacterTextSplitter
  - Default chunk size: 1000 characters
  - Default overlap: 200 characters

### 2. Vector Embeddings
- Uses Ollama's embedding model (mxbai-embed-large) to convert text chunks into vector embeddings
- These embeddings capture the semantic meaning of the text chunks

### 3. Vector Storage
- Stores the embeddings in Pinecone, a vector database
- Each chunk is stored with:
  - A unique ID
  - The vector embedding
  - The original text as metadata

### 4. Question Answering
When a user asks a question:
1. The question is converted into a vector embedding
2. The system searches Pinecone for the most similar text chunks
3. The retrieved chunks are used as context
4. Ollama's Gemma model generates an answer based on the context and question

## Setup Requirements

1. **Environment Variables**
   - `PINECONE_API_KEY`: Your Pinecone API key

2. **Dependencies**
   - PyPDF2
   - langchain
   - openai
   - pinecone-client
   - ollama
   - requests

3. **Ollama Setup**
   - Must have Ollama running locally (default port: 11434)
   - Required models:
     - mxbai-embed-large (for embeddings)
     - gemma3:4b (for text generation)

4. **pinecone Setup**
    -create an index i.e rag-python-pdf and make sure to add your model embedding Dimensions for this setup project its 1024

## Usage

1. Place your PDF file in the project directory
2. Update the `pdf_path` variable in `main.py` to point to your PDF
3. Run the script:
   ```bash
   python main.py
   ```
4. Enter your questions about the PDF content
5. Type 'exit', 'quit', or 'bye' to end the session

## Key Components

- `extract_text_from_pdf()`: Extracts text from PDF files
- `chunk_text()`: Splits text into manageable chunks
- `generate_embeddings()`: Creates vector embeddings from text
- `upsert_to_pinecone()`: Stores vectors in Pinecone
- `query_rag()`: Handles the question-answering process

## Notes

- The system uses a local Ollama instance for both embeddings and text generation
- Pinecone is used for vector storage and similarity search
- The chunking strategy can be adjusted by modifying the `chunk_size` and `chunk_overlap` parameters
- The number of retrieved chunks for answering questions can be adjusted via the `top_k` parameter in the query function
