# PDF Question-Answering System

A web application that allows users to upload PDF documents and ask questions about their content. The system uses natural language processing and machine learning to extract relevant answers from the documents.

## Features

- PDF document upload and processing
- Question-answering based on document content
- Semantic search to find relevant sections in large documents
- Confidence scoring for answers
- Document metadata extraction (title, author, page count, etc.)
- Chunking of large documents for efficient processing
- Debug mode for inspecting document processing

## How It Works

1. The system extracts and processes text from uploaded PDF documents
2. When a user asks a question, the system:
   - Splits the document into manageable chunks
   - Finds the most relevant chunks using semantic similarity
   - Uses a question-answering model to extract the best answer
   - Returns the answer with a confidence score

## Technologies Used

- **Backend**: Flask (Python)
- **PDF Processing**: PyMuPDF (fitz)
- **Machine Learning**:
  - Transformer models for question answering (RoBERTa)
  - Sentence embeddings for semantic search
- **Frontend**: HTML, JavaScript

## Prerequisites

- Python 3.6+
- PyTorch
- Flask
- Transformers library
- Sentence-Transformers
- PyMuPDF
- Werkzeug

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/pdf-qa-system.git
   cd pdf-qa-system
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create required directories (if they don't exist):
   ```
   mkdir -p uploads templates static
   ```

## Usage

1. Start the application:
   ```
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:3000
   ```

3. Upload a PDF document using the web interface
4. Ask questions about the document content
5. View the system's answers and confidence scores

## API Endpoints

- `GET /`: Main web interface
- `POST /upload`: Upload a PDF file
  - Returns: JSON with file metadata and success status
- `POST /query`: Ask a question about an uploaded document
  - Parameters: `question` (text) and `filename` (string)
  - Returns: JSON with answer, confidence score, and processing time
- `POST /debug`: Get detailed information about a processed document
  - Parameters: `filename` (string)
  - Returns: JSON with document metadata, text samples, and chunking information

## Configuration

You can modify the following constants in the code:

- `MAX_CHUNK_SIZE`: Maximum token length for text chunks (default: 512)
- `OVERLAP_SIZE`: Overlap between chunks to maintain context (default: 128)
- `UPLOAD_FOLDER`: Directory to store uploaded files

## Limitations

- Only supports PDF files
- Maximum file size: 16 MB
- Performance depends on document length and complexity
- Results may vary based on document formatting and content quality
