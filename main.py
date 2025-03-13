import os
import torch
import fitz
import time
from flask import Flask, request, jsonify, render_template, send_from_directory
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer, util
from werkzeug.utils import secure_filename
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
ALLOWED_EXTENSIONS = {'pdf'}
MAX_CHUNK_SIZE = 512
OVERLAP_SIZE = 128

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

logger.info("Loading models...")
qa_tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
qa_model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1F\x7F-\x9F]', '', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = text.replace('\r\n', ' ').replace('\n', ' ')
    text = text.replace('̶', '')
    return text.strip()

def extract_text_from_pdf(file_path):
    try:
        with fitz.open(file_path) as pdf_doc:
            metadata = {
                "title": pdf_doc.metadata.get('title', 'Unknown'),
                "author": pdf_doc.metadata.get('author', 'Unknown'),
                "subject": pdf_doc.metadata.get('subject', ''),
                "creator": pdf_doc.metadata.get('creator', ''),
                "producer": pdf_doc.metadata.get('producer', ''),
                "page_count": pdf_doc.page_count
            }

            all_text = []
            toc = pdf_doc.get_toc()

            for page_num in range(pdf_doc.page_count):
                page = pdf_doc[page_num]
                blocks = page.get_text("blocks")
                page_text = "\n".join([block[4] for block in blocks])
                all_text.append(clean_text(page_text))

            if toc:
                metadata["toc"] = [{"level": item[0], "title": item[1], "page": item[2]} for item in toc]

            return "\n".join(all_text), metadata

    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise ValueError(f"Could not extract text from PDF: {e}")


def chunk_text(text, chunk_size=MAX_CHUNK_SIZE, overlap=OVERLAP_SIZE):
    words = text.split()
    chunks = []

    if len(words) <= chunk_size:
        return [text]

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

        if i + chunk_size >= len(words):
            break

    return chunks


def get_answer_with_confidence(question, context, tokenizer, model):
    inputs = tokenizer(
        question,
        context,
        add_special_tokens=True,
        return_tensors="pt",
        max_length=MAX_CHUNK_SIZE,
        truncation=True,
        padding="max_length"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    start_idx = torch.argmax(start_logits)
    end_idx = torch.argmax(end_logits)

    start_confidence = torch.softmax(start_logits, dim=1)[0][start_idx].item()
    end_confidence = torch.softmax(end_logits, dim=1)[0][end_idx].item()
    confidence = (start_confidence + end_confidence) / 2

    if end_idx < start_idx:
        return "", 0.0

    input_ids = inputs.input_ids[0]
    answer_ids = input_ids[start_idx:end_idx + 1].tolist()
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True)

    if not answer or answer.isspace() or answer == '<s>' or answer == '</s>':
        return "", 0.0

    return answer, confidence


def answer_question(question, text, metadata=None):
    try:
        start_time = time.time()

        chunks = chunk_text(text, MAX_CHUNK_SIZE, OVERLAP_SIZE)
        logger.info(f"Split document into {len(chunks)} chunks")

        question_embedding = sentence_model.encode(question, convert_to_tensor=True)
        chunk_embeddings = sentence_model.encode(chunks, convert_to_tensor=True)

        similarity_scores = util.pytorch_cos_sim(question_embedding, chunk_embeddings)[0]

        top_k = min(5, len(chunks))
        top_indices = torch.topk(similarity_scores, k=top_k).indices.tolist()

        chunk_scores = [(idx, similarity_scores[idx].item()) for idx in top_indices]
        chunk_scores.sort(key=lambda x: x[1], reverse=True)

        relevant_chunks = []
        for idx, score in chunk_scores:
            relevant_chunks.append(chunks[idx])
            logger.info(f"Chunk {idx} selected with score {score:.4f}")

        answers = []

        for i, chunk in enumerate(relevant_chunks):
            answer, confidence = get_answer_with_confidence(question, chunk, qa_tokenizer, qa_model)
            if answer and len(answer) > 1:
                answers.append((answer, confidence, i))

        answers.sort(key=lambda x: x[1], reverse=True)

        if not answers or answers[0][1] < 0.3:
            if metadata and metadata.get("title"):
                enhanced_question = f"In the document titled '{metadata['title']}', {question}"
                for i, chunk in enumerate(relevant_chunks):
                    answer, confidence = get_answer_with_confidence(enhanced_question, chunk, qa_tokenizer, qa_model)
                    if answer and len(answer) > 1:
                        answers.append((answer, confidence, i))

            if not answers:
                return {
                    "answer": "I couldn't find a reliable answer to this question in the document.",
                    "confidence": "0%",
                    "success": False,
                    "processing_time": f"{time.time() - start_time:.2f} seconds"
                }

        best_answer, best_confidence, _ = answers[0]

        debug_info = {
            "chunks_analyzed": len(relevant_chunks),
            "total_chunks": len(chunks),
            "answers_found": len(answers)
        }

        return {
            "answer": best_answer,
            "confidence": f"{best_confidence:.2%}",
            "success": True,
            "processing_time": f"{time.time() - start_time:.2f} seconds",
            "debug": debug_info
        }

    except Exception as e:
        logger.error(f"Error answering question: {e}")
        return {
            "answer": f"An error occurred while processing your question: {str(e)}",
            "confidence": "0%",
            "success": False
        }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'pdf_file' not in request.files:
        return jsonify(success=False, message='No file part')

    file = request.files['pdf_file']

    if file.filename == '':
        return jsonify(success=False, message='No selected file')

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            pdf_text, metadata = extract_text_from_pdf(file_path)

            return jsonify(
                success=True,
                message='File uploaded and processed successfully',
                filename=filename,
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            return jsonify(success=False, message=f'Error processing file: {str(e)}')

    return jsonify(success=False, message='Invalid file format. Only PDF files are allowed.')


@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.json
        question = data.get('question')
        filename = data.get('filename')

        if not question or not filename:
            return jsonify(success=False, message='Question and filename are required')

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))

        if not os.path.exists(file_path):
            return jsonify(success=False, message='File not found')

        pdf_text, metadata = extract_text_from_pdf(file_path)

        result = answer_question(question, pdf_text, metadata)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify(
            success=False,
            answer=f"An error occurred: {str(e)}",
            confidence="0%"
        )


@app.route('/debug', methods=['POST'])
def debug_pdf():
    try:
        data = request.json
        filename = data.get('filename')

        if not filename:
            return jsonify(success=False, message='Filename is required')

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))

        if not os.path.exists(file_path):
            return jsonify(success=False, message='File not found')

        pdf_text, metadata = extract_text_from_pdf(file_path)

        text_sample = pdf_text[:500] + "..." if len(pdf_text) > 500 else pdf_text

        chunks = chunk_text(pdf_text)
        chunk_samples = []
        for i, chunk in enumerate(chunks[:3]):
            chunk_samples.append({
                "index": i,
                "length": len(chunk),
                "sample": chunk[:200] + "..." if len(chunk) > 200 else chunk
            })

        return jsonify({
            "success": True,
            "filename": filename,
            "metadata": metadata,
            "text_length": len(pdf_text),
            "text_sample": text_sample,
            "chunk_count": len(chunks),
            "chunk_samples": chunk_samples
        })

    except Exception as e:
        logger.error(f"Error debugging PDF: {e}")
        return jsonify(success=False, message=f"Error: {str(e)}")


if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)

    if not os.path.exists(os.path.join('templates', 'index.html')):
        with open(os.path.join('templates', 'index.html'), 'w') as f:
            pass

    js_file_path = os.path.join('static', 'app.js')
    if not os.path.exists(js_file_path):
        with open(js_file_path, 'w') as f:
            pass

    start_time = time.time()
    logger.info("Starting PDF Question-Answering System...")

    try:
        import psutil

        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        logger.info(f"Initial memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
    except ImportError:
        logger.info("psutil not installed, skipping memory usage reporting")

    # Start the Flask app
    logger.info(f"Initialization completed in {time.time() - start_time:.2f} seconds")
    app.run(debug=True, host='0.0.0.0', port=3000)