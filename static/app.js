let currentFilename = null;

document.addEventListener('DOMContentLoaded', function() {
    // Upload form handler
    document.getElementById('upload-form').addEventListener('submit', async function(e) {
        e.preventDefault();

        const formData = new FormData();
        const fileInput = document.getElementById('pdf_file');

        if (fileInput.files.length === 0) {
            document.getElementById('upload-status').textContent = 'Please select a file.';
            return;
        }

        formData.append('pdf_file', fileInput.files[0]);

        document.getElementById('upload-status').textContent = 'Uploading and processing...';

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                document.getElementById('upload-status').textContent = data.message;
                currentFilename = data.filename;

                // Display metadata
                if (data.metadata) {
                    const metadataElem = document.getElementById('metadata-display');
                    metadataElem.innerHTML = `
                        <p><strong>Document Info:</strong></p>
                        <p>Title: ${data.metadata.title || 'Unknown'}</p>
                        <p>Author: ${data.metadata.author || 'Unknown'}</p>
                        <p>Pages: ${data.metadata.page_count}</p>
                    `;
                    metadataElem.classList.remove('hidden');
                }

                // Show query section
                document.getElementById('query-section').classList.remove('hidden');
            } else {
                document.getElementById('upload-status').textContent = 'Error: ' + data.message;
            }
        } catch (error) {
            document.getElementById('upload-status').textContent = 'Error: ' + error.message;
        }
    });

    // Query form handler
    document.getElementById('query-form').addEventListener('submit', async function(e) {
        e.preventDefault();

        if (!currentFilename) {
            alert('Please upload a PDF file first.');
            return;
        }

        const question = document.getElementById('question').value.trim();

        if (!question) {
            alert('Please enter a question.');
            return;
        }

        const resultElem = document.getElementById('result');
        resultElem.innerHTML = 'Processing your question...';
        resultElem.classList.remove('hidden');

        try {
            const response = await fetch('/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    question: question,
                    filename: currentFilename
                })
            });

            const data = await response.json();

            let confidenceValue = 0;
            if (data.confidence) {
                confidenceValue = parseFloat(data.confidence.replace('%', '')) / 100;
            }

            const confidenceClass = confidenceValue < 0.5 ? 'low-confidence' : 'confidence';

            if (data.success) {
                resultElem.innerHTML = `
                    <p><strong>Answer:</strong> ${data.answer}</p>
                    <p class="${confidenceClass}">Confidence: ${data.confidence}</p>
                    <p class="processing-time">Processing time: ${data.processing_time || 'N/A'}</p>
                `;

                if (data.debug) {
                    const debugInfo = `
                        <details>
                            <summary>Debug Info</summary>
                            <p>Chunks analyzed: ${data.debug.chunks_analyzed}</p>
                            <p>Total chunks: ${data.debug.total_chunks}</p>
                            <p>Answers found: ${data.debug.answers_found}</p>
                        </details>
                    `;
                    resultElem.innerHTML += debugInfo;
                }
            } else {
                resultElem.innerHTML = `
                    <p><strong>No reliable answer found:</strong> ${data.answer}</p>
                    <p class="low-confidence">Confidence: ${data.confidence}</p>
                `;
            }
        } catch (error) {
            resultElem.innerHTML = 'Error: ' + error.message;
        }
    });

    const debugButton = document.createElement('button');
    debugButton.innerText = 'Verify PDF Extraction';
    debugButton.style.marginTop = '10px';
    debugButton.style.backgroundColor = '#607d8b';

    debugButton.addEventListener('click', async function() {
        if (!currentFilename) {
            alert('Please upload a PDF file first.');
            return;
        }

        try {
            const response = await fetch('/debug', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    filename: currentFilename
                })
            });

            const data = await response.json();

            if (data.success) {
                const resultElem = document.getElementById('result');
                resultElem.innerHTML = `
                    <h3>PDF Extraction Details</h3>
                    <p><strong>Total text length:</strong> ${data.text_length} characters</p>
                    <p><strong>Number of chunks:</strong> ${data.chunk_count}</p>
                    <details>
                        <summary>Text Sample</summary>
                        <pre style="white-space: pre-wrap; font-size: 0.8em;">${data.text_sample}</pre>
                    </details>
                `;
                resultElem.classList.remove('hidden');
            }
        } catch (error) {
            alert('Error: ' + error.message);
        }
    });

    document.getElementById('query-section').appendChild(debugButton);
});