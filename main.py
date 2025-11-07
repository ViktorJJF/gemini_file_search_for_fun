import os
import time
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google import genai
from google.genai import types

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="Gemini File Search Demo")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage directory for uploaded files
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Gemini client (will be initialized when API key is provided)
client = None
file_search_store = None


class ChatRequest(BaseModel):
    message: str


class ApiKeyRequest(BaseModel):
    api_key: str


class FileInfo(BaseModel):
    filename: str
    size: int
    uploaded_at: str


def get_client():
    """Get or create Gemini client"""
    global client
    if client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=400,
                detail="GEMINI_API_KEY not set. Please set it via /set-api-key endpoint or environment variable"
            )
        client = genai.Client(api_key=api_key)
    return client


def get_file_search_store():
    """Get or create file search store"""
    global file_search_store
    if file_search_store is None:
        client = get_client()
        file_search_store = client.file_search_stores.create(
            config={'display_name': 'demo-file-search-store'}
        )
    return file_search_store


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page"""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gemini File Search Demo</title>
    <style>
        :root {
            /* Modern purple palette - refined and professional */
            --color-background: #fafafa;
            --color-surface: #ffffff;
            --color-surface-hover: #f8f7fc;

            /* Primary purple shades */
            --color-primary: #7c3aed;
            --color-primary-dark: #6d28d9;
            --color-primary-light: #8b5cf6;
            --color-primary-lighter: #a78bfa;

            /* Purple accents for UI elements */
            --color-accent: #9333ea;
            --color-accent-light: #a855f7;
            --color-accent-bg: #f5f3ff;

            /* Neutral text colors */
            --color-text-primary: #1f2937;
            --color-text-secondary: #6b7280;
            --color-text-tertiary: #9ca3af;

            /* Border colors with purple tint */
            --color-border: #e5e7eb;
            --color-border-hover: #d1d5db;

            /* Status colors */
            --color-success-bg: #ecfdf5;
            --color-success-text: #065f46;
            --color-success-border: #a7f3d0;
            --color-error-bg: #fef2f2;
            --color-error-text: #991b1b;
            --color-error-border: #fecaca;

            /* Enhanced spacing scale - 2x original values */
            --space-xs: 12px;
            --space-sm: 16px;
            --space-md: 24px;
            --space-lg: 32px;
            --space-xl: 48px;
            --space-2xl: 64px;
            --space-3xl: 96px;
            --space-4xl: 128px;

            /* Typography */
            --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            --font-weight-normal: 400;
            --font-weight-medium: 500;
            --font-weight-semibold: 600;
            --font-weight-bold: 700;

            /* Border radius */
            --radius-sm: 8px;
            --radius-md: 12px;
            --radius-lg: 16px;
            --radius-xl: 20px;

            /* Refined shadows */
            --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.04);
            --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.05);
            --shadow-lg: 0 8px 24px rgba(0, 0, 0, 0.06);
            --shadow-xl: 0 12px 32px rgba(0, 0, 0, 0.08);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        body {
            font-family: var(--font-sans);
            background: var(--color-background);
            min-height: 100vh;
            padding: var(--space-3xl) var(--space-xl);
            color: var(--color-text-primary);
            line-height: 1.6;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        h1 {
            color: var(--color-text-primary);
            text-align: center;
            margin-bottom: var(--space-3xl);
            font-size: 2.75rem;
            font-weight: var(--font-weight-bold);
            letter-spacing: -0.03em;
            background: linear-gradient(135deg, var(--color-primary) 0%, var(--color-accent) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: var(--space-xl);
            margin-bottom: var(--space-xl);
        }

        .grid-3 {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: var(--space-xl);
            margin-bottom: var(--space-xl);
        }

        @media (max-width: 1200px) {
            .grid-3 {
                grid-template-columns: repeat(2, 1fr);
            }
        }

        @media (max-width: 768px) {
            body {
                padding: var(--space-lg) var(--space-md);
            }

            h1 {
                font-size: 2rem;
                margin-bottom: var(--space-2xl);
            }

            .grid,
            .grid-3 {
                grid-template-columns: 1fr;
                gap: var(--space-lg);
            }
        }

        .card {
            background: var(--color-surface);
            border-radius: var(--radius-xl);
            padding: var(--space-2xl);
            box-shadow: var(--shadow-md);
            border: 1px solid var(--color-border);
            transition: all 0.25s ease;
        }

        .card:hover {
            box-shadow: var(--shadow-lg);
            border-color: var(--color-border-hover);
        }

        .card h2 {
            color: var(--color-text-primary);
            margin-bottom: var(--space-xl);
            font-size: 1.25rem;
            font-weight: var(--font-weight-semibold);
            letter-spacing: -0.02em;
            display: flex;
            align-items: center;
            gap: var(--space-sm);
        }

        .card h2::before {
            content: '';
            width: 4px;
            height: 24px;
            background: var(--color-primary);
            border-radius: 2px;
        }

        .full-width {
            grid-column: 1 / -1;
        }

        .upload-area {
            border: 2px dashed var(--color-border);
            border-radius: var(--radius-lg);
            padding: var(--space-3xl);
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-bottom: var(--space-xl);
            background: var(--color-background);
            position: relative;
            overflow: hidden;
        }

        .upload-area::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 80px;
            height: 80px;
            background: var(--color-accent-bg);
            border-radius: 50%;
            opacity: 0;
            transition: opacity 0.3s ease;
        }

        .upload-area:hover::before {
            opacity: 1;
        }

        .upload-area p {
            color: var(--color-text-secondary);
            font-size: 1rem;
            font-weight: var(--font-weight-medium);
            position: relative;
            z-index: 1;
        }

        .upload-area:hover {
            background: var(--color-surface-hover);
            border-color: var(--color-primary);
            transform: translateY(-2px);
        }

        .upload-area.dragover {
            background: var(--color-accent-bg);
            border-color: var(--color-primary);
            border-style: solid;
            box-shadow: var(--shadow-lg);
        }

        input[type="file"] {
            display: none;
        }

        button {
            background: var(--color-primary);
            color: white;
            border: none;
            padding: var(--space-md) var(--space-xl);
            border-radius: var(--radius-md);
            cursor: pointer;
            font-size: 1rem;
            font-weight: var(--font-weight-semibold);
            font-family: var(--font-sans);
            transition: all 0.25s ease;
            width: 100%;
            box-shadow: var(--shadow-md);
            position: relative;
            overflow: hidden;
        }

        button::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(255, 255, 255, 0.1);
            opacity: 0;
            transition: opacity 0.25s ease;
        }

        button:hover:not(:disabled)::before {
            opacity: 1;
        }

        button:hover:not(:disabled) {
            background: var(--color-primary-dark);
            box-shadow: var(--shadow-lg);
            transform: translateY(-2px);
        }

        button:active:not(:disabled) {
            transform: translateY(0);
            box-shadow: var(--shadow-sm);
        }

        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        #fileList {
            list-style: none;
            display: flex;
            flex-direction: column;
            gap: var(--space-sm);
        }

        #fileList li {
            background: var(--color-background);
            padding: var(--space-lg) var(--space-xl);
            border-radius: var(--radius-md);
            display: flex;
            justify-content: space-between;
            align-items: center;
            border: 1px solid var(--color-border);
            transition: all 0.25s ease;
        }

        #fileList li:hover {
            border-color: var(--color-primary);
            background: var(--color-surface-hover);
            transform: translateX(4px);
            box-shadow: var(--shadow-sm);
        }

        .file-name {
            font-weight: var(--font-weight-semibold);
            color: var(--color-text-primary);
            font-size: 1rem;
        }

        .file-size {
            color: var(--color-text-tertiary);
            font-size: 0.9375rem;
            font-weight: var(--font-weight-medium);
        }

        .chat-container {
            display: flex;
            flex-direction: column;
            height: 600px;
        }

        #chatMessages {
            flex: 1;
            overflow-y: auto;
            margin-bottom: var(--space-xl);
            padding: var(--space-xl);
            background: var(--color-background);
            border-radius: var(--radius-lg);
            border: 1px solid var(--color-border);
            display: flex;
            flex-direction: column;
            gap: var(--space-lg);
        }

        #chatMessages::-webkit-scrollbar {
            width: 10px;
        }

        #chatMessages::-webkit-scrollbar-track {
            background: transparent;
        }

        #chatMessages::-webkit-scrollbar-thumb {
            background: var(--color-border-hover);
            border-radius: 5px;
        }

        #chatMessages::-webkit-scrollbar-thumb:hover {
            background: var(--color-primary-lighter);
        }

        .message {
            padding: var(--space-lg) var(--space-xl);
            border-radius: var(--radius-lg);
            max-width: 80%;
            font-size: 1rem;
            line-height: 1.7;
            animation: slideIn 0.3s ease;
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .user-message {
            background: var(--color-primary);
            color: white;
            margin-left: auto;
            box-shadow: var(--shadow-md);
            border-bottom-right-radius: var(--radius-sm);
        }

        .bot-message {
            background: var(--color-surface);
            color: var(--color-text-primary);
            border: 1px solid var(--color-border);
            box-shadow: var(--shadow-md);
            border-bottom-left-radius: var(--radius-sm);
        }

        .chat-input-area {
            display: flex;
            gap: var(--space-md);
        }

        input[type="text"],
        input[type="password"] {
            flex: 1;
            padding: var(--space-md) var(--space-lg);
            border: 2px solid var(--color-border);
            border-radius: var(--radius-md);
            font-size: 1rem;
            font-family: var(--font-sans);
            background: var(--color-surface);
            color: var(--color-text-primary);
            transition: all 0.25s ease;
            font-weight: var(--font-weight-medium);
        }

        input[type="text"]:hover,
        input[type="password"]:hover {
            border-color: var(--color-border-hover);
        }

        input[type="text"]:focus,
        input[type="password"]:focus {
            outline: none;
            border-color: var(--color-primary);
            box-shadow: 0 0 0 4px rgba(124, 58, 237, 0.1);
            background: var(--color-surface);
        }

        input[type="text"]::placeholder,
        input[type="password"]::placeholder {
            color: var(--color-text-tertiary);
        }

        .status {
            padding: var(--space-md) var(--space-lg);
            border-radius: var(--radius-md);
            margin-top: var(--space-lg);
            font-size: 1rem;
            font-weight: var(--font-weight-medium);
            border: 2px solid transparent;
        }

        .status.success {
            background: var(--color-success-bg);
            color: var(--color-success-text);
            border-color: var(--color-success-border);
        }

        .status.error {
            background: var(--color-error-bg);
            color: var(--color-error-text);
            border-color: var(--color-error-border);
        }

        .api-key-input {
            display: flex;
            gap: var(--space-md);
        }

        .loading {
            display: inline-block;
            width: 18px;
            height: 18px;
            border: 3px solid var(--color-border);
            border-top: 3px solid var(--color-primary);
            border-radius: 50%;
            animation: spin 0.7s linear infinite;
            margin-left: var(--space-sm);
            vertical-align: middle;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* Enhanced focus states for accessibility */
        button:focus-visible,
        input:focus-visible {
            outline: 3px solid var(--color-primary);
            outline-offset: 3px;
        }

        /* Empty state styling */
        #fileList li[style*="text-align: center"] {
            justify-content: center;
            color: var(--color-text-tertiary);
            font-style: italic;
            border-style: dashed;
            padding: var(--space-2xl);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Gemini File Search</h1>

        <div class="card full-width">
            <h2>API Key Configuration</h2>
            <div class="api-key-input">
                <input type="password" id="apiKeyInput" placeholder="Enter your Gemini API Key">
                <button onclick="setApiKey()" style="width: auto; padding: 16px 48px;">Set API Key</button>
            </div>
            <div id="apiKeyStatus"></div>
        </div>

        <div class="grid">
            <div class="card">
                <h2>Upload Files</h2>
                <div class="upload-area" id="uploadArea">
                    <p>Click to select or drag & drop files here</p>
                    <input type="file" id="fileInput" multiple>
                </div>
                <button onclick="uploadFiles()" id="uploadBtn">Upload Files</button>
                <div id="uploadStatus"></div>
            </div>

            <div class="card">
                <h2>Uploaded Files</h2>
                <ul id="fileList"></ul>
            </div>
        </div>

        <div class="card full-width">
            <h2>Chat with Your Documents</h2>
            <div class="chat-container">
                <div id="chatMessages"></div>
                <div class="chat-input-area">
                    <input type="text" id="chatInput" placeholder="Ask a question about your documents...">
                    <button onclick="sendMessage()" id="sendBtn" style="width: auto; padding: 16px 48px;">Send</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');

        uploadArea.onclick = () => fileInput.click();

        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            fileInput.files = e.dataTransfer.files;
        });

        async function setApiKey() {
            const apiKey = document.getElementById('apiKeyInput').value;
            const statusDiv = document.getElementById('apiKeyStatus');

            if (!apiKey) {
                statusDiv.innerHTML = '<div class="status error">Please enter an API key</div>';
                return;
            }

            try {
                const response = await fetch('/set-api-key', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({api_key: apiKey})
                });

                if (response.ok) {
                    statusDiv.innerHTML = '<div class="status success">API key set successfully!</div>';
                    document.getElementById('apiKeyInput').value = '';
                } else {
                    const error = await response.json();
                    statusDiv.innerHTML = `<div class="status error">Error: ${error.detail}</div>`;
                }
            } catch (error) {
                statusDiv.innerHTML = `<div class="status error">Error: ${error.message}</div>`;
            }
        }

        async function uploadFiles() {
            const files = fileInput.files;
            const statusDiv = document.getElementById('uploadStatus');
            const uploadBtn = document.getElementById('uploadBtn');

            if (files.length === 0) {
                statusDiv.innerHTML = '<div class="status error">Please select files to upload</div>';
                return;
            }

            uploadBtn.disabled = true;
            statusDiv.innerHTML = '<div class="status">Uploading...<span class="loading"></span></div>';

            const formData = new FormData();
            for (let file of files) {
                formData.append('files', file);
            }

            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });

                if (response.ok) {
                    const result = await response.json();
                    statusDiv.innerHTML = `<div class="status success">Uploaded ${result.uploaded} file(s) successfully!</div>`;
                    fileInput.value = '';
                    loadFiles();
                } else {
                    const error = await response.json();
                    statusDiv.innerHTML = `<div class="status error">Error: ${error.detail}</div>`;
                }
            } catch (error) {
                statusDiv.innerHTML = `<div class="status error">Error: ${error.message}</div>`;
            } finally {
                uploadBtn.disabled = false;
            }
        }

        async function loadFiles() {
            try {
                const response = await fetch('/files');
                const files = await response.json();

                const fileList = document.getElementById('fileList');
                fileList.innerHTML = '';

                if (files.length === 0) {
                    fileList.innerHTML = '<li style="text-align: center; color: #666;">No files uploaded yet</li>';
                } else {
                    files.forEach(file => {
                        const li = document.createElement('li');
                        li.innerHTML = `
                            <span class="file-name">${file.filename}</span>
                            <span class="file-size">${formatBytes(file.size)}</span>
                        `;
                        fileList.appendChild(li);
                    });
                }
            } catch (error) {
                console.error('Error loading files:', error);
            }
        }

        async function sendMessage() {
            const input = document.getElementById('chatInput');
            const message = input.value.trim();
            const sendBtn = document.getElementById('sendBtn');

            if (!message) return;

            addMessage(message, 'user');
            input.value = '';
            sendBtn.disabled = true;

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: message})
                });

                if (response.ok) {
                    const result = await response.json();
                    addMessage(result.response, 'bot');
                } else {
                    const error = await response.json();
                    addMessage(`Error: ${error.detail}`, 'bot');
                }
            } catch (error) {
                addMessage(`Error: ${error.message}`, 'bot');
            } finally {
                sendBtn.disabled = false;
            }
        }

        function addMessage(text, sender) {
            const messagesDiv = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}-message`;
            messageDiv.textContent = text;
            messageDiv.style.marginBottom = '0';
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        function formatBytes(bytes) {
            if (bytes === 0) return '0 Bytes';
            const k = 1024;
            const sizes = ['Bytes', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
        }

        document.getElementById('chatInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });

        // Load files on page load
        loadFiles();
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


@app.post("/set-api-key")
async def set_api_key(request: ApiKeyRequest):
    """Set the Gemini API key"""
    global client, file_search_store

    try:
        # Reset existing client and store
        client = None
        file_search_store = None

        # Set the API key in environment
        os.environ["GEMINI_API_KEY"] = request.api_key

        # Test the API key by creating a client
        test_client = genai.Client(api_key=request.api_key)

        return JSONResponse({"status": "success", "message": "API key set successfully"})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload files and import them to Gemini file search store"""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    try:
        client = get_client()
        store = get_file_search_store()

        uploaded_count = 0

        for file in files:
            # Save file locally
            file_path = UPLOAD_DIR / file.filename
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)

            # Upload to Gemini file search store
            operation = client.file_search_stores.upload_to_file_search_store(
                file=str(file_path),
                file_search_store_name=store.name,
                config={'display_name': file.filename}
            )

            # Wait for import to complete
            timeout = 60  # 60 seconds timeout
            start_time = time.time()
            while not operation.done:
                if time.time() - start_time > timeout:
                    raise HTTPException(status_code=408, detail=f"Timeout importing {file.filename}")
                time.sleep(2)
                operation = client.operations.get(operation)

            uploaded_count += 1

        return JSONResponse({
            "status": "success",
            "uploaded": uploaded_count,
            "message": f"Successfully uploaded {uploaded_count} file(s)"
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/files")
async def list_files():
    """List all uploaded files"""
    try:
        files = []
        for file_path in UPLOAD_DIR.iterdir():
            if file_path.is_file():
                stat = file_path.stat()
                files.append({
                    "filename": file_path.name,
                    "size": stat.st_size,
                    "uploaded_at": time.ctime(stat.st_mtime)
                })

        return files
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(request: ChatRequest):
    """Chat with Gemini using file search"""
    try:
        client = get_client()
        store = get_file_search_store()

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=request.message,
            config=types.GenerateContentConfig(
                tools=[
                    types.Tool(
                        file_search=types.FileSearch(
                            file_search_store_names=[store.name]
                        )
                    )
                ]
            )
        )
        
        print("Full response:", response)

        return JSONResponse({
            "response": response.text,
            "status": "success"
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
