# Website Search Assistant

A full-stack search application with AI-powered semantic search using AWS Bedrock, Flask backend, and React frontend.

## Quick Start

1. **Run the project:**
   ```bash
   run_project.bat
   ```

2. **Manual setup (if needed):**

   **Backend:**
   ```bash
   pip install -r requirements.txt
   python app.py
   ```

   **Frontend:**
   ```bash
   cd websearchfrontend
   npm install
   npm run dev
   ```

## Access Points

- **Frontend:** http://localhost:5173
- **Backend API:** http://127.0.0.1:8080
- **Search endpoint:** http://127.0.0.1:8080/search

## Configuration

- Copy `.env.example` to `.env` and configure AWS credentials
- Place your product data files in the `archive/` directory
- Supported formats: CSV, XLS, XLSX

## Features

- Semantic search using AWS Bedrock Titan embeddings
- Multi-currency price conversion
- Spell correction and query suggestions
- Real-time search with debouncing
- Responsive UI with Tailwind CSS

## Troubleshooting

- Ensure AWS credentials are configured
- Check that port 8080 (backend) and 5173 (frontend) are available
- Verify product data files exist in `archive/` directory