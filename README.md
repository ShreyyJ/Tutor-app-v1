# Tutor-RAG
1. Clone and Setup
bash# Create project directory
mkdir personal-tutor
cd personal-tutor

# Create the files (save each artifact as indicated below)
 - personal_tutor.py (main application)
 - requirements.txt
 - docker-compose.yml  
 - milvus.yaml
 - Dockerfile
 - .env.example
2. Environment Configuration
bash# Copy environment template
cp .env.example .env

# Edit .env file with your OpenAI API key
nano .env
Add your OpenAI API key:
bashOPENAI_API_KEY=sk-your-actual-api-key-here
3. Start Milvus Database
bash# Start Milvus with Docker Compose
docker-compose up -d

# Wait for services to be healthy (30-60 seconds)
docker-compose ps
Verify Milvus is running:

Milvus API: http://localhost:19530
Milvus Web UI (Attu): http://localhost:3000
MinIO Console: http://localhost:9001 (minioadmin/minioadmin)

4. Install Python Dependencies
bash# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
5. Run the Application
bash# Run Streamlit app
streamlit run personal_tutor.py
The app will be available at: http://localhost:8501
