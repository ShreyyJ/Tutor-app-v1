import streamlit as st
import os
import time
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import tempfile
import ssl
import urllib3
import warnings
from dotenv import load_dotenv

# Load environment variables from .env file with explicit path
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path, override=True)

# Force disable SSL verification for all connections
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

# Clear all SSL certificate environment variables
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['SSL_CERT_FILE'] = ''
os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'  # Disable SSL for HuggingFace Hub
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'  # Increase timeout to 5 minutes

# Patch urllib3 HTTPSConnectionPool class to disable SSL
import urllib3.connectionpool
original_https_init = urllib3.connectionpool.HTTPSConnectionPool.__init__

def patched_https_init(self, *args, **kwargs):
    # Set cert_reqs and assert_hostname which are valid HTTPSConnectionPool args
    kwargs['cert_reqs'] = ssl.CERT_NONE
    kwargs['assert_hostname'] = False
    result = original_https_init(self, *args, **kwargs)
    return result

urllib3.connectionpool.HTTPSConnectionPool.__init__ = patched_https_init

# Patch urllib3 to disable SSL at connection pool level
import urllib3.util.ssl_
original_ssl_wrap_socket = urllib3.util.ssl_.ssl_wrap_socket

def patched_ssl_wrap_socket(sock, *args, **kwargs):
    kwargs['cert_reqs'] = ssl.CERT_NONE
    kwargs['check_hostname'] = False
    return original_ssl_wrap_socket(sock, *args, **kwargs)

urllib3.util.ssl_.ssl_wrap_socket = patched_ssl_wrap_socket

# Additional urllib3 SSL patching for connection pools
from urllib3.util.ssl_ import create_urllib3_context

original_create_urllib3_context = create_urllib3_context

def patched_create_urllib3_context(*args, **kwargs):
    kwargs['cert_reqs'] = ssl.CERT_NONE
    kwargs['check_hostname'] = False
    context = original_create_urllib3_context(*args, **kwargs)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context

urllib3.util.ssl_.create_urllib3_context = patched_create_urllib3_context

# Import requests but DON'T patch it (causes issues with OpenAI client)
import requests

# Configure HuggingFace Hub to use requests Session without SSL verification
from huggingface_hub import configure_http_backend
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import socket

# Increase socket timeout globally
socket.setdefaulttimeout(300)

def backend_factory():
    import requests
    session = requests.Session()
    session.verify = False
    
    # Configure retry strategy with longer timeout
    retry_strategy = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_maxsize=10, pool_block=False)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Patch the session's request method to ensure verify=False and timeout are always used
    original_session_request = session.request
    def patched_session_request(method, url, **kwargs):
        kwargs['verify'] = False
        kwargs['timeout'] = (30, 300)  # (connect timeout, read timeout) in seconds
        return original_session_request(method, url, **kwargs)
    session.request = patched_session_request
    return session

configure_http_backend(backend_factory=backend_factory)

# Document processing
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
HUGGINGFACEHUB_API_TOKEN='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'
class PersonalTutor:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.memory = None
        self.conversation_chain = None
        self.index_name = "personal-tutor-docs"
        self.pinecone_client = None
        self.pinecone_api_key = None
        
    def initialize_components(self, 
                            openai_api_key: str, 
                            pinecone_api_key: str):
        """Initialize components with OpenAI embeddings and Pinecone"""
        try:
            # Initialize httpx client with SSL verification disabled for OpenAI connections
            logger.info("Creating HTTP client with SSL verification disabled...")
            import httpx
            
            # Create httpx client with SSL verification disabled for embeddings
            http_client_embeddings = httpx.Client(verify=False, timeout=60.0)
            
            # Initialize OpenAI embeddings
            logger.info("Loading OpenAI embeddings...")
            
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=openai_api_key,
                model="text-embedding-3-small",  # Fast and efficient
                http_client=http_client_embeddings
            )
            
            # Test embeddings
            logger.info("Testing embeddings connection...")
            test_embedding = self.embeddings.embed_query("test")
            embedding_dim = len(test_embedding)
            logger.info(f"Initialized OpenAI embeddings (dim={embedding_dim})")
            
            # Store API key and initialize Pinecone with SSL disabled
            self.pinecone_api_key = pinecone_api_key
            
            # Set environment variable to disable SSL for Pinecone
            os.environ['PINECONE_API_KEY'] = pinecone_api_key
            
            # Monkey patch urllib3 to disable SSL for Pinecone connections
            import urllib3.util.ssl_
            
            try:
                logger.info("Connecting to Pinecone...")
                self.pinecone_client = Pinecone(api_key=pinecone_api_key)
                
                # Try to list indexes with retry logic
                retry_count = 0
                max_retries = 3
                existing_indexes = None
                
                while retry_count < max_retries:
                    try:
                        logger.info(f"Listing Pinecone indexes (attempt {retry_count + 1}/{max_retries})...")
                        existing_indexes = [index.name for index in self.pinecone_client.list_indexes()]
                        logger.info(f"Found {len(existing_indexes)} existing indexes: {existing_indexes}")
                        break
                    except Exception as index_error:
                        retry_count += 1
                        logger.warning(f"Failed to list indexes (attempt {retry_count}/{max_retries}): {str(index_error)}")
                        if retry_count < max_retries:
                            import time
                            wait_time = 2 ** retry_count  # Exponential backoff: 2s, 4s, 8s
                            logger.info(f"Retrying in {wait_time} seconds...")
                            time.sleep(wait_time)
                
                if existing_indexes is None:
                    raise Exception("Failed to connect to Pinecone after 3 attempts")
                
                # Handle index creation/recreation
                if self.index_name not in existing_indexes:
                    # Index doesn't exist, create it with correct dimension
                    logger.info(f"Creating new Pinecone index: {self.index_name} with dimension {embedding_dim} (OpenAI embeddings)...")
                    self.pinecone_client.create_index(
                        name=self.index_name,
                        dimension=embedding_dim,
                        metric='cosine',
                        spec=ServerlessSpec(
                            cloud='aws',
                            region='us-east-1'
                        )
                    )
                    # Wait for index to be ready
                    import time
                    time.sleep(2)
                    logger.info(f"Created new Pinecone index: {self.index_name}")
                else:
                    # Index exists, check if dimension matches
                    try:
                        logger.info(f"Checking dimension of existing index: {self.index_name}")
                        index_desc = self.pinecone_client.describe_index(self.index_name)
                        current_dimension = index_desc.dimension
                        
                        if current_dimension != embedding_dim:
                            logger.warning(f"Index dimension mismatch! Current: {current_dimension}, Required: {embedding_dim}")
                            logger.info(f"Deleting old index '{self.index_name}' with dimension {current_dimension}...")
                            self.pinecone_client.delete_index(self.index_name)
                            
                            import time
                            time.sleep(2)  # Wait for deletion
                            
                            logger.info(f"Creating new index with correct dimension {embedding_dim}...")
                            self.pinecone_client.create_index(
                                name=self.index_name,
                                dimension=embedding_dim,
                                metric='cosine',
                                spec=ServerlessSpec(
                                    cloud='aws',
                                    region='us-east-1'
                                )
                            )
                            time.sleep(2)  # Wait for creation
                            logger.info(f"Successfully recreated Pinecone index with correct dimensions")
                        else:
                            logger.info(f"Index dimension matches: {current_dimension}")
                    except Exception as dimension_check_error:
                        logger.warning(f"Could not check index dimension: {str(dimension_check_error)}")
                        logger.info(f"Using existing index: {self.index_name}")
                    
            except Exception as pinecone_error:
                logger.error(f"Pinecone connection error: {str(pinecone_error)}")
                raise Exception(f"Failed to initialize Pinecone: {str(pinecone_error)}")
            
            # Initialize LLM with custom httpx client that has SSL disabled
            logger.info("Initializing ChatOpenAI LLM...")
            
            # Create another httpx client for LLM (separate from embeddings)
            http_client_llm = httpx.Client(verify=False, timeout=60.0)
            
            self.llm = ChatOpenAI(
                openai_api_key=openai_api_key,
                model_name="gpt-4o-mini",  # Much faster than gpt-4
                temperature=0.2,
                max_tokens=500,  # Reduced for faster responses
                request_timeout=30,  # Reduced timeout
                max_retries=2,
                http_client=http_client_llm
            )
            
            # Initialize memory
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            
            logger.info("Components initialized successfully")
            return True, embedding_dim
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False, None
    
    def process_pdf(self, pdf_file) -> List[Document]:
        """Extract text from PDF file with multiple fallback strategies"""
        documents = []
        
        try:
            # Reset file pointer
            pdf_file.seek(0)
            
            # Strategy 1: Try with strict=False (most lenient)
            try:
                logger.info(f"Attempting to read PDF with strict=False: {pdf_file.name}")
                pdf_reader = PyPDF2.PdfReader(pdf_file, strict=False)
                num_pages = len(pdf_reader.pages) if pdf_reader.pages else 0
                
                if num_pages > 0:
                    logger.info(f"Successfully read {num_pages} pages from {pdf_file.name}")
                    documents = self._extract_text_from_reader(pdf_reader, pdf_file.name)
                    if documents:
                        return documents
            except Exception as attempt1_error:
                logger.warning(f"Strategy 1 failed for {pdf_file.name}: {str(attempt1_error)}")
            
            # Strategy 2: Try skipping bad data with onwarning=LOGGER_WARN
            try:
                logger.info(f"Attempting Strategy 2 - seek and retry: {pdf_file.name}")
                pdf_file.seek(0)
                
                # Try opening in binary mode with minimal validation
                pdf_content = pdf_file.read()
                from io import BytesIO
                pdf_bytes = BytesIO(pdf_content)
                
                pdf_reader = PyPDF2.PdfReader(pdf_bytes, strict=False)
                num_pages = len(pdf_reader.pages) if pdf_reader.pages else 0
                
                if num_pages > 0:
                    logger.info(f"Strategy 2 successful - found {num_pages} pages")
                    documents = self._extract_text_from_reader(pdf_reader, pdf_file.name)
                    if documents:
                        return documents
            except Exception as attempt2_error:
                logger.warning(f"Strategy 2 failed for {pdf_file.name}: {str(attempt2_error)}")
            
            # Strategy 3: Try to salvage partial data by skipping corrupted pages
            try:
                logger.info(f"Attempting Strategy 3 - salvage mode: {pdf_file.name}")
                pdf_file.seek(0)
                pdf_reader = PyPDF2.PdfReader(pdf_file, strict=False)
                
                # Try to get whatever pages we can
                if hasattr(pdf_reader, 'pages'):
                    extracted_count = 0
                    for page_num in range(len(pdf_reader.pages)):
                        try:
                            page = pdf_reader.pages[page_num]
                            text = page.extract_text()
                            if text and text.strip():
                                doc = Document(
                                    page_content=text,
                                    metadata={
                                        "source": pdf_file.name,
                                        "page": page_num + 1,
                                        "type": "pdf",
                                        "note": "extracted_from_corrupted_file"
                                    }
                                )
                                documents.append(doc)
                                extracted_count += 1
                        except Exception as page_error:
                            logger.debug(f"Could not extract page {page_num + 1}: {str(page_error)}")
                            continue
                    
                    if extracted_count > 0:
                        logger.info(f"Strategy 3 partial success - extracted {extracted_count} pages from {pdf_file.name}")
                        return documents
            except Exception as attempt3_error:
                logger.warning(f"Strategy 3 failed for {pdf_file.name}: {str(attempt3_error)}")
            
            # If all strategies fail, log and return empty
            if not documents:
                logger.error(f"Unable to extract text from PDF {pdf_file.name} - file may be corrupted or in an unsupported format")
                return []
            
            return documents
            
        except Exception as e:
            logger.error(f"Critical error processing PDF {pdf_file.name}: {str(e)}")
            return []
    
    def _extract_text_from_reader(self, pdf_reader, filename: str) -> List[Document]:
        """Helper method to extract text from a PyPDF2 reader"""
        documents = []
        try:
            if not hasattr(pdf_reader, 'pages') or not pdf_reader.pages:
                return documents
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    text = page.extract_text()
                    if text and text.strip():
                        doc = Document(
                            page_content=text,
                            metadata={
                                "source": filename,
                                "page": page_num + 1,
                                "type": "pdf"
                            }
                        )
                        documents.append(doc)
                except Exception as page_error:
                    logger.debug(f"Error extracting text from page {page_num + 1} in {filename}: {str(page_error)}")
                    continue
            
            return documents
        except Exception as e:
            logger.error(f"Error in text extraction helper: {str(e)}")
            return []
    
    def process_text(self, text_content: str, source_name: str) -> List[Document]:
        """Process plain text content"""
        if not text_content.strip():
            return []
        
        doc = Document(
            page_content=text_content,
            metadata={
                "source": source_name,
                "page": 1,
                "type": "text"
            }
        )
        
        return [doc]
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Reduced for faster embedding
            chunk_overlap=50,  # Minimal overlap for faster processing
            length_function=len,
        )
        
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} document chunks")
        return chunks
    
    def store_documents(self, documents: List[Document]) -> bool:
        """Store documents in Pinecone vector database with retry logic"""
        try:
            # Ensure index_name is set
            index_name = self.index_name if self.index_name else "personal-tutor-docs"
            logger.info(f"Storing {len(documents)} documents in Pinecone index: {index_name}")
            
            # Log each document before storing
            for i, doc in enumerate(documents, 1):
                logger.info(f"  Doc {i}: {doc.page_content[:80]}... (source: {doc.metadata.get('source', 'unknown')})")
            
            # Initialize vector store with LangChain Pinecone
            # Set the PINECONE_API_KEY environment variable temporarily
            os.environ['PINECONE_API_KEY'] = self.pinecone_api_key
            
            # Try to store with retry logic
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    logger.info("Creating vector store from documents...")
                    self.vectorstore = PineconeVectorStore.from_documents(
                        documents=documents,
                        embedding=self.embeddings,
                        index_name=index_name,
                        namespace=""  # Use default namespace
                    )
                    
                    logger.info(f"Successfully stored {len(documents)} documents in Pinecone")
                    
                    # Verify storage by attempting a quick retrieval test
                    logger.info("Verifying storage by testing retrieval...")
                    test_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 1})
                    test_results = test_retriever.get_relevant_documents("paper error analysis")
                    logger.info(f"Verification successful: Retrieved {len(test_results)} documents")
                    
                    return True
                    
                except Exception as store_error:
                    retry_count += 1
                    error_str = str(store_error)
                    
                    if "gaierror" in str(type(store_error)) or "NameResolutionError" in error_str or "Failed to resolve" in error_str:
                        # DNS/network error - try again
                        if retry_count < max_retries:
                            import time
                            wait_time = 2 ** retry_count  # 2s, 4s, 8s
                            logger.warning(f"Network error storing documents (attempt {retry_count}/{max_retries}): {str(store_error)}")
                            logger.info(f"Retrying in {wait_time} seconds...")
                            time.sleep(wait_time)
                        else:
                            logger.error(f"Failed to store documents after {max_retries} attempts due to network issues")
                            raise Exception(f"Network connectivity issue: Cannot reach Pinecone server. Please check your internet connection and try again.")
                    else:
                        # Different error - don't retry
                        raise
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing documents: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def clear_vectorstore(self) -> bool:
        """Clear all documents from the vector store with retry logic"""
        try:
            if not self.pinecone_client or not self.index_name:
                logger.warning("Vector store not initialized, cannot clear")
                return False
            
            logger.info(f"Clearing all documents from Pinecone index: {self.index_name}")
            
            # Get the index
            index = self.pinecone_client.Index(self.index_name)
            
            # Try to delete with retry logic
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    # Delete all vectors from the index
                    index.delete(delete_all=True, namespace="")
                    logger.info(f"Successfully cleared all vectors from index: {self.index_name}")
                    
                    # Also clear the memory when clearing documents
                    self.reset_memory()
                    
                    return True
                    
                except Exception as delete_error:
                    retry_count += 1
                    if "gaierror" in str(type(delete_error)) or "NameResolutionError" in str(type(delete_error)) or "Failed to resolve" in str(delete_error):
                        # DNS/network error - try again
                        if retry_count < max_retries:
                            import time
                            wait_time = 2 ** retry_count  # 2s, 4s, 8s
                            logger.warning(f"Network error clearing vectors (attempt {retry_count}/{max_retries}): {str(delete_error)}")
                            logger.info(f"Retrying in {wait_time} seconds...")
                            time.sleep(wait_time)
                        else:
                            logger.error(f"Failed to clear vector store after {max_retries} attempts due to network issues")
                            logger.warning("Continuing anyway - will store new documents (old data may still exist)")
                            # Still clear memory even if delete fails
                            self.reset_memory()
                            return True  # Return True to allow upload to continue
                    else:
                        # Different error - don't retry
                        raise
            
            return True
            
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            logger.warning("Continuing anyway - will attempt to store new documents")
            # Still clear memory even on error
            self.reset_memory()
            return True  # Return True to allow upload to continue
    
    def load_existing_vectorstore(self) -> bool:
        """Load existing vector store if index exists"""
        try:
            # Check if index exists and has vectors
            logger.info("Attempting to connect to existing Pinecone index...")
            existing_indexes = [index.name for index in self.pinecone_client.list_indexes()]
            logger.info(f"Available indexes: {existing_indexes}")
            
            if self.index_name not in existing_indexes:
                logger.info(f"Index '{self.index_name}' not found. Creating new index...")
                return False
            
            # Get the index
            logger.info(f"Connecting to index: {self.index_name}")
            index = self.pinecone_client.Index(self.index_name)
            
            self.vectorstore = PineconeVectorStore(
                index=index,
                embedding=self.embeddings,
                text_key="text"
            )
            
            logger.info("Successfully connected to existing Pinecone index")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to existing index: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def initialize_conversation_chain(self):
        """Initialize the conversational retrieval chain"""
        if not self.vectorstore:
            logger.error("Vector store not initialized")
            return False
        
        try:
            # Reset memory when initializing chain
            if self.memory:
                self.memory.clear()
                logger.info("Memory cleared before initializing new conversation chain")
            
            # Create retriever
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}  # Reduced from 5 to 3 for faster retrieval
            )
            
            # Create conversation chain with properly configured memory
            self.conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                memory=self.memory,
                return_source_documents=True,
                verbose=False,  # Changed to False to reduce logging noise
                max_tokens_limit=2000  # Limit history length to prevent token overflow
            )
            
            logger.info("Conversation chain initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing conversation chain: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def add_message_to_memory(self, question: str, answer: str) -> None:
        """Add a question-answer pair to memory"""
        try:
            if self.memory:
                # Save to memory using the internal structure
                self.memory.save_context(
                    {"input": question},
                    {"output": answer}
                )
                logger.info(f"Added message to memory. Current memory variables: {self.memory.memory_variables}")
        except Exception as e:
            logger.error(f"Error adding message to memory: {str(e)}")
    
    def reset_memory(self) -> None:
        """Clear all conversation memory"""
        try:
            if self.memory:
                self.memory.clear()
                logger.info("Conversation memory cleared")
        except Exception as e:
            logger.error(f"Error clearing memory: {str(e)}")
    
    def get_recent_conversation_context(self, max_messages: int = 6) -> str:
        """Get recent conversation history formatted for the LLM"""
        try:
            if not self.memory:
                return ""
            
            # Access the actual message list from memory
            messages = []
            
            # Try to get messages from chat_memory first (LangChain ConversationBufferMemory)
            if hasattr(self.memory, 'chat_memory') and hasattr(self.memory.chat_memory, 'messages'):
                messages = self.memory.chat_memory.messages
            elif hasattr(self.memory, 'buffer'):
                # Fallback: try to parse the buffer string
                buffer_str = self.memory.buffer
                if not buffer_str:
                    return ""
                logger.info(f"Using buffer string fallback")
            
            if not messages and hasattr(self.memory, 'buffer'):
                buffer_str = self.memory.buffer
                if not buffer_str:
                    return ""
                # Parse the buffer string 
                lines = buffer_str.strip().split('\n')
                formatted_exchanges = []
                
                for line in lines:
                    line = line.strip()
                    if line.startswith('Human:') or line.startswith('User:'):
                        content = line.replace('Human:', '').replace('User:', '').strip()
                        if content:
                            formatted_exchanges.append(f"You: {content}")
                    elif line.startswith('AI:') or line.startswith('Assistant:'):
                        content = line.replace('AI:', '').replace('Assistant:', '').strip()
                        if content:
                            formatted_exchanges.append(f"AI Tutor: {content}")
                
                recent = formatted_exchanges[-max_messages:] if len(formatted_exchanges) > max_messages else formatted_exchanges
                result = "\n".join(recent) if recent else ""
                logger.info(f"Formatted context from buffer: {result}")
                return result
            
            # Process actual message objects
            if messages:
                logger.info(f"Found {len(messages)} messages in memory")
                formatted_exchanges = []
                
                for msg in messages:
                    try:
                        msg_type = type(msg).__name__
                        content = msg.content if hasattr(msg, 'content') else str(msg)
                        
                        if 'Human' in msg_type or 'User' in msg_type:
                            if content:
                                formatted_exchanges.append(f"You: {content}")
                        elif 'AI' in msg_type or 'Assistant' in msg_type:
                            if content:
                                formatted_exchanges.append(f"AI Tutor: {content}")
                    except Exception as e:
                        logger.warning(f"Error processing message: {str(e)}")
                        continue
                
                # Get the most recent messages
                recent = formatted_exchanges[-max_messages:] if len(formatted_exchanges) > max_messages else formatted_exchanges
                result = "\n".join(recent) if recent else ""
                
                logger.info(f"Formatted context from messages: {result}")
                return result
            
            return ""
            
        except Exception as e:
            logger.error(f"Error formatting conversation context: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return ""
    
    def get_answer(self, question: str) -> Dict[str, Any]:
        """Get answer from the RAG system with proper memory handling"""
        if not self.conversation_chain:
            return {
                "answer": "System not properly initialized. Please upload documents first.",
                "source_documents": [],
                "processing_time": 0
            }
        
        try:
            start_time = time.time()
            
            logger.info(f"Processing question: {question}")
            logger.info(f"Current memory before query: {self.memory.buffer if self.memory else 'No memory'}")
            
            # Check if this is a meta-question about the conversation
            is_meta_question = any(phrase in question.lower() for phrase in [
                "what was the", "did you ask", "did i ask", "what did i ask",
                "previous question", "last question", "before", "earlier",
                "remember", "do you remember"
            ])
            
            # For meta-questions, try to answer directly from memory first
            if is_meta_question and self.memory and self.memory.buffer:
                recent_context = self.get_recent_conversation_context(max_messages=6)
                if recent_context:
                    # Prepare a context-aware prompt
                    meta_prompt = f"""Based on the following conversation history, answer the user's question. Be specific and reference the actual questions and answers from the history.

Conversation History:
{recent_context}

User's question: {question}

Answer directly from the conversation history above:"""
                    
                    try:
                        meta_result = self.llm.predict(text=meta_prompt)
                        processing_time = time.time() - start_time
                        
                        logger.info(f"Meta-question answered from memory in {processing_time:.2f}s")
                        return {
                            "answer": meta_result,
                            "source_documents": [],
                            "processing_time": processing_time
                        }
                    except Exception as meta_error:
                        logger.warning(f"Meta-question handling failed, falling back to regular chain: {str(meta_error)}")
            
            # Check if question is vague and enhance with conversation context
            vague_indicators = ["this", "that", "more", "detail", "explain", "tell me", "how", "why"]
            is_vague = any(phrase in question.lower() for phrase in vague_indicators)
            
            enhanced_question = question
            if is_vague and self.memory and self.memory.buffer and len(question.split()) < 6:
                # Get last few exchanges for context
                try:
                    if hasattr(self.memory, 'chat_memory') and hasattr(self.memory.chat_memory, 'messages'):
                        messages = self.memory.chat_memory.messages
                        if len(messages) >= 2:
                            # Get the last user question for context
                            last_question = None
                            for msg in reversed(messages):
                                msg_type = type(msg).__name__
                                if 'Human' in msg_type or 'User' in msg_type:
                                    last_question = msg.content if hasattr(msg, 'content') else str(msg)
                                    break
                            
                            if last_question:
                                enhanced_question = f"{question} (in context of: {last_question})"
                                logger.info(f"Vague question enhanced: {enhanced_question}")
                except Exception as e:
                    logger.warning(f"Could not enhance vague question: {str(e)}")
            
            # For regular questions, use the retrieval chain
            logger.info(f"Calling conversation chain for question: {enhanced_question}")
            
            # Test the retriever directly to see what documents are found
            if hasattr(self.conversation_chain, 'retriever'):
                retrieved_docs = self.conversation_chain.retriever.get_relevant_documents(enhanced_question)
                logger.info(f"Retriever found {len(retrieved_docs)} relevant documents:")
                for i, doc in enumerate(retrieved_docs, 1):
                    logger.info(f"  Doc {i}: {doc.page_content[:100]}...")
            
            # Call the conversation chain with potentially enhanced question
            result = self.conversation_chain({
                "question": enhanced_question if is_vague else question
            })
            
            processing_time = time.time() - start_time
            
            logger.info(f"Answer: {result.get('answer', '')[:100]}...")
            logger.info(f"Source documents returned: {len(result.get('source_documents', []))}")
            logger.info(f"Current memory after query: {self.memory.buffer if self.memory else 'No memory'}")
            logger.info(f"Question processed successfully in {processing_time:.2f}s")
            
            return {
                "answer": result.get("answer", ""),
                "source_documents": result.get("source_documents", []),
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Error getting answer: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "answer": f"Error processing question: {str(e)}. Please check your OpenAI API key and internet connection.",
                "source_documents": [],
                "processing_time": 0
            }

def main():
    st.set_page_config(
        page_title="Personal Tutor - RAG AI",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-section {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .source-document {
        background-color: #fff3e0;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        border-left: 3px solid #ff9800;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🎓 Personal Tutor - RAG AI System</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'tutor' not in st.session_state:
        st.session_state.tutor = PersonalTutor()
        st.session_state.initialized = False
        st.session_state.documents_uploaded = False
        st.session_state.chat_history = []
    
    # Auto-initialize on first load with .env credentials
    if not st.session_state.initialized:
        openai_api_key = os.getenv("OPENAI_API_KEY", "")
        pinecone_api_key = os.getenv("PINECONE_API_KEY", "")
        
        if openai_api_key and pinecone_api_key:
            with st.spinner("Initializing system..."):
                success, embedding_dim = st.session_state.tutor.initialize_components(
                    openai_api_key, 
                    pinecone_api_key
                )
                
                if success:
                    st.session_state.initialized = True
                    st.session_state.embedding_dim = embedding_dim
                    st.success("✅ System initialized successfully!")
                    
                    # Try to load existing vectorstore
                    if st.session_state.tutor.load_existing_vectorstore():
                        if st.session_state.tutor.initialize_conversation_chain():
                            st.session_state.documents_uploaded = True
                else:
                    st.error("❌ Failed to initialize system. Please check:")
                    st.error("• Verify your OPENAI_API_KEY is valid")
                    st.error("• Verify your PINECONE_API_KEY is valid")
                    st.error("• Check your internet connection")
                    st.error("• Check your firewall/proxy settings")
                    st.info("Check the terminal/logs for detailed error messages.")
                    st.stop()
        else:
            st.error("❌ Missing API keys in .env file. Please add OPENAI_API_KEY and PINECONE_API_KEY to your .env file.")
            st.stop()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("📊 System Status")
        
        if st.session_state.initialized:
            st.success("✅ System Ready")
            if hasattr(st.session_state, 'embedding_dim'):
                st.info(f"📐 Embedding Dimension: {st.session_state.embedding_dim}")
        else:
            st.error("⚠️ System Not Initialized")
        
        if st.session_state.documents_uploaded:
            st.success("✅ Documents Loaded")
        else:
            st.warning("⚠️ No Documents Uploaded")
        
        st.divider()
        
        # Memory status
        st.header("💾 Memory Status")
        if st.session_state.documents_uploaded:
            chat_count = len(st.session_state.chat_history)
            st.metric("Chat Messages", chat_count)
            
            if st.session_state.tutor.memory:
                try:
                    memory_buffer = st.session_state.tutor.memory.buffer
                    buffer_words = len(memory_buffer.split()) if memory_buffer else 0
                    st.metric("Memory Buffer Words", buffer_words)
                except:
                    st.metric("Memory Status", "Active")
            
            st.info("💡 The AI can remember context from previous questions in this conversation.")
        else:
            st.info("Upload documents to start chatting with memory!")
        
        st.divider()
        
        # Clear vector store button
        st.header("🧹 Vector Store Management")
        if st.button("Clear Vector Store", type="secondary", help="Remove all documents from the vector store. Use this before uploading new documents."):
            with st.spinner("Clearing vector store..."):
                if st.session_state.tutor.clear_vectorstore():
                    st.session_state.documents_uploaded = False
                    st.session_state.chat_history = []
                    st.success("✅ Vector store cleared successfully!")
                    st.info("You can now upload new documents.")
                else:
                    st.error("❌ Failed to clear vector store")
    
    # Main content area
    if not st.session_state.initialized:
        st.error("❌ System failed to initialize. Please check your .env file contains valid OPENAI_API_KEY and PINECONE_API_KEY.")
        return
    
    # Document upload section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.header("📄 Upload Documents")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("PDF Upload")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload PDF documents for the AI to learn from"
        )
    
    with col2:
        st.subheader("Text Input")
        text_input = st.text_area(
            "Enter text content",
            height=150,
            help="Enter any text content you want the AI to learn from"
        )
        text_name = st.text_input("Text source name", value="User Input")
    
    # Process documents
    if st.button("🔄 Process Documents"):
        if not uploaded_files and not text_input.strip():
            st.warning("Please upload PDF files or enter text content")
        else:
            with st.spinner("Processing documents..."):
                all_documents = []
                
                # Process PDFs
                if uploaded_files:
                    for uploaded_file in uploaded_files:
                        docs = st.session_state.tutor.process_pdf(uploaded_file)
                        all_documents.extend(docs)
                
                # Process text input
                if text_input.strip():
                    text_docs = st.session_state.tutor.process_text(text_input, text_name)
                    all_documents.extend(text_docs)
                
                if all_documents:
                    # Clear old documents from vector store first
                    st.spinner("Clearing previous documents from vector store...")
                    logger.info("Clearing old documents before storing new ones")
                    if not st.session_state.tutor.clear_vectorstore():
                        st.warning("⚠️ Could not clear previous documents. You may see results from old documents.")
                    
                    # Chunk documents
                    chunked_docs = st.session_state.tutor.chunk_documents(all_documents)
                    
                    # Store in vector database
                    if st.session_state.tutor.store_documents(chunked_docs):
                        # Initialize conversation chain
                        if st.session_state.tutor.initialize_conversation_chain():
                            st.session_state.documents_uploaded = True
                            st.success(f"✅ Successfully processed {len(all_documents)} documents into {len(chunked_docs)} chunks!")
                            st.success("✅ All previous documents have been cleared. Vector store now contains only the new documents.")
                            
                            # Show index info
                            st.info(f"📊 Pinecone index '{st.session_state.tutor.index_name}' now contains your documents and is ready for queries!")
                        else:
                            st.error("❌ Failed to initialize conversation chain")
                    else:
                        st.error("❌ Failed to store documents in Pinecone")
                        st.error("📡 **Network Issue**: Your machine cannot reach the Pinecone server.")
                        st.error("**Troubleshooting:**")
                        st.error("• Check your internet connection")
                        st.error("• Check your firewall/antivirus settings")
                        st.error("• If you're behind a proxy, configure your .env file")
                        st.error("• Try again in a few moments (temporary network issue)")
                else:
                    st.error("❌ No documents were processed")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat interface
    if st.session_state.documents_uploaded:
        st.header("💬 Ask Questions")
        
        # Display memory status
        if st.session_state.tutor.memory:
            memory_status = f"📝 Memory: {len(st.session_state.chat_history)} messages in history"
            st.info(memory_status)
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["type"] == "user":
                st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message assistant-message"><strong>AI Tutor:</strong> {message["content"]}</div>', unsafe_allow_html=True)
                
                # Show sources if available
                if "sources" in message and message["sources"]:
                    with st.expander(f"📚 Sources ({len(message['sources'])} documents)"):
                        for i, source in enumerate(message["sources"], 1):
                            st.markdown(f"""
                            <div class="source-document">
                                <strong>Source {i}:</strong> {source.metadata.get('source', 'Unknown')} 
                                (Page {source.metadata.get('page', 'N/A')})<br>
                                <em>{source.page_content[:200]}...</em>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Show processing time
                if "processing_time" in message:
                    st.caption(f"⏱️ Processing time: {message['processing_time']:.2f} seconds")
        
        # Question input
        question = st.text_input(
            "Ask a question about your documents:",
            placeholder="What would you like to know?",
            key="question_input"
        )
        
        col1, col2, col3 = st.columns([1, 2, 2])
        with col1:
            ask_button = st.button("🤔 Ask Question", use_container_width=True)
        with col2:
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.tutor.reset_memory()
                st.success("✅ Chat history and memory cleared!")
                st.rerun()
        with col3:
            if st.button("🔄 Refresh Memory", use_container_width=True, help="Reinitialize conversation chain with current memory"):
                if st.session_state.tutor.initialize_conversation_chain():
                    st.success("✅ Conversation chain refreshed!")
                else:
                    st.error("❌ Failed to refresh conversation chain")
        
        if ask_button and question.strip():
            # Add user message to chat history
            st.session_state.chat_history.append({
                "type": "user",
                "content": question
            })
            
            # Get answer from RAG system
            with st.spinner("Thinking... (using conversation memory)"):
                result = st.session_state.tutor.get_answer(question)
                
                # Add assistant message to chat history
                st.session_state.chat_history.append({
                    "type": "assistant",
                    "content": result["answer"],
                    "sources": result["source_documents"],
                    "processing_time": result["processing_time"]
                })
            
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #666;">🎓 Personal Tutor RAG System - Built with Streamlit, OpenAI, and Pinecone</p>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()