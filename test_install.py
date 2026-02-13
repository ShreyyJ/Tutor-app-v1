"""
Test script to verify all dependencies are installed correctly
Run this before deploying to catch any issues early
"""
import sys

print("Testing Python version...")
print(f"Python {sys.version}")

required_modules = [
    "streamlit",
    "dotenv",
    "requests",
    "urllib3",
    "httpx",
    "PyPDF2",
    "langchain",
    "langchain_text_splitters",
    "langchain_openai",
    "langchain_pinecone",
    "pinecone",
    "huggingface_hub",
    "pydantic",
]

print("\nChecking required modules...")
failed = []

for module in required_modules:
    try:
        if module == "dotenv":
            __import__("dotenv")
        elif module == "PyPDF2":
            __import__("PyPDF2")
        else:
            __import__(module)
        print(f"✓ {module}")
    except ImportError as e:
        print(f"✗ {module} - {str(e)}")
        failed.append(module)

print("\n" + "="*50)
if not failed:
    print("✅ All dependencies installed successfully!")
    print("\nYou can now run: streamlit run app.py")
else:
    print(f"❌ Failed to import: {', '.join(failed)}")
    print("\nTry running: pip install -r requirements.txt")
    sys.exit(1)
