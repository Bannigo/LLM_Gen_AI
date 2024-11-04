import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streamlit import StreamlitCallbackHandler


def fetch_and_parse_webpage(url):
    """Fetch content from a webpage and extract text."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    
    # Get text content
    text = soup.get_text()
    
    # Clean up whitespace
    lines = (line.strip() for line in text.splitlines())
    text = ' '.join(chunk for chunk in lines if chunk)
    
    return text

def create_rag_system(url):
    """Create a RAG system from a webpage."""
    # 1. Fetch and prepare the document
    text = fetch_and_parse_webpage(url)
    
    # 2. Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # 3. Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    # 4. Create vector store
    vectorstore = FAISS.from_texts(chunks, embeddings)
    
    # 5. Initialize Llama model
    llm = Ollama(
        model="llama2:3.2",
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        temperature=0
    )
    
    # 6. Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    
    return qa_chain

def ask_question(qa_chain, question):
    """Ask a question to the RAG system."""
    result = qa_chain({"query": question})
    return result

# Example usage
if __name__ == "__main__":
    # Replace with your webpage URL
    url = "api"
    
    # Create the RAG system
    print("Creating RAG system...")
    qa_chain = create_rag_system(url)
    
    # Ask questions
    while True:
        question = input("\nCan you create a brief summary of this page: ")
        if question.lower() == 'quit':
            break
            
        print("\nSearching for answer...")
        response = ask_question(qa_chain, question)
        print("\nAnswer:", response["result"])