import streamlit as st
from pathlib import Path
import time

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from langchain_text_splitters  import MarkdownTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings, HuggingFaceEndpoint, ChatHuggingFace

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

def load_docs():
    content_dir = Path('content')
    docs = []
    
    for file in sorted(content_dir.glob('*.md')):
        text = file.read_text(encoding = 'utf-8')
        docs.append(Document(
            page_content = text,
            metadata = {'source': file.stem}
        ))
    
    return docs

def split_docs(docs):
    splitter = MarkdownTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )
    
    return splitter.split_documents(docs)

@st.cache_resource
def build_vector_store():
    docs = load_docs()
    chunks = split_docs(docs)
    
    embeddings = HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token = st.secrets['HUGGINGFACE_API_KEY'],
        model = 'sentence-transformers/all-MiniLM-L6-v2'
    )
    
    pc = Pinecone(api_key = st.secrets['PINECONE_API_KEY'])
    index_name = 'himanshu-portfolio'
    
    existing_indexes = [index.name for index in pc.list_indexes()]

    if index_name not in existing_indexes:
        pc.create_index(
            name = index_name,
            dimension = 384,
            metric = 'cosine',
            spec = ServerlessSpec(
                cloud = 'aws',
                region = 'us-east-1'
            )
        )
    
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)
        
    vector_store = PineconeVectorStore.from_documents(
        documents = chunks,
        embedding = embeddings,
        index_name = index_name,
        pinecone_api_key = st.secrets['PINECONE_API_KEY']
    )
    
    return vector_store

def build_rag_chain(_vector_store):
    llm_endpoint = HuggingFaceEndpoint(
        repo_id = 'meta-llama/Llama-3.1-8B-Instruct',
        task = 'conversational',
        huggingfacehub_api_token = st.secrets['HUGGINGFACE_API_KEY'],
        max_new_tokens = 512,
        temperature = 0.5
    )
    
    model = ChatHuggingFace(llm=llm_endpoint)
    
    retriever = _vector_store.as_retriever(
        search_type = 'similarity',
        search_kwargs = {'k': 3}
    )
    
    system_prompt = """You are a helpful assistant representing Himanshu Khandelwal's portfolio.
Answer questions about Himanshu based ONLY on the retrieved context below.
Be concise, friendly, and professional.
If the answer is not in the context, say "I don't have that information about Himanshu."

Retrieved Context:
{context}
"""

    prompt = ChatPromptTemplate.from_messages([
        ('system', system_prompt),
        MessagesPlaceholder(variable_name = 'chat_history'),
        ('human', '{input}')
    ])
    
    question_answer_chain = create_stuff_documents_chain(model, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain
    

def main():
    st.title("🤖 Ask me anything about Himanshu")
    st.caption("Powered by RAG · Mistral 7B · Pinecone · HuggingFace")
    
    with st.spinner('🔧 Setting up knowledge base...'):
        vector_store = build_vector_store()
        rag_chain = build_rag_chain(vector_store)
    
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
        
    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])
    
    if prompt := st.chat_input('Ask something like: What testing tools does Himanshu know?'):
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        
        with st.chat_message('user'):
            st.markdown(prompt)
        
        chat_history = []
        
        chat_history = [HumanMessage(content = msg['content']) if msg['role'] == 'user' else AIMessage(content = msg['content']) for msg in st.session_state.messages[:-1]]
        
        with st.chat_message('assistant'):
            with st.spinner('Thinking...'):
                result = rag_chain.invoke({
                    'input': prompt,
                    'chat_history': chat_history
                })
                response = result['answer']
                st.markdown(response)
        
        st.session_state.messages.append({'role': 'assistant', 'content': response})

if __name__ == "__main__":
    main()
