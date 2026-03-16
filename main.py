import streamlit as st
from pathlib import Path
import time

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from langchain_text_splitters  import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings

from langchain_anthropic import ChatAnthropic

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
    # 1. Headers identify karein jinpe split karna hai
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    final_chunks = []
    
    for doc in docs:
        # Header ke basis par split karein
        header_splits = header_splitter.split_text(doc.page_content)
        
        # Agar koi section bohot bada hai, toh usse further chota karein
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, # Size thoda badha diya taaki context rahe
            chunk_overlap=100
        )
        
        splits = text_splitter.split_documents(header_splits)
        
        # Purana metadata (like filename) preserve karein
        for split in splits:
            split.metadata.update(doc.metadata)
            final_chunks.append(split)
            
    return final_chunks

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
    
    # 1. Check/Create Index
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

    # 2. CLEAR THE INDEX (Crucial for testing)
    # This prevents old data from blocking your new details
    index = pc.Index(index_name)
    index.delete(delete_all=True) 
    
    # 3. Upload fresh data
    vector_store = PineconeVectorStore.from_documents(
        documents = chunks,
        embedding = embeddings,
        index_name = index_name,
        pinecone_api_key = st.secrets['PINECONE_API_KEY']
    )
    
    return vector_store

def build_rag_chain(_vector_store):
    model = ChatAnthropic(
        api_key=st.secrets['ANTHROPIC_API_KEY'],
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        temperature=0.5
    )

    
    retriever = _vector_store.as_retriever(
        search_type = 'similarity',
        search_kwargs = {'k': 10}
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
    st.caption("Powered by RAG · Anthropic · Pinecone · HuggingFace")
    
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
