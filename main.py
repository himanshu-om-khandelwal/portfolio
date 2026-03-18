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

from datetime import datetime

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
    # 1. Define the headers to split on
    headers_to_split_on = [
        ("#", "H1"),
        ("##", "H2")
    ]
    
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on = headers_to_split_on,
        strip_headers = False # Keep the # in the text so the LLM sees the structure
    )
    
    final_chunks = []
    
    for doc in docs:
        # Split purely by the markdown headers
        header_splits = header_splitter.split_text(doc.page_content)
        
        # We skip the RecursiveCharacterTextSplitter entirely
        for split in header_splits:
            # Preserve original metadata (filename/source)
            split.metadata.update(doc.metadata)
            final_chunks.append(split)
            
    return final_chunks

@st.cache_resource
def build_vector_store():
    docs = load_docs()
    chunks = split_docs(docs)
    
    embeddings = HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token = st.secrets['HUGGINGFACE_API_KEY'],
        model = 'sentence-transformers/all-mpnet-base-v2'
    )
    
    pc = Pinecone(api_key = st.secrets['PINECONE_API_KEY'])
    index_name = 'himanshu-portfolio'
    
    # 1. Check/Create Index
    existing_indexes = [index.name for index in pc.list_indexes()]
    if index_name not in existing_indexes:
        pc.create_index(
            name = index_name,
            dimension = 768,
            metric = 'cosine',
            spec = ServerlessSpec(
                cloud = 'aws',
                region = 'us-east-1'
            )
        )
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
    index = pc.Index(index_name)

    # Check if the index has any vectors before trying to delete
    stats = index.describe_index_stats()
    
    # Logic: Only delete if there is data to delete
    if stats.get('total_vector_count', 0) > 0:
        try:
            index.delete(delete_all=True)
            print("🧹 Purged old vector data...")
        except Exception as e:
            print(f"Error occurred while purging vector data: {e}")
    else:
        print("✨ Index is already fresh and empty.")
    
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

    # calculate current date and time
    now = datetime.now().strftime('%B %d, %Y %I:%M %p')

    retriever = _vector_store.as_retriever(
        search_type = 'mmr',          # Maximum Marginal Relevance
        search_kwargs = {
            'k': 10,
            'fetch_k': 20,
            'lambda_mult': 0.7
        }
    )
    
    system_prompt = f"""You are a helpful assistant representing Himanshu Khandelwal's portfolio.
Answer questions about Himanshu based ONLY on the retrieved context below.
Be concise, friendly, and professional.
If the answer is not in the context, say "I don't have that information about Himanshu."

Current date and time is {now}. Use this to calculate age, total experience, or any time-based questions.

Retrieved Context:
{{context}}
"""

    prompt = ChatPromptTemplate.from_messages([
        ('system', system_prompt),
        MessagesPlaceholder(variable_name = 'chat_history'),
        ('human', '{input}')
    ])
    
    question_answer_chain = create_stuff_documents_chain(model, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain, retriever, system_prompt
    
def main():
    st.title("🤖 Himanshu's Bot ")
    st.caption("Powered by RAG · Anthropic · Pinecone · HuggingFace")
    
    with st.spinner('🔧 Setting up knowledge base...'):
        vector_store = build_vector_store()
        rag_chain, retriever, system_prompt_text = build_rag_chain(vector_store)

    
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
                result = rag_chain.invoke(
                    {
                        'input': prompt,
                        'chat_history': chat_history
                    }
                )
                response = result['answer']
                st.markdown(response)
        
        # DEBUGGING: Show retrieved context and exact prompt sent to LLM
        '''
        with st.expander("🔍 Debug: What the AI retrieved"):
            if "context" in result:
                docs_with_scores = vector_store.similarity_search_with_score(prompt, k=10)
                for doc, score in docs_with_scores:
                    st.write(f"✅ **Similarity Score:** {round(score, 4)}")
                    st.write(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                    st.write(doc.page_content)
                    st.divider()
            else:
                st.write("No context was retrieved for this query.")

        with st.expander("📋 Debug: Exact Prompt Sent to LLM"):
            retrieved_docs = retriever.invoke(prompt)
            context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
            
            chat_history_text = ""
            for msg in chat_history:
                if isinstance(msg, HumanMessage):
                    chat_history_text += f"Human: {msg.content}\n"
                elif isinstance(msg, AIMessage):
                    chat_history_text += f"AI: {msg.content}\n"
            
            full_prompt = f"""{system_prompt_text.replace('{context}', context_text)}
        
CHAT HISTORY:
{chat_history_text if chat_history_text else "(none)"}

HUMAN: {prompt}"""
    
            st.text_area("Prompt", value=full_prompt, height=400)
        '''

        st.session_state.messages.append({'role': 'assistant', 'content': response})

if __name__ == "__main__":
    main()
