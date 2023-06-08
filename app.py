import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, FakeEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import pinecone
import os

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or "PINECONE_API_KEY"
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT") or "PINECONE_ENVIRONMENT"

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENVIRONMENT
)
index = pinecone.Index("fjgpt")

def get_pdf_text(pdf_docs):
    text = ""
    namespace = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
        namespace = pdf.name.split('/')[-1].split('.')[0]
    return text, namespace


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks, namespace):
    embeddings = OpenAIEmbeddings()
    index_name = "fjgpt"
    # embeddings = FakeEmbeddings(size=1536)
    vectorstore = Pinecone.from_texts(texts=text_chunks, embedding=embeddings, index_name=index_name, namespace=namespace)
    return vectorstore

def get_existing_vectorstore(namespace):
    embeddings = OpenAIEmbeddings()
    index_name = "fjgpt"
    # embeddings = FakeEmbeddings(size=1536)
    vectorstore = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings, namespace=namespace)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="tiiuae/falcon-40b", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    container = st.container()
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            container.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            container.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with FJ Knowledge Base", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with FJ Knowledge Base")
    user_question = st.text_input("Ask a question about the product:", key="user_question")
    if user_question:
        handle_userinput(user_question)

    def disable():
        st.session_state.disabled = True
    if "disabled" not in st.session_state:
        st.session_state.disabled = False
    
    @st.cache_data
    def populate():
        manuals_list = list(index.describe_index_stats().namespaces.keys())
        return manuals_list

    def clear_question():
        st.session_state["user_question"] = ''

    with st.sidebar:
        st.subheader("Your manual")
        doc_names = populate()
        previous_doc_name = st.session_state.doc_name if "doc_name" in st.session_state else ''
        doc_name = st.selectbox("Select Product", options=doc_names, on_change=clear_question)
        st.session_state.doc_name = doc_name 
        if doc_name != previous_doc_name:
            vectorstore = get_existing_vectorstore(doc_name)
            st.session_state.chat_history = None
            st.session_state.conversation = None
            st.session_state.conversation = get_conversation_chain(vectorstore)
        previous_doc_name = doc_name

        pdf_docs = st.file_uploader(
            "Upload your product manual PDF here and click on 'Process'", accept_multiple_files=True)
        for pdf in pdf_docs:
                if pdf.name.split('/')[-1].split('.')[0] in doc_names:
                    disable()
        if st.button("Process", on_click=disable, disabled=st.session_state.disabled):
            
            with st.spinner("Processing"):
                # get pdf text
                raw_text, doc_namespace = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks, doc_namespace)
                # # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
        
if __name__ == '__main__':
    main()
