import streamlit as st
import os
from io import BytesIO
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv(override=True)
prompt_template = """
Answer the question based on the context below. 
the context is delimited by <context> and </context> tags. the question is delimited by <question> and </question> tags.
If the question can't be answered based on the context, say "I don't know".

<context> {context} </context> 
<question>{question} </question> 
"""

llm = ChatOpenAI(
    model="liquid/lfm-2.5-2.6b:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    temperature=0.0,
)

def main():
    st.set_page_config(page_title="RAG Demo",layout="wide", page_icon=":robot_face:")
    st.subheader("RAG (Retrieval-Augmented Generation) Demo",divider="blue")
    

    with st.sidebar:
        st.sidebar.title("Data loader")
        st.image("https://raw.githubusercontent.com/special-space-computing-machine-6x47jrw4qjwfjgg/RAG/main/RAG.png", use_column_width=True)   
        pdf_docs = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)
        if st.button("submit"):
            if not pdf_docs:
                st.error("Upload at least one PDF before submitting.")
            else:
                with st.spinner("Processing..."):
                    documents = []
                    for pdf in pdf_docs:
                        reader = PdfReader(BytesIO(pdf.getvalue()))
                        for page_number, page in enumerate(reader.pages):
                            page_text = page.extract_text() or ""
                            if page_text.strip():
                                documents.append(
                                    Document(
                                        page_content=page_text,
                                        metadata={"source": pdf.name, "page": page_number},
                                    )
                                )

                    if not documents:
                        st.error("No extractable text was found in the uploaded PDFs.")
                    else:
                        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                            chunk_size=1000, chunk_overlap=200
                        )
                        chunks = text_splitter.split_documents(documents)

                        embedding_model = OpenAIEmbeddings(
                            model="text-embedding-ada-002",
                            base_url="https://openrouter.ai/api/v1",
                            api_key=os.environ["OPENROUTER_API_KEY"],
                        )
                        vectorstore = Chroma.from_documents(
                            chunks, embedding_model, collection_name="data_collection"
                        )
                        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

                        st.session_state.retriever = retriever
    st.subheader("Chatbot")
    user_question = st.text_input("Ask your question:")
    if user_question:
        context_doc = st.session_state.retriever.invoke(user_question)
        context_list = [doc.page_content for doc in context_doc]
        context_for_query = "\n".join(context_list)
        final_prompt = prompt_template.format(context=context_for_query, question=user_question)
        response = llm.invoke(final_prompt)
        st.write(response.content)


if __name__ == "__main__":
    main()