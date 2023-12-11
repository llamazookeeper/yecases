from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser

from llama_index import VectorStoreIndex, ServiceContext, StorageContext
from llama_index.vector_stores import FaissVectorStore
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.query_engine import SubQuestionQueryEngine
from llama_index.embeddings import OpenAIEmbedding
from llama_index.schema import Document
from llama_index.node_parser import UnstructuredElementNodeParser

from src.utils import get_model, process_pdf2

import streamlit as st
import os
import faiss
import time
from pypdf import PdfReader


st.set_page_config(page_title="Yield Case Analyzer", page_icon=":card_index_dividers:", initial_sidebar_state="expanded", layout="wide")

st.title(":card_index_dividers: Yield Case Analyzer")
st.info("""
Begin by uploading the case report in PDF format. Afterward, click on 'Process Document'. Once the document has been processed. You can enter question and click send, system will answer your question.
""")

def process_pdf(pdf):
    file = PdfReader(pdf)
    print("in process pdf")
    document_list = []
    for page in file.pages:
        document_list.append(Document(text=str(page.extract_text())))
    print("in process pdf 1")

    node_paser = UnstructuredElementNodeParser()
    print("in process pdf 1")

    nodes = node_paser.get_nodes_from_documents(document_list, show_progress=True)

    return nodes


def get_vector_index(nodes, vector_store):
    print(nodes)
    llm = get_model("openai")
    if vector_store == "faiss":
        d = 1536
        faiss_index = faiss.IndexFlatL2(d)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        # embed_model = OpenAIEmbedding()
        # service_context = ServiceContext.from_defaults(embed_model=embed_model)
        service_context = ServiceContext.from_defaults(llm=llm)
        index = VectorStoreIndex(nodes,
            service_context=service_context,
            storage_context=storage_context
        )
    elif vector_store == "simple":
        index = VectorStoreIndex.from_documents(nodes)


    return index



def generate_insight(engine, search_string):

    with open("prompts/report.prompt", "r") as f:
        template = f.read()

    prompt_template = PromptTemplate(
        template=template,
        input_variables=['search_string']
    )

    formatted_input = prompt_template.format(search_string=search_string)
    print(formatted_input)
    response = engine.query(formatted_input)
    return response.response


def get_query_engine(engine):
    llm = get_model("openai")
    service_context = ServiceContext.from_defaults(llm=llm)

    query_engine_tools = [
        QueryEngineTool(
            query_engine=engine,
            metadata=ToolMetadata(
                name="Case Report",
                description=f"Provides information about the cases from its case report.",
            ),
        ),
    ]


    s_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=query_engine_tools,
        service_context=service_context
    )
    return s_engine


if "process_doc" not in st.session_state:
        st.session_state.process_doc = False


OPENAI_API_KEY = "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"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


if OPENAI_API_KEY:
    pdfs = st.sidebar.file_uploader("Upload the case report in PDF format", type="pdf")
    st.sidebar.info("""
    Example pdf reports you can upload here:
    """)

    if st.sidebar.button("Process Document"):
        with st.spinner("Processing Document..."):
            nodes = process_pdf(pdfs)
            #st.session_state.index = get_vector_index(nodes, vector_store="faiss")
            st.session_state.index = get_vector_index(nodes, vector_store="simple")
            st.session_state.process_doc = True
        st.toast("Document Processsed!")

    #st.session_state.process_doc = True

    if st.session_state.process_doc:
        search_text = st.text_input("Enter your question")
        if st.button("Submit"):
            engine = get_query_engine(st.session_state.index.as_query_engine(similarity_top_k=3))
            start_time = time.time()

            with st.status("**Analyzing Report...**"):
                st.write("Case search result...")
                response = generate_insight(engine, search_text)
                st.session_state["end_time"] = "{:.2f}".format((time.time() - start_time))

                st.toast("Report Analysis Complete!")

        if st.session_state.end_time:
            st.write("Report Analysis Time: ", st.session_state.end_time, "s")

