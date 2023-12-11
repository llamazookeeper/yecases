from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.chat_models import ChatOpenAI
# from llama_index import VectorStoreIndex, SimpleDirectoryReader
# from llama_index.vector_stores import WeaviateVectorStore
from llama_index.schema import Document
from llama_index.llms import OpenAI
# from llama_index.node_parser import SimpleNodeParser


# from dotenv import dotenv_values
from pypdf import PdfReader
import streamlit as st
import requests
import time
import json
import plotly.graph_objects as go
from pydantic import create_model
from langchain.llms import OpenAI
import os

import logging
from logger_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)


USER_ID = 'openai'
APP_ID = 'chat-completion'
MODEL_ID = 'GPT-4'
MODEL_VERSION_ID = '4aa760933afa4a33a0e5b4652cfa92fa'

def get_model(model_name):
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    if model_name == "openai":
        model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo")
    return model

def process_pdf(pdfs):
    docs = []

    for pdf in pdfs:
        file = PdfReader(pdf)
        text = ""
        for page in file.pages:
            text += str(page.extract_text())
        # docs.append(Document(TextNode(text)))

    text_splitter = CharacterTextSplitter(separator="\n",
    chunk_size=2000,
    chunk_overlap=300,
    length_function=len)
    docs = text_splitter.split_documents(docs)
    # docs = text_splitter.split_text(text)

    return docs

def process_pdf2(pdf):
    file = PdfReader(pdf)
    text = ""
    for page in file.pages:
        text += str(page.extract_text())

    doc = Document(text=text)
    return [doc]


def faiss_db(splitted_text):
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = FAISS.from_texts(splitted_text, embeddings)
    db.save_local("faiss_db")
    return db

def safe_float(value):
        if value == "None" or value == None:
            return "N/A"
        return float(value)

def round_numeric(value, decimal_places=2):
    if isinstance(value, (int, float)):
        return round(value, decimal_places)
    elif isinstance(value, str) and value.replace(".", "", 1).isdigit():
        # Check if the string represents a numeric value
        return round(float(value), decimal_places)
    else:
        return value


def generate_pydantic_model(fields_to_include, attributes, base_fields):
    selected_fields = {attr: base_fields[attr] for attr, include in zip(attributes, fields_to_include) if include}

    return create_model("DynamicModel", **selected_fields)

def insights(insight_name, type_of_data, data, output_format):

    with open("prompts/iv2.prompt", "r") as f:
        template = f.read()


    prompt = PromptTemplate(
        template=template,
        input_variables=["insight_name","type_of_data","inputs", "output_format"],
        # partial_variables={"output_format": parser.get_format_instructions()}
    )

    model = get_model("openai")

    data = json.dumps(data)

    formatted_input = prompt.format(insight_name=insight_name,type_of_data=type_of_data, inputs=data, output_format=output_format)
    # print("-"*30)
    # print("Formatted Input:")
    # print(formatted_input)
    # print("-"*30)

    response = model.predict(formatted_input)
    return response










