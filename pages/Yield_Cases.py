import torch
from transformers import BitsAndBytesConfig
from llama_index.prompts import PromptTemplate
from llama_index.llms import HuggingFaceLLM
from llama_index import ServiceContext, VectorStoreIndex , SummaryIndex

import streamlit as st
import faiss
import time
from pypdf import PdfReader

from huggingface_hub import delete_file
from pathlib import Path
import requests
import os

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)


def messages_to_prompt(messages):
  prompt = ""
  for message in messages:
    if message.role == 'system':
      prompt += f"<|system|>\n{message.content}</s>\n"
    elif message.role == 'user':
      prompt += f"<|user|>\n{message.content}</s>\n"
    elif message.role == 'assistant':
      prompt += f"<|assistant|>\n{message.content}</s>\n"

  # ensure we start with a system prompt, insert blank if needed
  if not prompt.startswith("<|system|>\n"):
    prompt = "<|system|>\n</s>\n" + prompt

  # add final assistant prompt
  prompt = prompt + "<|assistant|>\n"

  return prompt

model_name="mistralai/Mistral-7B-v0.1"
#model_name="HuggingFaceH4/zephyr-7b-beta"

llm = HuggingFaceLLM(
    model_name=model_name,
    tokenizer_name=model_name,
    query_wrapper_prompt=PromptTemplate("<|system|>\n</s>\n<|user|>\n{query_str}</s>\n<|assistant|>\n"),
    context_window=3900,
    max_new_tokens=256,
    offload_folder="/notebooks/yecases/offload",
    #model_kwargs={"quantization_config": quantization_config},
    # tokenizer_kwargs={},
    generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
    messages_to_prompt=messages_to_prompt,
    device_map="auto",
)



st.set_page_config(page_title="Yield Case Analyzer", page_icon=":card_index_dividers:", initial_sidebar_state="expanded", layout="wide")

st.title(":card_index_dividers: Yield Case Analyzer")
st.info("""
Begin by uploading the case report in PDF format. Afterward, click on 'Process Document'. Once the document has been processed. You can enter question and click send, system will answer your question.
""")

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
            from llama_index import download_loader, SimpleDirectoryReader
            from llama_hub.file.pymu_pdf.base import PyMuPDFReader

            pdfs = st.sidebar.file_uploader("Upload the case report in PDF format", type="pdf")
            pdf_dir = './data'
            if not os.path.exists(pdf_dir):
                os.makedirs(pdf_dir)

            for pdf in pdfs:
                print(f'file named {pdf}')
                fname=f'{pdf_dir}/{pdf}'
                with open(fname, 'wb') as f:
                    f.write(pdf.read())


            def fmetadata(dummy: str):
                return None

            PyMuPDFReader = download_loader("PyMuPDFReader")
            loader =  SimpleDirectoryReader(input_dir=pdf_dir, file_extractor={".pdf": PyMuPDFReader(),}, file_metadata=fmetadata)
            documents = loader.load_data()

            service_context = ServiceContext.from_defaults(llm=llm, embed_model="local:BAAI/bge-small-en-v1.5")
            vector_index = VectorStoreIndex.from_documents(documents, service_context=service_context)
            summary_index = SummaryIndex.from_documents(documents, service_context=service_context)

        st.toast("Document Processsed!")

    #st.session_state.process_doc = True

    if st.session_state.process_doc:
        search_text = st.text_input("Enter your question")
        if st.button("Submit"):
            start_time = time.time()

            with st.status("**Analyzing Report...**"):
                st.write("Case search result...")
                query_engine = vector_index.as_query_engine(response_mode="compact")
                response = query_engine.query(search_text)
                st.session_state["end_time"] = "{:.2f}".format((time.time() - start_time))

                st.toast("Report Analysis Complete!")

        if st.session_state.end_time:
            st.write("Report Analysis Time: ", st.session_state.end_time, "s")

