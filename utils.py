from __future__ import annotations
import re
import os
from pathlib import Path
import ast
import pandas as pd
from io import BytesIO
from typing import Any, Dict, List

import docx2txt
import streamlit as st
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import VectorStore
from langchain.vectorstores.faiss import FAISS
from langchain.document_loaders import CSVLoader
from langchain.chains import RetrievalQA
from langchain.agents import create_csv_agent
from openai.error import AuthenticationError
from pypdf import PdfReader
import fitz

from embeddings import OpenAIEmbeddings
from prompts import STUFF_PROMPT, template
import config

def clear_all_cache():
    with st.spinner("Clearing all previous cache 🍪"):
        st.cache_data.clear()
        #st.write('')
    return

#@st.experimental_memo()
@st.cache_data()
def parse_docx(file: BytesIO) -> str:
    clear_all_cache()
    text = docx2txt.process(file)
    # Remove multiple newlines
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text


def preprocess(text):
    text = text.replace('\n', ' ')
    text = re.sub('\s+', ' ', text)
    return text

#@st.experimental_memo()
@st.cache_data()
def parse_pdf2(file: BytesIO) -> List[str]:

    pdf = fitz.open(stream=file.read(), filetype="pdf")
    total_pages = pdf.page_count
    
    st.markdown("""
    Document has {temp} pages  
    """.format(temp=total_pages))
    #st.text(pdf.metadata)
    
    doc = ""
    text = ""
    for page in pdf:  # iterate the document pages
        doc = doc + page.get_text().encode("utf8").decode()  # get plain text (is in UTF-8)
        #st.write(bytes((12,)))  # write page delimiter (form feed 0x0C)
    #doc = text
    #st.write(doc)
    #pdf.close()
    return doc

#@st.experimental_memo()
@st.cache_data()
def parse_pdf(file: BytesIO) -> List[str]:
    clear_all_cache()
    pdf = PdfReader(file)
    meta = pdf.metadata
    
    st.markdown("""
    Document has {temp} pages  
    """.format(temp=len(pdf.pages)))
           
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        # Merge hyphenated words
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Fix newlines in the middle of sentences
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)
        output.append(text)
    return output

#@st.experimental_memo()
@st.cache_data()
def parse_txt(file: BytesIO) -> str:
    clear_all_cache()
    text = file.read().decode("utf-8")
    # Remove multiple newlines
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text

#@st.experimental_memo()
@st.cache_data()
def parse_csv(fileIn):
    file_details = {"filename":fileIn.name, "filetype":fileIn.type, "filesize":fileIn.size}
    st.write(file_details)

    # Save uploaded file to 'content' folder.
    save_folder = 'data\csv_content'
    save_path = Path(save_folder, fileIn.name)
    with open(save_path, mode='wb') as w:
        w.write(fileIn.getvalue())

    if save_path.exists():
        st.success(f'File {fileIn.name} is successfully saved!')
    
    loader = CSVLoader(file_path=os.path.join('data\csv_content\\', fileIn.name))

    return loader

#@st.cache(allow_output_mutation=True)
@st.cache_data()
def text_to_docs(text: str | List[str]) -> List[Document]:
    """Converts a string or list of strings to a list of Documents
    with metadata."""
    if isinstance(text, str):
        # Take a single string as one page
        text = [text]
    page_docs = [Document(page_content=page) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # Split pages into chunks
    doc_chunks = []

    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=config.chunk_overlap,
        )
        #print("Splitting the uploaded document into chunks of size: ", config.chunk_size)
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            # Add sources a metadata
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)
    return doc_chunks

#@st.cache(allow_output_mutation=True, show_spinner=False)
@st.cache_data()
def embed_docs(_docs: List[Document]) -> VectorStore:
    """Embeds a list of Documents and returns a FAISS index"""
    docsp = _docs
    
    if(config.VERBOSE_MODE == "Enable"):
        with st.expander("Document Chunks"):
            st.write(docsp)
    
    if not st.session_state.get("OPENAI_API_KEY"):
        raise AuthenticationError(
            "Enter your OpenAI API key in the sidebar. You can get a key at"
            " https://platform.openai.com/account/api-keys."
        )
    else:
        # Embed the chunks
        embeddings = OpenAIEmbeddings(
            openai_api_key=st.session_state.get("OPENAI_API_KEY")
        )  # type: ignore
        #st.code(embeddings)
        index = FAISS.from_documents(docsp, embeddings)
        #st.write(index)
        return index


#@st.cache(allow_output_mutation=True)
@st.cache_data
def search_docs(_index: VectorStore, query: str) -> List[Document]:
    """Searches a FAISS index for similar chunks to the query
    and returns a list of Documents."""
    indexp = _index
    # Search for similar chunks
    docs = indexp.similarity_search(query, k=5)
    if(config.VERBOSE_MODE == "Enable"):
        with st.expander("Similar Chunks from ths Vector Index"):
            st.write(docs)
    return docs

def construct_prompt(question: str, context_embeddings: dict) -> str:    
    chosen_sections = []
    df = context_embeddings

    for row in df:
        #row = row.replace("\n", " ")
        chosen_sections.append(str(row))
        #st.write(type(row))
    #chosen_sections = str(chosen_sections)
    #st.write(template + "".join(chosen_sections) + "\n\n" + question)
    return template + "".join(str(context_embeddings)) + "\n\n" + question


#@st.cache(allow_output_mutation=True)
@st.cache_data()
def get_answer(_docs: List[Document], query: str) -> Dict[str, Any]:
    """Gets an answer to a question from a list of Documents."""
    #print("in get_answer")
    docsp = _docs
    
    # Get the answer
    prompt = ""
    #st.write(construct_prompt(query, docsp))
    chain = load_qa_with_sources_chain(        
        OpenAI(
            temperature=config.temperature, openai_api_key=st.session_state.get("OPENAI_API_KEY"), max_tokens=config.max_tokens, verbose=True, 
        ),  # type: ignore
        chain_type="stuff",
        prompt=STUFF_PROMPT,
        #prompt=(construct_prompt(query, docsp)),
    )
    if(config.VERBOSE_MODE == "Enable"):
        with st.expander("Chain Information"):
            st.write(chain)
            st.write(prompt)
    # Cohere doesn't work very well as of now.
    # chain = load_qa_with_sources_chain(
    #     Cohere(temperature=0), chain_type="stuff", prompt=STUFF_PROMPT  # type: ignore
    # )
        
    answer = chain(
        {"input_documents": docsp, "question": query}, return_only_outputs=True,
    )
    #if(config.VERBOSE_MODE == "Enable"):
    #    st.code(docsp)
    #    st.code(chain)
    return answer


#@st.cache(allow_output_mutation=True)
@st.cache_data()
def get_sources(_answer: Dict[str, Any], _docs: List[Document]) -> List[Document]:
    """Gets the source documents for an answer."""
    answerp = _answer
    docsp =  _docs
    # Get sources for the answer
    source_keys = [s for s in answerp["output_text"].split("SOURCES: ")[-1].split(", ")]

    source_docs = []
    for doc in docsp:
        if doc.metadata["source"] in source_keys:
            source_docs.append(doc)

    return source_docs


def wrap_text_in_html(text: str | List[str]) -> str:
    """Wraps each text block separated by newlines in <p> tags"""
    if isinstance(text, list):
        # Add horizontal rules between pages
        text = "\n<hr/>\n".join(text)
    return "".join([f"<p>{line}</p>" for line in text.split("\n")])
