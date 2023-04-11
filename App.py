import sys
import os
import config
import base64
import streamlit as st
from openai.error import OpenAIError
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.indexes import VectorstoreIndexCreator
import fitz

import components
from components.sidebar import sidebar
from utils import (
    embed_docs,
    get_answer,
    get_sources,
    parse_docx,
    parse_pdf,
    parse_pdf2,
    parse_txt,
    parse_csv,
    search_docs,
    text_to_docs,
    wrap_text_in_html,
)


# All of Streamlit config and customization
st.set_page_config(page_title="Use Generative AI on your own Documents", page_icon=":random:", layout="wide")
st.markdown(""" <style>
#MainMenu {visibility: visible;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)
padding = 0

st.markdown(f""" <style>

    div.stButton > button {{
        
        cursor:pointer;outline:0;display:inline-block;font-weight:400;line-height:1.5;text-align:center;background-color:transparent;border:2px solid transparent;padding:6px 12px;font-size:1rem;border-radius:.25rem;transition:color .15s ease-in-out,background-color .15s ease-in-out,border-color .15s ease-in-out,box-shadow .15s ease-in-out;color:#bf470a;border-color:#bf470a; display:block; margin: 0 auto;
        
    }}
    div.stButton > button:hover {{
        color:#fff;background-color:#bf470a;border-color:#bf470a
    }}

    .small-font {{
        font-size: 9px !important;
    }}
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)


with open( "style.css" ) as css: st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)
    
sys.path.append('./components')

def clear_submit():
    st.session_state["submit"] = False
    #st.cache_data.clear()

#st.set_page_config(page_title="KnowledgeGPT", page_icon="📖", layout="wide")
st.header("✴️ GPT on your Document")

sidebar()

placeholder_upload = st.empty()
with placeholder_upload.container():
    uploaded_file = st.file_uploader(
        "Upload a pdf, docx, or txt file. Ask questions and interact.",
        type=["pdf", "docx", "txt", "csv"],
        help="Scanned documents are not supported yet!",
        on_change=clear_submit,
    )
TABULAR_INPUT_FILE_FLAG = False
index = None
doc = None
if uploaded_file is not None:
    st.cache_data.clear()
    if uploaded_file.name.endswith(".pdf"):
        doc = parse_pdf(uploaded_file)
        #st.write(type(doc))
    elif uploaded_file.name.endswith(".docx"):
        doc = parse_docx(uploaded_file)
        #st.write(type(doc))
    elif uploaded_file.name.endswith(".txt"):
        doc = parse_txt(uploaded_file)
        #st.write(type(doc))
    elif uploaded_file.name.endswith(".csv"):
        TABULAR_INPUT_FILE_FLAG = True
        doc = parse_csv(uploaded_file)
        st.session_state["api_key_configured"] = True
    else:
        raise ValueError("File type not supported!")
    
    if TABULAR_INPUT_FILE_FLAG is False:
        text = text_to_docs(doc)
        try:
            with st.spinner("Indexing document... This may take a while ⏳"):
                index = embed_docs(text)
            st.session_state["api_key_configured"] = True
        except OpenAIError as e:
            st.error(e._message)

placeholder_config = st.empty()
with placeholder_config.container():
    with st.expander("Options & Config"):
        show_all_chunks = st.checkbox("Show all chunks retrieved from vector search" )
        show_full_doc = st.checkbox("Show parsed contents of the document", value =False)

        #st.metric(label="Query Model", value=config.text_model)
        #st.metric(label="Embedding Model", value=config.embedding_model)

    if show_full_doc and doc:
        with st.expander("Source Document"):        
            if uploaded_file.name.endswith(".pdf"):
                base64_pdf = base64.b64encode(uploaded_file.read()).decode('utf-8')
                #pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)
            elif uploaded_file.name.endswith(".docx"):
                base64_docx = uploaded_file #base64.b64encode(uploaded_file.read()).decode('utf-8')
                word_display = f'<embed src="https://view.officeapps.live.com/op/embed.aspx?src={base64_docx}" width="100%">'
                st.markdown(word_display, unsafe_allow_html=True)
            else:
                # Hack to get around st.markdown rendering LaTeX
                st.markdown(f"<p>{wrap_text_in_html(doc)}</p>", unsafe_allow_html=True)

st.markdown("---")

#You can check .empty documentation
placeholder = st.empty()
with placeholder.container():
    query = st.text_area("Ask a question about the document", on_change=clear_submit, value="Provide a summary of this document in a few bullets")
    
    button = st.button("Submit")
    if button or st.session_state.get("submit"):
        if TABULAR_INPUT_FILE_FLAG is True:
            # Create an index using the loaded documents
            index_creator = VectorstoreIndexCreator()
            docsearch = index_creator.from_loaders([doc])
            # Create a question-answering chain using the index
            chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.vectorstore.as_retriever(), input_key="question")
            response = chain({"question": query})
            st.write(response['result'])
        elif not st.session_state.get("api_key_configured"):
            st.info(st.session_state.get("api_key_configured"))
            st.error("Please configure your OpenAI API key!")
        elif not index:
            st.error("Please upload a document!")
        elif not query:
            st.error("Please enter a question!")
        else:
            st.session_state["submit"] = True
            # Output Columns
            answer_col, sources_col = st.columns(2)
            sources = search_docs(index, query)
            try:
                answer = get_answer(sources, query)
                if not show_all_chunks:
                    # Get the sources for the answer
                    sources = get_sources(answer, sources)

                with answer_col:
                    st.markdown("#### Answer")
                    st.write(answer["output_text"].split("SOURCES: ")[0])

                with sources_col:
                    st.markdown("#### Sources")
                    st.caption("Page Number - Chunk Number")
                    for source in sources:
                        st.markdown(source.page_content)
                        st.caption(source.metadata["source"])
                        st.markdown("---")

            except OpenAIError as e:
                st.error(e._message)
