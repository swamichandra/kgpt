# flake8: noqa
import streamlit as st
import config

#st.set_page_config(page_title=None, page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)

def faq():
    # model settings
    #models = ['text-davinci-003', 'gpt-3.5-turbo', 'text-curie-001']
    #st.selectbox("Query Model:", ('text-davinci-003', 'gpt-3.5-turbo', 'text-curie-001'), disabled=True)
    #st.text('Embedding Model: '+ config.embedding_model)
    #st.text('Query Model: ' + config.text_model)
    #st.selectbox("Embedding Model:", ('text-embedding-ada-002', ''), disabled=True)
    st.markdown("""
    **Embedding Model:** **:orange[{temp}]**  
    """.format(temp=config.embedding_model))
    #st.text(pdf.metadata)
    
    st.markdown("""
    **Query Model:** **:orange[{temp}]**
    """.format(temp=config.text_model))
    #st.text(pdf.metadata)
    
    radio_options = ["Enable", "Disable"]
    # verbose settings
    config.VERBOSE_MODE = st.radio("Verbose Mode", options=radio_options, horizontal=True, )  #index=radio_options.index(config.VERBOSE_MODE))
    #config.VERBOSE_MODE = st.select_slider("Verbose Mode:", radio_options)
        
    # chunk settings
    config.chunk_size = st.radio("Source Document Split Chunk Size", options=[600, 800, 1000], horizontal=True,)
    config.chunk_overlap = st.radio("Source Document Chunk Overlap", options=[0, 20], disabled=True, horizontal=True,)

    # temperature settings
    temp_range = [x / 10 for x in range(0, 11)]  #range(0, 1, 0.05)
    config.temperature = st.select_slider('Temperature', options=temp_range, value=config.temperature)

    # reply token settings
    reply_tokens_range = [x for x in range(200, 2000)]
    config.max_tokens = st.select_slider('Reply Tokens Size', options=reply_tokens_range, value=config.max_tokens)
    #st.markdown("---")