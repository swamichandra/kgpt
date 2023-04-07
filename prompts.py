# flake8: noqa
from langchain.prompts import PromptTemplate
import streamlit as st

## Use a shorter template to reduce the number of tokens in the prompt
template = """Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES"). Answer the question truthfully using the provided context. If you are not sure of the answer, say "Sorry, I don't know". Do not attempt to fabricate an answer and leave the SOURCES section empty. Create a final answer to the given questions using the provided document excerpts (in no particular order) as references. ALWAYS include a "SOURCES" section in your answer including only the minimal set of sources needed to answer the question.
{summaries}
---------

QUESTION: {question}

=========
FINAL ANSWER:
SOURCES:
    """

STUFF_PROMPT = PromptTemplate(
    template=template, input_variables=["summaries", "question"]
)