import streamlit as st
from PIL import Image


##### Introduction
image = Image.open('howitworks.png')
st.image(image, use_column_width='always', width=50, caption='This app demonstrates how to use OpenAI\'s GPT-3 to answer questions on your own document, by using document embeddings and retrieval.')
st.markdown(""" 
            
            ##### Introduction
This app demonstrates how to use OpenAI\'s GPT-3 to answer questions on your own document, by using document embeddings and retrieval. When you upload a document, it will be divided into smaller chunks and stored in a special type of database called a vector index that allows for semantic search and retrieval. When you ask a question, the app will search through the document chunks and find the most relevant ones using the vector index. Then, it will use GPT3 to generate a final answer.

The steps are:
1. On the uploaded document, pre-process the contextual information by splitting it into chunks.
2. Use GPT to create an embedding vector for each chunk.
3. Store the embeddings in a vector DB.
4. User asks a query in a natural laguage format.
5. On receiving a query, use GPT to embed the query in the same vector space as the context chunks.
6. Find the context embeddings which are most similar to the query. Prepend the most relevant context embeddings to the query prompt.
7. Submit the question along with the most relevant context to GPT, and receive a generated answer which makes use of the provided contextual information.
            """)


st.markdown(
        """
##### Language Model
<p class="small-font">
            A language model is a deep learning model that uses probabilistic methods to predict the likelihood of certain sequences occurring. 
            Often, this refers to the sequence of words occurring in a sentence but can be extended to things such as protein structures.

##### Generative AI
<p class="small-font">
Generative AI (GenAI) is a class of machine learning (ML) algorithms that can learn from content such as text, images, and audio
in order to generate new content. When compared to traditional / discriminative ML algorithms, GenAI models produce artifacts as output, which can have a wide range of variety and complexity.

##### Parameters
<p class="small-font">    
Parameters are the values that a model has learned during the training process. Think of a parameter as a string of related information such as “Ben Franklin born January 1706” or “Boston capital of Massachusetts.”
A model's parameter count essentially determines how skilled it is in solving a problem or query – it makes sense that the more information it has the better it can perform.
           
##### Embeddings
<p class="small-font">    
An embedding is a numerical representation (vector) of a text. Embeddings allow us to perform semantic similarity searche to identify documents, or parts of documents, that match say an input text or question. The embeddings that are the numerical representations of texts encode meaning (see e.g., Distributional Semantics) allowing for comparison of texts by comparing their vector representations. In this app we are using Open AI to embed the uploaded document chunks. 
           
##### Temperature
<p class="small-font">
         The temperature controls the "creativity" of the model\'s responses. The value set between from 0 to 1 controls the randomness of the response.
         A higher temperature value (closer to 1) will result in more creative and unpredictable responses. A higher value means the model evaluates possible responses that could fit into the context before spitting out the result
         A lower temperature value (closer to 0) will result in more conservative and predictable responses. A lower value of temperature means the API will respond with the first thing that the model sees.''

##### Response Length
<p class="small-font">
         The response length sets a limit on how much text the API includes in its completion. Because OpenAI charges by the length of text generated per API call, response length is a crucial parameter for anyone on a budget. A higher response length will cost more.
               

##### Top-P
<p class="small-font">
             Top P controls how many random results the model should consider for completion, as suggested by the temperature dial, thus determining the scope of randomness. Top P’s range is from 0 to 1. A lower value limits creativity, while a higher value expands its horizons.
            

##### Frequency and Presence Penalty
<p class="small-font">
            The frequency penalty decreases the likelihood that the model will repeat the same line verbatim by “punishing” it. The presence penalty increases the likelihood that it will talk about new topics.
           

##### Best of
<p class="small-font">
            This parameter lets you specify the number of completions (n) to generate on the server-side and returns the best of “n” completions.
           

##### Stop Sequence
<p class="small-font">
A stop sequence is a set of characters that signals the API to stop generating completions.
''

##### Inject Start & Restart Text
<p class="small-font">
            The inject start text and inject restart text parameters allow you to insert text at the beginning or end of the completion, respectively.
           

##### Show Probabilities
<p class="small-font">
            This option lets you debug the text prompt by showing the probability of tokens that the model can generate for a given input.

##### Is my data safe?
<p class="small-font">
Yes, your data is safe. This app does not store your documents or
questions. All uploaded data is deleted after you close the browser tab.

##### Why does it take so long to index my document?
<p class="small-font">
If you are using a free OpenAI API key, it will take a while to index
your document. This is because the free API key has strict [rate limits](https://platform.openai.com/docs/guides/rate-limits/overview).
To speed up the indexing process, you can use a paid API key.

##### What do the numbers mean under each source?
<p class="small-font">
For a PDF document, you will see a citation number like this: 3-12. 
The first number is the page number and the second number is 
the chunk number on that page. For DOCS and TXT documents, 
the first number is set to 1 and the second number is the chunk number.

##### Are the answers 100% accurate?
<p class="small-font">
No, the answers are not 100% accurate. The app uses GPT-3 to generate
answers. GPT-3 is a powerful language model, but it sometimes makes mistakes 
and is prone to hallucinations. Also, the app uses semantic search
to find the most relevant chunks and does not see the entire document,
which means that it may not be able to find all the relevant information and
may not be able to answer all questions (especially summary-type questions
or questions that require a lot of context from the document).
"""
    , unsafe_allow_html=True)
