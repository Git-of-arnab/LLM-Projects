#Integrate our code with openAI
#You can follow documentation in langchain
#langchain supports openAI model and local open model like llama2 via APIs. Documentaion can be found on langchain website

import os
import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain_core.output_parsers import StrOutputParser
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
#We store the conversation using below
from langchain.memory import ConversationBufferMemory


load_dotenv() #this will load all the environment variables defined in the .env file

#Initialize the google API key

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

#defining a function to read the text of the pdfs

def get_pdf_text(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text

#Define a function to divide the whole large text read from the above function into smaller chunks

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    text_chunks = text_splitter.split_text(text)

    return text_chunks

#Now, we will convert these text chunks and convert them into vectors

def get_chunk_vectors(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001') #embedding-001 is the model inside google generative ai
    vector_store = FAISS.from_texts(text_chunks,embedding=embeddings)
    #the vector_store can be stored in a database or local environment
    vector_store.save_local("faiss_index") #it will create a folder 'faiss_index' and store the vectors in the current local active directory

def get_conversational_chain():
    prompt_template ="""
    Answer the question as detailed as possible from the provided context, make sure to provide all the details including tables and images,
    if the answer is not in the provided context then just say ,"Answer is not available in the provided documents", don't provide wrong answer
    Context:\n{context}?\n
    Question:\n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)

    prompt = PromptTemplate(template=prompt_template,input_variables=['context','question'])
    #chain1 = load_qa_chain(model,chain_type="stuff",prompt=prompt) 
    chain1 = LLMChain(llm=model,prompt=prompt,verbose=True,output_key='summary',memory=summary_memory)

    second_input_prompt = ChatPromptTemplate.from_messages([
    ("system", "What are the procedures present in {summary}")] )

    chain2 = LLMChain(llm=model,prompt=second_input_prompt,verbose=True,output_key="procedures",memory=procedure_memory)

    third_input_prompt = ChatPromptTemplate.from_messages([
    ("system", "Give a summary of any one {procedures}")] )

    chain3 = LLMChain(llm=model,prompt=third_input_prompt,verbose=True,output_key="procsummary",memory=procsummary_memory)
    parent_chain = SimpleSequentialChain(chains=[chain1,chain2,chain3],input_key='input',output_key='procsummary',verbose=True)

    return parent_chain

#llm = ChatOpenAI(temperature=0.8,model_name="gpt-3.5-turbo") #temperature variable gives how much control the agent has while giving the answers. Range = 0 to 1

#Memory

summary_memory = ConversationBufferMemory(input_key='prompt', memory_key='summary_history')
procedure_memory = ConversationBufferMemory(input_key='summary',memory_key='procedure_history')
procsummary_memory = ConversationBufferMemory(input_key='procedures',memory_key='procsummary_history')

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index",embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain.run(
        {"input_documents":docs, "question":user_question},return_only_outputs=True
    )
    print(response)

    st.write("Reply:", response["output_text"])
    with st.expander('Summary'):
        st.info(summary_memory.buffer)

    with st.expander('Procedures'):
        st.info(procedure_memory.buffer)

    with st.expander('Procedure Summary'):
        st.info(procsummary_memory.buffer)
#defining our parameters of the streamlit app
    
def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with your PDFs")

    user_question = st.text_input("Ask a question from the PDF")

    #This section is to kickstart the code processing for answering the query as soon as it is entered in the question bar
    if user_question:
        user_input(user_question)

    #This module is to upload the PDF files,process it and create the FAISS index to use this for further actions 
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF files and click on SUBMIT",accept_multiple_files=True)

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_chunk_vectors(text_chunks)
                st.success("Processing Complete!")

#Execution
if __name__ == "__main__":
    main()

