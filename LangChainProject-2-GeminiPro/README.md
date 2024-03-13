# LLM Project using GeminiPro 

* This tool helps the user to upload custom PDF documents for the agent to learn and converse with it using the information

## User Interface

![image](https://github.com/Git-of-arnab/LLM-Projects/assets/138995898/185d8e06-7b89-4f83-ac12-fbfae195191d)

## What does the tool do?

![image](https://github.com/Git-of-arnab/LLM-Projects/assets/138995898/a1a517b2-6a9b-4f42-b12c-43870e6e4748)

## Loading and Pre-processing of Text data from PDF

* Load multiple PDFs on the tool from local machine
* Read the PDF using the PyPDF library of the Python
* Split the texts into chunks for each page in each document
* Embed the chunks into vectors and store it using FAISS vector store

![image](https://github.com/Git-of-arnab/LLM-Projects/assets/138995898/61bc9043-35d1-40e4-a77c-705699343d82)

## Query and Answer using the learned Data

* User provides the query in the query bar in the tool
* The query is then passed through the FAISS index for similarity search
* The relevant information are then passed through a langchain
* The langchain here is the prompt + LLM model (gemini-pro)
* The response is printed as output

![image](https://github.com/Git-of-arnab/LLM-Projects/assets/138995898/291d2428-3bbb-4364-a301-dce7a59f7cce)


