#Integrate our code with openAI
#You can follow documentation in langchain
#langchain supports openAI model and local open model like llama2 via APIs. Documentaion can be found on langchain website

import os
from constants import openai_key
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain_core.output_parsers import StrOutputParser

#We store the conversation using below
from langchain.memory import ConversationBufferMemory


#To use, you should have the openai python package installed, and the environment variable OPENAI_API_KEY set with your API key.

import streamlit as slit #streamlit is used when you are not worried about the UI

os.environ["OPENAI_API_KEY"] = openai_key

#initialize streamlit framework

slit.title('Celebrity Paparazzi')

#Create a input text box on the website
input_text = slit.text_input("Enter the topic to search")

#Prompt Templates

first_input_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a media head"),
    ("user", "Enlighten me about {input}")]  
)


#Initialize OPEN AI LLM

llm = ChatOpenAI(temperature=0.8,model_name="gpt-3.5-turbo") #temperature variable gives how much control the agent has while giving the answers. Range = 0 to 1

#Memory

person_memory = ConversationBufferMemory(input_key='input', memory_key='person_history')
dob_memory = ConversationBufferMemory(input_key='person',memory_key='dob_history')
achievement_memory = ConversationBufferMemory(input_key='person',memory_key='prize_history')
event_memory = ConversationBufferMemory(input_key='dob',memory_key='event_history')

#We can now combine these into a simple LLM chain:
chain = LLMChain(llm=llm,prompt=first_input_prompt,verbose=True,output_key='person',memory=person_memory)
#chain = first_input_prompt | llm 

#You can include an output parser to get a more presentable result
#output_parser = StrOutputParser()
#chain = first_input_prompt | llm | output_parser
second_input_prompt = ChatPromptTemplate.from_messages([
    ("system", "When was {person} born")]  
)

chain2 = LLMChain(llm=llm,prompt=second_input_prompt,verbose=True,output_key="dob",memory=dob_memory)


third_input_prompt = ChatPromptTemplate.from_messages([
    ("system", "What are {person} achievement")]  
)
chain3 = LLMChain(llm=llm,prompt=third_input_prompt,verbose=True,output_key="prize",memory=achievement_memory)

fourth_input_prompt = ChatPromptTemplate.from_messages([
    ("system", "What are major events happened in the world around {dob}")]  
)
chain4 = LLMChain(llm=llm,prompt=fourth_input_prompt,verbose=True,output_key="events",memory=event_memory)

parent_chain = SimpleSequentialChain(chains=[chain,chain2,chain3,chain4],input_key='input',output_key='dob',verbose=True)

if input_text:
    slit.write(parent_chain.run({"input":input_text}))

#chain.invoke is used execute the "chain" sequence only
#parent_chain.run is used to execute the sequential chain
    with slit.expander('About'):
        slit.info(person_memory.buffer)

    with slit.expander('Achievements'):
        slit.info(achievement_memory.buffer)

    with slit.expander('Major Events'):
        slit.info(event_memory.buffer)

