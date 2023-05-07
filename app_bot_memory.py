import os 
from apikey import apikey 
import openai 
import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory


llm = OpenAI(temperature=0.5)

# Set up ConversationChain
conversation = ConversationChain(
    llm=llm, 
    verbose=True, 
    memory=ConversationBufferMemory()
)

# Set up OpenAI API key
openai.api_key = apikey

# Define function to generate response
def generate_response(prompt):
    completions = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )
    message = completions.choices[0].text
    return message 

# Set up Streamlit app
st.title("👩‍⚕️🔎 Symptom Checker by Phoenix")
user_input = st.text_input("Are you not feeling well? Describe your symptoms:","")

# If user input is not empty, generate response and add to conversation chain
if user_input:
    response = generate_response(user_input)
    conversation_chain.add(user_input, response)

# Get previous conversation from conversation chain and display in reverse order
previous_conversation = conversation_chain.get_previous_conversation()
for i in range(len(previous_conversation)-1, -1, -1):
    message(previous_conversation[i]['response'], key=str(i))
    message(previous_conversation[i]['input'], is_user=True, key=str(i) + '_user')
    