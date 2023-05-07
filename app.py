# Bring in deps
import os 
from apikey import apikey 

import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 

os.environ['OPENAI_API_KEY'] = apikey

# App framework
st.title('👩‍⚕️🔎 Symptom Checker by Phoenix')
prompt = st.text_input('Are you not feeling well? Describe your symptoms:') 

# Prompt templates
template = PromptTemplate(
    input_variables = ['title'], 
    template='Find a disease with the following symptoms: {title}'
)

# script_template = PromptTemplate(
#     input_variables = ['title', 'wikipedia_research'], 
#     template='write me a youtube video script based on this title TITLE: {title} while leveraging this wikipedia reserch:{wikipedia_research} '
# )

# Memory 
# title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')


# Llms
llm = OpenAI(temperature=0.9, model_name="text-davinci-003") 
# title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=template, verbose=True, output_key='script', memory=script_memory)

# wiki = WikipediaAPIWrapper()

# Show stuff to the screen if there's a prompt
if prompt: 
    # title = title_chain.run(prompt)
    # wiki_research = wiki.run(prompt) 
    # script = script_chain.run(title=title, wikipedia_research=wiki_research)

    # script = script_chain.run(title="Random title", wikipedia_research = "Migraine (UK: /ˈmiːɡreɪn/, US: /ˈmaɪ-/)[12][13] is a common neurological disorder characterized by recurrent headaches.[1] Typically, the associated headache affects one side of the head, is pulsating in nature, may be moderate to severe in intensity, and could last from a few hours to three days.[1] Non-headache symptoms may include nausea, vomiting, and sensitivity to light, sound, or smell.[2] The pain is generally made worse by physical activity during an attack,[14] although regular physical exercise may prevent future attacks.[15] Up to one-third of people affected have aura: typically, it is a short period of visual disturbance that signals that the headache will soon occur.[14] Occasionally, aura can occur with little or no headache following, but not everyone has this symptom")

    script = script_chain.run(title="Random title")
    # st.write(title) 
    st.write(script) 

    # with st.expander('Title History'): 
    #     st.info(title_memory.buffer)

    with st.expander('Script History'): 
        st.info(script_memory.buffer)

    # with st.expander('Wikipedia Research'): 
    #     st.info(wiki_research)
