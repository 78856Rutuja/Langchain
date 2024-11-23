import os
from dotenv import load_dotenv
from langchain_ollama.llms import OllamaLLM  # Ensure this is the correct class for your library

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Langsmith tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the question asked"),
        ("user", "Question:{question}")
    ]
)

# Streamlit framework
st.title("Langchain Demo with Gemma Model")
input_text = st.text_input("What question do you have in mind?")

# Ollama LLM (corrected initialization)
llm = OllamaLLM(model="gemma2:2b")  # Adjust parameters if needed
Output_Parser = StrOutputParser()
chain = prompt | llm | Output_Parser

# Handling input and responses
if input_text:
    st.write(f"Input Question: {input_text}")  # Debugging line
    try:
        response = chain.invoke({"question": input_text})
        st.write(response)
    except Exception as e:
        st.write(f"Error: {e}")  # Display any errors encountered during invocation
