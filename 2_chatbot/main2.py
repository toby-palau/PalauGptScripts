import streamlit as st
import time
import chromadb
from io import StringIO
from langchain.chains import RetrievalQA
# from langchain_together.embeddings import TogetherEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.agents import Tool, AgentType, initialize_agent
from langchain.llms import Together
from langchain.document_loaders import (PyPDFLoader, JSONLoader)
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import HumanMessage, SystemMessage, AIMessage



# def metadata_func(record: dict, metadata: dict) -> dict:
#     metadata["chapterTitle"] = record.get("chapterTitle")
#     metadata["reference"] = record.get("reference")
#     return metadata


def main():
    # Initialize LLM
    llm = Together(
        together_api_key="73d9504f61ce7f7b552568845901023587a3728c25cc04849d0f0c5e276a5d34", 
        # model="mistralai/Mistral-7B-Instruct-v0.2"
        model="mistralai/Mixtral-8x7B-Instruct-v0.1"
        # model="togethercomputer/Llama-2-7B-32K-Instruct"
        # model="openchat/openchat-3.5-1210"
        # model="togethercomputer/GPT-NeoXT-Chat-Base-20B"
    )
    st.title("ChatGPT-like clone")
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello, I am the assistant."}
        ]
    print(st.session_state.messages)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            chat_history = "\n".join([f'{m["role"]}: {m["content"]}' for m in st.session_state.messages])
            message_placeholder = st.empty()
            response =  llm(
                prompt=f"""
                    You are an assistant and you are having a chat conversation with the user. You can read the chat history below. 
                    
                    [INST]
                    Now you're going to respond to the user's latest message with a response from the assistant. First write down your thoughts and decissions. Then your response as assistant. 
                    Never predict any messages that the user would send. Never write the identifiers "assistant:" or "user:" in your responses. 
                    [\INST]

                    Chat History:
                    {chat_history}
                """,
            )
            message_placeholder.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    
if __name__ == '__main__':
    main()