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


# def metadata_func(record: dict, metadata: dict) -> dict:
#     metadata["chapterTitle"] = record.get("chapterTitle")
#     metadata["reference"] = record.get("reference")
#     return metadata


def main():
    # Initialize LLM
    llm = Together(
        together_api_key="73d9504f61ce7f7b552568845901023587a3728c25cc04849d0f0c5e276a5d34", 
        # model="mistralai/Mistral-7B-Instruct-v0.2"
        model="mistralai/Mistral-7B-Instruct-v0.2"
        # model="togethercomputer/Llama-2-7B-32K-Instruct"
    )

    # vectorstore = chromadb.PersistentClient()

    # embeddings = TogetherEmbeddings(
    #     together_api_key="73d9504f61ce7f7b552568845901023587a3728c25cc04849d0f0c5e276a5d34",
    #     model="mistralai/Mistral-7B-v0.1"
    # )

    # docs = JSONLoader(
    #     file_path='./standards/reporting-standard.json',
    #     jq_schema='.requirements[]',
    #     content_key="content",
    #     metadata_func=metadata_func
    # ).load()
    # print(docs[0])

    # for doc in docs:
    #     esrs_store = Chroma.from_documents(client=vectorstore, collection_name="reporting-standard", documents=[doc], embedding=embeddings)
    #     time.sleep(1)

    # esrs_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=esrs_store.as_retriever(search_kwargs={"k": 1}))

    requirement = "The report should contain the word 'unaware'."
    # requirement = "The undertaking shall describe its policies adopted to manage its material impacts, risks and opportunities related to climate change mitigation and adaptation"
    # requirement = """
    #     The undertaking shall indicate whether and how its policies address the following areas:  
    #     (a) climate change mitigation;  
    #     (b) climate change adaptation;  
    #     (c) energy efficiency;   
    #     (d) renewable energy deployment; and   
    #     (e) other
    # """

    # Setup page
    st.set_page_config(page_title='Sustainability Report Analyser', page_icon=':mag:', layout='wide')
    st.header("Report Analyser")
    st.subheader("Upload your report")

    # Upload sustainability report
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is None:
        return
    
    st.spinner("Reading report...")
    
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    report = stringio.read()


    # texts = CharacterTextSplitter(
    #     # separators=["\n\n"],
    #     chunk_size = 600,
    #     chunk_overlap  = 100
    # ).split_text(text)
    # print(texts[0])

    # for doc in docs:
    #     print("requesting")
    #     report_store = Chroma.from_texts(client=vectorstore, collection_name="ecocorp-report", texts=texts, embedding=embeddings)
    #     time.sleep(10)

    # report_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=report_store.as_retriever())

    # tools = [
    #     Tool(
    #         name="Custom Palau Reporting Standard",
    #         func=esrs_chain.run,
    #         description="Contains all the requirements that a report needs to meet to adhere to the Custom Palau reporting standards.",
    #     ),
    #     Tool(
    #         name="Report EcoCorp",
    #         func=report_chain.run,
    #         description="Contains a report from company EcoCorp.",
    #     ),
    # ]
    # agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    st.spinner("Check requirement E1-1")
    
    response = llm(
    f"""
        Assess if the report section below meets the requirement below. If it meets the requirement, answer with "yes", if it doesn't meet the requirement, mention briefly what the problem is. Format your answer in a readable way.

        Requirement: 
        "{requirement}"

        Report section:
        "{report}"
    """
)


    # report_store = Chroma.from_documents(docs[0:10], embeddings)
    # report_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=report_store.as_retriever())

    # tools = [
    #     Tool(
    #         name="ESRS E1-1",
    #         func=esrs1_chain.run,
    #         description="Describes Disclosure requirement E1-1 for sustainability reports. All sustainability reports should meet this requirement. Input should be a fully formed question, not referencing any obscure pronouns from the conversation before.",
    #     ),
    #     Tool(
    #         name="O'Neil Sustainability report",
    #         func=report_chain.run,
    #         description="O'Neil's sustainability report from 2022. Input should be a fully formed question, not referencing any obscure pronouns from the conversation before.",
    #     ),
    # ]
    # agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    # agent.run("Does O'Neill's 2022 sustainability report meet requirement 15d in the disclousre requirement E1-1? Cite the corresponding section from O'Neil's sustainability report.")

    st.success("Done!")
    
    st.write(response)
    
if __name__ == '__main__':
    main()