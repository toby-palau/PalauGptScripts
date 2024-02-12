import re
import streamlit as st
from pydantic import BaseModel, Field, validator
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.agents.tools import Tool
from langchain.agents import load_tools
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent

import os
from dotenv import load_dotenv
load_dotenv()


class DisclosureRequirement(BaseModel): 
    standard: str = Field(description="The standard that the disclosure requirement is part of")
    chapterId: str = Field(description="The chapter ID that the disclosure requirement is part of")
    chapterTitle: str = Field(description="The chapter title that the disclosure requirement is part of")
    paragraphId: str = Field(description="The paragraph ID that contains the disclosure requirement")
    content: str = Field(description="The content of the disclosure requirement")
    @validator("paragraphId")
    def paragraphId_must_be_valid(cls, value):
        if not re.match(r"^\d+$", value):
            raise ValueError("paragraphId must be a number")
        return value

parser = PydanticOutputParser(pydantic_object=DisclosureRequirement)
# prompt = ChatPromptTemplate(
#     messages=[
#         HumanMessagePromptTemplate.from_template("""
#             Instruction: Read the user input below and if the user requests you to return a requirement, look it up by the requirementId and return the content.
#             Format Instructions: {format_instructions}
#             User input: {question}
#         """)
#     ],
#     input_variables=["question"],
#     partial_variables={
#         "format_instructions": parser.get_format_instructions(),
#     },
# )

llm = ChatOpenAI(temperature=0, api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4")


tools = load_tools([], llm=llm)

csv_agent = create_csv_agent(
    llm=llm,
    path="./standards/ESRS_E1.csv",
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    pandas_kwargs = {"sep": ",", "encoding": "utf-8", "quotechar":'"', "escapechar":'\\', "comment": "#"},
    memory=ConversationBufferMemory()
)
csv_tool = Tool(
    name="CSV Agent",
    func=csv_agent.run,
    description="Use this tool when the user asks for a requirement from the ESRS E1 standard. Before doing anything, set the pandas setting max_colwidth to None so that the value in the content column is not truncated. Note that the requirementId column only contains strings. So when you try to lookup a requirementId, make sure that you convert the lookup value to a string first.", 
)
tools.extend([csv_tool])
agent = initialize_agent(
    llm=llm,
    tools=tools,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    handle_parsing_errors=True,
    memory=ConversationBufferMemory(),
)



st.set_page_config(page_title="🦜🔗 ESRS Retriever bot")
st.title('🦜🔗 ESRS Retriever bot')

def generate_response(input_text):
    # csv_agent_executor = create_csv_agent(
    #     llm,
    #     "./standards/ESRS_E1.csv",
    #     verbose=True
    # )
    # st.info(csv_agent_executor.run(
    #     f"""
    #         Instruction: Read the user input below and if the user requests you to return a requirement, look it up by the requirementId and return the content.
            
    #         User input: {input_text}            
    #     """
    # ))
    st.info(agent.run(
        f"""
            Instruction: Read the user input below and if the user requests you to return a requirement, look it up by the requirementId and return the content literally and in its entirity (don't paraphrase and don't truncate).
            User input: {input_text} 
            Output Format Instructions: {parser.get_format_instructions()}
        """
    ))

with st.form('my_form'):
    text = st.text_area('Enter text:')
    submitted = st.form_submit_button('Submit')
    # if not openai_api_key.startswith('sk-'):
    #     st.warning('Please enter your OpenAI API key!', icon='⚠')
    if submitted:
        generate_response(text)