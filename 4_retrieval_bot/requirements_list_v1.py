import pandas as pd
import streamlit as st
import json
from langchain.chat_models import ChatOpenAI
from langchain.llms import Together
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

import os
from dotenv import load_dotenv

load_dotenv()


if __name__ == "__main__":
    st.title("ESRS Disclosure Requirements")
    st.markdown("""
        This is a list of all the requirements in the ESRS E1 standard. You can use this list to look up the content of a requirement by its requirementId.
    """)
    st.markdown("""
        ## Requirements List
    """)

    with st.spinner("Loading requirements list..."):
        df = pd.read_csv(
            "./standards/ESRS_E1.csv",
            encoding="utf-8",
            dtype={"requirementId": str, "chapterTitle": str, "content": str, "dataPointType": str},
        )
        df["completed"] = False
    
    llm = ChatOpenAI(temperature=0, api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4")
    # llm = Together(
        # together_api_key=os.getenv("TOGETHER_API_KEY"), 
        # model="mistralai/Mistral-7B-Instruct-v0.2"
        # model="mistralai/Mixtral-8x7B-Instruct-v0.1"
        # model="togethercomputer/llama-2-70b-chat"
    # )
    pd_agent = create_pandas_dataframe_agent(llm, df, verbose=True)

    # tabs = st.tabs(["ESRS E1", "ESRS E2", "ESRS S1", "ESRS S2"])


    # for tab in tabs:
    #     st.markdown(f"## {tab}")
    #     st.data_editor(df[df["requirementId"].str.startswith(tab)], disabled=["requirementId", "dataPointType", "content"], hide_index=True)


    with st.form('my_form'):
        text = st.text_area('What would you like to see?')
        submitted = st.form_submit_button('Submit')
        if submitted:
            response = pd_agent.run(
                f"""
                    Task: Analyze the user's request and return a list of requirementIds from the DataFrame that match the criteria mentioned in the user's input.

                    DataFrame Description: The DataFrame contains information about requirements. Each row represents a requirement and has several columns including requirementId (unique identifier stored as a string), chapterTitle (title of the chapter), content (text of the requirement), dataPointType (comma-seperated list of datatype of disclosure requested by the requirement), and completed (boolean indicating completion).

                    User input: {text}

                    Output: Return a list of requirementIds from the DataFrame that correspond to the user's request. Only the requirementIds should be provided as output. If there are no requirementIds that match the user's request, return an empty list. If the user's request is invalid, return an empty list.

                    Example:
                    User Input: "Show me all requirements in chapter Interactions with other ESRS"
                    Output: ["8", "9", "10", "11"]

                    Now, analyzing the current user input...
                """
            )

            requirementIds = response[response.find("[")+1:response.find("]")].replace("'", "").replace('"', "").replace(" ", "").split(",")

            if len(requirementIds) > 0:
                filtered_df = df.loc[df["requirementId"].isin(requirementIds)]
            else:
                filtered_df = df
    
            chapters = pd.unique(filtered_df.dropna(subset=["chapterTitle"]).sort_values(by="requirementId")["chapterTitle"])
            for chapter in chapters:
                st.markdown(f"### {chapter}")
                st.data_editor(filtered_df[filtered_df["chapterTitle"] == chapter][["completed", "requirementId", "content", "dataPointType"]], disabled=["requirementId", "dataPointType", "content"], hide_index=True)
