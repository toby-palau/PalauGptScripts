import streamlit as st
import pandas as pd


with st.spinner("Loading requirements list..."):
    df = pd.read_csv(
        "./standards/ESRS_E1_summarized.csv",
        dtype={"requirementId": str, "chapterTitle": str, "content": str, "dataPointType": str}
    )
    df["completed"] = False
    chapters = pd.unique(df.dropna(subset=["chapterTitle"]).sort_values(by="paragraphId")["chapterTitle"])

def main():
    st.title("ESRS Navigator")
    st.subheader("A tool to help you navigate the ESRS")

    for chapterTitle in chapters:
        chapter_df = df[df["chapterTitle"] == chapterTitle]
        st.subheader(chapterTitle)
        for paragraphId in chapter_df["paragraphId"]:
            paragraph = chapter_df.loc[chapter_df["paragraphId"] == paragraphId]
            summary = paragraph["summary"].iloc[0]
            content = paragraph["content"].iloc[0]
            if isinstance(summary, str):
                with st.expander(f"{paragraphId}: {summary}"):
                    if st.checkbox(content[:60] + "..."):
                        st.write(content)




if __name__ == "__main__":
    main()
