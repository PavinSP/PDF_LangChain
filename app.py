import streamlit as st
from dotenv import load_dotenv

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    
    st.header("Chat with multiple PDFs :books:")
    st.text_input("Ask questions about your PDFs")

    with st.sidebar:
        st.subheader("Your PDFs")
        pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing your PDFs..."):
                # Get the pdf text

                # Get the text chunks

                # Create vector store

if __name__ == "__main__":
    main()