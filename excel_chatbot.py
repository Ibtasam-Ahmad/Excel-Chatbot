import streamlit as st
import pandas as pd
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Function to load the file (Excel or CSV)
def load_file(file):
    if file.name.endswith('.csv'):
        data = pd.read_csv(file)
    elif file.name.endswith(('.xls', '.xlsx')):
        data = pd.read_excel(file)
    else:
        raise ValueError("Unsupported file type")
    return data

# Combine all columns into one text field for processing
def prepare_data(data):
    data['combined_text'] = data.apply(lambda row: ' | '.join(row.values.astype(str)), axis=1)
    return data[['combined_text']]

# Create an index from the data
def create_index(data, api_key):
    loader = DataFrameLoader(data, page_content_column='combined_text')
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(api_key=api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# Chat with the data
def chat_with_data(query, vectorstore, api_key):
    retriever = vectorstore.as_retriever()
    llm = OpenAI(temperature=0, api_key=api_key)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    response = qa_chain.run(query)
    return response

# Streamlit application
def main():
    st.title("Chat with Excel or CSV Files")
    
    # Step 1: Ask for the OpenAI API key
    api_key = st.text_input("Enter your OpenAI API key", type="password")
    if not api_key:
        st.warning("Please enter your API key to proceed.")
        return

    # Step 2: Upload file
    uploaded_file = st.file_uploader("Upload an Excel or CSV file", type=["csv", "xls", "xlsx"])
    if uploaded_file is not None:
        try:
            data = load_file(uploaded_file)
            st.success("File loaded successfully!")
            st.write("Preview of the file:")
            st.dataframe(data.head())
            
            # Step 3: Process and index the data
            data = prepare_data(data)
            st.write("Processing the file...")
            vectorstore = create_index(data, api_key)
            st.success("File indexed successfully! You can now ask questions.")

            # Step 4: Chat with the data
            query = st.text_input("Ask a question about your data:")
            if query:
                response = chat_with_data(query, vectorstore, api_key)
                st.write("Response:", response)
        
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
