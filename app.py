import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

# Sidebar contents
with st.sidebar:
    st.title('Guru AI')
    st.write("Guru AI is a tool that helps you to search for answers in your PDF files.")


    add_vertical_space(1)
    
    
    st.title('Previous Files')
    script_directory = os.path.dirname(__file__)
    files = [file for file in os.listdir(script_directory) if file.endswith('.pkl')]
    # writing file name without .pkl extension
     
    for file in files:
        st.write(file[:-4])

    add_vertical_space(5)
    st.write('Made with ❤️ by [Danish](https://danishzulfiqar.com)')

load_dotenv()


def main():
    st.header("Guru AI")

    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    # st.write(pdf)
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)

        # # embeddings
        store_name = pdf.name[:-4]
        st.write(f'{store_name}')
        # st.write(chunks)

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            # st.write('Embeddings Loaded from the Disk')s
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        # embeddings = OpenAIEmbeddings()
        # VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

        # Accept user questions/query
        # Add a text box for user to ask questions

        query = st.text_input("Ask questions about your PDF file:")

        # st.write(query)

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)

            llm = OpenAI(temperature=0,model_name="gpt-3.5-turbo")
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)

            # Display the references if user clicks on the button
            def stream_data():
                for doc in docs:
                    st.write(doc)
    
            if st.button("Show References"):
                stream_data()
            

if __name__ == '__main__':
    main()