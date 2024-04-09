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
from langchain.chat_models import ChatOpenAI
import os
import json
import datetime
import time

# Define the VectorStore directory if it doesn't exist already then create it
if not os.path.exists(os.path.join(os.path.dirname(__file__), 'VectorStore')):
    os.makedirs(os.path.join(os.path.dirname(__file__), 'VectorStore'))
vectorstore_directory = os.path.join(os.path.dirname(__file__), 'VectorStore')

ToolName = "Jericho"

# Set the following variables to True to enable the feature
uploadable = False
menu_hide = False
deploy_btn = False

# Hide the Streamlit menu if menu_hide is True
if  menu_hide:
    hide_menu_style = """
            <style>
            #MainMenu {visibility: hidden;}
            </style>
            """
    st.markdown(hide_menu_style, unsafe_allow_html=True)

if not deploy_btn:
    st.markdown("""
        <style>
            .reportview-container {
                margin-top: -2em;
            }
            .stDeployButton {display:none;}
        </style>
    """, unsafe_allow_html=True)

# Sidebar contents
with st.sidebar:
    st.title(f":red[{ToolName}]")
    st.write("An AI tool embedded upon vectors from specified dataset to answer your queries.")

    add_vertical_space(1)

    st.title('Available Cources')
    script_directory = os.path.dirname(__file__)
    # Get all .pkl files in the VectorStore directory
    files = [file for file in os.listdir(vectorstore_directory) if file.endswith('.pkl')]
    files = [file[:-4] for file in files]

    # Create a dropdown menu with all the .pkl files
    selected_file = st.selectbox('Select a Course', files)

    # moving element to bottom
    add_vertical_space(15)
    st.write('Made with ❤️ by [Danish](https://danishzulfiqar.com)')

load_dotenv()


def main():
    st.header(f":red[{ToolName}]")

    if(uploadable):
        pdf = st.file_uploader("Upload your Document", type='pdf', disabled=not uploadable, help="Upload a PDF file to get started" if uploadable else "Only admin can upload PDFs")
    else:
        pdf = None
        st.write("This Tool was designed specifically for solving multiple choice questions from hybrid cources in COMSATS Islamabad. If you have any MCQ, feel free to ask.")

    # Check if there are any .pkl files in the directory
    pkl_files = [file for file in os.listdir(vectorstore_directory) if file.endswith('.pkl')]

    if pdf is not None or pkl_files:
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

            store_name = os.path.join(vectorstore_directory, pdf.name[:-4])
            # writing file name without extension
            st.subheader(f'{pdf.name[:-4]}')

            if os.path.exists(f"{store_name}.pkl"):
                print("Loading from pickle file")
                with open(f"{store_name}.pkl", "rb") as f:
                    VectorStore = pickle.load(f)
            else:
                print("Creating new pickle file")
                embeddings = OpenAIEmbeddings()
                VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                with open(f"{store_name}.pkl", "wb") as f:
                    pickle.dump(VectorStore, f)

        elif pkl_files:
            # Load the selected pkl file, write the name of the file
            st.subheader(f'{selected_file}')
            with open(os.path.join(vectorstore_directory, f"{selected_file}.pkl"), "rb") as f:
                VectorStore = pickle.load(f)


        # Accept user questions/query with query file
        query = st.chat_input("Ask me anything from")

        if query:
            
            docs = VectorStore.similarity_search(query=query, k=3)

            model_name = "gpt-3.5-turbo"
            # model_name = "davinci-002"

            temperature = 0

            llm = OpenAI(temperature=temperature, model_name=model_name)
            chain = load_qa_chain(llm=llm, chain_type="stuff")

            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)

            # Loading the data from the Logs.json file
            try:
                with open('Logs.json', 'r') as f:
                    data = json.load(f)
            except FileNotFoundError:
                    data = {}
                
            if selected_file not in data:
                data[selected_file] = []
            
            callback = cb.to_dict() if hasattr(cb, 'to_dict') else str(cb)
            callback = callback.split('\n')

            callback = [i.replace('\t', '') for i in callback]
            callback = [i.replace('\n', '') for i in callback]

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            data[selected_file].append({
                "model_info": {
                    "model": model_name,
                    "temperature": temperature,
                },
                "query_data":{
                    "query": query,
                    "response": response,
                },
                "callback": callback,
                "timestamp": timestamp
            })

            # Save the data to the Logs.json file
            with open('Logs.json', 'w') as f:
                json.dump(data, f, indent=4)
            
            with st.chat_message("Jericho", avatar="https://avatars.githubusercontent.com/u/102870087?s=400&u=1c2dfa41026169b5472579d4d36ad6b2fe473b6d&v=4"):
                st.markdown(''':bold[Jericho]''')

                with st.spinner('Searchin for the answer...'):
                    time.sleep(5)
                st.write(f":blue[{response}]")
                with st.expander("References"):
                    for doc in docs:
                        st.info(doc.page_content)


if __name__ == '__main__':
    main()
