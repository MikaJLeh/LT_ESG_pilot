import os
import shutil
import streamlit as st
from fpdf import FPDF
from chromadb import Client
from chromadb.config import Settings
from langchain_community.utilities import SerpAPIWrapper
from llama_index.core import VectorStoreIndex
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.agents import AgentType, Tool, initialize_agent, AgentExecutor
from llama_parse import LlamaParse
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.core import SimpleDirectoryReader
from dotenv import load_dotenv, find_dotenv
import pandas as pd
from streamlit_chat import message
from langchain_community.vectorstores import Chroma
from langchain_community.utilities import SerpAPIWrapper
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import UnstructuredXMLLoader
from langchain_community.document_loaders import CSVLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import joblib
import nest_asyncio  # noqa: E402
nest_asyncio.apply()
import nltk

load_dotenv()
load_dotenv(find_dotenv())

os.environ["TOKENIZERS_PARALLELISM"] = "false"
SERPAPI_API_KEY = os.environ["SERPAPI_API_KEY"]
GOOGLE_CSE_ID = os.environ["GOOGLE_CSE_ID"]
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
LLAMA_PARSE_API_KEY = os.environ["LLAMA_PARSE_API_KEY"]
HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]
groq_api_key=os.getenv('GROQ_API_KEY')

st.set_page_config(layout="wide")

css = """
<style>
    [data-testid="stAppViewContainer"] {
        background-color: #f8f9fa; /* Very light grey */
    }
    [data-testid="stSidebar"] {
        background-color: white;
        color: black;
    }
    [data-testid="stAppViewContainer"] * {
        color: black; /* Ensure all text is black */
    }
    button {
        background-color: #add8e6; /* Light blue for primary buttons */
        color: black;
        border: 2px solid green; /* Green border */
    }
    button:hover {
        background-color: #87ceeb; /* Slightly darker blue on hover */
    }

    button:active {
        outline: 2px solid green; /* Green outline when the button is pressed */
        outline-offset: 2px; /* Space between button and outline */
    }

    .stButton>button:first-child {
        background-color: #add8e6; /* Light blue for primary buttons */
        color: black;
    }
    .stButton>button:first-child:hover {
        background-color: #87ceeb; /* Slightly darker blue on hover */
    }
    .stButton>button:nth-child(2) {
        background-color: #b0e0e6; /* Even lighter blue for secondary buttons */
        color: black;
    }
    .stButton>button:nth-child(2):hover {
        background-color: #add8e6; /* Slightly darker blue on hover */
    }
    [data-testid="stFileUploadDropzone"] {
        background-color: white; /* White background for file upload */
    }
    [data-testid="stFileUploadDropzone"] .stDropzone, [data-testid="stFileUploadDropzone"] .stDropzone input {
        color: black; /* Ensure file upload text is black */
    }
    
    .stButton>button:active {
        outline: 2px solid green; /* Green outline when the button is pressed */
        outline-offset: 2px;
    }
</style>
"""
nltk.download('punkt')
st.write(css, unsafe_allow_html=True)
#st.sidebar.image('lt.png', width=250)
#------------- 
llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="Llama-3.1-70b-Versatile", temperature = 0.0, streaming=True)   
#--------------
doc_retriever_ESG = None
doc_retriever_financials = None
#--------------
#@st.cache_data
def load_or_parse_data_ESG():
    data_file = "./data/parsed_data_ESG.pkl"

    parsingInstructionUber10k = """The provided document contain detailed information about the company's environmental, social and governance matters.
    It contains several tables, figures and statistical information. You must be precise while answering the questions and never provide false numeric or statistical data."""

    parser = LlamaParse(api_key=LLAMA_PARSE_API_KEY,
                        result_type="markdown",
                        parsing_instruction=parsingInstructionUber10k,
                        max_timeout=5000,
                        gpt4o_mode=True,
                        )

    file_extractor = {".pdf": parser}
    reader = SimpleDirectoryReader("./ESG_Documents", file_extractor=file_extractor)
    documents = reader.load_data()

    print("Saving the parse results in .pkl format ..........")
    joblib.dump(documents, data_file)

    # Set the parsed data to the variable
    parsed_data_ESG = documents

    return parsed_data_ESG

#@st.cache_data
def load_or_parse_data_financials():
    data_file = "./data/parsed_data_financials.pkl"

    parsingInstructionUber10k = """The provided document is the company's annual reports and includes financial statement, balance sheet, cash flow sheet and description of the company's business and operations.
    It contains several tabless, figures and statistical information. You must be precise while answering the questions and never provide false numeric or statistical data."""

    parser = LlamaParse(api_key=LLAMA_PARSE_API_KEY,
                        result_type="markdown",
                        parsing_instruction=parsingInstructionUber10k,
                        max_timeout=5000,
                        gpt4o_mode=True,
                        )

    file_extractor = {".pdf": parser}
    reader = SimpleDirectoryReader("./Financial_Documents", file_extractor=file_extractor)
    documents = reader.load_data()

    print("Saving the parse results in .pkl format ..........")
    joblib.dump(documents, data_file)

    # Set the parsed data to the variable
    parsed_data_financials = documents

    return parsed_data_financials

#@st.cache_data
def load_or_parse_data_portfolio():
    data_file = "./data/parsed_data_portfolio.pkl"

    parsingInstructionUber10k = """The provided document is the ESG and sustainability report of LocalTapiola (Lähitapiola) group including the funds it manages.
    It contains several tabless, figures and statistical information. You must be precise while answering the questions and never provide false numeric or statistical data."""

    parser = LlamaParse(api_key=LLAMA_PARSE_API_KEY,
                        result_type="markdown",
                        parsing_instruction=parsingInstructionUber10k,
                        max_timeout=5000,
                        gpt4o_mode=True,
                        )

    file_extractor = {".pdf": parser}
    reader = SimpleDirectoryReader("./ESG_Documents_Portfolio", file_extractor=file_extractor)
    documents = reader.load_data()

    print("Saving the parse results in .pkl format ..........")
    joblib.dump(documents, data_file)

    # Set the parsed data to the variable
    parsed_data_portfolio = documents

    return parsed_data_portfolio
#--------------
# Create vector database

@st.cache_resource
def create_vector_database_ESG():
    # Call the function to either load or parse the data
    llama_parse_documents = load_or_parse_data_ESG()

    with open('data/output_ESG.md', 'a') as f:  # Open the file in append mode ('a')
        for doc in llama_parse_documents:
            f.write(doc.text + '\n')

    markdown_path = "data/output_ESG.md"
    loader = UnstructuredMarkdownLoader(markdown_path)

    #loader = DirectoryLoader('data/', glob="**/*.md", show_progress=True)
    documents = loader.load()
    # Split loaded documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=15)
    docs = text_splitter.split_documents(documents)

    #len(docs)
    print(f"length of documents loaded: {len(documents)}")
    print(f"total number of document chunks generated :{len(docs)}")
    embed_model = HuggingFaceEmbeddings()
    #embed_model = OpenAIEmbeddings()
    # Create and persist a Chroma vector database from the chunked documents
    # Set up the Chroma client in local mode
    print('Vector DB not yet created !')
    persist_directory = os.path.join(os.getcwd(), "chroma_db_LT")
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)

    vs = Chroma.from_documents(
        documents=docs,
        embedding=embed_model,
        persist_directory=persist_directory,  # Local mode with in-memory storage only
        collection_name="rag",
    )

    doc_retriever_ESG = vs.as_retriever()
    
    print('Vector DB created successfully !')
    return doc_retriever_ESG

@st.cache_resource
def create_vector_database_financials():
    # Call the function to either load or parse the data
    llama_parse_documents = load_or_parse_data_financials()
    print(llama_parse_documents[0].text[:300])

    with open('data/output_financials.md', 'a') as f:  # Open the file in append mode ('a')
        for doc in llama_parse_documents:
            f.write(doc.text + '\n')

    markdown_path = "data/output_financials.md"
    loader = UnstructuredMarkdownLoader(markdown_path)

   #loader = DirectoryLoader('data/', glob="**/*.md", show_progress=True)
    documents = loader.load()
    # Split loaded documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=15)
    docs = text_splitter.split_documents(documents)

    #len(docs)
    print(f"length of documents loaded: {len(documents)}")
    print(f"total number of document chunks generated :{len(docs)}")
    embed_model = HuggingFaceEmbeddings()
    #embed_model = OpenAIEmbeddings()
    # Create and persist a Chroma vector database from the chunked documents
    persist_directory = os.path.join(os.getcwd(), "chroma_db_fin")
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)

    vs = Chroma.from_documents(
        documents=docs,
        embedding=embed_model,
        persist_directory=persist_directory,  # Local mode with in-memory storage only
        collection_name="rag"
    )
    doc_retriever_financials = vs.as_retriever()

    print('Vector DB created successfully !')
    return doc_retriever_financials

@st.cache_resource
def create_vector_database_portfolio():
    # Call the function to either load or parse the data
    llama_parse_documents = load_or_parse_data_portfolio()
    print(llama_parse_documents[0].text[:300])

    with open('data/output_portfolio.md', 'a') as f:  # Open the file in append mode ('a')
        for doc in llama_parse_documents:
            f.write(doc.text + '\n')

    markdown_path = "data/output_portfolio.md"
    loader = UnstructuredMarkdownLoader(markdown_path)

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=15)
    docs = text_splitter.split_documents(documents)

    print(f"length of documents loaded: {len(documents)}")
    print(f"total number of document chunks generated :{len(docs)}")
    embed_model = HuggingFaceEmbeddings()

    persist_directory = os.path.join(os.getcwd(), "chroma_db_portfolio")
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)

    vs = Chroma.from_documents(
        documents=docs,
        embedding=embed_model,
        persist_directory=persist_directory,  # Local mode with in-memory storage only
        collection_name="rag"
    )
    doc_retriever_portfolio = vs.as_retriever()

    print('Vector DB created successfully !')
    return doc_retriever_portfolio
#--------------
ESG_analysis_button_key = "ESG_strategy_button"
portfolio_analysis_button_key = "portfolio_strategy_button"

#---------------
def delete_files_and_folders(folder_path):
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for file in files:
            try:
                os.unlink(os.path.join(root, file))
            except Exception as e:
                st.error(f"Error deleting {os.path.join(root, file)}: {e}")
        for dir in dirs:
            try:
                os.rmdir(os.path.join(root, dir))
            except Exception as e:
                st.error(f"Error deleting directory {os.path.join(root, dir)}: {e}")
#---------------

uploaded_files_ESG = st.sidebar.file_uploader("Choose a Sustainability Report", accept_multiple_files=True, key="ESG_files")
for uploaded_file in uploaded_files_ESG:
    #bytes_data = uploaded_file.read()
    st.write("filename:", uploaded_file.name)
    def save_uploadedfile(uploadedfile):
     with open(os.path.join("ESG_Documents",uploadedfile.name),"wb") as f:
         f.write(uploadedfile.getbuffer())
     return st.success("Saved File:{} to ESG_Documents".format(uploadedfile.name))
    save_uploadedfile(uploaded_file)

uploaded_files_financials = st.sidebar.file_uploader("Choose an Annual Report", accept_multiple_files=True, key="financial_files")
for uploaded_file in uploaded_files_financials:
    #bytes_data = uploaded_file.read()
    st.write("filename:", uploaded_file.name)
    def save_uploadedfile(uploadedfile):
     with open(os.path.join("Financial_Documents",uploadedfile.name),"wb") as f:
         f.write(uploadedfile.getbuffer())
     return st.success("Saved File:{} to Financial_Documents".format(uploadedfile.name))
    save_uploadedfile(uploaded_file)

#---------------
def ESG_strategy():
    doc_retriever_ESG = create_vector_database_ESG()
    prompt_template = """<|system|>
    You are a seasoned specialist in environmental, social and governance matters. You write expert analyses for institutional investors. Always use figures, nemerical and statistical data when possible. Output must have sub-headings in bold font and be fluent.<|end|>
    <|user|>
    Answer the {question} based on the information you find in context: {context} <|end|>
    <|assistant|>""" 

    prompt = PromptTemplate(template=prompt_template, input_variables=["question", "context"])

    qa = (
    {
        "context": doc_retriever_ESG,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)   

    ESG_answer_1 = qa.invoke("Give a summary what ESG measures the company has taken and compare these to the best practices. Has the company issues green bonds or green loans.")
    ESG_answer_2 = qa.invoke("Do the company's main business fall under the European Union's taxonomy regulation? Is the company taxonomy compliant under European Union Taxonomy Regulation? Does the company follow the Paris Treaty's obligation to limit globabl warming to 1.5 celcius degrees? What are the measures to achieve this goal")
    ESG_answer_3 = qa.invoke("Explain what items of ESG information the company publishes. Describe what ESG transparency commitments the company has given?")
    ESG_answer_4 = qa.invoke("Does the company have carbon emissions reduction plan? Set out in a table the company's carbon footprint by location and its development over time. Set out carbon dioxide emissions in relation to turnover and whether the company has reached its carbod dioxide reduction objectives")
    ESG_answer_5 = qa.invoke("Describe and give a time series table of the company's carbon dioxide emissions (Scope 1), carbon dioxide emissions from purchased energy (Scope 2) and other indirect carbon dioxide emissions (Scope 3). Set out the company's objectives and material developments relating to these figures")
    ESG_answer_6 = qa.invoke("Set out in a table the company's energy and renewable energy usage for each activity coverning the last two or three years. Explain the energy efficiency measures taken by the company. Does the company have a plan to make its use of energy greemer?.")
    ESG_answer_7 = qa.invoke("Does the company follow UN Guiding Principles on Business and Human Rights, ILO Declaration on Fundamental Principles and Rights at Work or OECD Guidelines for Multinational Enterprises that involve affected communities? Set out the measures taken to have the gender balance on the upper management of the company.")
    ESG_answer_8 = qa.invoke("List the environmental permits and certifications held by the company. Set out and explain any environmental procedures and investigations and decisions taken against the company. Answer whether the company's locations or operations are connected to areas sensitive in relation to biodiversity.")
    ESG_answer_9 = qa.invoke("Set out waste produces by the company and possible waste into the soil by real estate. Describe if the company's real estates have hazardous waste.")
    ESG_answer_10 = qa.invoke("What policies has the company implemented to counter money laundering and corruption? What percentage of women are represented in the board, executive directors and upper management?")

    ESG_output = f"**__Summary of  ESG reporting and obligations:__** {ESG_answer_1} \n\n **__Compliance with taxonomy:__** \n\n {ESG_answer_2} \n\n **__Disclosure transparency:__** \n\n {ESG_answer_3} \n\n **__Carbon footprint:__** \n\n {ESG_answer_4} \n\n **__Carbon dioxide emissions:__** \n\n {ESG_answer_5} \n\n **__Renewable energy:__** \n\n {ESG_answer_6} \n\n **__Human rights compliance:__** \n\n {ESG_answer_7} \n\n **__Management and gender balance:__** \n\n {ESG_answer_8} \n\n **__Waste and other emissions:__** {ESG_answer_9} \n\n **__Money laundering and corruption:__** {ESG_answer_10}"
    financial_output = ESG_output
    
    with open("ESG_analysis.txt", 'w') as file:
        file.write(financial_output)
    
    return financial_output

def portfolio_strategy():
    persist_directory_ESG = "chroma_db_LT"
    embeddings = HuggingFaceEmbeddings()
    doc_retriever_ESG = Chroma(persist_directory=persist_directory_ESG, embedding_function=embeddings).as_retriever()

    doc_retriever_portfolio = create_vector_database_portfolio()
    prompt_portfolio = PromptTemplate.from_template(
        template="""<|system|>
            You are a seasoned finance specialist and a specialist in environmental, social and governance matters. You write expert portofolion analyses fund management. Always use figures, numerical and statistical data when possible. Output must have sub-headings in bold font and be fluent.<|end|>
            <|user|> Based on the {context}, write a summary of LähiTapiola's investment policy. Set out also the most important ESG and sustainability aspects of the policy.<|end|>"\
            <|assistant|>""")
    
    prompt_strategy = PromptTemplate.from_template(
        template="""<|system|>
            You are a seasoned specialist in environmental, social and governance matters. You analyse companies' ESG matters. Always use figures, numerical and statistical data when possible. Output must have sub-headings in bold font and be fluent.<|end|>
            <|user|> Based on the {context}, give a summary of the target company's ESG policy. Set out also the most important ESG and sustainability aspects of the policy.<|end|>"\
            <|assistant|>""")
    
    prompt_analysis = PromptTemplate.from_template(
        template="""<|system|>
            You are a seasoned finance specialist and a specialist in environmental, social and governance matters. You write expert portofolio analyses fund management. Always use figures, numerical and statistical data when possible. Output must have sub-headings in bold font and be fluent.<|end|>
            <|user|> Answer the {question} based on {company_ESG} and {fund_policy}.<|end|>"\
            <|assistant|>""")
    
    portfolio_chain = (
            {
            "context": doc_retriever_portfolio,
            #"question": RunnablePassthrough(),
            }
            | prompt_portfolio
            | llm
            | StrOutputParser()
            )
    strategy_chain = (
            {
            "context": doc_retriever_ESG,
            #"question": RunnablePassthrough(),
            }
            | prompt_strategy
            | llm
            | StrOutputParser()
            )
    
    analysis_chain = (
        {
            "company_ESG": strategy_chain, 
            "fund_policy": portfolio_chain, 
            "question": RunnablePassthrough(),
        }
        | prompt_analysis
        | llm
        | StrOutputParser()
            )

    portfolio_answer = analysis_chain.invoke("is the company's ESG such that it fits within LähiTapiola's investment policy of: {fund_policy}? Give a policy rating")
    portfolio_output = f"**__Summary of fit with LähiTapiola's sustainability policy:__** {portfolio_answer} \n"
    
    with open("portfolio_analysis.txt", 'w') as file:
        file.write(portfolio_output)

    return portfolio_output

#-------------
@st.cache_data
def generate_ESG_strategy() -> str:
    ESG_output = ESG_strategy()
    st.session_state.results["ESG_analysis_button_key"] = ESG_output
    return ESG_output

@st.cache_data
def generate_portfolio_analysis() -> str:
    portfolio_output = portfolio_strategy()
    st.session_state.results["portfolio_analysis_button_key"] = portfolio_output
    return portfolio_output
#---------------
#@st.cache_data
def create_pdf():
    text_file = "ESG_analysis.txt"
    pdf = FPDF('P', 'mm', 'A4')
    pdf.add_page()
    pdf.set_margins(10, 10, 10)
    pdf.set_font("Arial", size=15)
    #image = "lt.png"
    #pdf.image(image, w = 40)
    # Add introductory lines
    #pdf.cell(0, 10,  txt="Company name", ln=1, align='C')
    pdf.cell(0, 10, txt="Structured ESG Analysis", ln=2, align='C')
    pdf.ln(5)

    pdf.set_font("Arial", size=11)
    try:
        with open(text_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Replace '\u2019' with a different character or string
                #line = line.replace('\u2019', "'")  # For example, replace with apostrophe
                #line = line.replace('\u2265', "'")  # For example, replace with apostrophe
                #pdf.multi_cell(0, 6, txt=line, align='L')
                pdf.multi_cell(0, 6, txt=line.encode('latin-1', 'replace').decode('latin-1'), align='L')
            pdf.ln(5)
    except UnicodeEncodeError:
        print("UnicodeEncodeError: Some characters could not be encoded in Latin-1. Skipping...")
        pass  # Skip the lines causing UnicodeEncodeError

    output_pdf_path = "ESG_analysis.pdf"
    pdf.output(output_pdf_path)

#----------------
#llm = build_llm()

if 'results' not in st.session_state:
    st.session_state.results = {
        "ESG_analysis_button_key": {}
    }

loaders = {'.pdf': PyMuPDFLoader,
           '.xml': UnstructuredXMLLoader,
           '.csv': CSVLoader,
           }

def create_directory_loader(file_type, directory_path):
    return DirectoryLoader(
        path=directory_path,
        glob=f"**/*{file_type}",
        loader_cls=loaders[file_type],
    )


strategies_container = st.container()
with strategies_container:
    mrow1_col1, mrow1_col2 = st.columns(2)

    st.sidebar.info("To get started, please upload the documents from the company you would like to analyze.")
    button_container = st.sidebar.container()
    if os.path.exists("ESG_analysis.txt"):
        create_pdf()
        with open("ESG_analysis.pdf", "rb") as pdf_file:
            PDFbyte = pdf_file.read()

        st.sidebar.download_button(label="Download Analyses",
                    data=PDFbyte,
                    file_name="strategy_sheet.pdf",
                    mime='application/octet-stream',
                    )

    if button_container.button("Clear All"):
        
        st.session_state.button_states = {
        "ESG_analysis_button_key": False,
        }
        st.session_state.button_states = {
        "portfolio_analysis_button_key": False,
        }
        st.session_state.results = {}

        st.session_state['history'] = []
        st.session_state['generated'] = ["Let's discuss the ESG issues of the company 🤗"]
        st.session_state['past'] = ["Hey ! 👋"]
        st.cache_data.clear()
        st.cache_resource.clear()

        # Check if the subfolder exists
        if os.path.exists("ESG_Documents"):
            for filename in os.listdir("ESG_Documents"):
                file_path = os.path.join("ESG_Documents", filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    st.error(f"Error deleting {file_path}: {e}")
        else:
            pass

        if os.path.exists("Financial_Documents"):
            # Iterate through files in the subfolder and delete them
            for filename in os.listdir("Financial_Documents"):
                file_path = os.path.join("Financial_Documents", filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    st.error(f"Error deleting {file_path}: {e}")
        else:
            pass
            # st.warning("No 'data' subfolder found.")

        folders_to_clean = ["data", "chroma_db_portfolio", "chroma_db_LT", "chroma_db_fin"]

        for folder_path in folders_to_clean:
            if os.path.exists(folder_path):
                for item in os.listdir(folder_path):
                    item_path = os.path.join(folder_path, item)
                    try:
                        if os.path.isfile(item_path) or os.path.islink(item_path):
                            os.unlink(item_path)  # Remove files or symbolic links
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)  # Remove subfolders and all their contents
                    except Exception as e:
                        st.error(f"Error deleting {item_path}: {e}")
            else:
                pass
                # st.warning(f"No '{folder_path}' folder found.")

    with mrow1_col1:
        st.subheader("Summary of the ESG Analysis")
        st.info("This tool is designed to provide a comprehensive ESG risk analysis for institutional investors.")
        button_container2 = st.container()
        if "button_states" not in st.session_state:
            st.session_state.button_states = {
            "ESG_analysis_button_key": False,
            }
        
        if "results" not in st.session_state:
            st.session_state.results = {}

        if button_container2.button("ESG Analysis", key=ESG_analysis_button_key):
            st.session_state.button_states[ESG_analysis_button_key] = True
            result_generator = generate_ESG_strategy()  # Call the generator function
            st.session_state.results["ESG_analysis_output"] = result_generator
            
        if "ESG_analysis_output" in st.session_state.results:           
            st.write(st.session_state.results["ESG_analysis_output"])
        st.divider()

    with mrow1_col2:
        st.subheader("Analyze the ESG summary and the investment policy")
        st.info("This tool enables analysing the company's ESG policy with respect to the portfolio and investment policy.")
        uploaded_files_portfolio = st.file_uploader("Choose a pdf file", accept_multiple_files=True, key="portfolio_files")
        for uploaded_file in uploaded_files_portfolio:
            st.write("filename:", uploaded_file.name)
            def save_uploadedfile(uploadedfile):
                with open(os.path.join("ESG_Documents_Portfolio",uploadedfile.name),"wb") as f:
                    f.write(uploadedfile.getbuffer())
                return st.success("Saved File:{} to ESG_Documents_Portfolio".format(uploadedfile.name))
            save_uploadedfile(uploaded_file)
        button_container3 = st.container()
        #st.button("Portfolio Analysis")
        if "button_states" not in st.session_state:
            st.session_state.button_states = {
            "portfolio_analysis_button_key": False,
            }
        if button_container3.button("Portfolio Analysis", key=portfolio_analysis_button_key):
            st.session_state.button_states[portfolio_analysis_button_key] = True
            portfolio_result_generator = generate_portfolio_analysis()
            st.session_state.results["portfolio_analysis_output"] = portfolio_result_generator
            st.write(portfolio_result_generator)

        if "portfolio_analysis_output" in st.session_state.results:
            st.write(st.session_state.results["portfolio_analysis_output"])

        st.divider()
        
        with mrow1_col2:
            if "ESG_analysis_button_key" in st.session_state.results and st.session_state.results["ESG_analysis_button_key"]:
                doc_retriever_ESG = create_vector_database_ESG()
                doc_retriever_financials = create_vector_database_financials()

                persist_directory = os.path.join(os.getcwd(), "chroma_db_portfolio")
                if not os.path.exists(persist_directory):
                    os.makedirs(persist_directory)
                
                # Load the Chroma retriever from the persisted directory
                embeddings = HuggingFaceEmbeddings()
                doc_retriever_portfolio = Chroma(persist_directory=persist_directory, embedding_function=embeddings).as_retriever()


                memory = ConversationBufferMemory(memory_key="chat_history", k=3, return_messages=True)
                search = SerpAPIWrapper()

                # Updated prompt templates to include chat history
                def format_chat_history(chat_history):
                    """Format chat history as a single string for input to the chain."""
                    formatted_history = "\n".join([f"User: {entry['input']}\nAI: {entry['output']}" for entry in chat_history])
                    return formatted_history

                prompt_portfolio = PromptTemplate.from_template(
                    template="""
                        You are a seasoned finance specialist and a specialist in environmental, social, and governance matters.
                        Use figures, numerical, and statistical data when possible.

                        Conversation history:
                        {chat_history}

                        Based on the context: {context}, write a summary of LähiTapiola's investment policy. Set out also the most important ESG and sustainability aspects of the policy.
                    """
                )

                prompt_financials = PromptTemplate.from_template(
                    template="""
                        You are a seasoned corporate finance specialist.
                        Use figures, numerical, and statistical data when possible.

                        Conversation history:
                        {chat_history}

                        Based on the context: {context}, answer the following question: {question}.
                    """
                )

                prompt_ESG = PromptTemplate.from_template(
                    template="""
                        You are a seasoned finance specialist and a specialist in environmental, social, and governance matters.
                        Use figures, numerical, and statistical data when possible.

                        Conversation history:
                        {chat_history}

                        Based on the context: {context}, write a summary of LähiTapiola's ESG policy. Set out also the most important sustainability aspects of the policy.
                    """
                )

                # LCEL Chains with memory integration
                financials_chain = (
                    {
                        "context": doc_retriever_financials,
                        # Lambda function now accepts one argument (even if unused)
                        "chat_history": lambda _: format_chat_history(memory.load_memory_variables({})["chat_history"]),
                        "question": RunnablePassthrough(),
                    }
                    | prompt_financials
                    | llm
                    | StrOutputParser()
                )

                portfolio_chain = (
                    {
                        "context": doc_retriever_portfolio,
                        "chat_history": lambda _: format_chat_history(memory.load_memory_variables({})["chat_history"]),
                        "question": RunnablePassthrough(),
                    }
                    | prompt_portfolio
                    | llm
                    | StrOutputParser()
                )

                ESG_chain = (
                    {
                        "context": doc_retriever_ESG,
                        "chat_history": lambda _: format_chat_history(memory.load_memory_variables({})["chat_history"]),
                        "question": RunnablePassthrough(),
                    }
                    | prompt_ESG
                    | llm
                    | StrOutputParser()
                )

                # Define the tools with LCEL expressions
                tools = [
                    Tool(
                        name="ESG QA System",
                        func=ESG_chain.invoke,
                        description="Useful for answering questions about environmental, social, and governance (ESG) matters related to the target company, but not LähiTapiola.",
                    ),
                    Tool(
                        name="Financials QA System",
                        func=financials_chain.invoke,
                        description="Useful for answering questions about financial or operational information concerning the target company, but not LähiTapiola.",
                    ),
                    Tool(
                        name="Policy QA System",
                        func=portfolio_chain.invoke,
                        description="Useful for answering questions about LähiTapiola's ESG policy and sustainability measures.",
                    ),
                    Tool(
                        name="Search Tool",
                        func=search.run,
                        description="Useful when other tools do not provide the answer.",
                    ),
                ]

                # Initialize the agent with LCEL tools and memory
                agent = initialize_agent(
                    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, memory=memory, handle_parsing_errors=True)
                def conversational_chat(query):
                    # Get the result from the agent
                    result = agent.invoke({"input": query, "chat_history": st.session_state['history']})
                    
                    # Handle different response types
                    if isinstance(result, dict):
                        # Extract the main content if the result is a dictionary
                        result = result.get("output", "")  # Adjust the key as needed based on your agent's output
                    elif isinstance(result, list):
                        # If the result is a list, join it into a single string
                        result = "\n".join(result)
                    elif not isinstance(result, str):
                        # Convert the result to a string if it is not already one
                        result = str(result)
                    
                    # Add the query and the result to the session state
                    st.session_state['history'].append((query, result))
                    
                    # Update memory with the conversation
                    memory.save_context({"input": query}, {"output": result})
                    
                    # Return the result
                    return result

                # Ensure session states are initialized
                if 'history' not in st.session_state:
                    st.session_state['history'] = []

                if 'generated' not in st.session_state:
                    st.session_state['generated'] = ["Let's discuss the ESG matters and financial matters 🤗"]

                if 'past' not in st.session_state:
                    st.session_state['past'] = ["Hey ! 👋"]

                if 'input' not in st.session_state:
                    st.session_state['input'] = ""

                # Streamlit layout
                st.subheader("Discuss the ESG and financial matters")
                st.info("This tool is designed to enable discussion about the ESG and financial matters concerning the company and also LocalTapiola's own comprehensive sustainability policy and guidance.")
                response_container = st.container()
                container = st.container()

                with container:
                    with st.form(key='my_form'):
                        user_input = st.text_input("Query:", placeholder="What would you like to know about ESG and financial matters", key='input')
                        submit_button = st.form_submit_button(label='Send')
                    if submit_button and user_input:
                        output = conversational_chat(user_input)
                        st.session_state['past'].append(user_input)
                        st.session_state['generated'].append(output)
                        user_input = "Query:"
                    #st.session_state['input'] = ""
                # Display generated responses
                if st.session_state['generated']:
                    with response_container:
                        for i in range(len(st.session_state['generated'])):
                            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="shapes")
                            message(st.session_state["generated"][i], key=str(i), avatar_style="icons")
