from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import os
from PyPDF2 import PdfReader
import requests
from bs4 import BeautifulSoup
# from dotenv import load_dotenv
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain

#saved the Google api key in env file
# load_dotenv()
# os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = 'AIzaSyCvw_aGHyJtLxpZ4Ojy8EyaEDtPOzZM29s'

# Retrieve the Google API key from the environment variable
google_api_key = os.getenv("GOOGLE_API_KEY")


def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as pdfFile:
        pdfReader = PdfReader(pdfFile)#used pdfReader from pypdf2 to read the pdf
        numPages = len(pdfReader.pages)
        # Extracting text from each page
        all_text = ""
        for page_num in range(numPages):
            page = pdfReader.pages[page_num]
            text = page.extract_text()
            if text:
                all_text += text.encode('ascii', 'ignore').decode('ascii') + "\n" # Written this so as ignor Non- Ascii characters.
    return all_text

def extract_text_from_url(url):
    response = requests.get(url)
    # Parsing the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    # Extracting the main content
    article_content = soup.find_all('p')
    text = '\n'.join([p.get_text() for p in article_content])#combined all paaragraphs
    
    # Return the text with non-ASCII characters removed
    return text.encode('ascii', 'ignore').decode('ascii')

def get_text_chunks(text):
    #created Chunks of the Text data so as to get a personalized search
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    # Stored vector embeddings in FAISS vector db
    embeddings = GoogleGenerativeAIEmbeddings(google_api_key = google_api_key, model = "models/embedding-001")#model for creating vector embeddings
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")#I have saved the vector database so that I don't have to create and save embeddings agin and again !

def get_conversational_chain():
    # It is a popular way to instruct the LLM to Give right answers ,So I used it.
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    # Using gemini-pro model as LLM
    model = ChatGoogleGenerativeAI(google_api_key = google_api_key , model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)#created a Stuff type chain to get a Q/A model
    return chain

def user_input(user_question):

    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")#same model for creating embedding for the query...
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)#loaded the previously saved vector db
    docs = new_db.similarity_search(user_question)#getting all similiar text present in the Database with the query

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    return response , docs

    
def load_in_db():#This function loads all the text and do the pre-processing Stuff !
    all_text = ""
    all_text += extract_text_from_pdf('Cracking the Granularity Problem - Siluet Case Study.pdf')
    all_text += extract_text_from_pdf('Proving Efficacy of Marketing Mix Model through the Difference in Difference (DID) Technique.pdf')
    all_text += extract_text_from_url('https://open.substack.com/pub/arymalabs/p/marketing-mix-modeling-mmm-101?r=2p7455&utm_campaign=post&utm_medium=web')
    all_text += extract_text_from_url('https://open.substack.com/pub/arymalabs/p/market-mix-modeling-101-part-2?r=2p7455&utm_campaign=post&utm_medium=web')
    all_text += extract_text_from_url('https://open.substack.com/pub/arymalabs/p/why-you-cant-rct-marketing-mix-models?r=2p7455&utm_campaign=post&utm_medium=web')
    all_text += extract_text_from_pdf('Investigation of Marketing Mix Models Business Error using KL Divergence and Chebyshev.pdf')
    # print(all_text)
    text_chunks = get_text_chunks(all_text)
    # print(text_chunks)
    get_vector_store(text_chunks)
