import os
import streamlit as st  
import shutil
import pdfplumber
import cassio
import tempfile
import uuid
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Cassandra
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

os.environ["HUGGINGFACEHUB_API_TOKEN"]=os.getenv("HUGGINGFACE_API_TOKEN")
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
cassio.init(database_id=os.getenv("ASTRA_DB_ID"), token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"))


st.set_page_config(
    page_title="PDF Talker",
    page_icon=":speech_balloon:",
    layout="wide",
    initial_sidebar_state="collapsed",
)



@st.cache_resource(show_spinner="Creating vector store...")
def create_vector_store(file_upload,id):
    temp_dir=tempfile.mkdtemp()
    temp_file_path=os.path.join(temp_dir, file_upload.name)

    with open(temp_file_path, "wb") as f:
        f.write(file_upload.getvalue())
        loader=PyPDFLoader(temp_file_path)
        documents=loader.load()

    for doc in documents:
        doc.metadata["source"] = file_upload.name
        doc.metadata["id"]=id
    
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    texts=text_splitter.split_documents(documents)

    astra_db=Cassandra.from_documents(
    embedding=HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2"),
    documents=texts,
    table_name="PDFTalker",
    )

    shutil.rmtree(temp_dir)
    return astra_db

def process_query(question,vector_store,id):
    llm=ChatGroq(model="llama-3.3-70b-versatile",temperature=0)
    query_prompt=PromptTemplate(
    input_variables=["question"],
    template="""You are a helpful AI Assistant.Your Task is to generate multiple variation of the given user question
    to retrieve relevant information from the vector store. Generate 3 different variations of the question to ensure a comprehensive search.
     By generating multiple user perspective questions, you can enhance the retrieval process and increase the chances of finding relevant information in the vector store.
     Give question seperated by new line and no numbering.
     Question:{question}""",
    )

    retriever_from_llm=MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(search_kwargs={"filter":{"id":id}}),
    llm=llm,
    prompt=query_prompt,
    parser_key="lines",
    )

    context_prompt=ChatPromptTemplate.from_template(
        """ Answer the question based on the following context:
            Think step by step in detail before giving the final answer.
            I will give you 5 star if you answer correctly.
            If question is not related to context, always mention not in context and generate a correct answer accordingly.
            <context>
            {context}
            </context>                                        
            question:{input} """
    )

    doc_chain=create_stuff_documents_chain(
        llm=llm,
        prompt=context_prompt
    )

    retrieval_chain=create_retrieval_chain(
        retriever_from_llm,doc_chain
    )

    response=retrieval_chain.invoke({"input":question})
    return response["answer"]

@st.cache_data(show_spinner="Processing PDF...")
def extract_all_pages_as_images(file_upload):
    images=[]
    with pdfplumber.open(file_upload) as pdf:
        for page in pdf.pages:
            pil_image=page.to_image().original
            images.append(pil_image)
    return images

def reset_collection(vector_store,id):
 if vector_store is not None:
        with st.spinner("Resetting collection..."):
         try:
            print(f"Deleting collection for id:{id}")
            vector_store.delete_by_metadata_filter(filter={"id":id})
         except Exception as e:
            print(e)
            pass
        st.session_state["vector_store"] = None
        st.session_state["pdf_pages"] = []
        st.session_state["messages"] = []
        st.session_state["file_upload"] = None
        st.session_state["uploader_key"] += 1
        st.rerun()
 else:
        st.error("No collection to delete.")

@st.fragment
def chat_interface():
    message_conatiner=st.container(height=500,border=True)

    
    if st.session_state["messages"] ==[]:
            st.session_state["messages"].append(
                {"role":"assistant","content":"Hello! How can I help you?"}
            )
        

    for messages in st.session_state["messages"]:
            avatar="🤖" if messages["role"]=="assistant" else "👤"
            with message_conatiner.chat_message(messages["role"],avatar=avatar):
                st.markdown(messages["content"])

    if prompt:=st.chat_input("Enter your prompt here"):
            try:
                st.session_state["messages"].append(
                    {"role":"user","content":prompt}
                )
                message_conatiner.chat_message("user",avatar="👤").markdown(prompt)

                with message_conatiner.chat_message ("assistant",avatar="🤖"):
                    with st.spinner("Thinking..."):
                        if st.session_state["vector_store"] is None:
                            st.error("Please upload a PDF first.")
                        else:
                            response=process_query(prompt,st.session_state["vector_store"],st.session_state["id"])
                            st.markdown(response)
                
                if st.session_state["vector_store"] is not None:
                    st.session_state["messages"].append(
                        {"role":"assistant","content":response}
                    )
                
            except Exception as e:
                st.error(e,icon="🚨")
        
    else:
            if st.session_state["vector_store"] is None:
                st.error("Please upload a PDF first.")
            else:
                st.error("Please enter a prompt.")



def main():

    st.subheader("PDF Talker")


    if "vector_store" not in st.session_state:
        st.session_state["vector_store"]=None
    if "messages" not in st.session_state:
        st.session_state["messages"]=[]
    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0
    if "pdf_pages" not in st.session_state:
        st.session_state["pdf_pages"] = []
    if "file_upload" not in st.session_state:
        st.session_state["file_upload"] = None
    if "id" not in st.session_state:
        st.session_state["id"] = str(uuid.uuid4())


    file_upload=st. file_uploader(
        "Upload a PDF",
        type=["pdf"],
        accept_multiple_files=False,
        key=st.session_state['uploader_key'],
    )
    col1,col2=st.columns([1.5,2])

    if file_upload is None and st.session_state["file_upload"] is not None:
        reset_collection(st.session_state["vector_store"],st.session_state["id"])
    
    
    if st.button("Delete Collection"):
        if st.session_state["vector_store"] is None:
            st.error("No collection to delete.")
            return
        else:
           reset_collection(st.session_state["vector_store"],st.session_state["id"])

    if file_upload :
        st.session_state["file_upload"]=file_upload
        if st.session_state["pdf_pages"] ==[]:
            pdf_pages=extract_all_pages_as_images(file_upload)
            st.session_state["pdf_pages"]=pdf_pages
        if st.session_state["vector_store"] is None:
            st.session_state["vector_store"]=create_vector_store(file_upload,st.session_state["id"])
      

        zoom_factor=col1.slider("Zoom factor", min_value=100, max_value=1000, value=700, step=100)

        with col1:
            with st.container(height=410,border=True):
                for page_image in st.session_state["pdf_pages"]:
                    st.image(page_image,width=zoom_factor)
    


    if file_upload:
      with col2:
         chat_interface()


                    
if __name__=="__main__":
    main()
        
    

