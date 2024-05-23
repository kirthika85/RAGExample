from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Useful to add documents to the chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Useful to load the URL into documents
from langchain_community.document_loaders import WebBaseLoader

# Split the Web page into multiple chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Create Embeddings
from langchain_openai import OpenAIEmbeddings

# Vector Database FAISS
from langchain_community.vectorstores.faiss import FAISS

# USeful to create the Retrieval part
from langchain.chains import create_retrieval_chain

# Retrieve Data from the webpage
def get_docs(weburl):
    loader = WebBaseLoader('{weburl}')
    docs = loader.load()
  
    # WE need to split the web page data
    # We create chunks of 200 and overlap so no data is missed out
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    splitDocs = text_splitter.split_documents(docs)
    return splitDocs
  
# Create Embeddings and Vector database
def create_vector_store(docs):
    embedding = OpenAIEmbeddings(api_key=openaikey)
    vectorStore = FAISS.from_documents(docs, embedding=embedding)
    return vectorStore

# Create cahin for execution
def create_chain(vectorStore):
    model = ChatOpenAI(api_key=openaikey,temperature=0.4,model='gpt-3.5-turbo-1106')
    prompt = ChatPromptTemplate.from_template("""
             Answer the user's question.
             Context: {context}
             Question: {input}
              """)
    print(prompt)
    document_chain = create_stuff_documents_chain(llm=model,prompt=prompt)
    # Retrieving the top 1 relevant document from the vector store , We can change k to 2 and get top 2 and so on
    retriever = vectorStore.as_retriever(search_kwargs={"k": 1})
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

st.title("Simple Search Using RAG")
openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')
if not openai_api_key.startswith('sk-'):
   st.warning('Please enter your OpenAI API key!', icon='⚠')
if openai_api_key.startswith('sk-'):
   url=st.text_area("Enter the URL to search:")
   user_question=st.text_input("Enter your question: ")
   if url and user_question:
      with
   st.spinner('Processing...'):
       docs=get_docs(url)
       vectorStore = create_vector_store(docs)
       chain=create_chain(vectorStore)
       response = chain.invoke({"input":{user_question}})
       st.write(response) 
   else:
       st.error("Please enter  URL and the query.")

     
     
