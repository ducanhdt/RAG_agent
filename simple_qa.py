import os
from llama_index.embeddings.google import GeminiEmbedding as Embedding
from llama_index.llms.gemini import Gemini
from llama_index.core import SimpleDirectoryReader, Settings, StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import PromptTemplate
import chromadb
from dotenv import load_dotenv
load_dotenv()


gemini_api_key = os.environ['GEMINI_API_KEY']
embed_model_name = os.environ['EMBED_MODEL_NAME']
llm_model_name = os.environ['LLM_MODEL_NAME']
load_data = True

llm = Gemini(api_key=gemini_api_key, model_name=llm_model_name, generation_config = {"temperature": 0.7})
embed_model = Embedding(model_name=embed_model_name, api_key=gemini_api_key)
# Set Global settings
Settings.llm = llm
Settings.embed_model = embed_model

# Create a client and a new collection
client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = client.get_or_create_collection("quickstart")
# Create a vector store
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

if load_data:
    reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    documents = reader.load_data()
    # Create a storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # Create an index from the documents and save it to the disk.
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )
    print("Load data Done")
else:    
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        

template = (
    """ You are an assistant for question-answering tasks.
Use the following context to answer the question.
If you don't know the answer, just say that you don't know.
Use five sentences maximum and keep the answer concise.\n
Question: {query_str} \nContext: {context_str} \nAnswer:"""
)
llm_prompt = PromptTemplate(template)


# Query data from the persisted index
query_engine = index.as_query_engine(text_qa_template=llm_prompt)
# response = query_engine.query("Give me education record of Duc Anh?")
# print(response)
while 1:
    req = input("Ask me something:")
    response = query_engine.query(req)
    print(response)
    