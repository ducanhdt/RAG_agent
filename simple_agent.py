import os
from llama_index.embeddings.google import GeminiEmbedding as Embedding
from llama_index.llms.gemini import Gemini
from llama_index.core import (
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import PromptTemplate, load_index_from_storage
import chromadb

from llama_index.core.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
)
from llama_index.core.node_parser import TokenTextSplitter


from dotenv import load_dotenv

load_dotenv()


class GenimiAgent:

    llm_prompt = PromptTemplate(
        """ You are an assistant for question-answering tasks.
    Use the following context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use five sentences maximum and keep the answer concise.\n
    Question: {query_str} \nContext: {context_str} \nAnswer:"""
    )

    gemini_api_key = os.environ["GEMINI_API_KEY"]
    embed_model_name = os.environ["EMBED_MODEL_NAME"]
    llm_model_name = os.environ["LLM_MODEL_NAME"]

    def __init__(
        self,
        data_folder="./data",
        vecter_storage="./chroma_db",
        persist_dir="./storage",
        reload_data=False,
    ):
        self.persist_dir = persist_dir
        self.data_folder = data_folder
        self.vecter_storage= vecter_storage
        
        llm = Gemini(
            api_key=self.gemini_api_key,
            model_name=self.llm_model_name,
            generation_config={"temperature": 0.5},
        )
        embed_model = Embedding(
            model_name=self.embed_model_name, api_key=self.gemini_api_key
        )
        # Set Global settings
        Settings.llm = llm
        Settings.embed_model = embed_model

        # Create a client and a new collection
        client = chromadb.PersistentClient(path=vecter_storage)
        chroma_collection = client.get_or_create_collection("default")
        # Create a vector store
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        transformations = [
            TokenTextSplitter(chunk_size=512, chunk_overlap=64),
            TitleExtractor(nodes=5, num_workers=1),
            QuestionsAnsweredExtractor(questions=3, num_workers=1),
        ]

        try:
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store, persist_dir=persist_dir
            )
            index = load_index_from_storage(
                storage_context=storage_context,
                transformations=transformations,
                store_nodes_override=True,
            )
            load_success = True

            print(f"Agent loaded data from storage")
        except Exception as e:
            load_success = False
            print(f"Agent fail to load data storage: {e}")
        if reload_data or not load_success:
            reader = SimpleDirectoryReader(
                input_dir=self.data_folder, recursive=True, filename_as_id=True
            )
            documents = reader.load_data()
            # Create a storage context
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            # Create an index from the documents and save it to the disk.
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                transformations=transformations,
                store_nodes_override=True,
            )
            storage_context.persist(persist_dir)
            print("Reload data done")
        self.index = index

        self.query_engine = self.index.as_query_engine(
            text_qa_template=self.llm_prompt,
            similarity_top_k=5,
            # response_mode="no_text",
        )

    def add_files_to_index(self, paths: list[str]):
        documents = SimpleDirectoryReader(
            input_files=paths, filename_as_id=True
        ).load_data()
        self.index.refresh_ref_docs(documents)
        # for document in documents:
            # self.index.insert(document)
        self.index.storage_context.persist(self.persist_dir)
        
    def remove_files_from_index(self, file_name):
        ids = self.index.vector_store._collection.get(where={"file_name": file_name})[
            "metadatas"
        ]
        print("DELETE", ids)
        # print(self.index.ref_doc_isnfo)
        for id in ids:
            self.index.delete_ref_doc(id["ref_doc_id"], delete_from_docstore=True)
        self.index.storage_context.persist(self.persist_dir)

        
    def query(self, query):
        return self.query_engine.query(query)

    def get_all_file_list(self):
        return list(self.index.docstore.get_all_ref_doc_info().keys())

    # Query data from the persisted index
    # query_engine = index.as_chat_engine(chat_mode="react", text_qa_template=llm_prompt, verbose=True)
    # print(response)


if __name__ == "__main__":
    agent = GenimiAgent(reload_data=False)
    response = agent.query("Give me education record of Duc Anh?")
    print(response)
    print(len(response.source_nodes))
    print(response.source_nodes)
    # while 1:
    #     req = input("Ask me something:")
    #     response = query_engine.query(req)
    #     print(response)
