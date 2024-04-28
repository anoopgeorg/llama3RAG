from werkzeug.datastructures import FileStorage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_chroma import Chroma
from pathlib import Path

from src.configurations.config import ConfigManager
from src.entities.data_entity import UpdateVault


class Vault:
    def __init__(self):
        # pull all the configurations
        self.config_mgr = ConfigManager()
        self.config = self.config_mgr.getVaultConfig()

    # Update the vault folder
    def updateVault(self, obj: UpdateVault):
        if obj.type == "file":
            print("Updating Knowledge Base")
            file_name = obj.file.filename
            save_file = str(self.config.knowledge_dir) + file_name
            obj.file.save(save_file)
            print(f"File saved : {save_file}")
            chunks = self.getDocChunks(file_path=save_file)
            chroma_db = self.chromaInsert(
                chunks=chunks,
                collection=self.config.knowledge_coll_name,
                db_path=self.config.chroma_db,
            )
            return {"status": "Successful"}
        elif obj.type == "memory":
            print("entry Memory")
            print("Updating memory")
            return {"status": "Successful"}

    def getDocChunks(self, file_path: Path):
        print("Document chunks extract started....")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=80,
            length_function=len,
            is_separator_regex=False,
        )
        loader = PDFPlumberLoader(file_path=file_path)
        docs = loader.load_and_split()
        chunks = text_splitter.split_documents(docs)
        print(f"Number of docs :{len(docs)}, number of chunks: {len(chunks)}")
        print("Document chunks extract ended....")
        return chunks

    def chromaInsert(self, chunks: list, collection: str, db_path: Path):
        print("Chroma DB insert started....")
        fast_embed = FastEmbedEmbeddings()
        chroma_db = Chroma.from_documents(
            collection_name=collection,
            embedding=fast_embed,
            persist_directory=str(db_path),
            documents=chunks,
        )
        print("Chroma DB insert ended....")
        return chroma_db

    def loadChroma(self, db_path: Path, coll_name: str):
        print("loading Vector store")
        fast_embed = FastEmbedEmbeddings()
        chroma = Chroma(
            persist_directory=str(db_path),
            collection_name=coll_name,
            embedding_function=fast_embed,
        )
        print("Vector store Loaded")
        return chroma

    def getRetriever(self, retriever_type="knowledge"):
        db_path = self.config.chroma_db
        coll_name = (
            self.config.knowledge_coll_name
            if retriever_type == "knowledge"
            else self.config.memory_coll_name
        )
        chroma_db = self.loadChroma(db_path=db_path, coll_name=coll_name)
        retriever = chroma_db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 5, "score_threshold": 0.6},
        )
        print("RETRIEVER CREATED FROM VECTOR STORE")
        return retriever
