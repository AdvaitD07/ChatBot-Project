import time
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import uuid
import os
from dotenv import load_dotenv


load_dotenv()

DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"


embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)
loader = PyPDFDirectoryLoader(DATA_PATH)
raw_documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)
chunks = text_splitter.split_documents(raw_documents)
uuids = [str(uuid.uuid4()) for _ in range(len(chunks))]
print(f"SUCCESS: Split the PDF into {len(chunks)} chunks!")
print("Now saving to ChromaDB... this might take a minute...")
batch_size = 50
print(f"Adding documents to database in batches of {batch_size}...")
for i in range(0, len(chunks), batch_size):
    batch_chunks = chunks[i : i + batch_size]
    batch_uuids = uuids[i : i + batch_size]

    vector_store.add_documents(documents=batch_chunks, ids=batch_uuids)
    print(
        f"Successfully processed batch {i // batch_size + 1} of {(len(chunks) // batch_size) + 1}"
    )

    if i + batch_size < len(chunks):
        print(
            "Sleeping for 45 seconds to respect Google's free API limits... (Do not close terminal)"
        )
        time.sleep(45)

print("\n--- Database successfully created! ---")
