from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
import gradio as gr
from dotenv import load_dotenv


load_dotenv()


DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"


embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.5)


vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)


num_results = 5
retriever = vector_store.as_retriever(search_kwargs={'k': num_results})

def stream_response(message, history):
 
    docs = retriever.invoke(message)

    
    knowledge = ""
    for doc in docs:
        knowledge += doc.page_content + "\n\n"
    print("\n--- WHAT THE DATABASE FOUND ---")
    print(knowledge)


    rag_prompt = f"""
    You are an assistant which answers questions based on knowledge which is provided to you.
    While answering, you don't use your internal knowledge,
    but solely the information in the "The knowledge" section.
    You don't mention anything to the user about the provided knowledge.

    The question: {message}

    Conversation history: {history}

    The knowledge: {knowledge}
    """

    partial_message = ""
    
    for response in llm.stream(rag_prompt):
      
        if isinstance(response.content, str):
            partial_message += response.content
        elif isinstance(response.content, list) and len(response.content) > 0:
            if isinstance(response.content[0], dict):
                partial_message += response.content[0].get("text", "")
            else:
                partial_message += str(response.content[0])
                
        yield partial_message


chatbot = gr.ChatInterface(stream_response, textbox=gr.Textbox(placeholder="Send to the LLM..."))

chatbot.launch(share=True)