# ChatBot-Project
This is a locally hosted Retrieval-Augmented Generation (RAG) chatbot built using Python, LangChain, ChromaDB, Gradio, and Google's Gemini API. This project processes PDF documents, stores their embeddings in a vector database, and uses an LLM to accurately answer questions based strictly on the provided text.

**Setup Instructions**
1) Clone Repository
   Download or clone the repository to local machine.
2) Install necessary libraries of python
   Ensure you have Python installed, then run the following command in your terminal
   `pip install langchain-google-genai langchain-chroma gradio pypdf python-dotenv`
3) Set your Gemini API Key
   Create a `.env` file in the root directory and add your Google Gemini API key:
   `GOOGLE_API_KEY="your_api_key_here"`
4) Add data
  Create a new folder named data and place the PDF document you want to chat with inside the `data/` folder.
5) Build database
   Run the ingestion script to process the PDF and build the Chroma vector database:
   `python ingest_database.py`
6) Start Chatbot
   Launch the Gradio user interface:
   `python chatbot.py`
   Click the local URL


   Example Interaction
   Question 1 : What is the Transformer architecture?
   Answer : The Transformer is a network architecture based solely on attention mechanisms, dispensing with recurrence and convolutions. It follows an encoder-decoder structure using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder. 

The architecture replaces the recurrent layers most commonly used in encoder-decoder models with multi-headed self-attention. The encoder and decoder are connected through an attention mechanism. For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent layers.

Question 2:How does self-attention work?
Answer : Self-attention, also referred to as intra-attention, is an attention mechanism that relates different positions of a single sequence to compute a representation of that sequence. 

In this mechanism, the queries, keys, and values all originate from the same place, such as the output of the previous layer. Within the encoder, this allows each position to attend to all positions in the previous layer. Similarly, self-attention layers in the decoder allow each position in the decoder to attend to other positions within the decoder sequence.
   
