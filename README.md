# Kenyan_Constitution_RAG
A Retrieval-Augmented Generation (RAG) application that allows users to interactively query the Kenyan Constitution using natural language. It leverages vector embeddings and a language model to provide accurate, context-aware answers backed by constitutional references.
This project creates a smart assistant that answers questions about the Kenyan Constitution.
It reads the entire constitution from a PDF file, breaks it down into small pieces, and then allows you to ask questions and get accurate answers based on what is written inside the document — without guessing.

It also builds a simple website interface where you can interact with the assistant.

How it Works (In Simple Terms)
1. PDF Loading
The project starts by opening and reading the Constitution of Kenya stored in a PDF file.

2. Breaking Down the Document
Because the constitution is very big, the system cuts it into smaller sections (called "chunks") that are easier for the computer to understand.

3. Making the Text Searchable (Embeddings)
Each small chunk is turned into a special format (called an embedding) that allows the computer to quickly find the right information when you ask a question.

4. Saving the Information (Database)
All the chunks and their searchable formats are stored inside a mini database on the computer using a tool called ChromaDB.

5. Asking a Question
When you ask a question like "What is the structure of the Kenyan government?", the system does the following:

Searches through the mini database for parts of the Constitution that are related to your question.

Sends those parts to an advanced AI assistant (Groq's Llama 3 model).

The AI is instructed to answer only using information from the Constitution, not from the internet or its own opinions.

6. Showing the Answer
The system then prints the answer for you — based strictly on the Constitution!

7. Website Interface
The project also creates a small website using FastAPI and Jinja2 templates where:

You can eventually connect the smart assistant.

Right now, it generates a basic page called output.html.


