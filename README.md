# EduMate - Your Study Buddy

## Overview
This project implements a **Retrieval-Augmented Generation (RAG) system** using **PDF data** and serves it via a **FastAPI** endpoint. Additionally, an **Agent** is integrated to utilize tools and orchestrate the whole system. This was built using **langgraph and langchain**.

## Features
1. **RAG System**: A FastAPI endpoint to handle user queries, using a Vector Database for document retrieval.
2. **Agent**: An agent capable of determining when to query the VectorDB and performing additional actions.
3. **Additional Tools**: Integrated an extra tool- a quiz/test tool that provides questions of 2 marks + 5 marks + 10 marks that the Agent can invoke based on the query.
4. **Voice Integration**: Integrated voice feature using Sarvam API - Text to Speech - to add voice responses to the Agent.

## Demo

https://github.com/user-attachments/assets/ab6790d3-0dfe-4468-b456-25da8268f6c1

