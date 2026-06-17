# PlantWise 

## Overview

PlantWise is an AI-powered plant knowledge search engine that helps users discover plant information through natural language queries. Instead of relying on exact keyword matching, PlantWise uses semantic search and Retrieval-Augmented Generation (RAG) to understand user intent and provide context-aware answers.

The project was built to demonstrate modern AI search systems by combining NLP embeddings, vector databases, information retrieval, and large language models into a practical application.

---

## Problem Statement

Plant information is often scattered across agricultural databases, medicinal plant repositories, and gardening resources. Traditional search systems require users to know exact plant names or keywords.

PlantWise allows users to ask questions naturally, such as:

* Which medicinal plants help with cough?
* Plants suitable for indoor environments
* Crops requiring less water
* Plants suitable for tropical climates
* Beginner-friendly medicinal plants

The system retrieves relevant information and generates concise, understandable responses.

---

## Features

### Semantic Search

Uses transformer-based embeddings to understand the meaning behind user queries rather than relying only on keywords.

### Retrieval-Augmented Generation (RAG)

Retrieves relevant plant knowledge from a vector database and uses Gemini to generate grounded responses.

### AI-Powered Question Answering

Provides contextual answers for plant care, medicinal uses, cultivation requirements, and environmental suitability.

### Fast Search Infrastructure

Uses ChromaDB vector storage and Sentence Transformers embeddings for efficient semantic retrieval.

### Interactive Web Interface

Built with Next.js and Tailwind CSS to provide a responsive user experience.

---

## Example Queries

* Medicinal plants for cough
* Indoor plants for beginners
* Plants requiring low maintenance
* Crops suitable for high rainfall
* Plants with edible uses
* Plants suitable for tropical regions

---

## Architecture

Dataset Collection
→ Data Processing
→ Text Embeddings (Sentence Transformers)
→ ChromaDB Vector Storage
→ Semantic Retrieval
→ Gemini RAG Layer
→ FastAPI Backend
→ Next.js Frontend

---

## Tech Stack

### Backend

* Python
* FastAPI

### AI / NLP

* Sentence Transformers
* all-MiniLM-L6-v2
* Gemini API
* Retrieval-Augmented Generation (RAG)

### Vector Database

* ChromaDB

### Frontend

* Next.js
* React
* Tailwind CSS

### Data Processing

* Pandas

---

## Dataset

PlantWise is built on a knowledge base containing over 21,000 plant-related records collected from multiple datasets, including medicinal plant information, crop recommendation data, and plant care records.

---

## Future Enhancements

* Hybrid Search (Keyword + Semantic Search)
* Plant Recommendation System
* Climate-Aware Recommendations
* Knowledge Graph Integration
* Image-Based Plant Search
* Personalized Plant Suggestions
* Explainable Recommendations

---

