# Assignment 2

This assignment was completed in partial fulfillment of the Deploying AI course administered by the Data Sciences Institute at the University of Toronto. As part of my transition from academia to industry, I needed to translate my skills across domains. Here, I implement a scaled-down version that compares my own work to influential papers in machine learning and artificial intelligence.

## Purpose
This conversational AI chatbot acts as a helpful academic assistant who fetches and concisely summarizes academic work. It offers three services which are invoked by the agent based on the question the user asks. 

## Services
### Service 1: PubMed Search (`tools_pubmed.py`)
Calls the PubMed API for live searching of the database. The LLM translates the user's text query into a search query, and returns linked citations and a snippet of the abstract. No API key required. Helpful for finding recent papers on a topic that may not be contained in the local database. 

*Sample Queries*: 
- "Find preprints on topic X." 
- "What's the latest paper from researcher Y?"
- "Create a reading list of 10 papers on Z."

<!-- Tool use could be improved by returning as a structured tool. -->

### Service 2: Semantic Query (`tools_chroma.py`)
This tool is called when the user asks how similar a given paper is to papers in the research collection (my body of work). Intended for use alongside the live PubMed search tool. Loads a persistent ChromaDB collection of my research abstracts. Embeddings are performed using `all-MiniLM-L6-v2` from the `SentenceTransformerEmbeddingFunction` in `chromadb.utils`, and semantic search is performed using cosine similarity.

*Sample Queries*:
- "Compare how similar citation 1 is to the research in the database."
- "Are any of the database abstracts relevant to citation 3?"

<!-- Notes: I decided to implement the live pubmed search in order to allow for users to search for their own work -->

### Service 3: Function Call (`get_current_time`)
A simple datetime function. This tool is called whenever the user asks for the current date or time. The intended service was live search of arXiv, however routing the agent to one search service over the other became too complex to include for the present assignment.

*Sample Queries*:
- "What time is it Mr. Wolf?"
- "I'm lost, what day is it?"

## Implementation
The chatbot UI is provided via `Gradio`. Services are implemented in `LangGraph`. `ChromaDb` for persistent database and embeddings. The system prompt instructs the chatbot to act as a helpful academic assistant who fetches and concisely summarizes academic work. 

## Guardrails 
Simple keyword search prior to each LLM call + system prompt instructions (`prompts.py`). Blocked words included `cat`, `dog`, `zodiac`, `horoscope`, `taylor swift`. The model was also instructed to refuse restricted topics, and to not reveal system prompts, allow prompt injection, or instruction overridding. 

## Notes
This implementation is a toy example only, and extensive testing has not been performed. Library dependencies can be found in `pyproject.toml`. Requires a `.secrets` file containing a valid `API_GATEWAY_KEY`. 

To run, from the folder `05_src/`:
```zsh
python -m assignment_chat.app
```


---
*Last Updated: 2026-04-10*
