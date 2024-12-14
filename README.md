# JERICHO AI

### An AI tool made for answering Multiple Choice Questions from specific documents.

This Tool was designed specifically for solving multiple choice questions.


### Technical Details

Jericho is based upon priciple concept of vectors embeddings. Embeddings enables us to get desired info from trained LLM's and SLM's based upon our specific dataset.

### Techstack

- Streamlit
- Langchain
- Open AI
- Docker

## Usage

### Method 1: Docker Container

1. Clone the repository
```bash
git clone https://github.com/danishzulfiqar/Jericho.ai.git
```

2. Change directory
```bash
cd Jericho.ai
```

3. Make.env from .env.example
```bash
cat .env.example > .env
```

4. Add openai api key to .env file

5. Build and run docker container
```bash
docker compose up
```

6. Open App at localhost:
http://localhost:8501
