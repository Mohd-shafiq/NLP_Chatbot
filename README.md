# NLP_Chatbot
```markdown
# Eldoria Story Chatbot

A lightweight Python chatbot that answers questions about a short fantasy story set in the kingdom of Eldoria. The bot uses sentence-level TF-IDF retrieval with cosine similarity to locate the most relevant sentence in the story and applies a simple heuristic to extract a concise answer.

## Project summary
This repository demonstrates an explainable, local Q&A pipeline tailored to a single narrative. It's intended as a compact reference implementation for retrieval-based question answering on short texts, educational demos, and iterative experimentation.

## Key features
- Fast local retrieval with TF-IDF and cosine similarity (scikit-learn).
- Sentence and word tokenization via NLTK.
- Simple, explainable extraction heuristic for concise answers (names, places, short phrases).
- Interactive CLI for quick question-and-answer sessions.
- Easy to extend: swap the retrieval or extractor for embeddings or NER-based components.

## Prerequisites
- Python 3.8 or newer
- pip

Recommended Python packages:
- nltk
- scikit-learn

## Installation
1. Clone or copy the repository into your working directory.
2. (Optional but recommended) Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   .venv\Scripts\activate      # Windows (PowerShell)
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   # or
   pip install nltk scikit-learn
   ```

## NLTK data
Download required tokenizer data before first run:
```python
import nltk
nltk.download('punkt')
```

## Usage
1. Put the story text in `story.txt` (UTF-8 plain text).
2. Run the chatbot:
   ```bash
   python chatbot.py
   ```
3. Ask natural-language questions about the story at the prompt. Type `exit` to quit.

## How it works
1. The story is split into sentences using NLTK's sentence tokenizer.
2. Sentences are vectorized using scikit-learn's TfidfVectorizer.
3. The user question is vectorized and compared to every sentence via cosine similarity.
4. The top-scoring sentence is returned as a supporting sentence.
5. A lightweight extractor uses nearby keywords and capitalization heuristics to produce a concise answer.

This architecture keeps the pipeline transparent and easy to debug, making it ideal for learning and small-scale projects.

## Example
Question:
- "Who is Elara?"

Sample output:
```
Answer (precise): Elara
Supporting sentence: At the center of this tale was Elara, a young elf archer with silver hair and eyes like emeralds.
```

## Suggestions & improvements
- Replace TF-IDF with semantic sentence embeddings (e.g., sentence-transformers) for better recall on paraphrased questions.
- Replace the capitalization-based extractor with NER or span-selection models for more robust answer extraction.
- Add unit tests for tokenization, vectorization, ranking, and extraction components.
- Support multiple documents and incremental indexing for larger corpora.

## Development & contributing
Contributions and improvements are welcome. A suggested workflow:
1. Fork the repository.
2. Create a descriptive feature branch (e.g., `feat/embeddings`).
3. Add tests and update the README with any new configuration steps.
4. Open a pull request describing the motivation and changes.

Please include unit tests and keep changes modular so the core retrieval pipeline remains easy to reason about.

## Credits & acknowledgements
- Sonet D Thomas — https://github.com/SONET12  
- Naveen Joy — https://github.com/naveen-joy-18

## License
This project is delivered as a demonstration. For public distribution, consider the MIT License — add a `LICENSE` file if you intend to publish this repository.
```
