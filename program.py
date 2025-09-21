# --- Step 0: Import libraries ---
import sys
print("Using Python from:", sys.executable)

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Download necessary NLTK data
nltk.download('punkt')  # For sentence tokenization
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize, word_tokenize

# --- Step 1: Load the story ---
with open("story.txt", "r", encoding="utf-8") as f:
    story = f.read()

# --- Step 2: Split story into sentences ---
sentences = sent_tokenize(story)
print(f"Story loaded with {len(sentences)} sentences.")

# --- Step 3: Train TF-IDF model ---
# This converts sentences into numerical vectors representing their word importance
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(sentences)
print("TF-IDF model trained and ready!")

# --- Step 4: Define helper function to extract precise answer ---
def extract_precise_answer(sentence, question):
    """
    This function tries to return the most relevant word(s) from the sentence based on the question.
    1. It tokenizes the sentence.
    2. Looks for keywords in the question (e.g., 'kingdom', 'city', 'forest').
    3. Returns the first capitalized word after the keyword.
    4. Falls back to the first capitalized word in the sentence if no keyword is found.
    """
    words = word_tokenize(sentence)
    # Convert to lowercase for keyword matching
    lower_words = [w.lower() for w in words]
    
    # Identify keywords from the question
    question_keywords = [w.lower() for w in word_tokenize(question) if w.isalpha()]
    
    # Search for capitalized words after keywords
    for i, word in enumerate(lower_words):
        if word in question_keywords and i+1 < len(words):
            # Look for the next capitalized word
            for j in range(i+1, len(words)):
                if words[j][0].isupper():
                    return words[j]
    
    # Fallback: return first capitalized word in the sentence
    for word in words:
        if word[0].isupper():
            return word
    
    # If nothing found, return the longest word (rare case)
    return max(words, key=len)

# --- Step 5: Interactive question-answer loop ---
while True:
    question = input("\nAsk a question (or type 'exit'): ")
    if question.lower() == "exit":
        break

    # Step 5a: Convert the question into TF-IDF vector
    q_vec = vectorizer.transform([question])

    # Step 5b: Compute similarity between question and all sentences
    sim = cosine_similarity(q_vec, X)

    # Step 5c: Find the index of the most relevant sentence
    idx = sim.argmax()
    best_sentence = sentences[idx]

    # Step 5d: Extract candidate answer
    answer = extract_precise_answer(best_sentence, question)

    # Step 5e: Show answer + supporting sentence
    print("Answer (precise):", answer)
    print("Supporting sentence:", best_sentence)
