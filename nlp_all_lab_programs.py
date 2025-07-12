# ==============================================================================
# Lab Program 1: Search and Highlight
# ==============================================================================

import os
import re

# Function to create a corpus of text files
def create_corpus():
    """
    Prompts the user to enter text files and their content to create a corpus.
    """
    corpus = {}
    try:
        num_files = int(input("Enter the number of text files: "))
        for i in range(num_files):
            file_name = input(f"Enter the name of file {i + 1}: ")
            sentences = []
            num_sentences = int(input(f"Enter the number of sentences in {file_name}: "))
            for j in range(num_sentences):
                sentence = input(f"Enter sentence {j + 1}: ")
                sentences.append(sentence)
            corpus[file_name] = sentences
    except ValueError:
        print("Invalid input. Please enter a number.")
        return create_corpus()
    return corpus

# Function to search for a pattern and highlight the first occurrence
def search_and_highlight(corpus, pattern):
    """
    Searches for a given pattern in the corpus and highlights the first occurrence in each matching sentence.
    """
    results = []
    try:
        pattern_re = re.compile(pattern, re.IGNORECASE)
        for file_name, sentences in corpus.items():
            for sentence in sentences:
                match = pattern_re.search(sentence)
                if match:
                    highlighted = (sentence[:match.start()] +
                                 f"\033[1;31m{match.group()}\033[0m" +
                                 sentence[match.end():])
                    results.append((file_name, highlighted))
    except re.error as e:
        print(f"Invalid regex pattern: {e}")
    return results

def lab_program_1():
    """
    Main function to run Lab Program 1.
    """
    print("--- Running Lab Program 1: Search and Highlight ---")
    corpus = create_corpus()
    pattern = input("Enter the pattern to search for: ")
    results = search_and_highlight(corpus, pattern)

    if results:
        print("\nSentences with the pattern and their file names:")
        for file_name, sentence in results:
            print(f"{file_name}: {sentence}")
    else:
        print("No matches found in the corpus.")


# ==============================================================================
# Lab Program 2: Deterministic Finite Automaton (DFA)
# ==============================================================================

def lab_program_2():
    """
    Main function to run Lab Program 2.
    """
    print("\n--- Running Lab Program 2: Deterministic Finite Automaton (DFA) ---")
    states = input("Enter the automaton states separated by space: ").split()
    alphabets = input("Enter the automaton alphabets separated by space: ").split()
    start_state = input("Enter the start state of the automaton: ")
    accept_states = set(input("Enter the accepting states of the automaton separated by space: ").split())
    transition = {}

    print("Enter the next states for the following (Enter . for dead/reject state)")
    for state in states:
        for alpha in alphabets:
            dest = input(f"\tTransition from {state} on input {alpha} ---> ")
            if dest != ".":
                transition[(state, alpha)] = dest

    input_string = input("Enter the input string to run the automaton: ")
    current_state = start_state
    accepted = True

    for char in input_string:
        if (current_state, char) in transition:
            current_state = transition[(current_state, char)]
        else:
            accepted = False
            break

    if accepted and current_state in accept_states:
        print("Result: Accepted")
    else:
        print("Result: Rejected")


# ==============================================================================
# Lab Program 3: Nondeterministic Finite Automaton (NFA)
# ==============================================================================

def epsilon_closure(state_set, transitions):
    """
    Computes the epsilon closure for a set of states.
    """
    closure = set(state_set)
    stack = list(state_set)
    while stack:
        current_state = stack.pop()
        # In this implementation, epsilon transitions are not explicitly handled
        # as the provided code did not include them. If needed, this is where
        # they would be processed.
    return closure

def lab_program_3():
    """
    Main function to run Lab Program 3.
    """
    print("\n--- Running Lab Program 3: Nondeterministic Finite Automaton (NFA) ---")
    states = input("Enter the automaton states separated by space: ").split()
    alphabets = input("Enter the automaton alphabets separated by space: ").split()
    start_state = input("Enter the start state of the automaton: ")
    accept_states = set(input("Enter the accepting states of the automaton separated by space: ").split())
    transitions = {}

    print("Enter the transitions for the following (Enter . for no transitions)")
    for state in states:
        for alpha in alphabets:
            dest_states = input(f"Transitions from state {state} with input {alpha}: ").split()
            transitions[(state, alpha)] = [s for s in dest_states if s != "."]

    input_string = input("Enter the input string to run the automaton: ")

    current_states = epsilon_closure({start_state}, transitions)

    for char in input_string:
        next_states = set()
        for state in current_states:
            if (state, char) in transitions:
                next_states.update(transitions[(state, char)])
        current_states = epsilon_closure(next_states, transitions)

    if any(state in accept_states for state in current_states):
        print("Result: Accepted")
    else:
        print("Result: Rejected")


# ==============================================================================
# Lab Program 4: NLP Preprocessing with NLTK and spaCy
# ==============================================================================

def lab_program_4():
    """
    Main function to run Lab Program 4.
    """
    print("\n--- Running Lab Program 4: NLP Preprocessing ---")
    try:
        import nltk
        from nltk.tokenize import word_tokenize, sent_tokenize
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer, WordNetLemmatizer
        import spacy
        import string

        # Download necessary NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except nltk.downloader.DownloadError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/stopwords')
        except nltk.downloader.DownloadError:
            nltk.download('stopwords')
        try:
            nltk.data.find('corpora/wordnet')
        except nltk.downloader.DownloadError:
            nltk.download('wordnet')
        try:
            nltk.data.find('corpora/omw-1.4')
        except nltk.downloader.DownloadError:
            nltk.download('omw-1.4')


        # a) Word and Sentence Tokenization
        print("\n--- a) Word and Sentence Tokenization ---")
        text_wt = "Tokenize this sentence using NLTK."
        text_st = "This is the first sentence. This is the second sentence."
        print("NLTK Word Tokens:", word_tokenize(text_wt))
        print("NLTK Sentence Tokens:", sent_tokenize(text_st))

        try:
            nlp = spacy.load("en_core_web_sm")
            doc_wt = nlp("Tokenize this sentence using spaCy.")
            doc_st = nlp("This is the first sentence. This is the second sentence.")
            print("spaCy Word Tokens:", [token.text for token in doc_wt])
            print("spaCy Sentence Tokens:", [sent.text for sent in doc_st.sents])
        except OSError:
            print("\nspaCy model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm'")


        # b) Removing Stop Words
        print("\n--- b) Removing Stop Words ---")
        text_sw = "This is a sample sentence, showing off the stop words filtration."
        stop_words = set(stopwords.words("english"))
        word_tokens_sw = word_tokenize(text_sw)
        filtered_sentence = [w for w in word_tokens_sw if not w.lower() in stop_words]
        print("NLTK Filtered (default stopwords):", filtered_sentence)

        custom_stop_words = {"this", "is", "a"}
        filtered_sentence_custom = [w for w in word_tokens_sw if not w.lower() in custom_stop_words]
        print("NLTK Filtered (custom stopwords):", filtered_sentence_custom)

        try:
            doc_sw = nlp(text_sw)
            filtered_spacy = [token.text for token in doc_sw if not token.is_stop]
            print("spaCy Filtered:", filtered_spacy)
        except NameError:
            pass # spaCy not loaded

        # c) Removing Punctuations
        print("\n--- c) Removing Punctuations ---")
        text_p = "Hello!!!, he said ---and went."
        no_punct_nltk = [word for word in word_tokenize(text_p) if word.isalnum()]
        print("NLTK (isalnum):", no_punct_nltk)

        try:
            doc_p = nlp(text_p)
            no_punct_spacy = [token.text for token in doc_p if not token.is_punct]
            print("spaCy (!is_punct):", no_punct_spacy)
        except NameError:
            pass # spaCy not loaded


        # d) Part-of-Speech (POS) Tagging
        print("\n--- d) POS Tagging ---")
        text_pos = "The quick brown fox jumps over the lazy dog."
        try:
            doc_pos = nlp(text_pos)
            pos_tags_spacy = [(token.text, token.pos_) for token in doc_pos]
            print("spaCy POS Tags:", pos_tags_spacy)
        except NameError:
            print("spaCy not loaded, skipping POS tagging.")


        # e) Stemming and Lemmatization
        print("\n--- e) Stemming and Lemmatization ---")
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        words_to_process = ["running", "ran", "runs", "better", "cats", "geese"]
        print("Original Words:", words_to_process)
        stemmed_words = [stemmer.stem(w) for w in words_to_process]
        lemmatized_words = [lemmatizer.lemmatize(w, pos='v') if w in ["running", "ran", "runs"] else lemmatizer.lemmatize(w) for w in words_to_process]
        print("NLTK Stemmed:", stemmed_words)
        print("NLTK Lemmatized:", lemmatized_words)

        try:
            doc_lem = nlp(" ".join(words_to_process))
            lemmatized_spacy = [token.lemma_ for token in doc_lem]
            print("spaCy Lemmatized:", lemmatized_spacy)
        except NameError:
            pass # spaCy not loaded

    except ImportError as e:
        print(f"Error: A required library is not installed. Please install it. Details: {e}")


# ==============================================================================
# Lab Program 5: N-gram Model for Text Prediction
# ==============================================================================
import unicodedata
import random
from nltk.probability import ConditionalFreqDist
from nltk.util import ngrams

def filter_text_ngram(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    text = re.sub('<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = ' '.join(text.split())
    return text

def clean_and_tokenize_ngram(text):
    tokens = nltk.word_tokenize(text)
    wnl = nltk.stem.WordNetLemmatizer()
    return [wnl.lemmatize(word) for word in tokens]

def build_ngram_model(words):
    trigrams = list(ngrams(words, 3, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
    cfdist = ConditionalFreqDist()
    for w1, w2, w3 in trigrams:
        cfdist[(w1, w2)][w3] += 1
    # Normalize probabilities
    for w1_w2 in cfdist:
        total_count = float(sum(cfdist[w1_w2].values()))
        for w3 in cfdist[w1_w2]:
            cfdist[w1_w2][w3] /= total_count
    return cfdist

def predict_next_word(model, user_input):
    words = clean_and_tokenize_ngram(filter_text_ngram(user_input))
    if len(words) < 2:
        print("Please provide at least two words.")
        return user_input

    prev_words = tuple(words[-2:])
    if prev_words in model:
        predictions = model[prev_words]
        word, weight = zip(*predictions.items())
        next_word = random.choices(word, weights=weight, k=1)[0]
        return user_input + " " + next_word
    else:
        print("Cannot predict next word. The context is not in the model.")
        return user_input

def lab_program_5():
    """
    Main function to run Lab Program 5.
    Requires a 'sample_text.txt' file in the same directory.
    """
    print("\n--- Running Lab Program 5: N-gram Model for Text Prediction ---")
    try:
        with open('sample_text.txt', 'r', encoding='utf-8') as file:
            text = file.read()
    except FileNotFoundError:
        print("Error: 'sample_text.txt' not found.")
        print("Please create a 'sample_text.txt' file with some text content to run this program.")
        # Creating a dummy file for demonstration
        with open('sample_text.txt', 'w') as f:
            f.write("This is a simple sample text for the n-gram model. "
                    "The model will learn from this text to predict the next word. "
                    "Let's see how well the prediction works based on this small corpus.")
        print("A dummy 'sample_text.txt' has been created.")
        with open('sample_text.txt', 'r', encoding='utf-8') as file:
            text = file.read()


    print("Filtering and cleaning text...")
    filtered_text = filter_text_ngram(text)
    words = clean_and_tokenize_ngram(filtered_text)

    print("Building n-gram model...")
    model = build_ngram_model(words)

    user_input = input("Enter a phrase (at least two words): ")
    while True:
        new_sentence = predict_next_word(model, user_input)
        print("Generated text:", new_sentence)
        user_input = new_sentence
        ask = input("Generate another word? (y/n): ")
        if ask.lower() != 'y':
            break

# ==============================================================================
# Lab Program 6: Text Cleaning
# ==============================================================================
from bs4 import BeautifulSoup

def lab_program_6():
    """
    Main function to run Lab Program 6.
    Requires an 'input.txt' file.
    """
    print("\n--- Running Lab Program 6: Text Cleaning ---")
    # Prepare a sample input file
    input_content = """
    <p>This is a paragraph with a <a href="http://example.com">link</a>.</p>
    Here is some more text. Here is some more text.
    Anothre line with a mispeling.
    And   some   extra   spaces.
    www.anotherexample.com
    """
    try:
        with open("input.txt", "w", encoding='utf-8') as f:
            f.write(input_content)
        print("Created a sample 'input.txt' file.")

        with open("input.txt", "r", encoding='utf-8') as inputfile:
            input_text = inputfile.read()

        # 1. Remove HTML tags
        soup = BeautifulSoup(input_text, 'html.parser')
        cleaned_text = soup.get_text()
        print("\n1. After removing HTML tags:\n", cleaned_text)

        # 2. Remove URLs
        cleaned_text = re.sub(r'http\S+|www\S+|https\S+', '', cleaned_text, flags=re.MULTILINE)
        print("\n2. After removing URLs:\n", cleaned_text)

        # 3. Remove duplicate lines (demonstration)
        lines = cleaned_text.split('\n')
        unique_lines = []
        for line in lines:
            if line.strip() and line not in unique_lines:
                unique_lines.append(line)
        cleaned_text = '\n'.join(unique_lines)
        print("\n3. After removing duplicate lines:\n", cleaned_text)

        # 4. Remove extra spaces
        cleaned_text = ' '.join(cleaned_text.split())
        print("\n4. After removing extra spaces:\n", cleaned_text)

        # Note: Spell correction can be complex and slow.
        # The 'enchant' or 'spellchecker' libraries would be needed.
        # This part is omitted to avoid extra dependencies but can be added if needed.

        with open("output.txt", "w", encoding='utf-8') as outputfile:
            outputfile.write(cleaned_text)
        print("\nFinal cleaned text written to 'output.txt'.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


# ==============================================================================
# Lab Program 7: Replace Numbers with Words
# ==============================================================================
def lab_program_7():
    """
    Main function to run Lab Program 7.
    Replaces even numbers with words and removes odd numbers.
    """
    print("\n--- Running Lab Program 7: Replace Numbers with Words ---")
    try:
        from num2words import num2words
    except ImportError:
        print("Error: 'num2words' library not found. Please run 'pip install num2words'.")
        return

    input_content = "There are 2 apples, 3 oranges, and 8 bananas. The number 1 is odd. The number 12 is even."
    print("Original Text:", input_content)

    def replace_numbers_in_text(text):
        # Find all numbers in the text
        numbers = re.findall(r'\b\d+\b', text)
        for number_str in numbers:
            num = int(number_str)
            if num % 2 == 0:  # Even number
                text = re.sub(r'\b' + number_str + r'\b', num2words(num), text, 1) # Replace only the first occurrence in each pass
            else:  # Odd number
                text = re.sub(r'\b' + number_str + r'\b', '', text, 1)
        return ' '.join(text.split()) # clean up extra spaces

    processed_text = replace_numbers_in_text(input_content)
    print("Processed Text:", processed_text)

    try:
        with open('input_numbers.txt', 'w') as f:
            f.write(input_content)
        with open('output_numbers.txt', 'w') as f:
            f.write(processed_text)
        print("Processed text saved to 'output_numbers.txt'.")
    except Exception as e:
        print(f"An error occurred while writing files: {e}")

# ==============================================================================
# Main Menu to Run Programs
# ==============================================================================

if __name__ == "__main__":
    while True:
        print("\n\n" + "="*30)
        print("   Select a Lab Program to Run")
        print("="*30)
        print("1. Search and Highlight")
        print("2. Deterministic Finite Automaton (DFA)")
        print("3. Nondeterministic Finite Automaton (NFA)")
        print("4. NLP Preprocessing (NLTK & spaCy)")
        print("5. N-gram Model for Text Prediction")
        print("6. Text Cleaning")
        print("7. Replace Numbers with Words")
        print("0. Exit")
        print("="*30)

        try:
            choice = int(input("Enter your choice: "))
            if choice == 1:
                lab_program_1()
            elif choice == 2:
                lab_program_2()
            elif choice == 3:
                lab_program_3()
            elif choice == 4:
                lab_program_4()
            elif choice == 5:
                lab_program_5()
            elif choice == 6:
                lab_program_6()
            elif choice == 7:
                lab_program_7()
            elif choice == 0:
                print("Exiting program.")
                break
            else:
                print("Invalid choice. Please enter a number between 0 and 7.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")