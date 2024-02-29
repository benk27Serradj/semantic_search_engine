# Import------------------------------------------------------------------
import spacy
import string
import gensim
import operator
import re


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Create a list of stopwords
stop_words = set(stopwords.words('english'))

# Initialize limitizers
lemmatizer = WordNetLemmatizer()

#-------------------------------------------------------------------------
def text_tokenizer(text):
    
    """
    Tokenizes and processes a given text.
    
    Parameters:
    - text(str) : The input text to be tokenized.
    
    Returns:
    - list of str: A list of lemmatized tokens after processing the input text.
    
    Example:
    text = "This is a simple text for tokenization and processing."
    result = text_tokenizer(text)
    print(result)
    
    >>> ['simple', 'text', 'tokenization', 'processing']
    
    """
    
    # Remove any characters that are not uppercase letters, lowercase letters, or white space character.
    cleaned_text = re.sub(r'[^A-Za-z\s]', '', text) 
    
    # Replace conecutive spaces with a single space.
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    # Creating token objects      
    tokens = word_tokenize(cleaned_text)
    
    
    lowercase_tokens = [token.lower() for token in tokens]
    
    # remove stop words
    filtered_tokens = [token for token in lowercase_tokens if token not in stop_words]
    
    # limitize the tokens
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    
    return lemmatized_tokens

#-------------------------------------------------------------------------


#-------------------------------------------------------------------------

def text_cleaner(text):
    
    """
    Cleans a given text by removing non-alphabetic characters and consecutive spaces.
    
    Parameters:
    - text(str): The input text to be cleaned.
    
    Returns:
    - str: The cleaned text
    
    Example:
    dirty_text = "This  ^&^&   is     !^^&*(()  a  11    no longer a     dirty    text 0- 323       ....... "
    cleaned_text = text_cleaner(dirty_text)
    print(cleaned_text)
    
    >>> This is a no longer a dirty text
    
    """
    
    # Remove any characters that are not uppercase letters, lowercase letters, or white space character.
    cleaned_text = re.sub(r'[^A-Za-z\s]', '', text) 
    
    # Replace consecutive spaces with a single space.
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text   
#-------------------------------------------------------------------------

