import re
import emoji
import pandas as pd

# ---------------------------------------------------------------------

# REGEX stuff (ill add explains later)
HTML_TAG_RE = re.compile(r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
URL_RE = re.compile(r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)")
SPEC_CHAR_RE = re.compile(r"[^a-zA-Z0-9\s:_]")

# ---------------------------------------------------------------------

# main text cleaning function
def clean_review_text(text, data_dict=None):

    text = text.lower() # change to lower case

    # html and url related
    text = HTML_TAG_RE.sub("", text) # removes html tags 
    text = URL_RE.sub("", text) # removes urls

    # emojis and special char related
    text = emoji.demojize(text, language="id") # removes emojis
    text = SPEC_CHAR_RE.sub("", text) # removes special characters
    text = text.replace(":", " ") # replaces colons with spaces (mostly for emojis)
    text = text.replace("_", " ") # replaces underscores with spaces (mostly for emojis)

    # white space related
    text = " ".join(text.split()) # removes multiple spaces
    text = text.strip() # removes leading and trailing spaces
    
    # slang / abbreviation cleaning
    if data_dict: # if the optional dict_path is provided
        text_split = text.split() # split the text into individual words
        cleaned_split = [data_dict.get(i, i) for i in text_split] # from the text split, replace each word with its formal counterpart if it exists in the dictionary, otherwise keep the word as is
        text = " ".join(cleaned_split) # join the cleaned words back into a single string with spaces in between

                

    return text

# ---------------------------------------------------------------------

# main guard

if __name__ == "__main__":
    pass