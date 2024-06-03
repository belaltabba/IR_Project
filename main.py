import random
import tkinter as tk
from tkinter import ttk
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import ir_datasets
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re
import string
import pickle
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

class DataSet:
   def __init__(self, dataset, vectorizor, tfidf_matrix) -> None:
        self.dataset = dataset
        self.vectorizor = vectorizor
        self.tfidf_matrix = tfidf_matrix

selected_dataset = None


def load_object(name):
  with open(f'{name}.pkl', 'rb') as file:
    obj = pickle.load(file)
  return obj

tfidf_matrix_1 = load_object('tfidf_matrix_1')
vectorizer_1 = load_object('vectorizer_1')
tfidf_matrix_2 = load_object('tfidf_matrix_2')
vectorizer_2 = load_object('vectorizer_2')

dataset1 = ir_datasets.load('antique/train')
dataset2 = ir_datasets.load('lotte/lifestyle/dev/search')

datasets = [
   DataSet(dataset=dataset1, vectorizor=vectorizer_1, tfidf_matrix=tfidf_matrix_1),
   DataSet(dataset=dataset2, vectorizor=vectorizer_2, tfidf_matrix=tfidf_matrix_2),
]


lemmatizer = WordNetLemmatizer()
stopwords = stopwords.words('english')

shortcut = {
    'p.p.s':'post postscript',
    'u.s.a': 'united states of america',
    'a.k.a': 'also known as',
    'm.a.d': 'Mutually Assured Destruction',
    'a.b.b': 'Asea Brown Boveri',
    's.c.o': 'Santa Cruz Operation',
    'e.t.c': 'etcetera',
    'm.i.t': 'Massachusetts Institute of Technology',
    'v.i.p': 'very important person',
    'us':'united states of america',
    'u.s.':'united states of america',
    'usa':'united states of america',
    'cobol':'common business oriented language',
    'rpm':'red hat package manager',
    'ap':'associated press',
    'gpa':'grade point average',
    'npr':'national public radio',
    'fema':'federal emergency',
    'crt':'cathode ray tube',
    'gm':'grandmaster',
    'fps':'frames per second',
    'pc':'personal computer',
    'pms':'premenstrual syndrome',
    'cia':'central intelligence agency',
    'aids':'acquired immune deficiency syndrome',
    'it\'s':'it is',
    'you\'ve':'you have',
    'what\'s':'what is',
    'that\'s':'that is',
    'who\'s':'who is',
    'don\'t':'do not',
    'haven\'t':'have not',
    'there\'s':'there is',
    'i\'d':'i would',
    'it\'ll':'it will',
    'i\'m':'i am',
    'here\'s':'here is',
    'you\'ll':'you will',
    'cant\'t':'can not',
    'didn\'t':'did not',
    'hadn\'t':'had not',
    'kv':'kilovolt',
    'cc':'cubic centimeter',
    'aoa':'american osteopathic association',
    'rbi':'reserve bank',
    'pls':'please',
    'dvd':'digital versatile disc',
    'bdu':'boise state university',
    'dvd':'digital versatile disc',
    'mac':'macintosh',
    'tv':'television',
    'cs':'computer science',
    'cse':'computer science engineering',
    'iit':'indian institutes of technology',
    'uk':'united kingdom',
    'eee':'electrical and electronics engineering',
    'ca':'california',
    'etc':'etcetera',
    'ip':'internet protocol',
    'bjp':'bharatiya janata party',
    'gdp':' gross domestic product',
    'un':'unitednations',
    'ctc':'cost to company',
    'atm':'automated teller machine',
    'pvt':'private',
    'iim':'indian institutes of management'
    
    }

def expand_contractions(text, shortcut):
    contractions_pattern = re.compile('({})'.format('|'.join(re.escape(key) for key in shortcut.keys())), flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = shortcut.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    return expanded_text

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    

def preprocess(text):

  text = expand_contractions(text, shortcut)

  filtered_tokens = []
  for token in word_tokenize(text):
    token = re.sub(r'\b[0-9]+\b', '', token)
    token = token.translate(str.maketrans('', '', string.punctuation))
    token = token.lower()
    if len(token) > 0 and token not in stopwords:
      filtered_tokens.append(token)

  # lemmatization
  tagged_tokens = pos_tag(filtered_tokens)

  # Lemmatize based on POS tags
  lemmatized_words = [lemmatizer.lemmatize(token, get_wordnet_pos(tag)) for token, tag in tagged_tokens]
  processed_text = ' '.join(lemmatized_words)
  return processed_text


def search(query, dataset, tfidf_matrix, vectorizer, top_n=5):
  normalized_query = preprocess(query)
  query_vec = vectorizer.transform([normalized_query])
  cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
  most_similar_docs_indices = cosine_similarities.argsort()[-top_n:][::-1]

  results = [0] * top_n
  docs_list = list(most_similar_docs_indices)
  for i, doc in enumerate(dataset.docs_iter()):
    if i in docs_list:
      results[docs_list.index(i)] = doc.doc_id

  return results


# Function to simulate a search and return a list of IDs
def search_function(query):
    dataset = datasets[selected_dataset]
    if query:
        results = search(query, dataset.dataset, dataset.tfidf_matrix, dataset.vectorizer)
        return results  # Dummy IDs
    return []


# Function to be called when the search button is pressed
def on_search():
    query = search_entry.get()
    if not query:
        result_label.config(text="Please enter a search query.")
        return
    result_ids = search_function(query)
    result_text = "Results:\n" + "\n".join(str(id) for id in result_ids)
    result_label.config(text=result_text)


# Function to enable the search bar and button when a dataset is chosen
def enable_search(*args):
    global selected_dataset
    selected_dataset = dataset_var.get()
    search_entry.config(state='normal')
    search_button.config(state='normal')


######################### Create the main application window #######################################
root = tk.Tk()
root.title("Elegant Search Application")

# Get screen width and height
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Set window size to 3/4 of screen width and height
window_width = int(screen_width * 0.75)
window_height = int(screen_height * 0.75)

# Center the window on the screen
position_top = int(screen_height / 2 - window_height / 2)
position_right = int(screen_width / 2 - window_width / 2)
root.geometry(f"{window_width}x{window_height}+{position_right}+{position_top}")

# Style configurations
bg_color = "#282c34"
fg_color = "#61dafb"
font_large = ("Helvetica", 24)
font_small = ("Helvetica", 18)
font_toggle = ("Helvetica", 20)

# Set the background color
root.configure(bg=bg_color)

# Create and place the label
choose_label = tk.Label(root, text="Choose a dataset", font=font_large, bg=bg_color, fg=fg_color)
choose_label.pack(pady=20)
choose_label.place(relx=0.5, rely=0.2, anchor='center')

# Create and place the toggle buttons
dataset_var = tk.StringVar(value="none")

toggle_frame = tk.Frame(root, bg=bg_color)
toggle_frame.pack(pady=20)
toggle_frame.place(relx=0.5, rely=0.3, anchor='center')

dataset1_button = ttk.Radiobutton(toggle_frame, text="Dataset 1", variable=dataset_var, value=0,
                                  command=enable_search, style="TRadiobutton")
dataset2_button = ttk.Radiobutton(toggle_frame, text="Dataset 2", variable=dataset_var, value=1,
                                  command=enable_search, style="TRadiobutton")

dataset1_button.grid(row=0, column=0, padx=10)
dataset2_button.grid(row=0, column=1, padx=10)

# Style the toggle buttons
style = ttk.Style()
style.configure("TRadiobutton", font=font_toggle, background=bg_color, foreground=fg_color)

# Create and place the search bar in the middle of the screen
search_entry = tk.Entry(root, width=50, font=font_large, bd=2, state='disabled')
search_entry.pack(pady=20)
search_entry.place(relx=0.5, rely=0.5, anchor='center')

# Create and place the search button
search_button = tk.Button(root, text="Search", command=on_search, font=font_large, bg=fg_color, fg=bg_color, bd=0,
                          padx=20, pady=10, state='disabled')
search_button.pack(pady=20)
search_button.place(relx=0.5, rely=0.6, anchor='center')

# Create and place the result label
result_label = tk.Label(root, text="", font=font_large, bg=bg_color, fg=fg_color)
result_label.pack(pady=20)
result_label.place(relx=0.5, rely=0.84, anchor='center')

# Start the Tkinter event loop
root.mainloop()
