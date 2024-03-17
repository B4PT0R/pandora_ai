from openai import OpenAI # for calling the OpenAI API
import tiktoken  # for counting tokens
import numpy as np
import json
import os

encoding=tiktoken.get_encoding("cl100k_base")

def get_text(file):
    if os.path.isfile(file):
        with open(file) as f:
            content=f.read()
    else:
        content=''
    return content

def normalize(vect):
    inv_norm=1.0/np.linalg.norm(vect,ord=2)
    return [x_i*(inv_norm) for x_i in vect]

def token_count(string):
    """
    count tokens in a string
    """
    return len(encoding.encode(string))

def split_string(string, delimiters):
    """
    splits a string according to a chosen set of delimiters
    """
    substrings = []
    current_substring = ""
    i = 0
    while i < len(string):
        for delimiter in delimiters:
            if string[i:].startswith(delimiter):
                current_substring += delimiter
                if current_substring:
                    substrings.append(current_substring)
                    current_substring = ""
                i += len(delimiter)
                break
        else:
            current_substring += string[i]
            i += 1
    if current_substring:
        substrings.append(current_substring)
    return substrings

def split_text(text, max_tokens):
    """
    split a text into chunks of maximal token length, not breaking sentences in halves.
    """
    # Tokenize the text into sentences
    sentences = split_string(text,delimiters=["\n",". ", "! ", "? ", "... ", ": ", "; "])
    
    chunks = []
    current_chunk = ""
    current_token_count = 0
    
    for sentence in sentences:
        sentence_token_count = token_count(sentence)
        
        # If adding the next sentence exceeds the max_tokens limit,
        # save the current chunk and start a new one
        if current_token_count + sentence_token_count > max_tokens:
            chunks.append(current_chunk.strip())
            current_chunk = ""
            current_token_count = 0
        
        current_chunk += sentence
        current_token_count += sentence_token_count
    
    # Add the remaining chunk if it's not empty
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

class Retriever:

    def __init__(self,chunk_size=100,dimensions=250,folder="./documents"):
        self.store={}
        self.chunk_size=chunk_size
        self.dimensions=dimensions
        self.folder=folder
        self.client=OpenAI(timeout=3)
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    def get_loaded(self):
        return [dict(title=doc["title"],description=doc["description"]) for doc in self.store.values()]
    
    def get_titles(self):
        return [os.path.basename(file).split('.')[0] for file in os.listdir(self.folder)]

    def embed(self,chunks):
        success=False
        while not success:
            try:
                response=self.client.embeddings.create(
                    input=chunks,
                    model="text-embedding-3-small",
                    dimensions=self.dimensions
                )
            except Exception as e:
                print(str(e))
                success=False
            else:
                success=True
        embeddings=[normalize(response.data[i].embedding) for i in range(len(chunks))]
        return embeddings
    
    def save_document(self,title):
        if title in self.store:
            with open(os.path.join(self.folder,f"{title}.json"),'w') as f:
                json.dump(self.store[title],f)

    def load_document(self,title):
        if title not in self.store:
            path=os.path.join(self.folder,f"{title}.json")
            if os.path.isfile(path):
                with open(path,'r') as f:
                    self.store[title]=json.load(f)
            else:
                self.store[title]=[]

    def close_document(self,title):
        if title in self.store:
            del self.store[title]

    def add_chunks(self,chunks,title):
        self.load_document(title)
        self.store[title]["chunks"].extend(chunks)
        self.store[title]["embeddings"].extend(self.embed(chunks))
        self.save_document(title)
        print(f'Chunks successfully added to document : {self.folder}/{title}.json')


    def new_document(self,text,title,description):
        chunks=split_text(text,self.chunk_size)
        self.store[title]=dict(title=title,description=description,chunks=chunks,embeddings=self.embed(chunks))
        self.save_document(title)
        print(f"Successfully created document '{title}' : path='{self.folder}/{title}.json'")

    def search(self,query,titles='all',num=3,threshold=0.5):
        vect=self.embed([query])[0]
        if titles=='all':
            titles=self.store.keys()
        results={}
        for title in titles:
            doc_results=[]
            for i in range(len(self.store[title]["chunks"])):
                doc_results.append((self.store[title]["chunks"][i],np.dot(vect,self.store[title]["embeddings"][i])))
            doc_results.sort(key=lambda result: result[1], reverse=True)
            doc_results=list(filter(lambda result:result[1]>=threshold,doc_results))
            results[title]=doc_results[:num]
        return results
    
if __name__=='__main__':
    
    ret=Retriever()