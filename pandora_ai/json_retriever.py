from openai import OpenAI # for calling the OpenAI API
import tiktoken  # for counting tokens
import numpy as np
import json
import os

tokenizer=tiktoken.get_encoding("cl100k_base")

def normalize(vect,precision=5):
    inv_norm=1.0/np.linalg.norm(vect,ord=2)
    return [round(x_i*(inv_norm),precision) for x_i in vect]

def flattener(data):
    def _traverse(obj, keys, output):
        if isinstance(obj, dict):
            if not obj:
                output.append((keys, {}))  # Handle empty dict
            for k, v in obj.items():
                _traverse(v, keys + (k,), output)
        elif isinstance(obj, list):
            if not obj:
                output.append((keys, []))  # Handle empty list
            for idx, item in enumerate(obj):
                _traverse(item, keys + (idx,), output)
        else:
            output.append((keys, obj))

    output = []
    # Start with an empty tuple for the key sequence of the root object
    _traverse(data, tuple(), output)
    # Handle single values and empty structures directly
    if not output:
        output.append((tuple(), data))
    return output

def builder(flat_list):
    # Handle single value case
    if len(flat_list) == 1 and flat_list[0][0] == ():
        return flat_list[0][1]

    # Determine the root type from the first key sequence if the flat list is not empty
    root = [] if flat_list and isinstance(flat_list[0][0][0], int) else {}
    
    for keys, value in flat_list:
        # Handle empty structure case
        if not keys:
            return value

        current_level = root
        for i, key in enumerate(keys):
            # If it's the last key in the sequence, set the value
            if i == len(keys) - 1:
                if isinstance(key, int):
                    # Ensure the current level is a list for integer keys
                    while len(current_level) <= key:
                        current_level.append(None)
                    current_level[key] = value
                else:
                    current_level[key] = value
            else:
                # Prepare next level structure
                if isinstance(key, int):
                    while len(current_level) <= key:
                        current_level.append({} if (i + 1 < len(keys) and not isinstance(keys[i + 1], int)) else [])
                    current_level = current_level[key]
                else:
                    if key not in current_level:
                        current_level[key] = {} if (i + 1 < len(keys) and not isinstance(keys[i + 1], int)) else []
                    current_level = current_level[key]
    return root

def is_in(keys,content):
    return any(is_prefix(keys,entry["keys"]) for entry in content)

def is_prefix(keys1,keys2):
    if keys1==tuple():
        return True
    else:
        return keys2[:len(keys1)]==keys1

def to_str(data):
    if isinstance(data,str):
        return "'"+data+"'"
    else:
        return str(data)

def keys_as_str(keys):
    return ''.join(['['+to_str(key)+']' for key in keys])

def as_string(title,entry):
    return title+keys_as_str(entry[0])+"="+to_str(entry[1])

def subdict(original_dict, keys):
    return {k: original_dict[k] for k in keys if k in original_dict}

class Item:

    def __init__(self,nested=None,keys=None):
        self.keys=keys
        self.nested=nested

    @property
    def content(self):
        content = {}
        for entry in self.nested.table['content'].values():
            if is_prefix(self.keys,entry["keys"]):
                content[keys_as_str(entry["keys"])]=entry
        return content
    
    @property
    def value(self):
        # Reconstruct the nested substructure from the flat content
        flat_list = [(entry['keys'][len(self.keys):], entry['value']) for entry in self.content.values()]
        return builder(flat_list)
    
    def __getitem__(self,key):
        keys=self.keys+(key,)
        
        if is_in(keys,self.content):
            return Item(nested=self.nested, keys=keys)
        else:
            raise KeyError(f"Key {key} does not exist.")
    
    def __setitem__(self, key, value):
        # Construct the full key sequence for the new or updated value
        keys = self.keys + (key,)
        self.nested.set_value(keys, value)

    def __delitem__(self, key):
        keys=self.keys+(key,)

        if is_in(keys,self.content):
            self.nested.delete_value(keys)
        else:
            raise KeyError(f"Key {key} does not exist.")
    
    def __repr__(self):
        return repr(self.value)
    
    def __str__(self):
        return str(self.value)
    
    def search(self,query,num=3,threshold=0.3):
        vect=self.nested.embed([query])[0]
        results=[]
        for entry in self.content.values():
            results.append((entry["string"],np.dot(vect,entry["embedding"])))
        results.sort(key=lambda result: result[1], reverse=True)
        results=list(filter(lambda result:result[1]>=threshold,results))
        return results[:num]


class Nested(Item):

    def __init__(self,openai_api_key=None,title=None,description=None,dimensions=128,precision=5):
        Item.__init__(self,nested=self,keys=tuple())
        self.title=title
        self.description=description
        self.dimensions=dimensions
        self.precision=precision
        self.client=OpenAI(api_key=openai_api_key or os.getenv('OPENAI_API_KEY'),timeout=3)
        self.table=dict(
            title=self.title,
            description=self.description,
            dimensions=self.dimensions,
            precision=self.precision,
            content=dict()
        )

    def embed(self,strings):
        success=False
        while not success:
            try:
                response=self.client.embeddings.create(
                    input=strings,
                    model="text-embedding-3-small",
                    dimensions=self.dimensions
                )
            except Exception as e:
                print(str(e))
                success=False
            else:
                success=True
        embeddings=[normalize(response.data[i].embedding,self.precision) for i in range(len(strings))]
        return embeddings
    
    def load_document(self,file):
        if os.path.isfile(file) and file.endswith('.json'):
            with open(file) as f:
                self.table=json.load(f)
            self.title=self.table["title"]
            self.description=self.table["description"]
            self.precision=self.table["precision"]
            self.dimensions=self.table["dimensions"]

    def save_document(self,file):
        if file.endswith('.json'):
            with open(file,'w') as f:
                self.table["title"]=self.title
                self.table["description"]=self.description
                self.table["precision"]=self.precision
                self.table["dimensions"]=self.dimensions
                json.dump(self.table,f)
            

    def load_content(self,content):
        if isinstance(content,str) and content.endswith(".json") and os.path.isfile(content):
            self.load_json_file(json_file=content)
        elif isinstance(content,str):
            self.load_json_string(json_string=content)
        else:
            self.load_json_data(content)

    def load_json_data(self,json_data):
        entries=flattener(json_data)
        strings=[as_string(self.title,entry) for entry in entries]
        embeddings=self.embed(strings)
        for i in range(len(entries)):
            keys,value=entries[i]
            self.table['content'][keys_as_str(keys)]=dict(
                keys=keys,
                value=value,
                string=strings[i],
                embedding=embeddings[i]
            )

    def load_json_string(self,json_string):
        json_data=json.loads(json_string)
        self.load_json_data(json_data)

    def load_json_file(self,json_file):
        if os.path.isfile(json_file) and json_file.endswith('.json'):
            with open(json_file,'w') as f:
                json_data=json.load(f)
            self.load_json_data(json_data)

    def set_value(self, keys, value):
        # Prepare the string and embedding for the new value
        # If the value is structured, it is first converted to a flat list of entries
        if isinstance(value, dict) or isinstance(value, list):
            entries = [(keys+entry[0],entry[1]) for entry in flattener(value)]
            strings=[as_string(self.title,entry) for entry in entries]
            embeddings=self.embed(strings)
            for i in range(len(entries)):
                keys,value=entries[i]
                self.table['content'][keys_as_str(keys)]=dict(
                    keys=keys,
                    value=value,
                    string=strings[i],
                    embedding=embeddings[i]
                )
        else:
            string = as_string(self.title,(keys, value))
            embedding = self.embed([string])[0]
            # Otherwise, add a new entry
            self.table['content'][keys_as_str(keys)]=dict(
                keys= keys,
                value= value,
                string=string,
                embedding=embedding
            )

    def delete_value(self, keys):
        #removes any entries that are prefixed with the key sequence of the deleted item
        self.table['content'] = {keys_as_str(entry['keys']):entry for entry in self.table['content'].values() if not is_prefix(keys,entry['keys'])}


class JsonRetriever:

    def __init__(self,openai_api_key=None,folder='./json_documents'):
        self.openai_api_key=openai_api_key
        self.folder=folder
        self.store={}
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    def get_loaded(self):
        return [dict(title=doc.title,description=doc.description) for doc in self.store.values()]
    
    def get_titles(self):
        return [os.path.basename(file).split('.')[0] for file in os.listdir(self.folder)]
    
    def save_document(self,title):
        if title in self.store:
            file=os.path.join(self.folder,f"{title}.json")
            self.store[title].save_document(file)

    def load_document(self,title):
        if title not in self.store:
            file=os.path.join(self.folder,f"{title}.json")
            if os.path.isfile(file):
                self.store[title]=Nested(self.openai_api_key,title=title)
                self.store[title].load_document(file)
            else:
                self.store[title]=Nested(self.openai_api_key,title=title)

    def close_document(self,title):
        if title in self.store:
            del self.store[title]

    def new_document(self,title,content,description):
        self.store[title]=Nested(self.openai_api_key,title=title,description=description)
        if content:
            self.store[title].load_content(content=content)
        self.save_document(title)
        print(f"Successfully created document '{title}' : path='{self.folder}/{title}.json'")

    def search(self,query,titles='all',num=5,threshold=0.3):
        if titles=='all':
            titles=self.store.keys()
        results={}
        for title in titles:
            results[title]=self.store[title].search(query,num=num,threshold=threshold)
        return results


if __name__=='__main__':

    data=dict(
        users=dict(
            Baptiste=dict(
                age=38,
                job="Programmer",
                city="Vibeuf",
                hobby="Guitar playing",
                email="bferrand.maths@gmail.com"
            ),
            Manon=dict(
                age=35,
                job="Nurse",
                city="Guignen",
                hobby="Going to the cinema.",
                email="manon.ferrand@laposte.net"
            )
        )
    )

    store=JsonRetriever()

    #store.new_document(title="test",content=data,description="A test data structure.")
    store.load_document("test")

    print(store.search("Where does Manon live ?"))




    





