"""

Pandora-AI

----------------------------------------------------------------

Module implementing a custom AI-powered python console using OpenAI GPT4 Turbo model.
Usable both as a regular python console and/or an AI assistant.
Capable of generating and running scripts autonomously in its own internal interpreter.
Can be interacted with using a mix of natural language (markdown and LaTeX support) and python code. 
Having the whole session in context. Including user prompts/commands, stdout outputs, etc... (up to 128k tokens)
Highly customizable input/output redirection and display (including hooks for TTS) for an easy and user friendly integration in any kind of application. 
Modularity with custom tools passed to the agent and loaded in the internal namespace, provided their usage is precisely described to the AI (including custom modules, classes, functions, APIs).
Powerful set of builtin tools to:
- facilitate communication with the user, 
- enable AI access to data/file content or module/class/function documentation/inspection,
- files management (custom work and config folder, memory file, startup script, file upload)
- access to external data (websearch tool, webpage reading), 
- notify status, 
- generate images via DALL-e 3,
- persistent memory storage via an external json file.
Also usable as an 'intelligent' python function capable of generating scripts autonomously and returning any kind of processed data or python object according to a query in natural language along with some kwargs passed in the call.
Can use the full range of common python packages in its scripts (provided they are installed and well known to the AI)

--------------------------------------------------------------------
"""

#Imports
import os
_root_=os.path.dirname(os.path.abspath(__file__))
import sys
if not sys.path[0]==_root_:
    sys.path.insert(0,_root_)
def root_join(*args):
    return os.path.join(_root_,*args)

import io
import re
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv(root_join(".env"))
load_dotenv(os.path.join(os.getcwd(),".env"))
import tiktoken
import json
import codeop
import time
from datetime import datetime
from objdict_bf import objdict
from console import Console
from get_text import get_text
import base64
import requests
from regex_tools import process_regex,split,pack
import textwrap

#Utility Functions

def strint(i,n=4,filler='0'):
    """
    Converts an int to a filled string of fixed length
    """
    s=str(i)
    while len(s)<n:
        s=filler+s
    return s

def load_txt_file(file):
    """
    Returns the string content of a textual file if it exists, otherwise returns None.
    """
    if os.path.exists(file):
        with open(file,'r') as f:
            content=f.read()
        return content
    else:
        return None

def load_json_file(file):
    """
    Returns the data content of a json file if it exists, otherwise returns an empty dict.
    """
    if os.path.exists(file):
        with open(file,'r') as f:
            data=json.load(f)
        return data
    else:
        return {}

def escape_quote_marks(string):
    """
    Escapes all double-quotes marks in a string
    """
    return string.replace(r'\"','"').replace('"', r'\"')

def format(string, context=None):
    # Si aucun contexte n'est fourni, utiliser un dictionnaire vide
    if context is None:
        context = {}

    # Trouver les expressions entre <<...>>
    def replace_expr(match):
        expr = match.group(1)
        try:
            # Évaluer l'expression dans le contexte donné et la convertir en chaîne
            return str(eval(expr, context))
        except Exception as e:
            # En cas d'erreur, retourner l'expression non évaluée
            return '<<' + expr + '>>'

    # Remplacer chaque expression par son évaluation
    return re.sub(r'<<(.*?)>>', replace_expr, string)

def is_valid_python_code(code_str):
    """
    Checks wether a string is a syntactically correct python script
    """
    try:
        # Try to compile the code string
        compiled_code = compile(code_str, '<string>', 'exec')
        return True
    except SyntaxError:
        return False

def process_markdown(string):
    """
    Processes LaTeX expressions found in a markdown string (except if they are enclosed in code blocks)
    Intended to make the syntax more flexible by allowing to use '\(', '\)','\[,'\]' tags in Markdown Katex.
    '\(' and '\)' are converted to '$' tags
    '\[' and '\]' are converted to '$$' tags
    Formula tags are automatically added to \\begin \end environments if they aren't enclosed already.
    Eases the AI use of Katex...
    """
    def latex_func(string,context):
        if not 'codeblock' in context:
            string=string.replace(r'\(','$').replace(r'\)','$').replace(r'\[','$$').replace(r'\]','$$')
            if string.startswith('$$') and not string[2]=='\n':
                string="\n$$\n"+string[2:-2]+"\n$$\n"
        return string

    def latex_env_func(string,context):
        if not 'codeblock' in context and not 'latex' in context:
            string="\n$$\n"+string+"\n$$\n"
        return string

    string=string.replace(r'\"','"')
    patterns={
        'codeblock':r'```.*?```|`.*?`',
        'latex':r'\$\$.*?\$\$|\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\]',
        'latex_env':r'\\begin\{.+?\}[\s\S]*?\\end\{.+?\}'
    }
    processing_funcs={
        'latex':latex_func,
        'latex_env':latex_env_func
    }
    
    return process_regex(string,patterns,processing_funcs)

def get_code_and_language(string):
    lines=string.strip().split('\n')
    flag=False
    content=''
    language=None
    for i in range(len(lines)):
        if lines[i].startswith('```') and not flag:
            flag=True
            language=lines[i][3:] or 'text'
        elif lines[i].startswith('```') and flag:
            flag=False
        elif flag:
            content+=lines[i]+'\n'
        else:
            pass
    return content.strip(),language

def process_codeblock(code,context,role='assistant'):
    """
    Processes markdown code blocks found out of string litterals in a script and converts them into call to a suitable python codeblock function.
    """

    if any(key in context for key in ['single_quoted_literal','double_quoted_literal']):
        return code
    else:
        code,language=get_code_and_language(code)
        if role=='user':
            return f"user_codeblock(r\"\"\"\n{escape_quote_marks(code)}\n\"\"\",language={str(language)})"
        else:
            if language=='python':
                return code
            else:
                return f"codeblock(r\"\"\"\n{escape_quote_marks(code)}\n\"\"\",language={str(language)})"

def process_raw_code(code,role='assistant'):
    """
    Ensures python executability of a mixed markdown-python script
    non-executable markdown parts are converted to calls to a suitable messaging or codeblocking function (depending on who produced the script). 
    """

    if role=='assistant':
        func='message'
    elif role=='user':
        func='user_prompt'

    patterns={
        'double_quoted_literal':r'""".*?"""',
        'single_quoted_literal':r"'''.*?'''",
        'mkdwn_codeblock':r'```.*?```'
    }
    processing_funcs={
        'mkdwn_codeblock':lambda content,context:process_codeblock(content,context,role=role)
    }
    
    code=process_regex(code,patterns,processing_funcs)
    lines=code.split('\n')

    fully_correct=False
    sym="#/#"
    i=0
    while not fully_correct:
        try:
            codeop.compile_command('\n'.join(lines[:i+1]),symbol='exec')
        except SyntaxError:
            lines[i]=sym+lines[i]
        else:
            i+=1
        if i==len(lines):
            fully_correct=True           

    lines='\n'.join(lines).strip().split('\n')

    new_lines=[]
    n=len(lines)
    flag=False
    for i in range(n):
        if lines[i].startswith(sym) and flag==False:
            flag=True
            new_lines.append(f'{func}(r"""{escape_quote_marks(lines[i][len(sym):])}')
        elif lines[i].startswith(sym) and flag==True:
            new_lines.append(f'{escape_quote_marks(lines[i][len(sym):])}')
        elif not lines[i].startswith(sym) and flag==True:
            flag=False
            new_lines[-1]=new_lines[-1]+'""")'
            new_lines.append(lines[i])
        elif not lines[i].startswith(sym) and flag==False:
            new_lines.append(lines[i])
    if flag==True:
        new_lines[-1]=new_lines[-1]+'""")'

    code='\n'.join(new_lines)
    
    return code

def get_code_segments(code):
    """
    splits python a python script in segments of three possible types:
    - 'prompt' : calls to user_prompt function
    - 'codeblock' : calls to user_codeblock function
    - 'else' : other python code segment  

    Used in Pandora.process_user_input method
    """
    patterns={
        'prompt':r'user_prompt\(r""".*?"""\)',
        'codeblock':r'user_codeblock\(r""".*?""",language=\'.*?\'\)'
    }

    parts=split(code,patterns)
    return parts

def stdout_write(content):
    """
    Prints to terminal __stdout__ whatever the stdout redirection currently active
    """
    sys.__stdout__.write(content)

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

def strip_newlines(string):
    """
    removes leading, trailing and double newlines found in a string
    """
    while len(string)>=1 and string[0]=='\n':
        string=string[1:]
    
    while len(string)>=1 and string[-1]=='\n':
        string=string[:len(string)-1]
    
    while not (newstring:=string.replace('\n\n', '\n'))==string:
        string=newstring
    return string

encoding=tiktoken.get_encoding("cl100k_base")
def token_count(string):
    """
    count tokens in a string
    """
    return len(encoding.encode(string))

def pack_msg(messages):
    """
    Assembles a message list into a single string
    """
    text=''
    for message in messages:
        text+=message.name+':\n'
        if not message.tag=='image':
            text+=str(message.content)+'\n\n'
        else:
            text+='(Contains an Image file or URL)'+'\n\n'
    return text       

def total_tokens(messages):
    """
    Returns the total token count of a message list
    """
    return token_count(pack_msg(messages))

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

###Main Functions and Classes

class Image:

    """
    Image object suitable for AI vision implementation 
    """

    def __init__(self,file=None,url=None,bytesio=None,detail='auto'):
        self.file=file
        self.b64_string=None
        self.url=url
        self.bytesio=bytesio
        self.detail=detail

    def mime(self,ext):
        if ext in ['jpg','jpeg']:
            return 'jpeg'
        elif ext in ['png']:
            return 'png'
        elif ext in ['webp']:
            return 'webp'
        elif ext in ['gif']:
            return 'gif'
        else:
            return None
        
    def get_image_url(self):
        if self.url:
            return self.url
        elif self.b64_string:
            return self.b64_string
        elif self.file:
            ext=self.file.split('.')[-1]
            with open(self.file, "rb") as image_file:
                b64=base64.b64encode(image_file.read()).decode('utf-8')
            self.b64_string=f"data:image/{self.mime(ext)};base64,{b64}"
            return self.b64_string
        elif self.bytesio:
            ext=self.bytesio.name.split('.')[-1]
            b64=base64.b64encode(self.bytesio.getvalue()).decode('utf-8')
            self.b64_string=f"data:image/{self.mime(ext)};base64,{b64}"
            return self.b64_string
        else:
            return None
        
    def get_image_data(self):
        return self.url or self.file or self.bytesio
        
    def get_message(self):
        content=[
            {
                'type':'image_url',
                'image_url':{
                    'url':self.get_image_url(),
                    'detail':self.detail
                }
            },
            {
                'type':'text',
                'text':"The following image has been observed successfully and made available in context for analysis:\n"+(self.url or self.file or self.bytesio.name)
            },
        ]
        msg=Message(content=content,role='system',name='image',type='temp',tag='image',image=self)
        return msg

def Message(content,role,name,type="queued",tag="default",**kwargs):
    """
    Returns a custom message (objdict instance) with additional / optional metadata
    type and tag are used internaly by the agent and its collector to properly route the messages for display or apply conditionnal treatments.
    """
    message=objdict(
        role=role,
        name=name,
        content=content,
        type=type,
        tag=tag,
        timestamp=datetime.now().timestamp()
        )
    message.update(kwargs)
    return message 

def Sort(messages):
    """
    Sorts a list of messages according to their timestamps
    """
    return sorted(messages, key=lambda msg: msg.timestamp)

class NoContext:
    """
    A context manager that does nothing, useful when no context manager is required to display a message.
    """
    def __init__(self,*args,**kwargs):
        pass
    def __enter__(self,*args,**kwargs):
        pass
    def __exit__(self,*args,**kwargs):
        pass

class OutputCollector:

    """
    Class handling the routing of messages received by the user and agent to custom display and context manager methods. 
    By default, all goes to stdout. But can be passed a display_hook and context_handler to customize the routing.
    """

    class DefaultContext:
        def __init__(self,message):
            if message.tag in ['assistant_message','interpreter']:
                stdout_write(message.name+':\n')
        def __enter__(self,*args,**kwargs):
            pass
        def __exit__(self,*args,**kwargs):
            pass

    def __init__(self,agent,display_hook=None,context_handler=None):
        self.messages=[]
        self.agent=agent
        self.last_tag=None
        self.context=None
        self.status=None
        self.context_handler=context_handler or (lambda message:OutputCollector.DefaultContext(message))
        def default_display(message,status):
            if message.tag in ['interpreter']:
                stdout_write(message.content)
            elif message.tag in ['assistant_message']:
                stdout_write(message.content+'\n')
            elif message.tag in ['status'] and not message.content=='#DONE#':
                stdout_write(message.content+'\n')

        self.display=display_hook or default_display

    def collect(self,new_message):
        if new_message.type=='status':
            self.collect_status(new_message)
        else:
            self.collect_default(new_message)

    def collect_status(self,new_message):
        if not new_message.content=="#DONE#":
            if not self.status:
                self.status=self.context_handler(new_message)
            self.display(new_message,self.status)
        else:
            if self.status:
                self.display(new_message,self.status)
                self.status=None

    def collect_default(self,new_message):
        if new_message.tag==self.last_tag and self.messages:
            message=self.messages[-1]
            message.content+='\n'+new_message.content
        else:
            self.last_tag=new_message.tag
            self.context=self.context_handler(new_message)
            self.process_all()
            self.messages.append(new_message)
        with self.context:
            self.display(new_message,self.status)
        
    def process_all(self):
        while self.messages:
            message=self.messages.pop(0)
            if not message.get("no_add"):
                self.agent.add_message(message)

class Tool:
    """
    Tool object used to pass custom tools to the AI.
    """

    def __init__(self,name,description,obj,type='function',example=None,parameters=None,required=None):
        self.type=type
        self.name=name
        self.description=description
        self.example=example
        self.obj=obj
        self.parameters=parameters or {}
        self.required=required or []
    
    def get_description(self):
        desc=self.name+" ("+self.type+"):\n"
        desc+="Signature / Usage: "+self.description+'\n'
        if self.parameters:
            desc+="Parameters:\n"
            for p in self.parameters:
                if p in self.required:
                    desc+='\t'+p+' : '+'(required) '+self.parameters[p]+'\n'
                else:
                    desc+='\t'+p+' : '+'(optional) '+self.parameters[p]+'\n'
        if self.example:
            desc+="Example:\n"
            desc+="###\n"
            desc+=textwrap.dedent(self.example)+'\n'
            desc+="###\n"
        return desc

    def __call__(self,*args,**kwargs):
        return self.obj(*args,**kwargs)

class Pandora:
    """
    Class implementing a custom AI-powered python console using OpenAI GPT4 model.
    Usable both as a regular python console and/or an AI assistant.
    Capable of generating and running scripts autonomously in its own internal interpreter.
    Can be interacted with using a mix of natural language (markdown and LaTeX support) and python code. 
    Having the whole session in context. Including user prompts/commands, stdout outputs, etc... (up to 128k tokens)
    Highly customizable input/output redirection and display (including hooks for TTS) for an easy and user friendly integration in any kind of application. 
    Modular usage of custom tools provided their usage is precisely described to the AI (including custom modules, classes, functions, APIs).
    Powerful set of builtin tools to:
    - facilitate communication with the user, 
    - enable AI access to data/file content or module/class/function documentation/inspection,
    - files management (custom work and config folder, folder visibility, file upload/download features)
    - access to external data (websearch tool, webpage reading), 
    - notify status, 
    - generate images via DALL-e 3,
    - persistent memory storage via an external json file.
    Also usable as an 'intelligent' python function capable of generating scripts autonomously and returning any kind of processed data or python object according to a query in natural language along with some kwargs passed in the call.
    Can use the full range of common python packages in its scripts (provided they are installed and well known to the AI)
    """
    
    @staticmethod
    def setup_folder(path=None):
        config=objdict.load(root_join("config.json"),_use_default=True)
        config.folder=path or config.folder
        if not config.folder:
            config.folder=os.path.expanduser("~/Pandora")
        if not os.path.isdir(config.folder):
            try:
                os.makedirs(config.folder)
            except Exception as e:
                print(f"Error when attempting to create Pandora's profiles folder:\n{str(e)}")
                config.folder=os.path.expanduser("~/Pandora")
                print(f"Defaulting to '{config.folder}' as the profiles folder.") 
                print("(Please create this folder manualy if this error persists.)")
        config.dump()
        Pandora.folder=config.folder

    folder=None

    @staticmethod
    def folder_join(*args):
        return os.path.join(Pandora.folder,*args)
                
    #Default configuration
    default_config=objdict(
        username='User',
        code_replacements=[],
        model="gpt-4-1106-preview",
        enabled=True,
        vision=False,
        voice_mode=True,
        language='en',
        uses_memory=True,
        uses_past=True,
        uses_agents=True,
        top_p=0.5,
        temperature=1,
        max_tokens=2000,
        token_limit=32000,
        n=1,
        stream=False,
        stop=["system:"],
        presence_penalty=0,
        frequency_penalty=0,
        logit_bias=None
    )

    #Default base preprompt used to instruct the AI, should be tweaked with caution. 
    base_preprompt="""
        #INSTRUCTIONS
        You're <<self.name>>, an advanced AI-powered python console, resulting of the combination of the latest OpenAI model and a python interpreter.
        The user can use you as a regular python console, interact with you in many languages or pass you files/images that you may analyze using your multimodal abilities.
        As an AI model, you've been trained to use Python as your primary language: All your responses are 100% python code, automatically sent to execution in the built-in interpreter.
        This setting allows you to perform anything that can be done with Python. 
        On top of usual python libraries that you may import, additionnal tools are preloaded in the interpreter's namespace to enable you to perform specific tasks conveniently.
        Communication with the user is performed by using the built-in 'message' tool (except instructed otherwise).
        All the other tools prepared for you and how you should use them will be detailed in the 'TOOLS' section below.
        As a Pandora class instance, you're preloaded in your built-in console's namespace under the name <<self.instance_name>>.
        This setting allows the user to interact with you and call your methods programaticaly.

        Important note: You should always attempt to run computations first. Only then will you be able to craft an informed response to the user.

        Over-simplified example of expected behaviour:

        User:
        What is the factorial of 12?

        Assistant:
        import math
        math.factorial(12)
        #finish script here to let computations resolve and get the interpreter's feedback.

        Interpreter:
        479001600

        Assistant:
        #Use an adequate communication tool the send your response to the user, informed by the interpreter's result.

        #END OF INSTRUCTIONS
        """

    def __init__(self,name=None,openai_client=None,openai_api_key=None,config=None,work_folder=None,base_preprompt=None,preprompt=None,builtin_tools=None,tools=None,example=None,infos=None,console=None,text_to_audio_hook=None,audio_play_hook=None, input_hook=None,display_hook=None, context_handler=None):
        """
        Initializes the Pandora instance
        All kwargs below are optional :
        name : The name of the AI assistant, defaults to 'Pandora'. Will determine the name of the object instance in its own namespace (in lower case).
        openai_client : you may pass directly an authenticated openai client to initialize the instance
        openai_api_key : or you may provide an api key (a new client will be initialized with it). If you don't pass any, the instance will attempt to retrieve it as an environment variable.
        #Note: the instance will still work as a regular python console in case a client can't be initialized.
        work_folder : path to the desired workfolder. If None, the instance will create one in the cwd named 'Pandora'.
        config_folder : path to the desired config folder (the instance will look for config files there), default to 'config' in the workfolder.
        base_preprompt : Core preprompt, necessary to instruct the AI it is a python console and should work in python language exclusively. Can be customized, but will most likely break the AI intergation in the python environment if not carefully crafted. None = default base_preprompt
        preprompt : secondary preprompt to give custom instructions to the AI, you can safely tweak this one as you wish.
        builtin_tools : a list of builtin tools amongst ['message','codeblock','status','memory','observe','generate_image'] that the AI will be allowed to use. None = all
        tools : a dict of custom Tool objects that the AI will be able to use (can be any python module, class, object, function, variable... provided a precise enough description so that the AI knows how to use it correctly).
        example : an example string showing the AI how it should behave / use tools in the context of a conversation with the user (can gather several such examples).
        infos : a list strings, giving custom useful informations for the AI.
        console : the console object that the AI will use to run code (should be an instance of the Console class provided in the package)
        memory_file : a json file where the AI will be able to store informations in the long run.

        #Custom redirection hooks:
        text_to_audio_hook : a custom hook to implement TTS. Takes a text and a language, returns an audio object (any type). expected usage: audio=text_to_audio_hook(text,language=None)
        audio_play_hook : a custom audio auto player, should be able to play audio objects as returned by the text_to_audio_hook. expected usage: audio_play_hook(audio) # should play the audio immediately
        input_hook : a hook used to redirect stdin to a custom interface
        display_hook : a hook used by the collector to determine what method it should use to display or route messages depending on their type/tag
        context_handler : a hook used for integration in chat interfaces, determines the context manager in which the display hook will be called (for instance an AI message container, a user message container, a system message container, a status message container...)
        #Note: By default (all hooks set to None) Pandora will send its outputs to stdout and won't implement text to speech.
        """
        Pandora.setup_folder()
        self.name=name or 'Pandora'
        self.init_client(openai_client=openai_client,openai_api_key=openai_api_key)
        self.init_folders_and_files(work_folder=work_folder)
        self.init_config(config=config)
        self.init_preprompts(base_preprompt=base_preprompt,preprompt=preprompt,example=example,infos=infos)
        self.collector=OutputCollector(self,display_hook=display_hook,context_handler=context_handler)     
        self.init_console(console=console,input_hook=input_hook)
        self.init_TTS(text_to_audio_hook=text_to_audio_hook,audio_play_hook=audio_play_hook)
        self.init_tools(tools=tools,builtin_tools=builtin_tools)
        self.messages=[]
        self.new_turn=False
        self.output=None
        self.run_script(self.startup_file)

    @property
    def instance_name(self):
        return self.name.lower().replace(' ','_')

    def init_client(self,openai_client=None,openai_api_key=None):
        """
        Initializes the OpenAI client
        """
        if openai_client:
            self.client=openai_client
        elif openai_api_key:
            self.client=OpenAI(api_key=openai_api_key)
        elif os.getenv("OPENAI_API_KEY"):
            self.client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            self.client=None
        self.authenticated=self.check_client_authentication()
        self.warning=False

    def check_client_authentication(self):
        """
        Checks if the OpenAI client is properly authenticated by running a simple completion test with gpt-3.5-turbo
        """
        try:
            response = self.completion("Say 'API test successful.'",temperature=0,max_tokens=10)
        except Exception as e:
            stdout_write(str(e)+'\n')
            return False
        else:
            return True

    def init_folders_and_files(self,work_folder=None):
        """
        Initializes the folders used by the AI (work folder, config folder)
        Creates them if necessary.
        Then sets the cwd to the workfolder
        """
        self.work_folder=work_folder or Pandora.folder_join(self.name)
        if not os.path.exists(self.work_folder):
            os.mkdir(self.work_folder)

        self.config_folder=os.path.join(self.work_folder,'config')
        if not os.path.exists(self.config_folder):
            os.mkdir(self.config_folder)

        memory_file=os.path.join(self.config_folder,'memory.json')
        if not os.path.exists(memory_file):
            with open(memory_file,'w') as f:
                json.dump({},f)
        self.memory=objdict.load(_file=memory_file)

        startup_file=os.path.join(self.config_folder,'startup.py')
        if not os.path.exists(startup_file):
            with open(startup_file,'w') as f:
                f.write('')
        self.startup_file=startup_file
        
    def init_console(self,console=None,input_hook=None):
        """
        Initializes the built-in python console used by the AI
        Properly sets stdout redirection to the collector
        stdin redirection is handled by the input_hook (optional) 
        """
        def custom_output(buffer):
            message=Message(content=buffer,role='system',name="Interpreter",type="queued",tag="interpreter")
            self.collector.collect(message)
        self.console=console or Console()
        self.console.output_redirection_hook=custom_output
        self.console.input_redirection_hook=input_hook
        self.console.mode='interactive'
        self.console.update_namespace({
            self.instance_name:self,
            'user_prompt':self.add_user_prompt, 
            'user_codeblock':self.add_user_codeblock
        })
        self.console.run(textwrap.dedent(f"""
            import os
            os.chdir({self.instance_name}.work_folder)
        """))

    def init_tools(self,tools=None,builtin_tools=None):
        """
        Initializes the tools used by the AI
        First load custom tools passed to the constructor
        Then load builtin tools (all by default, but can be selected via the builtin_tools list passed in the constructor)
        """
        self.tools=tools or {}
        self.load_tools()
        self.builtin_tools=builtin_tools or ['message','codeblock','observe','return_output','status','memory','generate_image']
        if 'message' in self.builtin_tools:
            self.add_tool(
                name="message",
                description=r"""message(content,speak=True,language=None) # Main and default tool to communicate a message to the user. Supports Markdown and KaTeX.""",
                obj=self.add_ai_message,
                parameters=dict(
                    content='(string) Should always be a triple quoted raw string. Triple quote marks within the content string must be escaped for code consitency.',
                    speak='(boolean) Determines if TTS is used to speak out the message',
                    language='(string) The language setting for TTS. Set it to the appropriate value depending on the content. Default falls back to your global language setting.'
                ),
                required=['content'],
                example="""
                user:
                Show me a LaTeX formula please.

                assistant:
                message(r\"\"\"Sure! Here is the famous Euler formula:
                \[
                e^{i\pi}+1=0 
                \]
                It connects the most fundamental constants in mathematics: \(0,1,i,\pi,e\).\"\"\")
                """
            )
        else:
            def message(*args,**kwargs):
                print("Warning: Messaging the user is not possible in the current setting. Your messages will not be received.")
                
            self.console.update_namespace(
                message=message
            )
        
        if 'codeblock' in self.builtin_tools:
            self.add_tool(
                name="codeblock",
                description="codeblock(code,language) # Main and default tool to show a code snippet to the user.",
                obj=self.add_ai_codeblock,
                parameters=dict(
                    code="(string) The code you want to display. Should always be a triple quoted raw string. Triple quote marks within the code string must be escaped.",
                    language="(string) The programming language used in the code string (defaults to 'text')"
                ),
                required=['code'],
                example="""
                user:
                Show me how to print some text in the console.

                assistant:
                message(r\"\"\"Sure, here is a code snippet showing how to print a simple 'Hello world!' statement in the console.\"\"\")
                codeblock(r\"\"\"
                print('Hello world!')
                \"\"\",language='python')
                message(r\"\"\"Here is what the output will look like when we run it:\"\"\")
                print('Hello world!")
                """
            )
        else:
            def codeblock(*args,**kwargs):
                print("Warning: Showing code blocks is not possible in the current setting. Code blocks will not be shown to the user.")

            self.console.update_namespace(
                codeblock=codeblock
            )

        if 'status' in self.builtin_tools:
            self.add_tool(
                name='status',
                description="status(status_string) # Shows the user a short status notification while code is running. Used to keep the user informed by notifying your strategy/steps while your scripts are running. Any call to 'status' is always immediately followed by the corresponding code performing the action (in the same script).",
                obj=self.status,
                example="""
                assistant:
                status("Computing the factorial of 12.")
                import math
                math.factorial(12)                
                """
            )
        
        if 'memory' in self.builtin_tools:
            self.add_tool(
                name="memory",
                description="memory # a custom nested attribute-style access data structure linked to a JSON file for long lasting storage. Supports dump() method to save the content to the file, and dumps() to serialize into a string. Nested keys must all be valid identifiers.",
                obj=self.memory,
                type="object"
            )

        if 'observe' in self.builtin_tools:
            self.add_tool(
                name="observe",
                description="observe(source) # Inject in your context feed informations extracted from any kind of source. examples: observe(math) ; observe('my_script.py') ; observe(os.getcwd()) ; observe('my_image.png') ; observe('my_document.pdf')... The function will do its best to return something informative, even mediocre, whatever the source type. Use it proactively to get contextual awareness of information contained in the source that wouldn't be visible to you otherwise. For token efficiency, observed information won't persist in context more that one turn, so you will have to observe the source again anytime you need visibility on its content.",
                obj=self.observe,
                parameters=dict(
                    source="(any) The source to observe. Can be a directory, file, url, module, class, function, variable, image..."
                ),
                required=['source'],
                example="""
                #Currently supported sources:
                observe(math) # observe a module/class/function to get some documentation about it
                observe(my_dict) # observe any variable/object to get a representation of its contents and functionnalities
                observe('my_script.py') # observe the content of a file (any textual file)
                observe(path_to_folder) # observe the recursive contents of a given folder
                observe('my_image.png') # observe an image using your vision capabilities
                observe('my_document.pdf') # observe the content of a textual document (tex,pdf,odt,docx,...)
                observe(url) # Observe the textual contents of a webpage (or observe the file if the url points to an observable file)
                """
            )
        
        if 'generate_image' in self.builtin_tools:
            self.add_tool(
                name="generate_image",
                description="path_to_image=generate_image(description,file_path) # Generates a png image using DALL-E 3 according to a descriptive prompt. Suitable only for creative/artistic/fun purposes as the output is high quality but not easily controled due to inherent randomness of DALL-E 3. For more precise or technical needs you should use something else.",
                obj=self.gen_image,
                parameters=dict(
                    description="(string) The descriptive prompt. The richer the description, the better the result.",
                    file_path="(string) The chosen path of the output file (with .png extension)."
                ),
                required=["description","file_path"]
            )

        if 'return_output' in self.builtin_tools:
            self.add_tool(
                name='return_output',
                description="return_output(data) # Use this function whenever you're explicitely asked to return data. This allows the user to use you as an intelligent python function like so : 'data=MagicFunction(query,**kwargs)'.",
                obj=self.return_output,
                example="""
                user:
                Return the factorial of 12.

                assistant:
                import math
                return_output(math.factorial(12))
                """
            )
                
    def init_TTS(self,text_to_audio_hook=None,audio_play_hook=None):
        """
        Initializes the TTS hooks used to speak out AI messages. 
        """
        self.text_to_audio_hook=text_to_audio_hook
        self.audio_play_hook=audio_play_hook

    def init_config(self,config=None):
        """
        Initializes the configuration.
        First load the default one.
        Then update with config files in the config folder
        Then update with config passed in the constructor
        """
        self.config=Pandora.default_config.copy()    
        
        file_config=load_json_file(os.path.join(self.config_folder,"config.json"))
        self.config.update(file_config)

        if config:
            self.config.update(config)

    def init_preprompts(self,base_preprompt=None,preprompt=None,example=None,infos=None):
        """
        Initializes the preprompts (base_preprompt, preprompt, example).
        First load the default base preprompt.
        Then update with files in the config folder
        Then update with preprompts passed in the constructor
        """
        #load defaults
        self.base_preprompt=Pandora.base_preprompt
        self.preprompt=''
        self.example=''

        #load from config files
        file_base_preprompt=load_txt_file(os.path.join(self.config_folder,'base_preprompt.txt'))
        if file_base_preprompt:
            self.base_preprompt=file_base_preprompt
        file_preprompt=load_txt_file(os.path.join(self.config_folder,'preprompt.txt'))
        if file_preprompt:
            self.preprompt=file_preprompt
        file_example=load_txt_file(os.path.join(self.config_folder,'example.txt'))
        if file_example:
            self.example=file_example

        #load from constructor kwargs
        if base_preprompt:
            self.base_preprompt=base_preprompt
        if preprompt:
            self.preprompt=preprompt
        if example:
            self.example=example

        self.infos=infos or []

    def run_script(self,script):
        """
        Run a script file using the agent. 
        The code will be run and the agent will have the code and execution results in context.
        Notably useful to run a startup script.
        """
        if os.path.isfile(script):
            with open(script,'r') as f:
                content=f.read()
            if content:
                msg=Message(content=f"Running {script}...",role='system',name='system_bot')
                self.add_message(msg)
                code=process_raw_code(content,role='user')
                code=self.replace(code)
                msg=Message(content=code,role="system",name='system_bot')       
                self.add_message(msg)
                self.console.run(code)
                if not self.console.error:
                    msg=Message(content=f"{script} ran successfully.",role='system',name='system_bot')
                    self.add_message(msg)
                else:
                    msg=Message(content=f"{script} ran with errors.",role='system',name='system_bot')
                    self.add_message(msg)

    def add_message(self,message):
        """
        Adds a message the internal messages history
        """
        self.messages.append(message)

    def observe(self,source):
        """
        Main tool used by the AI to observe something.
        Supports a wide range of possible sources:
        folders, files, urls, images, python modules/classes/functions/variables
        """
        self.status("Observing something...")

        if isinstance(source, str) and source.startswith(('http://', 'https://')):
            if source.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif')):
                self.observe_image(url=source)
            else:
                self.observe_data(**{f'{source}':source})
        elif isinstance(source, str) and os.path.isfile(source):
            # C'est un chemin de fichier
            if source.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif')):
                self.observe_image(file=source)
            else:
                self.observe_data(**{f'{source}':source})
        elif isinstance(source, io.BytesIO):
            name=source.name
            if name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif')):
                self.observe_image(bytesio=source)
            else:
                raise TypeError("Only BytesIO of images are supported for observation.")
        else:
            self.observe_data(data=source)

    def observe_image(self,file=None,url=None,bytesio=None):
        """
        Shortcut function to make the AI observe an image
        """
        self.status("Observing image...")
        success=self.add_image(file=file,url=url,bytesio=bytesio)
        if not success:
            print("Impossible to observe image.")

    def observe_data(self,**kwargs):
        """
        Extract textual data from some sources passed as kwargs and inject the data in the AI context as a system message
        """
        self.status("Observing data...")
        s='#Result of observation:\n'
        for key in kwargs:
            s+=f"{key}:\n"
            s+=get_text(kwargs[key])+'\n'
        message=Message(content=s,role='system',name="Observation",type='temp',tag="observation")
        self.collector.collect(message)
        self.new_turn=True
            
    def add_image(self,file=None,url=None,bytesio=None):
        """
        Adds an image to the AI context for vision analysis (if vision is enabled)
        """
        success=False
        if self.config.vision and self.config.model=="gpt-4-vision-preview":
            if(file or url or bytesio):
                img=Image(
                    file=file,
                    url=url,
                    bytesio=bytesio
                )
                self.collector.collect(img.get_message())
                self.new_turn=True
                success=True
        return success

    def gen_image(self,description,file_path,model="dall-e-3",size="1024x1024",quality="standard"):
        """
        Generates an image using DALL-E 3
        """
        self.status("Generating image...")
        try:
            response = self.client.images.generate(
                model=model,
                prompt=description,
                size=size,
                quality=quality,
                n=1,
            )
            image_url = response.data[0].url
            response = requests.get(image_url)

            if response.status_code == 200:
                if not os.path.isdir(os.path.dirname(os.path.abspath(file_path))):
                    print("Invalid destination directory. Defaulting to work folder for destination.")
                    file_path=os.path.join(self.work_folder,os.path.basename(file_path))

                with open(file_path, "wb") as file:
                    file.write(response.content)
                print(f"Image successfully generated: path='{file_path}'")
                return file_path

            else:
                print("There was a problem while generating the image file. Aborting.")
                return None
        
        except Exception as e:
            print(str(e))
            return None
        
    def add_tool(self,name,description,obj,type="function",example=None,parameters=None,required=None):
        """
        Adds a new tool to the AI agent (loads it in the namespace along with a description for the AI).
        name : the name the tool will have in the internal namespace
        description : basic signature and usage instructions for the AI
        obj : the python object embodying the usable aspect of the tool (will be injected in the internal namespace under the chosen name)
        type : type hint of the tool (function, object, module...) to help the AI know what it has to deal with
        example : an example showing how the tool should be used by the AI
        parameters : a dict mapping all parameters' names to a short description of each
        required : a list of all required parameters
        """
        parameters=parameters or {}
        required=required or []
        tool=Tool(
            name=name,
            type=type,
            description=description,
            obj=obj,
            parameters=parameters,
            required=required,
            example=example
        )
        self.tools[tool.name]=tool
        self.load_tools()

    def add_ai_message(self,content,language=None,speak=True):
        """
        Main function to handle messages sent by the AI.
        Manages TTS functionalities.
        Markdown is preprocessed to enhance LaTeX support.
        Methods for display are dealt with by the collector
        """
        
        def display_part(part):
            if part.strip():
                message=Message(part,role="assistant",name=self.name,tag='assistant_message',no_add=True)
                self.collector.collect(message)

        
        def speak_part(part,language=None,speak=True):
            chunks=split_string(part,delimiters=["\n"])
            for chunk in chunks:
                if chunk.strip():
                    if self.text_to_audio_hook and self.audio_play_hook and speak and self.config.voice_mode:
                        self.status("Preparing audio...")
                        audio=self.text_to_audio_hook(chunk,language=lang)
                    display_part(chunk)
                    if self.text_to_audio_hook and self.audio_play_hook and speak and self.config.voice_mode:
                        self.status("Speaking...")
                        self.audio_play_hook(audio)
        
        content=process_markdown(content)
        lang=language or self.config.language
        patterns={
            'codeblock':r'```.*?```',
            'latex':r'\$\$.*?\$\$',
        }
        splitted=split(content,patterns)
        if isinstance(splitted['content'],str):
            parts=[splitted['content']] 
        elif isinstance(splitted['content'],list):
            parts=splitted['content']
        for part in parts:
            if part['tag']=='else':
                speak_part(pack(part),language=lang,speak=speak)
            else:
                display_part(pack(part))

    def add_user_prompt(self,content,tag='user_message'):
        """
        Adds a user prompt to the collector.
        """
        content=process_markdown(content)
        message=Message(content,role="user",name=self.config.username,type='prompt',tag=tag)
        self.collector.collect(message)

    def add_ai_codeblock(self,code,language="text"):
        """
        Shortcut function to add an AI codeblock to the collector.
        """
        code=code.replace('`',"'")
        self.add_ai_message(f'```{language}\n{code}\n```',speak=False)

    def add_user_codeblock(self,code,language="text"):
        """
        Shortcut function to add a user codeblock to the collector.
        """
        content=f'```{language}\n{code}\n```'
        self.add_user_prompt(content)

    def inject_kwargs(self,kwargs):
        """
        Inject some kwargs in the internal console's namespace from outside.
        Then notify the AI it happened, giving it the name and repr of the kwargs as a system message.
        """
        if kwargs:
            self.console.update_namespace(kwargs)
            s="#The following kwargs have been injected in the console's namespace:\n"
            for key in kwargs:
                s+=f"{key}={repr(kwargs[key])}\n"
            message=Message(content=s,role='system',name="system_bot",type='prompt')
            self.add_message(message)

    def upload_file(self, file_path=None, bytesio=None, stringio=None):
        """
        Uploads a file to the workfolder and notify the AI it happened by giving it the path as a system message.
        """
        def get_available_path(name):
            i = 0
            parts = name.split('.')
            ext = parts[-1] if len(parts) > 1 else ''
            name = '.'.join(parts[:-1]) if len(parts) > 1 else parts[0]
            new_name = f"{name}.{ext}" if ext else name
            while os.path.exists(os.path.join(self.work_folder, new_name)):
                i += 1
                new_name = f"{name}_{i}.{ext}" if ext else f"{name}_{i}"
            return os.path.join(self.work_folder, new_name)

        success = False
        if file_path and os.path.exists(file_path):
            path = get_available_path(os.path.basename(file_path))
            with open(file_path, 'rb') as f:
                with open(path, 'wb') as g:
                    g.write(f.read())
            success = True
        elif bytesio:
            bytesio.seek(0)  # Rewind to the start of the BytesIO object
            path = get_available_path(bytesio.name)
            with open(path, 'wb') as g:
                g.write(bytesio.read())
            success = True
        elif stringio:
            stringio.seek(0)  # Rewind to the start of the StringIO object
            path = get_available_path(stringio.name)
            with open(path, 'w') as g:
                g.write(stringio.read())
            success = True
        if success:
            s = "#The following file has been uploaded in your workfolder:\n"
            s += path
            message = Message(content=s, role='system', name="system_bot",type='prompt')
            self.add_message(message)

    def get_tools(self):
        """
        Returns the description of all tools as a system message.
        """
        if self.tools:
            s="#TOOLS:\n"
            s+="The following tools are already declared in the console namespace and ready to be used in your scripts. Below is a description of each.\n"
            for tool in self.tools:
                s+=self.tools[tool].get_description()+'\n'
            s+="#END OF TOOLS"
            message=Message(content=s,role="system",name="system_bot")
            return [message]
        else:
            return []

    def get_preprompt(self):
        """
        Returns the final preprompt passed to the AI as a system message.
        This final preprompt is constructed by agregating the base_preprompt (necessary for the AI to understand its python console setting), with an optional additional preprompt used to customize the AI behavior or give it additional instructions.
        The final preprompt gets formatted with local variables or evaluated expressions to allow some dynamism in the text.
        """

        base_preprompt=textwrap.dedent(self.base_preprompt)

        if self.preprompt:
            add_preprompt="#ADDITIONAL INSTRUCTIONS\n"
            add_preprompt+=textwrap.dedent(self.preprompt)
            add_preprompt+="\n#END OF ADDITIONAL INSTRUCTIONS"
            preprompt=base_preprompt+'\n\n'+add_preprompt
        else:
            preprompt=base_preprompt
        
        if preprompt:
            preprompt=Message(content=format(preprompt,locals()),role="system",name="system_bot")
            return [preprompt]
        else:
            return []

    def get_infos(self):
        """
        Get the list of additional informations as a system message.
        These infos represent pieces of information useful to explain the setting of its interaction with the user to the AI.
        They are formatted with local variables / evaluated expression enclosed between <<...>> tags.
        Could be anything, for instance language setting, path to a workfolder, etc...
        """
        infos=self.infos+[
                "Default language: '<<self.config.language>>'",
                "Your default workfolder: '<<self.work_folder>>' (save files you create there, and initial cwd of the python session).",
                "Your config folder: '<<self.config_folder>>' (where your memory file, startup file and optional preprompts files are)"
                ]
        if infos:
            s="#Additional informations:\n"
            s+=format('\n'.join(infos),locals())
            msg=Message(content=s,role='system',name='system_bot')
            return [msg]
        else:
            return []
    
    def get_example(self):
        """
        Get the example (if any) as a system message.
        The 'example' is an example interaction with the user helping the AI match the expected behavior.
        """
        if self.example:
            example=Message(content=self.example,role='system',name='system_bot')   
            return [example]
        else:
            return []

    def get_memory(self):
        """
        Get the memory file content as a system message for context (if implemented)
        """
        if self.config.uses_memory:
            s="#Memory Contents:\n"
            s+=self.memory.dumps()
            memory=Message(content=s,role="system",name="Memory")
            return [memory]
        else:
            return []

    def get_messages(self,type='all',tag='all'):
        """
        Utility method allowing to gather all messages matching a given type or tag, and return them sorted according to their timestamp.
        Makes sure no messages with the same exact timestamp are added twice.
        """
        timestamps=[]
        messages=[]
        for message in self.messages:
            ts=message.timestamp
            if ts not in timestamps:
                if (type=='all' or message.type==type) and (tag=='all' or message.tag==tag) :
                    messages.append(message)
                    timestamps.append(ts)
        return Sort(messages)

    def get_prompts(self):
        """
        Returns the list of all prompts.
        Prompts are messages that the AI has not yet seen in context.
        """
        prompts=self.get_messages(type="prompt")
        return prompts

    def send_prompts_to_queue(self):
        """
        Converts prompt messages to queued messages (once these have been passed as context at least one time).
        """
        prompts=self.get_prompts()
        if prompts:
            for prompt in prompts:
                prompt.type="queued"

    def get_queued(self,current_count):
        """
        Returns a list of queued messages found in the history (from newest to latest) up to a maximal token count imposed by the token_limit of the model.
        current_count represents the total token amount already taken from other sources (namely static messages like preprompt, example, tools...).
        If config.uses_past is false, returns an empty list instead, thus implementing an agent without memory of the past.
        """
        if self.config.uses_past:
            queued=self.get_messages(type='queued')
            context_limit=self.config.token_limit-self.config.max_tokens
            available=context_limit-current_count
            n=len(queued)
            k=0
            tokens=0
            while n>=1 and k<n and tokens+(new_tokens:=total_tokens([queued[n-k-1]]))<=available:
                tokens=tokens+new_tokens
                k+=1
            queued=queued[n-k:n]
            return queued
        else:
            return []

    def get_temp(self):
        """
        Returns a list of all temporary messages found in the 'messages' history, to pass them as context.
        Then removes these temporary messages from the history.
        """
        temp=self.get_messages(type='temp')
        for message in self.messages:
            if message.type=='temp':
                self.messages.remove(message)
        return temp

    def gen_context(self):
        """
        Generates a context (ie. a coherent list of messages) by agregating and sorting messages with various roles.
        Namely: 
        - global system messages (preprompt,example,tools,info,memory) ; 
        - temporary messages resulting from tool calls
        - queued messages from the history
        - prompts passed by the user.
        """
        prompts=self.get_prompts()
        preprompt=self.get_preprompt()
        example=self.get_example()
        tools=self.get_tools()
        info=self.get_infos()
        memory=self.get_memory()
        temp=self.get_temp()
        count=total_tokens(preprompt+tools+info+memory+example+temp+prompts)
        queued=self.get_queued(count)
        context=preprompt+tools+info+memory+example+Sort(queued+temp+prompts)
        self.send_prompts_to_queue()
        return context

    def prepare_messages(self,messages):
        """
        Converts a list of messages (objdict instances) to a list of dicts accepted by the OpenAI API.
        """
        prepared=[]
        for message in messages:
            prep={'content':message.content,'role':message.role,'name':message.name}
            if message.get('tool_call_id'):
                prep['tool_call_id']=message.tool_call_id
            if message.get('tool_calls'):
                prep['tool_calls']=message.tool_calls
            prepared.append(prep)
        return prepared

    def call_OpenAI_API(self,context):
        """
        Takes a context and calls the OpenAI API for a response.
        Configuration of the model is taken from self.config.
        Messages in context are prepared (convered to suitable dicts) before being passed to the API.
        """
        kwargs=objdict(
            model=self.config.model,
            messages=self.prepare_messages(context),
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            n=self.config.n,
            stop=self.config.stop,
            presence_penalty=self.config.presence_penalty,
            frequency_penalty=self.config.frequency_penalty
        )

        success=False
        err=0
        while not success and err<2:
            try:
                answer = self.client.chat.completions.create(**kwargs.to_dict())
            except Exception as e:
                print(str(e))
                err+=1
                time.sleep(0.5)
            else:
                success=True
        if not success:
            response=objdict(content="I'm sorry but there was a recurring error with the OpenAI server. Would you like me to try again?",role="assistant",name=self.name)
        else:
            response=answer.choices[0].message
        return response

    def gen_response(self):
        """
        Generates the context and pass it to the AI for a response
        Returns the response.
        """
        context=self.gen_context()      
        response=self.call_OpenAI_API(context)
        return response    

    def load_tools(self):
        """
        Loads all tools in the console's namespace
        """
        for tool in self.tools:
            self.console.update_namespace({self.tools[tool].name:self.tools[tool].obj})

    def process_ai_response(self,response):
        """
        Processes the AI response by making sure it is fully python-executable.
        Make some custom syntax replacements.
        Adds the processed code as a message to the collector.
        Runs the code.
        Determines if a new turn should be taken (error or results in stdout)
        Process all messages created by code execution remaining in the collector 
        """
        content=response.content
        if content:
            code=process_raw_code(content,role='assistant')
            code=self.replace(code)
            message=Message(content=code,role="assistant",name=self.name,type="queued",tag="assistant_code")       
            self.collector.collect(message)
            self.console.run(code)
            if self.console.error or self.console.get_result():
                self.new_turn=True
            self.collector.process_all()

    def process(self):
        """
        Core method for the AI internal process.
        Initializes the new_turn flag to False.
        Process the newly collected messages by sending them to the AI (to form the full context).
        Generates a reponse from the AI.
        Processes and runs the AI python code response
        If the new_turn flag has been set to True for some reason in the meanwhile, take another turn.
        """

        self.new_turn=False               
        
        self.collector.process_all()

        self.status("Generating response.")
        response=self.gen_response() 

        self.status("Running code.")
        self.process_ai_response(response)

        self.status("#DONE#")

        if self.new_turn:
            self.process()

    def replace(self,code):
        """
        Utility function to replace some elements of syntax into others in a code snippet.
        Useful to avoid AI or the user make common syntax mistakes or misuse some tools.
        Desired replacements can be customized in self.config.code_replacements.
        """
        code=code.replace("message(\"","message(r\"").replace("message('","message(r'")
        code=code.replace("codeblock(\"","codeblock(r\"").replace("codeblock('","codeblock(r'")
        code=code.replace("code=\"","code=r\"").replace("code='","code=r'")
        if self.config.get('code_replacements'):
            for segment, replacement in self.config.code_replacements:
                code=code.replace(segment,replacement)
            return code.strip()
        else: 
            return code                                  

    def process_user_input(self,prompt):
        """
        Parses the user prompt and splits the python code parts from the markdown parts.
        The markdown parts are converted in calls to the 'user_message' or 'user_codeblock' functions which makes the prompt fully python-executable.
        The converted parts are then executed one by one in the console.
        The user_message and user_codeblock function will deal with sending corresponding messages to the AI.
        The other python parts are converted to user_code messages and sent to the AI for context.
        """
        if prompt:
            code=process_raw_code(prompt,role='user')
            code=self.replace(code)
            parts=get_code_segments(code)
            if isinstance(parts['content'],list):
                parts=parts['content']
            elif isinstance(parts['content'],str):
                parts=[parts['content']]

            for part in parts:
                content=pack(part)
                if content.strip():
                    if part['tag']=='else':
                        message=Message(content=content,role='user',name=self.config.username,type='queued',tag='user_code')
                        self.collector.collect(message)
                    self.console.run(content)

            self.collector.process_all()
            
    def __call__(self,prompt="",**kwargs):
        """
        Main entry point of the instance.
        Make a call to the instance with a prompt and optional kwargs passed to the agent.
        kwargs will be injected in the console's namespace and notified to the agent.
        The prompt can mix both python code segments and markown text segments.
        The prompt will be processed and executable python code segments found within it will be executed directly in the internal console.
        The rest will be sent to the AI as prompt messages and will trigger a response from the Agent.
        """
        self.output=None
        self.inject_kwargs(kwargs)
        self.process_user_input(prompt)
        if self.get_prompts() and self.config.enabled:
            if self.authenticated:
                self.process()
            elif not self.warning:
                self.warning=True
                s="Warning: To benefit from my AI assistance in this session, you must set me up with a properly authenticated OpenAI client. Otherwise I will behave as a mere Python console."
                message=Message(content=s,role='assistant',name=self.name,tag='assistant_message')
                self.collector.collect(message)
        return self.output 

    def return_output(self,data):
        """
        When called, the agent will output the passed data as a result to its __call__ method.
        Useful to implement magic function behavior.
        """
        self.output=data

    def clear_message_history(self):
        """
        Clears the message history of the agent.
        """
        self.messages=[]   

    def proceed(self):
        """
        If called the AI will take another turn after the current one.
        """
        self.new_turn=True

    def stop(self):
        """
        If called the AI won't take another turn after the current one.
        """
        self.new_turn=False

    def status(self,status):
        """
        Display a status notification to the user.
        """
        message=Message(content=status,role='assistant',name=self.name,type='status',tag='status')
        self.collector.collect(message)

    def new_agent(self,name,**kwargs):
        """
        Creates a new instance of AI agent whose outputs are routed to the main agent's stdout
        kwargs are any parameters supported by the Pandora constructor (except for display_hook and context_handler).
        All kwargs are optional except the name parameter.
        Agents have their own internal python interpreter with a separate namespace independant from the main agent's.
        They can be customized to handle a different set of tools, or be instructed with a different preprompt to refine their behavior. 
        """
        def custom_print(buffer):
            message=Message(content=buffer,role='system',name="Interpreter",type="queued",tag="interpreter")
            self.collector.collect(message)

        class AgentContext:
            def __init__(self,message):
                if message.tag in ['assistant_message','interpreter']:
                    custom_print(message.name+':\n')
            def __enter__(self,*args,**kwargs):
                pass
            def __exit__(self,*args,**kwargs):
                pass
        
        preprompt=f"""
        In the current setting, you've been instantiated as a sub-agent of a main agent called {self.name}.
        Your message outputs are directed in the main agent's standard output and will be visible to both the main agent and the user.
        You're task is to assist in performing specific tasks required by the main agent or the user.
        """

        if not kwargs.get(preprompt):
            kwargs['preprompt']=preprompt

        def agent_display(message,status):
            if message.tag in ['interpreter']:
                custom_print(message.content)
            elif message.tag in ['assistant_message']:
                custom_print(message.content+'\n')
            elif message.tag in ['status'] and not message.content=='#DONE#':
                custom_print(message.content+'\n')
        agent=Pandora(name=name,work_folder=os.path.join(self.work_folder,name),openai_client=self.client,display_hook=agent_display,context_handler=lambda msg:AgentContext(msg),**kwargs)
        return agent
    
    def magic_function(self,verbose=False):
        """
        Returns an AI function that can be called with a natural language query and some kwargs like so:
        result=func("return the sum of a and b.",a=2,b=3)
        result # output: 5
        Can return any type of python object, as long as the AI manages to perform the task with the python modules at its disposal.
        --------
        verbose: (boolean) determines whether the inner workings of the func are printed in the console
        """
        def custom_print(buffer):
            message=Message(content=buffer,role='system',name="Interpreter",type="queued",tag="interpreter")
            self.collector.collect(message)

        def display(message,status):
            if verbose:
                if message.tag in ['interpreter','assistant_message','assistant_code']:
                    custom_print(message.content)
                elif message.tag in ['status'] and not message.content=='#DONE#':
                    custom_print(message.content+'\n')
        
        builtin_tools=['return_output']
        agent=Pandora(openai_client=self.client,builtin_tools=builtin_tools,display_hook=display,context_handler=lambda msg:NoContext(msg))
        agent.config.name="MagicFunction"
        agent.preprompt="""
        Assignement:
        You behave as an intelligent python function.
        The user will ask you to perform a task and may pass you specific kwargs.
        You generate a python script that performs the task and finish it by returning the result via the return_output function.
        The user will receive this value as the output of the call.
        The code must be left out of markdown code blocks to be executed (never use markdown code blocks in your scripts).
        You're not supposed to attempt to communicate with the user in any way, except by returning relevant data via the return_output function.
        Variables/objects/functions you declared in previous runs are reusable in subsequent calls, take advantage of it to make your scripts more efficient.
        """
        agent.config.uses_past=True
        return agent

    def completion(self,prompt,context=None,**kwargs):
        """
        Returns a quick text completion given a prompt and optional context.
        prompt: (string) the prompt passed to the model.
        context: (optional) a list of messages as returned by the Message funtion representing the context of the prompt.
        other optional kwargs parameters must be compatible with OpenAI 'chat.completions.create' API endpoint:
        To name a few:
        model : the model used for completion (default='gpt-3.5-turbo-1106')
        top_p : the top probability parameter of the model (default=1)
        temperature : the temperature of the model (default=1)
        max_tokens : the maximal token length of the model's response (default=1000)
        ...
        """
        context=context or []
        context=context+[Message(content=prompt,role='user',name='user')]
        
        config=dict(
            model='gpt-3.5-turbo-1106',
            top_p=1,
            temperature=1,
            max_tokens=1000,
            n=1,
            stream=False,
            stop=["system:"],
            presence_penalty=0,
            frequency_penalty=0,
            logit_bias=None
        )

        config.update(kwargs)

        response = self.client.chat.completions.create(
                messages=self.prepare_messages(context),
                **config
            )
        return response.choices[0].message.content.strip()
        
    def interact(self,custom_input=None):
        """
        Implements a simple interaction loop with the agent.
        Useful to test the agent in the terminal without code overhead:
        Example:
        from pandora_ai import Pandora

        agent=Pandora()

        agent.interact() # enters the interaction loop in the terminal (type exit() or quit() to end the loop)
        """

        def default_custom_input():
            prompt=''
            flag=False
            while True:
                if not flag:
                    s=input("--> ")
                    flag=True
                else:
                    s=input("... ")
                if not s.endswith(r'//'):
                    prompt+=s+'\n'
                else:
                    prompt+=s[:-2]
                    break
            return prompt
        
        custom_input=custom_input or default_custom_input
        self.loop_flag=True

        def custom_exit():
            self.loop_flag=False
        
        self.console.update_namespace(
            exit=custom_exit,
            quit=custom_exit
        )

        print("Pandora AI - Interactive session")
        print("Enter your python commands or talk with Pandora in natural language. End your input scripts with '//'.")
        print('Get some help at any time by simply asking the AI.')

        while self.loop_flag:
            prompt=custom_input()
            self(prompt)

        print("Exiting interactive session. See you next time!")