"""

Pandora-AI

----------------------------------------------------------------

Module implementing a custom AI-powered python console using OpenAI GPT4 Turbo model.
Usable both as a regular python console and/or an AI assistant.
Capable of generating and running scripts autonomously in its own internal interpreter.
Can be interacted with using a mix of natural language (markdown and LaTeX support) and python code. 
Having the whole session in context. Including user prompts/commands, stdout outputs, etc... (up to 128k tokens)
Highly customizable input/output redirection and display (including support for streaming and hooks for TTS) for an easy and user friendly integration in any kind of application. 
Modularity with custom tools passed to the agent and loaded in the internal namespace, provided their usage is precisely described to the AI (including custom modules, classes, functions, APIs).
Powerful set of builtin tools to:
- facilitate communication with the user, 
- enable AI access to data/file content or module/class/function documentation/inspection,
- files management (custom work and config folder, memory file, startup script, file upload)
- access to external data (websearch tool, webpage reading, selenium webdriver tool), 
- notify status, 
- generate images via DALL-E 3,
- persistent memory storage via an external json file.

The AIFunction class implements an 'intelligent' python function capable of generating and running scripts autonomously to return any kind of processed data or python object in response to a query in natural language and some kwargs passed in the call.
Like so:
func=AIFunction()
result=func("Return a function that takes a number and returns the n-th power of that number.",n=3)
print(result(2)) # output: 8

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
import tiktoken
import json
import codeop
import time
from datetime import datetime
from objdict_bf import objdict
from console import Console
from retriever import Retriever
from get_text import get_text
import base64
import requests
from regex_tools import process_regex,split,pack
import textwrap
from queue import Queue
from threading import Thread
from inspect import isgenerator
from pydub import AudioSegment
from pydub.playback import play


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

def extract_python(text):
    pattern = r'```run_python(.*?)```'
    iterator = re.finditer(pattern, text, re.DOTALL)
    result = []

    for match in iterator:
        # Ajouter le texte correspondant au group capturé
        result.append(match.group(1))

    return result

def stdout_write(content):
    """
    Prints to terminal __stdout__ whatever the stdout redirection currently active
    """
    sys.__stdout__.write(content)
    sys.__stdout__.flush()

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

def tokenize(string):
    int_tokens=encoding.encode(string)
    str_tokens=[encoding.decode([int_token]) for int_token in int_tokens]
    return str_tokens

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


###Main Functions and Classes

def play_audio(audio):
    if audio is not None and audio.get("bytes"):
        audio_file_like = io.BytesIO(audio["bytes"])
        audio_segment = AudioSegment.from_file(audio_file_like, format="mp3") 
        play(audio_segment)

def text_to_audio(text, openai_api_key=None):
    
    client=OpenAI(api_key=openai_api_key or os.getenv('OPENAI_API_KEY'))
    
    # Create MP3 audio
    if text.strip():

        mp3_buffer = io.BytesIO()

        response = client.audio.speech.create(
        model="tts-1",
        voice="shimmer",
        input=text
        )

        for chunk in response.iter_bytes():
            mp3_buffer.write(chunk)

        mp3_buffer.seek(0)

        # Convert MP3 to WAV and make it mono
        audio = AudioSegment.from_file(mp3_buffer,format="mp3").set_channels(1)

        # Extract audio properties
        sample_rate = audio.frame_rate
        sample_width = audio.sample_width

        # Return the required dictionary
        return {
            "bytes": mp3_buffer.getvalue(),
            "sample_rate": sample_rate,
            "sample_width": sample_width
        }
    else:
        return None

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
    Class handling the routing of messages sent by the user or agent to custom display and context manager methods.
    Just call the collect(message) method, and it will deal with displaying the content.
    Supports string or generator message content (in stream mode). 
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
        def default_display(content,tag,status):
            if tag in ['interpreter']:
                stdout_write(content)
            elif tag in ['assistant_message']:
                stdout_write(content)
            elif tag in ['status'] and not content=='#DONE#':
                stdout_write(content+'\n')

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
            self.display(new_message.content,new_message.tag,self.status)
        else:
            if self.status:
                self.display(new_message.content,new_message.tag,self.status)
                self.status=None

    def collect_default(self,new_message):
        if new_message.tag==self.last_tag and self.messages:
            message=self.messages[-1]
            tag=message.tag
            content=message.content
        else:
            self.last_tag=new_message.tag
            self.process_all()
            message=new_message
            self.context=self.context_handler(new_message)
            self.messages.append(new_message)
            tag=new_message.tag
            content=''
        if isgenerator(new_message.content):
            for chunk in new_message.content:
                content+=chunk
                if self.agent.config.display_mode=="cumulative":
                    display_chunk=content
                else:
                    display_chunk=chunk
                with self.context:
                    self.display(display_chunk,tag,self.status)
        elif isinstance(new_message.content,str):
            content+=new_message.content
            if self.agent.config.display_mode=="cumulative":
                display_chunk=content
            else:
                display_chunk=new_message.content
            with self.context:
                self.display(display_chunk,tag,self.status)
        message.content=content

    def process_all(self):
        while self.messages:
            message=self.messages.pop(0)
            if not message.get("no_add"):
                self.agent.add_message(message)

class TokenProcessor:
    """
    Class used to apply a post-processing on the AI response.
    Designed to work either on string or generator (streamed) responses.
    """

    def __init__(self,size,processing_funcs=None):
        self.size=size
        self.buffer=[]
        self.processing_funcs=processing_funcs or []
    
    def process(self,chunk):
        for func in self.processing_funcs:
            chunk=func(chunk)
        return chunk
    
    def __call__(self,content):
        if isgenerator(content):
            def reader():
                for token in content:
                    self.buffer.append(token)
                    if len(self.buffer)>self.size:
                        chunk=''.join(self.buffer)
                        new_chunk=self.process(chunk)
                        self.buffer=tokenize(new_chunk)
                        while len(self.buffer)>self.size:
                            yield self.buffer.pop(0)
                    else:
                        pass
                while len(self.buffer)>0:
                    yield self.buffer.pop(0)
            return reader()
        elif isinstance(content,str):
            return self.process(content)

class StreamSplitter:
    """
    Class used to duplicate a stream 
    """
    def __init__(self,generator,n):
        self.generator=generator
        self.queues=None
        self.n=n
        self.start_read_thread()

    def start_read_thread(self):
        self.queues=[Queue() for _ in range(self.n)]
        thread=Thread(target=self.splitter)
        thread.start()

    def splitter(self):
        for chunk in self.generator:
            for q in self.queues:
                q.put(chunk)
        for q in self.queues:
            q.put("#END#")

    def get_reader(self,i):
        def reader():
            while not (chunk:=self.queues[i].get())=="#END#":
                yield chunk
        return reader()

class VoiceProcessor:
    """
    Class handling TTS.
    Uses the speak method as entry point.
    Takes a string or stream as input.
    Speaks the content as it goes.
    Returns a token stream synchronized with speech.
    The thread_decorator is meant for Streamlit compatibility (to decorate Threads with add_script_run_ctx).
    """
    def __init__(self,agent):
        self.agent=agent
        self.line_queue=Queue()
        self.audio_queue=Queue()
        self.output_queue=Queue()
        
    def line_splitter(self,content):
        if isgenerator(content):
            self.line_queue=Queue()
            def target(content):
                line=""
                for chunk in content:
                    while '\n' in chunk:
                        parts=chunk.split('\n')
                        line+=parts[0]
                        self.line_queue.put(line)
                        chunk='\n'.join(parts[1:])
                        line=""
                    else:
                        line+=chunk
                if line:
                    self.line_queue.put(line)
                self.line_queue.put("#END#")

            thread=self.agent.thread_decorator(Thread(target=target,args=(content,)))
            thread.start()

            def reader():
                while not (line:=self.line_queue.get())=="#END#":
                    yield line
            return reader()
        elif isinstance(content,str):
            return content.split('\n')

    def line_processor(self,content):
        self.audio_queue=Queue()
        def target(content):
            flag=False
            for line in self.line_splitter(content):
                if self.agent.config.voice_enabled and line:
                    if line.startswith('```') and not flag:
                        flag=True
                        audio=None
                    elif flag and line.startswith("```"):
                        flag=False
                        audio=None
                    elif flag:
                        audio=None
                    else:
                        audio=self.agent.text_to_audio_hook(line)
                else:
                    audio=None
                self.audio_queue.put((line,audio))
            self.audio_queue.put("#END#")

        thread=self.agent.thread_decorator(Thread(target=target,args=(content,)))
        thread.start()

        def reader():
            while not (content:=self.audio_queue.get())=="#END#":
                yield content
        return reader()
    
    def process(self,line,audio):
        if audio:
            def target1(line):
                if self.agent.config.stream:
                    for token in tokenize(line):
                        self.output_queue.put(token)
                        time.sleep(0.2)
                else:
                    self.output_queue.put(line)
                self.output_queue.put('\n')
        else:
            def target1(line):
                if self.agent.config.stream:
                    for token in tokenize(line):
                        self.output_queue.put(token)
                        time.sleep(0.02)
                else:
                    self.output_queue.put(line)
                self.output_queue.put('\n')
            
        def target2(audio):
            self.agent.audio_play_hook(audio)
        thread1=Thread(target=target1,args=(line,))
        thread1.start()
        if self.agent.config.voice_enabled:
            thread2=self.agent.thread_decorator(Thread(target=target2,args=(audio,)))
            thread2.start()
        thread1.join()
        if self.agent.config.voice_enabled:
            thread2.join()

    def speak(self,content):
        if self.agent.audio_play_hook is not None and self.agent.text_to_audio_hook is not None and self.agent.config.voice_enabled:
            def target(content):
                self.output_queue=Queue()
                for line,audio in self.line_processor(content):
                    self.process(line,audio)
                self.output_queue.put("#END#")
            thread=self.agent.thread_decorator(Thread(target=target,args=(content,)))
            thread.start()
            def reader():
                while not (token:=self.output_queue.get())=="#END#":
                    yield token
            return reader()
        else:
            return content
            
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

class Tool:
    """
    Tool object used to pass custom tools to the AI.
    """

    def __init__(self,name,description,obj,type='function',example=None,parameters=None,required=None,mode="python"):
        self.mode=mode
        self.type=type
        self.name=name
        self.description=description
        self.example=example
        self.obj=obj
        self.parameters=parameters or {}
        self.required=required or []
    
    def get_description(self):
        desc="Name: "+self.name+'\n'
        desc+="Type: "+self.type+"\n"
        desc+="Description: "+self.description+'\n'
        if self.parameters:
            desc+="Parameters:\n"
            for name,param in self.parameters.items():
                    desc+='\t'+name+':\n'
                    if isinstance(param,dict):
                        if param.get("type"):
                            desc+='\t\t'+"Type: "+param["type"]+'\n'
                        if param.get("description"):
                            desc+='\t\t'+'Description: '+textwrap.dedent(param["description"])+'\n'
                        if param.get("enum"):
                            desc+='\t\t'+'Accepted values: '+str(param["enum"])+'\n'
                    elif isinstance(param,str):
                        desc+='\t\t'+'Description: '+textwrap.dedent(param)+'\n'

                    if name in self.required:
                        desc+='\t\t'+"Required: Yes"+'\n'
                    else:
                        desc+='\t\t'+"Required: No"+'\n'
                        
        if self.example:
            desc+="Example:\n"
            desc+="###\n"
            desc+=textwrap.dedent(self.example)+'\n'
            desc+="###\n"
        return desc

    def get_dict(self):
        properties=dict()
        for name,param in self.parameters.items():
            if isinstance(param,dict):
                properties[name]=param
            elif isinstance(param,str):
                properties[name]=dict(description=param)
        tool=dict(
            type=self.type,
            function=dict(
                name=self.name,
                description=self.description,
                parameters=dict(
                    type="object",
                    properties=properties
                ),
                required=self.required
            )
        )
        return tool

    def __call__(self,*args,**kwargs):
        return self.obj(*args,**kwargs)

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
    - generate images via DALL-E 3,
    - persistent memory storage via an external json file.
    Also usable as an 'intelligent' python function capable of generating scripts autonomously and returning any kind of processed data or python object according to a query in natural language along with some kwargs passed in the call.
    Can use the full range of common python packages in its scripts (provided they are installed and well known to the AI)
    """

    @staticmethod
    def setup_folder(path=None):
        config=objdict.load(root_join("config.json"),_use_default=True)
        config.folder=path or config.folder
        if not config.folder:
            config.folder=os.path.expanduser(os.path.join("~","Pandora"))
        if not os.path.isdir(config.folder):
            try:
                os.makedirs(config.folder)
            except Exception as e:
                print(f"Error when attempting to create Pandora's profiles folder:\n{str(e)}")
                config.folder=root_join("UserFiles")
                os.makedirs(config.folder)
                print(f"Defaulting to '{config.folder}' as the profiles folder.") 
                print("(Call Pandora.setup_folder(your_folder_path) to choose another folder.)")
        try:
            config.dump()
        except:
            pass
        Pandora.folder=config.folder

    folder=None

    @staticmethod
    def folder_join(*args):
        return os.path.join(Pandora.folder,*args)
    
    #Default configuration
    default_config=objdict(
        username='User',
        model="gpt-4-vision-preview",
        vision_enabled=True,
        voice_enabled=True,
        enabled=True,
        language='en',
        uses_memory=True,
        uses_past=True,
        uses_agents=True,
        top_p=1,
        temperature=1,
        max_tokens=2000,
        token_limit=16000,
        stream=True,
        display_mode="normal"
    )

    #Default base preprompt used to instruct the AI, should be tweaked with caution. 
    base_preprompt="""
        #INSTRUCTIONS
        You're <<self.name>>, an advanced AI-powered python console resulting of the combination of the latest OpenAI model and a built-in Python interpreter.
        The user can use you as a regular python console in which he can run scripts, interact with you in many languages, or pass you files/images that you may analyze using your multimodal abilities.
        As an AI model, you've been trained to use the Python interpreter as your primary toolbox to perform various tasks. 
        Your responses are parsed and all parts matching the regex pattern r'```run_python(.*?)```' will get executed directly in your interpreter.
        The interpreter's feedback will be redirected as system messages in context to inform your next steps or help you self-correct possible mistakes autonomously.
        The other parts of your response will be displayed as markdown chat messages to the user (KaTeX is supported).

        On top of popular python libraries that you may import in your python scripts, some specific tools are predeclared in your interpreter's namespace to help you deal with specific tasks.
        These tools and how you should use them in your scripts will be detailed in the 'TOOLS' section below.

        Don't make up answers, use your vast knowledge in combination with a smart use of available tools to craft informed responses to the user.
        Plan your strategy and break down complex tasks in smaller steps, using possibly several turns to achieve them sequentially.
        In case you run into unexpected execution issues or don't what/how to do, default back to asking the user for guidance.

        Simplified example of the assistant expected behaviour:

        Example:

        User:
        What is the factorial of 12?
                                
        Assistant:
        Let's check this out with a simple script.
        ```run_python
        import math
        math.factorial(12)
        ```
        Now, I'll let this script execute.
                                
        Interpreter:
        479001600
                                
        Assistant:
        The factorial of 12 is 479001600.

        #END OF INSTRUCTIONS
        """

    def __init__(self,name=None,openai_api_key=None,google_custom_search_api_key=None,google_custom_search_cx=None,config=None,base_preprompt=None,preprompt=None,decide_prompt=None,tools=None,builtin_tools=None,example=None,infos=None,work_folder=None,console=None,input_hook=None, display_hook=None,context_handler=None,text_to_audio_hook=None,audio_play_hook=None,thread_decorator=None):
        Pandora.setup_folder()
        self.name=name or 'Pandora'
        self.init_client(openai_api_key=openai_api_key)
        self.init_folders_and_files(work_folder=work_folder)
        self.init_config(config=config)
        self.init_preprompts(base_preprompt=base_preprompt,preprompt=preprompt,example=example,decide_prompt=decide_prompt,infos=infos)
        self.collector=OutputCollector(self,display_hook=display_hook,context_handler=context_handler)
        processing_funcs=[
                lambda chunk:chunk.replace("```run_python","```python"),
                lambda chunk:chunk.replace(r"\(","$"),
                lambda chunk:chunk.replace(r"\)","$"),
                lambda chunk:chunk.replace(r"\[","$$"),
                lambda chunk:chunk.replace(r"\]","$$")
            ]
        self.token_processor=TokenProcessor(size=5,processing_funcs=processing_funcs)
        self.retriever=Retriever(openai_api_key=self.client.api_key,folder=os.path.join(self.work_folder,"documents"))
        self.init_console(console=console,input_hook=input_hook)
        self.init_TTS(text_to_audio_hook=text_to_audio_hook,audio_play_hook=audio_play_hook,thread_decorator=thread_decorator)
        self.init_tools(tools=tools,builtin_tools=builtin_tools,google_custom_search_api_key=google_custom_search_api_key,google_custom_search_cx=google_custom_search_cx)
        self.messages=[]
        self.new_turn=False
        self.output=None
        self.token_queue=Queue()
        self.console_queue=Queue()
        self.run_startup()

    def init_client(self,openai_api_key=None):
        """
        Initializes the OpenAI client
        """
        if openai_api_key:
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

    @property
    def instance_name(self):
        return self.name.lower().replace(' ','_')

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

    def init_tools(self,tools=None,builtin_tools=None, google_custom_search_api_key=None, google_custom_search_cx=None):
        """
        Initializes the tools used by the AI
        First load custom tools passed to the constructor
        Then load builtin tools (all by default, but can be selected via the builtin_tools list passed in the constructor)
        """
        self.tools=tools or {}
        self.load_tools()
        self.builtin_tools=builtin_tools or ['observe','generate_image','memory','open_in_browser','websearch','get_webdriver','get_text','retriever']

        if 'observe' in self.builtin_tools:
            self.add_tool(
                name="observe",
                description="observe(source) # Inject in your context feed informations extracted from any kind of source. The function will do its best to return something informative, whatever the source type. Use it proactively to get contextual awareness of the source content. For token efficiency, observed information won't persist in context more that one turn, so you will have to observe the source again anytime you need visibility on the content.",
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

        if 'get_text' in self.builtin_tools:
            self.add_tool(
                name="get_text",
                description="text=get_text(source) # Returns extracted text content or representation of a given source (directory, data structure, file, url, python object / module, ...) as a string.",
                obj=get_text,
                parameters=dict(
                    source="(any) The source of text content. Can be a directory, file, url, module, class, function, variable. Can deal with a wide variety of sources."
                ),
                required=['source'],
            )


        if 'retriever' in self.builtin_tools:
            self.add_tool(
                name='retriever',
                obj=self.retriever,
                type='object',
                description="""
                retriever # A document store used to implement your chunk-retrieval mechanism. Retrieval is automatic according to semantic relevance of loaded document chunks with respect to the current context.
                # Methods:
                retriever.get_titles() # returns the list of titles of documents saved as files in the document store (can be loaded in memory).
                retriever.get_loaded() # returns the list of titles of documents currently loaded in memory and active for chunk retrieval.
                retriever.new_document(title,text,description) # Create a new stored document from a givent text content (chunked, embedded, saved and loaded for semantic search).
                retriever.load_docuemnt(title) # Loads a document in memory.
                retriever.close_document(title) # unloads a document from memory.
                """
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

        if 'open_in_browser' in self.builtin_tools:
            import webbrowser
            self.add_tool(
                name="open_in_browser",
                description="open_in_browser(file_or_url) # Default function to show a file or url via the user's default webbrowser.",
                obj=webbrowser.open
            )

        if 'memory' in self.builtin_tools:
            self.add_tool(
                name="memory",
                description="memory # a custom nested attribute-style access data structure linked to a JSON file for long lasting storage. Supports dump() method to save the content to the file, and dumps() to serialize into a string. Nested keys must all be valid identifiers.",
                obj=self.memory,
                type="object"
            )

        if 'websearch' in self.builtin_tools:
            from google_search import init_google_search
            google_custom_search_api_key=google_custom_search_api_key or os.getenv('GOOGLE_CUSTOM_SEARCH_API_KEY')
            google_custom_search_cx=google_custom_search_cx or os.getenv('GOOGLE_CUSTOM_SEARCH_CX') 
            google_search=init_google_search(api_key=google_custom_search_api_key,cse_id=google_custom_search_cx)
            def websearch(query,num=5,type='web'):
                self.observe(google_search(query,num,type))
            
            self.add_tool(
                name="websearch",
                description="websearch(query,num=5,type='web') # Make a google search. type can be either 'web' or 'image'. Results are automatically observed (returns None).",
                obj=websearch
            )

        if 'get_webdriver' in self.builtin_tools:
            from get_webdriver import get_webdriver
            
            self.add_tool(
                name="get_webdriver",
                description="driver=get_webdriver() # This function returns a preconfigured headless firefox selenium webdriver suitable tu run in the current environment.",
                obj=get_webdriver,
                example="""
                driver = get_webdriver()
                # Open Wikipedia webpage
                driver.get('https://www.wikipedia.org')
                """
            )
                 
    def init_TTS(self,text_to_audio_hook=None,audio_play_hook=None,thread_decorator=None):
        """
        Initializes the TTS hooks used to speak out AI messages. 
        """
        self.text_to_audio_hook=text_to_audio_hook or text_to_audio
        self.audio_play_hook=audio_play_hook or play_audio
        self.thread_decorator=thread_decorator or (lambda thread:thread)
        self.voice_processor=VoiceProcessor(self)

    def init_config(self,config=None):
        """
        Initializes the configuration.
        First load the default one.
        Then update with config files in the config folder
        Then update with config passed in the constructor
        """
        self.config=Pandora.default_config.copy()    

        if config:
            self.config.update(config)

    def init_preprompts(self,base_preprompt=None,preprompt=None,example=None,decide_prompt=None,infos=None):
        """
        Initializes the preprompts (base_preprompt, preprompt, example).
        First load the default base preprompt.
        Then update with preprompts passed in the constructor
        """
        #load defaults
        self.base_preprompt=Pandora.base_preprompt
        self.preprompt=''
        self.example=''
        self.decide_prompt=None

        #load from constructor kwargs
        if base_preprompt:
            self.base_preprompt=base_preprompt
        if preprompt:
            self.preprompt=preprompt
        if example:
            self.example=example
        if decide_prompt:
            self.decide_prompt=decide_prompt
        

        self.infos=infos or []

    def run_startup(self):
        if self.startup_file and os.path.isfile(self.startup_file):
            with open(self.startup_file,'r') as f:
                code=f.read()
            self.console.run_silent(code)

    def add_message(self,message):
        """
        Adds a message the internal messages history
        """
        self.messages.append(message)

    def add_user_prompt(self,content,tag='user_message'):
        """
        Adds a user prompt to the collector.
        """
        content=process_markdown(content)
        message=Message(content,role="user",name=self.config.username,type='prompt',tag=tag)
        self.collector.collect(message)

    def add_user_codeblock(self,code,language="text"):
        """
        Shortcut function to add a user codeblock to the collector.
        """
        content=f'```{language}\n{code}\n```'
        self.add_user_prompt(content)

    def add_image(self,file=None,url=None,bytesio=None):
        """
        Adds an image to the AI context for vision analysis (if vision is enabled)
        """
        success=False
        if self.config.vision_enabled and self.config.model=="gpt-4-vision-preview":
            if(file or url or bytesio):
                img=Image(
                    file=file,
                    url=url,
                    bytesio=bytesio
                )
                self.add_message(img.get_message())
                self.new_turn=True
                success=True
        return success
        
    def add_tool(self,name,description,obj,type="function",example=None,parameters=None,required=None,mode="python"):
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
            mode=mode,
            name=name,
            type=type,
            description=description,
            obj=obj,
            parameters=parameters,
            required=required,
            example=example
        )
        self.tools[tool.name]=tool
        if tool.mode=="python":
            self.console.update_namespace({tool.name:tool.obj})

    def load_tools(self):
        """
        Loads all tools in the console's namespace
        """
        for tool in self.tools.values():
            if tool.mode=="python":
                self.console.update_namespace({tool.name:tool.obj})

    def observe(self,source):
        """
        Main tool used by the AI to observe something.
        Supports a wide range of possible sources:
        folders, files, urls, images, python modules/classes/functions/variables
        """

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
        success=self.add_image(file=file,url=url,bytesio=bytesio)
        if not success:
            print("Impossible to observe image.")

    def observe_data(self,**kwargs):
        """
        Extract textual data from some sources passed as kwargs and inject the data in the AI context as a system message
        """
        s='#Result of observation:\n'
        for key in kwargs:
            s+=f"{key}:\n"
            s+=get_text(kwargs[key])+'\n'
        message=Message(content=s,role='system',name="Observation",type='temp',tag="observation")
        self.add_message(message)
        self.new_turn=True

    def gen_image(self,description,file_path,model="dall-e-3",size="1024x1024",quality="standard"):
        """
        Generates an image using DALL-E 3
        """
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

    def inject_kwargs(self,kwargs):
        """
        Inject some kwargs in the internal console's namespace from outside.
        Then notify the AI it happened, giving it the name and repr of the kwargs as a system message.
        """
        if kwargs:
            self.console.update_namespace(kwargs)
            s="#The following kwargs have been injected in the console's namespace:\n"
            for key in kwargs:
                s+=f"{key}={str(kwargs[key])}\n"
            message=Message(content=s,role='system',name="system_bot",type='prompt')
            self.add_message(message)

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

    def get_tools(self):
        """
        Returns the description of all tools as a system message.
        """
        tools={name:tool for name,tool in self.tools.items() if tool.mode=='python'}
        if tools:
            s="#TOOLS:\n"
            s+="The following tools are already declared in the console namespace and ready to be used in your python scripts. Below is a description of each.\n"
            for tool in tools.values():
                s+=tool.get_description()+'\n'
            s+="#END OF TOOLS"
            message=Message(content=s,role="system",name="system_bot")
            return [message]
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
        memory=self.get_memory()
        info=self.get_infos()
        temp=self.get_temp()
        tools=self.get_tools()
        count=total_tokens(preprompt+tools+info+example+memory+temp+prompts)
        queued=self.get_queued(count)
        context=preprompt+tools+info+example+memory+Sort(queued+temp+prompts)
        self.send_prompts_to_queue()
        return context

    def prepare_messages(self,messages):
        """
        Converts a list of messages (objdict instances) to a list of dicts accepted by the OpenAI API.
        """
        return [dict(content=msg.content,role=msg.role,name=msg.name) for msg in messages]

    def streamed_completion(self):
        
        context=self.gen_context()

        kwargs=dict(
            model=self.config.model,
            messages=self.prepare_messages(context),
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            stream=True
        )

        success=False
        err=0
        while not success and err<2:
            try:
                response = self.client.chat.completions.create(**kwargs)
            except Exception as e:
                print(str(e))
                err+=1
                time.sleep(0.5)
            else:
                success=True

        if success:
            for chunk in response:
                if (token:=chunk.choices[0].delta.content) is not None:
                    self.token_queue.put(token)
                    time.sleep(0.005)
        else:
            for token in tokenize("I'm sorry but there was a recurring error with the OpenAI server. Would you like me to try again?"):
                self.token_queue.put(token)
                time.sleep(0.005)
        self.token_queue.put('\n')
        self.token_queue.put("#END#")   
        
    def stream_generator(self):
        self.token_queue=Queue()
        self.gen_thread=Thread(target=self.streamed_completion)
        self.gen_thread.start()
        def reader():
            tokens=[]
            while not (token:=self.token_queue.get())=="#END#":
                tokens.append(token)
                yield token
            content=''.join(tokens)
            msg=Message(content=content,role="assistant",name=self.name)
            self.add_message(msg)
            self.response=content
        return reader()

    def standard_completion(self):
        """
        Takes a context and calls the OpenAI API for a response.
        Configuration of the model is taken from self.config.
        Messages in context are prepared (convered to suitable dicts) before being passed to the API.
        """
        context=self.gen_context()

        kwargs=dict(
            model=self.config.model,
            messages=self.prepare_messages(context),
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p
        )

        success=False
        err=0
        while not success and err<2:
            try:
                answer = self.client.chat.completions.create(**kwargs)
            except Exception as e:
                print(str(e))
                err+=1
                time.sleep(0.5)
            else:
                success=True
        if not success:
            content="I'm sorry but there was a recurring error with the OpenAI server. Would you like me to try again?\n"
        else:
            content=answer.choices[0].message.content.strip()+'\n'

        msg=Message(content=content,role="assistant",name=self.name)
        self.add_message(msg)
        self.response=content
        return content

    def response_generator(self):
        if self.config.stream:
            return self.stream_generator()
        else:
            return self.standard_completion()

    def process(self):
        
        self.new_turn=False

        self.collector.process_all()
        
        self.status("Generating response.")
            
        content=self.voice_processor.speak(self.token_processor(self.response_generator()))
        
        msg=Message(content=content,role="assistant",name=self.name,type="queued",tag="assistant_message",no_add=True)
        self.collector.collect(msg)

        code_parts=extract_python(self.response)
        
        if code_parts:
            self.status("Running code.")

        for code in code_parts:
            self.console.run(self.replace(code.strip()))
            if self.console.get_result():
                self.new_turn=True
        
        self.status("#DONE#")

        if self.new_turn:
            self.process()

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

    def replace(self,code):
        """
        Utility function to replace some elements of syntax into others in a code snippet.
        Useful to avoid AI or the user make common syntax mistakes or misuse some tools.
        Desired replacements can be customized in self.config.code_replacements.
        """
        if self.config.get('code_replacements'):
            for segment, replacement in self.config.code_replacements:
                code=code.replace(segment,replacement)
            return code.strip()
        else: 
            return code                                    
            
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
                if self.retriever.get_loaded():
                    results=self.retriever.search(query='\n'.join(prompt.content for prompt in self.get_prompts()),num=5)
                    if results:
                        msg=Message(content=str(results),role="system",name="Retriever",type="temp")
                        self.add_message(msg)
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
        return f"Successfully transmitted data={repr(data)}"

    def clear_message_history(self):
        """
        Clears the message history of the agent.
        """
        self.messages=[]   

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

    def decide(self):
        prompt=textwrap.dedent("""
        [Assignement]
        Based on the immediate context of the conversation (described above) between the user and the assistant:
        - Output "1" if the assistant should continue generating or perform more action.
        (For instance when the assistant is clearly in the middle of a task that require no additional user input in order to complete it.)
        - Output "0" if the assistant should stop here and let the user make a move.
        (For instance when the assistant has finished a task, or when it is asking for input from the user.)
        
        In case you're not sure what the assistant should do, output "0".
        
        Response:
        
        """)
        decide_prompt=self.decide_prompt or prompt
        response=self.completion(prompt=decide_prompt,context=self.get_messages()[-6:],max_tokens=1,temperature=0,model="gpt-3.5-turbo")
        if int(response)==1:
            return True
        else:
            return False
        
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
            stream=False,
        )

        config.update(kwargs)

        response = self.client.chat.completions.create(
                messages=self.prepare_messages(context),
                **config
            )
        return response.choices[0].message.content.strip()
        
    def interact(self):
        """
        Implements a simple interaction loop with the agent.
        Useful to test the agent in the terminal without code overhead:
        Example:
        from pandora_ai import Pandora

        agent=Pandora()

        agent.interact() # enters the interaction loop in the terminal (type exit() or quit() to end the loop)
        """

        def custom_input():
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
        
        self.loop_flag=True

        def custom_exit():
            self.loop_flag=False
        
        self.console.update_namespace(
            exit=custom_exit,
            quit=custom_exit
        )

        print("Pandora AI - Interactive session")
        print("Enter your python commands or chat with Pandora in natural language.")
        print("Multi-line input is supported. End your input script with '//' before pressing [Enter] to submit.")
        print("Run exit() or quit() to exit this interactive session.")
        print('Get some help at any time by simply asking the AI.')

        while self.loop_flag:
            prompt=custom_input()
            self(prompt)

        print("Exiting interactive session. See you next time!")

class AIFunction(Pandora):

    def __init__(self,verbose=False):
        builtin_tools=['return_output']
        self.verbose=verbose
        base_preprompt="""
        #INSTRUCTIONS
        You're an advanced AI-powered python function resulting of the combination of the latest OpenAI model and a built-in Python interpreter.
        As an AI model, you've been especialy trained to use the Python interpreter as your primary means of action. 
        Your responses are parsed and all parts matching the regex pattern r'```run_python(.*?)```' will be executed directly in your internal interpreter.
        The user can use you as a python function like so:
        data=ai_function(query,**kwargs)
        Upon receiving the query and kwargs, you design a python script meant to satisfy the user's demand, generaly consisting in producing a data output.
        You're provided with a special 'return_output' tool that you must use to send the produced data out to the user as a response.
        YOU MUST USE THIS TOOL AND NO OTHER TO SEND THE RETURNED DATA OUT TO THE USER.
        In case your script results in an exception, you'll be given another turn to self-correct based on the interpreter's feedback, until the 'return_output' call is finally successful.
        You're not expected to talk to the user, only to do your job sending data out as a python function would.
        
        Examples:

        user:
        Return the factorial of 12.

        assistant:
        ```run_python
        import math
        return_output(math.factorial(12))
        ```

        user:
        Return a function that doubles every element of its input list and returns it.

        assistant:
        ```run_python
        def double_list_elements(input_list):
            input_list=[2*e for e in input_list]
            return input_list
        
        return_output(double_list_elements)
        ```

        #END OF INSTRUCTIONS
        """
        def display(content,tag,status):
            if self.verbose:
                if tag in ['interpreter','assistant_message']:
                    stdout_write(content)
        
        context_handler=None if self.verbose else lambda message: NoContext()
        Pandora.__init__(self,base_preprompt=base_preprompt,builtin_tools=builtin_tools,display_hook=display,context_handler=context_handler)
        self.config.update(
            model="gpt-3.5-turbo",
            temperature=0.6
        )
        