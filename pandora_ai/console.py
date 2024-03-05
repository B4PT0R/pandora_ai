import code
from code import InteractiveConsole
import sys
import os
from contextlib import contextmanager
import re

_root_ = os.path.dirname(os.path.abspath(__file__))

# Redirect inputs/outputs to a target I/O object
@contextmanager
def redirect_IOs(target):
    stdout_fd = sys.stdout
    stderr_fd = sys.stderr
    stdin_fd = sys.stdin
    sys.stdout = target
    sys.stderr = target
    sys.stdin = target
    yield
    sys.stdout = stdout_fd
    sys.stderr = stderr_fd
    sys.stdin = stdin_fd

class IO_Interceptor:
    # The I/O object intercepting the interpreter's inputs/outputs.
    def __init__(self, console):
        self.console = console
        self.buffer = ''

    def write(self, text):
        self.buffer += text
        if text.endswith('\n'):
            self.console.results[-1].append(self.buffer)
            if self.console.output_redirection_hook:
                self.console.output_redirection_hook(self.buffer)
            else:
                sys.__stdout__.write(self.buffer)
            self.buffer = ''

    def readline(self):
        if not self.buffer == '':
            self.write('\n')
        if self.console.input_redirection_hook:
            string = self.console.input_redirection_hook()
        else:
            string = sys.__stdin__.readline()
        return string

    def flush(self):
        pass

class SilentIO:
    def __init__(self,console):
        self.console = console
        self.buffer = ''

    def write(self, text):
        # pour un comportement silencieux
        self.buffer += text
        if text.endswith('\n'):
            self.console.results[-1].append(self.buffer)
            self.buffer = ''

    def readline(self):
        # Pour simuler une entrée, retourner simplement une chaîne vide
        return '\n'

    def flush(self):
        # Pas d'action nécessaire ici
        pass

def remove_empty_lines(script):
    # Regular expression for multiline string literals
    multiline_string_pattern = r"('''[\s\S]*?'''|\"\"\"[\s\S]*?\"\"\")"

    # Placeholder dictionary for multiline string literals
    placeholders = {}
    def replace_with_placeholder(match):
        placeholder = f"__MULTILINE_STRING_{len(placeholders)}__"
        placeholders[placeholder] = match.group(0)
        return placeholder

    # Replace multiline string literals with placeholders
    script_with_placeholders = re.sub(multiline_string_pattern, replace_with_placeholder, script)

    # Remove empty lines from the rest of the script
    non_empty_lines = [line for line in script_with_placeholders.split('\n') if line.strip()]

    # Reinsert multiline string literals
    processed_script = '\n'.join(non_empty_lines)
    for placeholder, string_literal in placeholders.items():
        processed_script = processed_script.replace(placeholder, string_literal)

    return processed_script

class Console(InteractiveConsole):
    # The python interpreter in which the code typed in the input cell will be run
    def __init__(self, namespace=None, startup=None,mode='scripted',input_display_hook=None,input_redirection_hook=None, output_redirection_hook=None):
        self.namespace = namespace or {}
        self.namespace['__console__'] = self
        self.outer_cwd=os.getcwd()
        self.inner_cwd=os.getcwd()
        self.input_display_hook=input_display_hook or (lambda content:None)
        self.output_redirection_hook=output_redirection_hook
        self.input_redirection_hook=input_redirection_hook
        self.interceptor = IO_Interceptor(self)
        self.silence=SilentIO(self)
        InteractiveConsole.__init__(self, self.namespace)
        self.mode=mode
        self.current_code=None
        self.inputs = []
        self.results = []
        self.error = False
        if startup:
            self.run_file(startup)
            if self.inputs:
                self.inputs.pop(-1)

    @contextmanager
    def switch_cwd(self):
        self.outer_cwd=os.getcwd()
        os.chdir(self.inner_cwd)
        try:
            yield
        finally:
            self.inner_cwd=os.getcwd()
            os.chdir(self.outer_cwd)

    def run(self, code):
        if self.mode == "scripted":
            self.run_exec(code)
        elif self.mode == "interactive":
            self.run_eval(code)
        elif self.mode == "silent":
            self.run_silent(code)
        else:
            raise Exception('This execution mode is not supported by the Console object.')

    def run_file(self,source):
        if source.endswith('.py'):
            try:
                with open(source,'r') as f:
                    code=f.read()
                self.run_exec(code)
            except Exception as e:
                with redirect_IOs(self.interceptor):
                    print(str(e))
        else:
            raise ValueError("The provided source should be a python file.")

    def run_exec(self, source): 
        # The original run function
        self.error = False
        self.inputs.append(source)
        if self.input_display_hook:
            self.input_display_hook(source)
        self.results.append([])
        with self.switch_cwd():
            with redirect_IOs(self.interceptor):
                try:
                    output = code.compile_command(source, 'user', 'exec')
                except Exception as e:
                    self.error = True
                    print(str(e))
                else:
                    if output is not None:
                        self.current_code=source
                        self.runcode(output)
                    else:
                        self.error = True
                        e = SyntaxError("Incomplete code isn't allowed to be executed.")
                        print(str(e))

    def run_eval(self, source): 
        # The new run function that processes line by line
        self.error = False
        self.inputs.append(source)
        self.results.append([])
        source=remove_empty_lines(source)
        lines = source.split('\n')+['']
        current_lines = []
        line_index=0
        with self.switch_cwd():
            with redirect_IOs(self.interceptor):  # context manager for I/O redirection
                while line_index < len(lines) and not self.error:
                    current_lines.append(lines[line_index])
                    current_code="\n".join(current_lines)
                    try:
                        compiled_code = code.compile_command(current_code, 'user', 'single')
                        if compiled_code is not None:
                            self.current_code=current_code
                            self.runcode(compiled_code)
                            current_lines = []
                    except SyntaxError as e:
                        if len(current_lines) > 1:
                            # Attempt to fix and recompile code by adding a newline
                            current_lines = current_lines[:-1] + ['']
                            current_code="\n".join(current_lines)
                            try:
                                compiled_code = code.compile_command(current_code, 'user', 'single')
                                if compiled_code is not None:
                                    self.current_code=current_code
                                    self.runcode(compiled_code)
                                    current_lines = [lines[line_index]]  # Start with the last line for the next iteration
                                    continue
                            except SyntaxError as e:
                                pass  # If recompilation fails, fall through to the error printing
                        print(f"Syntax Error: {e}")
                        self.error = True
                        break
                    except Exception as e:
                        print(f"Error: {e}")
                        self.error = True
                        break
                    finally:
                        line_index += 1

    def run_silent(self, source):
        self.error = False
        self.inputs.append(source)
        self.results.append([])
        with self.switch_cwd():
            with redirect_IOs(self.silence):
                try:
                    output = code.compile_command(source, 'user', 'exec')
                except Exception as e:
                    self.error = True
                    print(str(e), file=sys.__stderr__)  # Imprimer les erreurs vers stderr réel pour le débogage
                else:
                    if output is not None:
                        self.current_code = source
                        self.runcode(output)
                    else:
                        self.error = True
                        e = SyntaxError("Incomplete code isn't allowed to be executed.")
                        print(str(e), file=sys.__stderr__)  # Imprimer vers stderr réel

    def update_namespace(self, *args,**kwargs): 
        # Updates the interpreter's namespace with a name:object dictionary
        self.namespace.update(*args,**kwargs)

    def showtraceback(self):
        self.error = True
        InteractiveConsole.showtraceback(self)

    def get_current_code(self):
        return self.current_code

    def get_result(self): 
        # Quickly get the last output of the interpreter as a string
        return '\n'.join(self.results[-1])