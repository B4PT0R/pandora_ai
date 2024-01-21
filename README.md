
# Pandora

Pandora is an advanced AI Python console object, resulting from the combination of the latest GPT-4 Turbo model from OpenAI and an interactive Python interpreter. This console allows users to execute Python commands/scripts in real time like a conventional Python console, but also allows to interact in natural language with the assistant and offers rich interactive and multimodal capabilities based on the real-time execution of AI-generated python scripts.

## Main Features

- Implements a custom AI-powered python console using OpenAI GPT4 Turbo model.
- Usable both as a regular python console and/or an AI assistant.
- Capable of generating and running scripts autonomously in its own internal interpreter.
- Can be interacted with using a mix of natural language (markdown and LaTeX support) and python code. 
- Having the whole session in context. Including user prompts/commands, stdout outputs, etc... (up to 128k tokens)
- Highly customizable input/output redirection and display (including hooks for TTS) for an easy and user friendly integration in any kind of application. 
- Modularity with custom tools passed to the agent and loaded in the internal namespace, provided their usage is precisely described to the AI (including custom modules, classes, functions, APIs).

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

## Installation

```bash
$ pip install pandora-ai
```

## Usage

Minimal setting:

```python
from pandora_ai import Pandora

agent=Pandora(openai_api_key=<your_api_key>)

agent.interact()

```

## License

This project is licensed. Please see the LICENSE file for more details.

## Contributions

Contributions are welcome. Please open an issue or a pull request to suggest changes or additions.

## Contact

For any questions or support requests, please contact Baptiste Ferrand at the following address: bferrand.maths@gmail.com.
