import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pandora_ai",
    version="0.0.1",
    author="Baptiste Ferrand",
    author_email="bferrand.maths@gmail.com",
    description="GPT4-powered python interpreter / AI assistant",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/B4PT0R/pandora_ai",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "selenium",
        "get-gecko-driver",
        "beautifulsoup4",
        "google-api-python-client",
        "objdict_bf",
        "odfpy",
        "openai",
        "PyPDF2",
        "python-dotenv",
        "python-docx",
        "requests",
        "tiktoken"
    ],
    python_requires='>=3.6',
)
