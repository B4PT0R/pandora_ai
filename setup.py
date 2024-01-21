import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pandora_ai",
    version="0.0.1",
    author="Baptiste Ferrand",
    author_email="bferrand.maths@gmail.com",
    description="AI-powered python interpreter",
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
    ],
    python_requires='>=3.6',
)
