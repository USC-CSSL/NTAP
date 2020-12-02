import setuptools
from os import path

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    author="Brendan Kennedy",
    author_email="btkenned@usc.edu",
    description="The Neural Text Analysis Pipeline (NTAP) provides high-level access to cutting-edge NLP methods for text analysis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/USC-CSSL/NTAP",
    packages=setuptools.find_packages(),
    install_requires = ['gensim', 'numpy', 'pandas', 'scikit-learn'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
