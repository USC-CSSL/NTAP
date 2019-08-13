import setuptools
from os import path


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ntap",
    version="1.0.6",
    author="Praveen Patil",
    author_email="pspatil@usc.edu",
    description="NTAP - CSSL",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/USC-CSSL/NTAP",
    packages=setuptools.find_packages(),
    install_requires = ['absl-py==0.7.1', 'astor==0.8.0', 'boto==2.49.0', 'boto3==1.9.199', 'botocore==1.12.199', 'certifi==2019.6.16', 'chardet==3.0.4', 'docutils==0.14', 'gast==0.2.2', 'gensim==3.8.0', 'google-pasta==0.1.7', 'grpcio==1.22.0', 'h5py==2.9.0', 'idna==2.8', 'jmespath==0.9.4', 'joblib==0.13.2', 'Keras-Applications==1.0.8', 'Keras-Preprocessing==1.1.0', 'Markdown==3.1.1', 'nltk==3.4.4', 'numpy==1.17.0', 'pandas==0.25.0', 'protobuf==3.9.0', 'python-dateutil==2.8.0', 'pytz==2019.2', 'requests==2.22.0', 's3transfer==0.2.1', 'scikit-learn==0.21.3', 'scipy==1.3.0', 'six==1.12.0', 'sklearn==0.0', 'smart-open==1.8.4', 'tensorboard==1.14.0', 'tensorflow==1.14.0', 'tensorflow-estimator==1.14.0', 'termcolor==1.1.0', 'urllib3==1.25.3', 'Werkzeug==0.15.5', 'wrapt==1.11.2'],
    classifiers=[ 
        "Programming Language :: Python :: 3",
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
