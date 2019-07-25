import setuptools
from os import path


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ntap",
    version="1.0.3",
    author="Praveen Patil",
    author_email="pspatil@usc.edu",
    description="NTAP - CSSL",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/USC-CSSL/NTAP",
    packages=setuptools.find_packages(),
    install_requires = ['bleach==1.5.0', 'numpy==1.16.0', 'tensorflow==1.14.0','tensorflow-tensorboard==1.5.1', 'Markdown==3.0.1',\
     'html5lib==0.9999999', 'nltk==3.4', 'pandas==0.24.2', 'backports.weakref==1.0.post1', 'boto==2.49.0', 'boto3==1.9.60', \
     'botocore==1.12.60', 'bz2file==0.98', 'certifi==2018.11.29', 'chardet==3.0.4', 'Cython==0.29.1', 'docutils==0.14', \
     'emoji==0.5.1', 'emot==2.0', 'enum34==1.1.6', 'funcsigs==1.0.2', 'future==0.17.1', 'gensim==3.6.0', 'idna==2.7', \
     'jmespath==0.9.3', 'mock==2.0.0', 'pbr==5.1.1', 'protobuf==3.6.1', 'python-dateutil==2.7.5', 'pytz==2018.7', \
     'requests==2.20.1', 's3transfer==0.1.13', 'scikit-learn==0.20.1', 'scipy==1.1.0', 'singledispatch==3.4.0.3', \
     'six==1.11.0', 'sklearn==0.0', 'sklearn-pandas==1.8.0', 'smart-open==1.7.1', 'urllib3==1.24.1', 'Werkzeug==0.14.1',\
      'stanfordcorenlp', 'progressbar2','tensorboard==1.14.0'],
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
