from pandas import Series
import os
import pytest
from ntap.bagofwords import DocTerm, LDA, Dictionary, TFIDF

"""
FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'test_files',
    )

LABEL_DATASETS = {'imdb': os.path.join(FIXTURE_DIR, 
                                       'imdb_sentiment_train.zip'),
                  'reviews': os.path.join(FIXTURE_DIR, 
                                          'review_polarity.tar.gz')}

@pytest.mark.datafiles(LABEL_DATASETS['imdb'])
def test_load_imdb(datafiles):
    for fn in datafiles.listdir():
        data = load_imdb(fn)
        assert len(data['pos']) == 12500
        assert len(data['neg']) == 12500
"""

@pytest.fixture
def text8_docterm():
    from gensim import downloader
    text8 = downloader.load('text8')
    text8 = [' '.join(words) for words in text8]
    text8 = text8[:10]
    dt = DocTerm()
    dt.fit(text8)
    return dt

@pytest.fixture
def text8():
    from gensim import downloader
    text8 = downloader.load('text8')
    text8 = [' '.join(words) for words in text8]
    text8 = text8[:10]
    return list(text8)

@pytest.fixture
def text8_series():
    from gensim import downloader
    text8 = downloader.load('text8')
    text8 = [' '.join(words) for words in text8]
    text8 = text8[:10]
    return Series(list(text8))

def test_docterm_attributes(text8_docterm):

    assert isinstance(text8_docterm, DocTerm)
    assert text8_docterm.N == 10
    assert len(text8_docterm.vocab) == 357
    assert text8_docterm.__str__() == ("DocTerm Object (documents: 10, terms: 357)\n"
                            "Doc Lengths (165-193): mean 177.00, median 178.50\n"
                            "Top Terms: academy oil aircraft appears illinois metal "
                            "bc territory characters roman property ground data isbn "
                            "method local ethical france agricultural republican")

def test_docterm_exceptions():
    with pytest.raises(AssertionError):
        text8_docterm = DocTerm()
        text8_docterm.fit([1,2,3])
    with pytest.raises(AssertionError):
        text8_docterm = DocTerm(tokenizer='some_tokenizer')
    with pytest.raises(AssertionError):
        text8_docterm = DocTerm(vocab_size='bad_vocab')

def test_dic(text8):

    d = Dictionary(os.environ["LIWC_PATH"])
    
    counts = d.transform(text8)
    assert float(counts[0,0].__str__()) == pytest.approx(0.115848)

def test_tfidf_from_docterm(text8_docterm):

    tfidf = TFIDF()
    vecs = tfidf.transform(text8_docterm)

    assert float(vecs[0,0].__str__()) == pytest.approx(0.097042796)

def test_tfidf_from_list(text8):

    tfidf = TFIDF()
    vecs = tfidf.transform(text8)

    assert float(vecs[0,0].__str__()) == pytest.approx(0.097042796)

def test_tfidf_from_series(text8_series):

    tfidf = TFIDF()
    vecs = tfidf.transform(text8_series)

    assert float(vecs[0,0].__str__()) == pytest.approx(0.097042796)

@pytest.mark.parametrize("topic_method,mallet_path", 
                         [("online", None), 
                          ("gibbs",os.environ['MALLET_PATH'])])
def test_lda(text8_docterm, topic_method, mallet_path):
    k = 5
    lda_m = LDA(method=topic_method, num_topics=k, num_iterations=10, mallet_path=mallet_path)
    lda_m.fit(text8_docterm)
    vec = [v for _, v in lda_m.transform(["Hi there"])[0]]
    expected_vec = [1./k] * k

    assert expected_vec == pytest.approx(vec)


def test_lda_exceptions(text8_docterm):
    k = 5
    with pytest.raises(ValueError):
        lda_m = LDA(method="gibbs", num_topics=k, num_iterations=10)
        lda_m.fit(text8_docterm)
    with pytest.raises(ValueError):
        lda_m = LDA(method="gibbs", num_topics=k, num_iterations=10, mallet_path="some/path")
        lda_m.fit(text8_docterm)


