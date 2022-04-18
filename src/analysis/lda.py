import nltk
import spacy
from nltk.corpus import stopwords
import pandas as pd
import gensim
from gensim.utils import simple_preprocess


def get_vectors(lda_model, corpus):
    vectors = []
    for i in range(len(corpus)):
        top_topics = lda_model.get_document_topics(corpus[i], minimum_probability=0.0)
        vectors.append([top_topics[i][1] for i in range(10)])
    return vectors


def create_tokenized_column(df):
    df["tokenized"] = df["text"].apply(lambda x: simple_preprocess(x))
    return df


def create_bigrams(texts):
    bigram = gensim.models.Phrases(texts, min_count=5, threshold=100)
    bigram_maker = gensim.models.phrases.Phraser(bigram)
    texts = remove_stopwords(texts)
    bigrams = [bigram_maker[doc] for doc in texts]
    bigrams = lemmatization(bigrams)
    return bigrams


def lemmatization(texts, allowed_postags=None):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    if allowed_postags is None:
        allowed_postags = ["NOUN", "ADJ", "VERB", "ADV"]
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append(
            [token.lemma_ for token in doc if token.pos_ in allowed_postags]
        )
    return texts_out


def remove_stopwords(texts):
    stop_words = stopwords.words("english")
    stop_words.extend(["from", "subject", "re", "edu", "use"])
    return [[word for word in doc if word not in stop_words] for doc in texts]


def get_dict_and_corpus(bigrams):
    dictionary = gensim.corpora.Dictionary(bigrams)
    corpus = [dictionary.doc2bow(doc) for doc in bigrams]
    return dictionary, corpus


def lda_model(dictionary, corpus):
    return gensim.models.ldamodel.LdaModel(
        alpha="auto",
        corpus=corpus,
        id2word=dictionary,
        num_topics=10,
        random_state=100,
        update_every=1,
        chunksize=100,
        passes=10,
        per_word_topics=True,
    )


def coherence(lda_model, dictionary, texts):
    coherence_model_lda = gensim.models.CoherenceModel(
        model=lda_model,
        texts=texts,
        dictionary=dictionary,
        coherence="c_v",
    )
    return coherence_model_lda.get_coherence()


def lda(df, load_from=None):
    df = create_tokenized_column(df)
    bigrams = create_bigrams(df["tokenized"])
    dictionary, corpus = get_dict_and_corpus(bigrams)
    if load_from:
        lda = gensim.models.ldamodel.LdaModel.load(load_from)
    else:
        lda = lda_model(dictionary, corpus)
    coherence_lda = coherence(lda, dictionary, bigrams)
    return lda, get_vectors(lda, corpus), coherence_lda
