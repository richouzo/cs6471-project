from lda import lda
import pandas as pd

def gridsearch_lda():

    implicit_df_train = pd.read_csv("./../../data/implicithate_train.csv")

    max_implicit_coherence = 0
    best_k_implicit = 0
    for i in range(3,25,2):
        lda_implicit, vectors_implicit_train, coherence_implicit = lda(implicit_df_train, k=i)
        if coherence_implicit > max_implicit_coherence:
            max_implicit_coherence = coherence_implicit
            best_k_implicit = i
    print("Best k for implicit:", best_k_implicit)
    print("Best coherence for implicit:", max_implicit_coherence)

    covid_df_train = pd.read_csv("./../../data/covidhate_train.csv")

    max_covid_coherence = 0
    best_k_covid = 0
    for i in range(3,25,2):
        lda_covid, vectors_covid_train, coherence_covid = lda(covid_df_train, k=i)
        if coherence_covid > max_covid_coherence:
            max_covid_coherence = coherence_covid
            best_k_covid = i

    print("Best k for covid:", best_k_covid)
    print("Best coherence for covid:", max_covid_coherence)

    offenseval_df_train = pd.read_csv("./../../data/offenseval_train.csv")

    max_offenseval_coherence = 0
    best_k_offenseval = 0
    for i in range(3,25,2):
        lda_offenseval, vectors_offenseval_train, coherence_offenseval = lda(offenseval_df_train, k=i)
        if coherence_offenseval > max_offenseval_coherence:
            max_offenseval_coherence = coherence_offenseval
            best_k_offenseval = i

    print("Best k for offenseval:", best_k_offenseval)
    print("Best coherence for offenseval:", max_offenseval_coherence)

    return best_k_implicit, best_k_covid, best_k_offenseval
