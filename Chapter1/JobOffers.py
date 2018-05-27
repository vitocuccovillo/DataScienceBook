from bs4 import BeautifulSoup
import requests
from sklearn.feature_extraction.text import CountVectorizer

if __name__ == '__main__':
    texts = []

    for index in range(0,100,10):
        page = 'http://indeed.com/jobs?q=data+scientist&start='+str(index)
        web_result = requests.get(page).text
        soup = BeautifulSoup(web_result)

        for listing in soup.findAll('span', {'class': 'summary'}):
            texts.append(listing.text)

    vect = CountVectorizer(ngram_range=(1,2),stop_words='english')
    matrix = vect.fit_transform(texts) # restituisce la matrice termini-doc
    print(len(vect.get_feature_names()))
    freqs = [(word, matrix.getcol(idx).sum()) for word, idx in vect.vocabulary_.items()]
    # sort from largest to smallest
    for phrase, times in sorted(freqs, key=lambda x: -x[1])[:25]:
        print(phrase, times)