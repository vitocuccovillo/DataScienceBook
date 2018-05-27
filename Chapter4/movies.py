import numpy as np
import time

if __name__ == '__main__':
    user_pref = np.array([5,1,3]) #vettore preferenze utente: commedie, romantici, azione (gradimento da 1 a 5)
    movies = np.random.randint(5,size=(3,10000))+1 #randint va da 0 a 4 e metto +1 per ottenere 5
    np.dot(user_pref,movies)

    print(movies.shape)

    for num_movies in (10000, 100000, 1000000, 10000000, 100000000):
        cur_movies = np.random.randint(5,size=(3,num_movies))+1
        now = time.time()
        np.dot(user_pref,cur_movies)
        print(time.time()-now,"secondi per ", num_movies, " film")