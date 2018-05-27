
def jaccard(u1, u2):
    in_common = len(u1 & u2)
    union = len(u1 | u2)
    return in_common/float(union)


if __name__ == '__main__':
    '''
    NOTAZIONE:
        - liste: lista = ["a","a","b"]
        - insieme: set = {"a","b"} senza ripetizioni
        - no_duplic = set(lista) trasforma da lista a set
    '''
    countries = ["ita","ita","eng","eng","usa","fra","fra","lux"]
    dist_countries = set(countries)
    print(dist_countries)

    # crea un dizionario, chiave:valore
    dict = {"a":"nicola","b":"vito","c":"paola"}
    print(dict["a"])

    u1 = {"a","b","c","d"}
    u2 = {"a","f","d"}
    print(jaccard(u1,u2))

