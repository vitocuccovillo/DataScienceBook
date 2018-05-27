import numpy as np

if __name__ == '__main__':
    numbers = [3,5,3,2,2,1,5,7,5,4,3,5,7,4,3,4,6,6,23,12,4,3,9,34,6,4,3,6,7,8,7,67,4,3]
    print(sorted(numbers))
    mean = np.mean(numbers)
    print("Media: " + str(mean))
    median = np.median(numbers)
    print("Mediana: " + str(median))