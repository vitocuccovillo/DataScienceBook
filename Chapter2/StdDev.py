import numpy as np

if __name__ == '__main__':
    data = [31, 32, 31, 28, 29, 31, 38, 32, 31, 30, 29, 32, 35, 32]
    mean = np.mean(data)
    squared_diff = [] # differenze dalla media di ogni valore
    for t in data:
        diff = t - mean
        diff_sq = diff**2
        squared_diff.append(diff_sq)

    avg_squared_diff = np.mean(squared_diff) # scostamento quadratico medio (dalla media) VARIANZA!
    std_dev = np.sqrt(squared_diff) # scostamento medio dalla media DEVIAZIONE STANDARD!

    print("VARIANZA:" + str(avg_squared_diff))
    print("DEV_STD:" + str(avg_squared_diff))

