import random
from matplotlib import pyplot as plt

def random_variable_of_dice_roll():
    return random.randint(1,7)


trials = []
num_trials = 100
for trial in range(num_trials):
    trials.append(random_variable_of_dice_roll())
print(sum(trials) / float(num_trials))

num_trials = range(100,10000,10)
avgs = []
for num_trial in num_trials:
    trials = []
    for trial in range(1,num_trial):
        trials.append(random_variable_of_dice_roll())
    avgs.append(sum(trials) / float(num_trial))

plt.plot(num_trials,avgs)
plt.xlabel("Number of trials")
plt.ylabel("Average")
plt.show()
