import csv
import matplotlib.pyplot as plt
import sys
from datetime import datetime

times = list()
emotions_anger = list()
emotions_contempt = list()
emotions_disgust = list()
emotions_fear = list()
emotions_happiness = list()
emotions_neutral = list()
emotions_sadness = list()
emotions_surprise = list()

with open("29_01_2024 03h50m42s.csv", newline='') as csvfile:
    filereader = csv.reader(csvfile, delimiter=';')
    for row in filereader:
        times.append(float(row[0]))
        emotions_anger.append(float(row[1]))
        emotions_contempt.append(float(row[2]))
        emotions_disgust.append(float(row[3]))
        emotions_fear.append(float(row[4]))
        emotions_happiness.append(float(row[5]))
        emotions_neutral.append(float(row[6]))
        emotions_sadness.append(float(row[7]))
        emotions_surprise.append(float(row[8]))
    plt.plot(times, emotions_happiness)
    plt.show()
