import numpy as np
import matplotlib.pyplot as plt
import csv

plt.rcParams["font.family"] = "serif"
epoch = []
training_loss = []
validation_loss = []
precision = []
recall = []
f1 = []
accuracy = []
with open('result/graph.txt') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=' ')
    for row in readCSV:
        epoch.append(int(row[0])+1)
        training_loss.append(float(row[1]))
        validation_loss.append(float(row[2]))
        precision.append(float(row[3])*100)
        recall.append(float(row[4])*100)
        f1.append(float(row[5])*100)
        accuracy.append(float(row[6])*100)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col')
# fig.suptitle('A tale of 2 subplots')

axes = plt.gca()
axes.set_xlim([0,100])
axes.set_ylim([0,1])
ax1.set_ylabel('Average loss')
# ax1.set_xlabel('Epoch')
ax1.plot(epoch, training_loss, 'k^-', markevery=10, markersize=5, linewidth=1.2, label="Training loss")
ax1.plot(epoch, validation_loss, 'kx-', markevery=10, markersize=5, linewidth=1.2, label="Validation loss")
ax1.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))

axes.set_ylim([0,100])
ax2.set_ylabel('Percentage')
ax2.set_xlabel('Epoch')
ax2.plot(epoch, precision, 'ks-', markevery=10, markersize=5, linewidth=1.2, label="Precision")
ax2.plot(epoch, recall, 'kx-', markevery=10, markersize=5, linewidth=1.2, label="Recall")
ax2.plot(epoch, f1, 'k^-', markevery=10, markersize=5, linewidth=1.2, label="F1")
ax2.plot(epoch, accuracy, 'ko-', markevery=10, markersize=5, linewidth=1.2, label="Accuracy")
ax2.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()