#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from io import StringIO



# Parse data for one layer network
oneLayerTrainAcc = np.zeros((0, 1))
oneLayerValAcc = 0
oneLayerTestAcc = 0
with open('oneLayerData.out', 'r') as file:
    for line in file:
        if re.search(r'Averaged Accuracy', line):
            out = float(re.sub(r'^Averaged Accuracy: (\d+\.\d+)$', r'\1', line))
            print(f"out: {out}")
            oneLayerTrainAcc = np.append(oneLayerTrainAcc, out)
        if re.search(r'Averaged Val Accuracy', line):
            out = re.sub(r'^Averaged Val Accuracy: (\d+\.\d+)$', r'\1', line)
            oneLayerValAcc = float(out)
        if re.search(r'Averaged Test Accuracy', line):
            out = re.sub(r'^Averaged Test Accuracy: (\d+\.\d+)$', r'\1', line)
            oneLayerTestAcc = float(out)

print(f"oneLayerTrainAcc: {oneLayerTrainAcc}")
print(f"oneLayerValAcc: {oneLayerValAcc}")
print(f"oneLayerTestAcc: {oneLayerTestAcc}")

# Create the plot
plt.figure()
plt.plot(oneLayerTrainAcc, label='One Layer Train Accuracy')

# Add fixed horizontal lines at y=2 and y=4
plt.axhline(y=oneLayerValAcc, color='r', linestyle='--', label='One Layer Val Accuracy')
plt.axhline(y=oneLayerTestAcc, color='g', linestyle='-', label='One Layer Test Accuracy')

# Add labels and legend
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Single Layer Network Results')
#plt.yticks(np.linspace(np.min(oneLayerTrainAcc), np.max(oneLayerTrainAcc), num=5))
plt.legend()

# Show the plot
plt.tight_layout()
plt.savefig('OneLayerResults.png')
plt.show()


# Parse data for two layer network
twoLayerTrainAcc = np.zeros((0, 1))
twoLayerValAcc = 0
twoLayerTestAcc = 0
with open('twoLayerData.out', 'r') as file:
    for line in file:
        if re.search(r'Averaged Accuracy', line):
            out = float(re.sub(r'^Averaged Accuracy: (\d+\.\d+)$', r'\1', line))
            print(f"out: {out}")
            twoLayerTrainAcc = np.append(twoLayerTrainAcc, out)
        if re.search(r'Averaged Val Accuracy', line):
            out = re.sub(r'^Averaged Val Accuracy: (\d+\.\d+)$', r'\1', line)
            twoLayerValAcc = float(out)
        if re.search(r'Averaged Test Accuracy', line):
            out = re.sub(r'^Averaged Test Accuracy: (\d+\.\d+)$', r'\1', line)
            twoLayerTestAcc = float(out)

print(f"twoLayerTrainAcc: {twoLayerTrainAcc}")
print(f"twoLayerValAcc: {twoLayerValAcc}")
print(f"twoLayerTestAcc: {twoLayerTestAcc}")

# Create the plot
plt.figure()
plt.plot(twoLayerTrainAcc, label='Two Layer Train Accuracy')

# Add fixed horizontal lines at y=2 and y=4
plt.axhline(y=twoLayerValAcc, color='r', linestyle='--', label='Two Layer Val Accuracy')
plt.axhline(y=twoLayerTestAcc, color='g', linestyle='-', label='Two Layer Test Accuracy')

# Add labels and legend
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Dual Layer Network Results')
#plt.yticks(np.linspace(np.min(twoLayerTrainAcc), np.max(twoLayerTrainAcc), num=5))
plt.legend()

# Show the plot
plt.tight_layout()
plt.savefig('DualLayerResults.png')
plt.show()





