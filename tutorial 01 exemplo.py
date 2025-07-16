import os, sys
project_directory = os.getcwd()
sys.path.append(project_directory)

from bciflow.modules.core.kfold import kfold
from bciflow.modules.tf.bandpass.chebyshevII import chebyshevII
from bciflow.modules.fe import logpower
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda

# tutorial 01 exemplo

from bciflow.datasets.cbcic import cbcic

dataset = cbcic(subject=1, path='data/cbcic/')

print("EEG signals shape:", dataset["X"].shape)
print("Labels:", dataset["y"])
print("Class dictionary:", dataset["y_dict"])
print("Events:", dataset["events"])
print("Channel names:", dataset["ch_names"])
print("Sampling frequency (Hz):", dataset["sfreq"])
print("Start time (s):", dataset["tmin"])