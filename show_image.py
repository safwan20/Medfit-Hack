import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display as ipd


def plot_spectogram(file):

  x, sr = librosa.load(file, sr=44100)

#   print(type(x), type(sr))
#   print(x.shape, sr)
#   plt.figure(figsize=(14, 5))
#   librosa.display.waveplot(x, sr=sr)

  #below snippet is without log transform spectogram
  
  X = librosa.stft(x)
  Xdb = librosa.amplitude_to_db(abs(X))

  librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
  plt.colorbar()
  plt.savefig('Spectograms/spectogram_image.png', bbox_inches='tight')