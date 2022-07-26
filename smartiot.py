# Smart IoT Library
import requests
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio, display

def version():
  ''' Show Smart IoT library version'''
  print('Smart IoT Library ver. 1.3')
  print('torchaudio ver.', torchaudio.__version__)


def load_audio(url, fname):
  ''' 
  Obtiene un archivo de audio a través de un URL y lo guarda 
  Regresa audio-wave, sample-rate, metadata, bytes-size
  '''
  r = requests.get(url)
  with open(fname, 'wb') as f:
    f.write(r.content)
  sz = len(r.content)
  meta = torchaudio.info(fname)
  wave, sr = torchaudio.load(fname)
  return (wave, sr, meta, sz)

def print_info(info, fname = None):
  ''' Muestra los metadatos de un archivo de audio con la tupla que regresa load_audio'''
  if fname: 
    print('-' * 30)
    print('Filename: ', fname)
    print('-' * 30)
  wave, sr, meta, sz = info 
  channels, frames = wave.shape 
  print(f'     Frames: {frames}')
  print(f'   Channels: {channels}')
  print(f'  File size: {sz} bytes')
  print(f'Tensor size: {wave.element_size() * channels * frames} bytes')
  print(f'      Dtype: {wave.dtype}')
  print(f'        Max: {wave.max().item():6.3f}')
  print(f'        Min: {wave.min().item():6.3f}')
  print(f'       Mean: {wave.mean().item():6.3f}')
  print(f'    Std Dev: {wave.std().item():6.3f}')
  print(wave)

def plot_wave(wave, torch=True):
  ''' Graficando señal de audio de PyTorch o NumPy'''
  plt.figure()
  plt.plot(wave[0].numpy() if torch else wave)

def play_audio(waveform, sample_rate, torch=True):
  ''' Reproduciendo señal de audio de PyTorch o NumPy'''
  if torch:
    wave = wave.numpy()
    num_channels, _ = waveform.shape
    if num_channels == 1:
      display(Audio(waveform[0], rate=sample_rate))
    elif num_channels == 2:
      display(Audio((waveform[0], waveform[1]), rate=sample_rate))
    else:
      raise ValueError("Waveform with more than 2 channels are not supported.")
  else: # numpy array
    display(Audio(waveform, rate=sample_rate))
