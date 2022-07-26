# Smart IoT Library
import requests
import torchaudio

def version():
  ''' Show Smart IoT library version'''
  print('Smart IoT Library ver. 1.0')


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
