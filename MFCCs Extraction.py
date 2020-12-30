#!/usr/bin/env python
# coding: utf-8

# In[3]:


import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np


# In[4]:


#Load an audio file
audio_file="1.wav"


# In[5]:


ipd.Audio(audio_file)


# In[6]:


signal,sr=librosa.load(audio_file)


# In[7]:


signal.shape


# In[8]:


#Extract MFCCs
mfccs=librosa.feature.mfcc(signal, n_mfcc=13, sr=sr)


# In[9]:


mfccs.shape


# In[10]:


#Visualize MFCCs
plt.figure(figsize=(25,10))
librosa.display.specshow(mfccs,
                        x_axis="time",
                        sr=sr)
plt.colorbar(format="%+2f")
plt.show()


# In[11]:


#Calculate delta and delta2 MFCCs
delta_mfccs=librosa.feature.delta(mfccs)
delta2_mfccs=librosa.feature.delta(mfccs,order=2)


# In[12]:


delta_mfccs.shape


# In[13]:


delta2_mfccs.shape


# In[14]:


plt.figure(figsize=(25,10))
librosa.display.specshow(delta_mfccs,
                        x_axis="time",
                        sr=sr)
plt.colorbar(format="%+2f")
plt.show()


# In[15]:


plt.figure(figsize=(25,10))
librosa.display.specshow(delta2_mfccs,
                        x_axis="time",
                        sr=sr)
plt.colorbar(format="%+2f")
plt.show()


# In[16]:


comprehensive_mfccs=np.concatenate((mfccs, delta_mfccs, delta2_mfccs))


# In[17]:


comprehensive_mfccs.shape


# In[ ]:




