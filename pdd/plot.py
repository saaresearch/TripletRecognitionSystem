import matplotlib.pyplot as plt
import numpy as np

def plot(embeddings,labels,classes_name,classes_count,name,colors):
  plt.figure(figsize=(10,10))
  
  for i in range(classes_count):
      inds = np.where(labels==i)[0]
      plt.scatter(embeddings[inds,0],embeddings[inds,1],color=colors[i])
  plt.legend(classes_name,loc='center left', bbox_to_anchor=(1, 0.5))
  plt.title(name)
  plt.show()