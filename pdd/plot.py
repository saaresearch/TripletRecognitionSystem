import matplotlib.pyplot as plt
import numpy as np

def plot(embeddings,labels,class_name,name,colors):
  plt.figure(figsize=(10,10))
  for i in range(len(class_name)):
      inds = np.where(labels==i)[0]
      plt.scatter(embeddings[inds,0],embeddings[inds,1],color=colors[i])
  plt.legend(class_name,loc='center left', bbox_to_anchor=(1, 0.5))
  plt.title(name)
  plt.show()
  plt.figure.canvas.draw()
  plt.figure.savefig( f'./PDD.png' )