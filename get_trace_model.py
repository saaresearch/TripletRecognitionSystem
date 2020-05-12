import os 

os.system('python train.py')
os.system('rm -r data')
os.system('python classifier_train.py')

