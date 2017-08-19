import numpy as np
from sklearn.manifold import TSNE

model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)

with open('data/casia_resnet.csv') as fp:
    data = [[float(t) for t in line.strip().split(',')] for line in fp]
data = model.fit_transform(data)
print 'casia:',data.shape
np.savetxt('data/casia_resnet_reduced.csv', data, delimiter=',')

with open('data/renren_resnet.csv') as fp:
    data = [[float(t) for t in line.strip().split(',')] for line in fp]
data = model.fit_transform(data)
print 'renren:',data.shape
np.savetxt('data/renren_resnet_reduced.csv', data, delimiter=',')
