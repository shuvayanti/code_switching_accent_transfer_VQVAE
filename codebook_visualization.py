import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import LatentDirichletAllocation as LDA
import random
import os

spk_data ={}
spk={}
clusters ={}

PHN_CODE_PATH = "/home/s1995633/s1995633/dissertation/codes/vq_phn_codes_updated/sys5_lang/siwis_552024/train/"
PHN_VEC_PATH = "/home/s1995633/s1995633/dissertation/codes/vq_phn_vecs_updated/sys5_lang/siwis_552024/train/"
SPK_CODE_PATH = "/home/s1995633/s1995633/dissertation/codes/vq_spk_codes_updated/sys5_lang/siwis_552024/all_siwis/"
SPK_VEC_PATH = "/home/s1995633/s1995633/dissertation/codes/vq_spk_vecs_updated/sys5_lang/siwis_552024/all_siwis/"
SPK_MAP = "/home/s1995633/s1995633/dissertation/siwis_database/speaker_index_updated/index.pkl"
ALIGN_PATH = "/home/s1995633/s1995633/dissertation/siwis_database/alignment_updated/"
DATA_PATH = "/home/s1995633/s1995633/dissertation/siwis_database/normalised_output_updated/"

filename = random.choice(os.listdir(PHN_CODE_PATH))
vec = np.load(open(PHN_VEC_PATH + filename, 'rb'))

'''
for line in spk_map:
    spk_id = line.split(',')[0]
    #print(spk_id)
    mapping = line.split(',')[-1]
    #print(mapping)
    spk[spk_id] = mapping
'''
#print(spk)
vectors = list()
print('vec=', vec)
print('length of vec=',len(vec))
for v in vec:
    for item in v:
        #print(item)
        vectors.append(item)

vectors = np.array(vectors) 
#print(vectors)
#vectors_std = StandardScaler().fit_transform(vec)


#PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(vectors)
#print('components=',principal_components)
features = range(pca.n_components_)
print('features = ',features)

#print(principal_components)
plt.scatter(principal_components[0], principal_components[1])
plt.show()
#LDA
#lda = LDA(n_components=11, n_jobs=-1)
#print(lda.fit(vec))

'''
Kmean = KMeans(n_clusters=11)
Kmean.fit(principal_components)
#print('cluster centres: ',Kmean.cluster_centers_)
#print('labels:', Kmean.labels_)

spk_id =list(spk_data.keys())

for i, label in enumerate(Kmean.labels_):
    #print(spk_id[i], label)
    try:
        clusters[label].append(spk_id[i])
    except:
        clusters[label] = [spk_id[i]]

print(clusters)
'''
