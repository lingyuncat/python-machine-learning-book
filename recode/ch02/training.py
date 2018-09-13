#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-09-12 16:26:59
# @Author  : Hong Houren (hourenhong@gmail.com)
# @Link    : http://example.org
# @Version : $Id$

import numpy as np
import pandas as pd
import perceotron
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
df = pd.read_csv('./iris.data',header=None)
y = df.iloc[0:100,4].values
y = np.where(y == "Iris-setosa",-1,1)
X = df.iloc[0:100,[0,2]].values

ppn= perceotron.Perceptron(eta=0.1,n_iter=10)
ppn.fit(X,y)

plt.plot(range(1,len(ppn.errors_)+1),ppn.errors_,marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.tight_layout()
plt.savefig('./perceptron_1.png',dpi=300)

def plot_decision_region(X,y,classifier,resolution=0.02):
	#setup marker generator and color map
	markers = ('s','x','o','^','v')
	colors = ('red','blue','lightgreen','grey','cyan')
	cmap = ListedColormap(colors[:len(np.unique(y))])

	#plot decision surface
	x1_min,x1_max = X[:,0].min()-1,X[:,0].max()+1
	x2_min,x2_max = X[:,1].min()-1,X[:,1].max()+1
	xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),
						  np.arange(x2_min,x2_max,resolution))
	Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
	Z = Z.reshape(xx1.shape)
	plt.contourf(xx1,xx2,Z,alpha=0.4,cmap=cmap)
	plt.xlim(xx1.min(),xx1.max())
	plt.ylim(xx2.min(),xx2.max())

	#plot class samples
	for idx,cl in enumerate(np.unique(y)):
		plt.scatter(x=X[y == cl,0],y=X[y == cl,1],
					alpha=0.8,c=cmap(idx),
					edgecolor='black',
					marker=markers[idx],
					label=cl)

plot_decision_region(X,y,classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend('upper left')
plt.tight_layout()
plt.savefig('./perceptron_2.png',dpi=300)
plt.show()