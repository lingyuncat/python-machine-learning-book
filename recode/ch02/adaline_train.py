#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-09-13 15:43:00
# @Author  : Hong Houren (hourenhong@gmail.com)
# @Link    : http://example.org
# @Version : $Id$

import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import adaline
from training import plot_decision_region

df = pd.read_csv('iris.data')
y = df.iloc[0:100,4].values
y = np.where(y == 'Iris-setosa',-1,1)
X = df.iloc[0:100,[0,2]].values

fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(8,4))
ada1 = adaline.AdalineGD(n_iter=10,eta=0.01).fit(X,y)
ax[0].plot(range(1,len(ada1.cost_) + 1),np.log(ada1.cost_),marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum Squared of Error)')
ax[0].set_title('Adaline-Learning Rate 0.01')

ada2 = adaline.AdalineGD(eta=0.0001, n_iter=10).fit(X,y)
ax[1].plot(range(1,len(ada2.cost_) + 1),ada2.cost_,marker ='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum Squared Error')
ax[1].set_title('Adaline-Learning Rate 0.0001')
plt.tight_layout()
plt.savefig('./adaline_1.png',dpi=300)
# standardize features
