#%matplotlib inline 
import sys
import matplotlib.pyplot as plt
from matplotlib import animation as ani
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.special import ndtri

# Data Import
df = pd.read_table('Mansion2.data')
df2 = pd.DataFrame(df.values, columns=['Walk_min','distance','Price','Type','Area','Direction','Year'])
data_size = len(df2)

#plt.figure(figsize=(12,11))
price = df2['Price']
mins = df2['Walk_min']
area = df2['Area']

# plt.subplot(221)
# plt.hist(price,bins=20)
# plt.title("Histgram of House Price")
# plt.xlabel("Price")
# plt.ylabel("Count")

# plt.subplot(222)
# plt.title("Scatter plot (Price - Area)")
# plt.xlim(6000, 20000)
# plt.ylabel("Area")
# plt.xlabel("Price")
# plt.scatter(price, area)

# plt.subplot(223)
# plt.title("Scatter plot (Price - Walk mins)")
# plt.scatter(price, mins)
# plt.xlim(6000, 20000)
# plt.xlabel("Price")
# plt.ylabel("Walk mins")

# plt.subplot(224)
# plt.title("Scatter plot (Area - Walk mins)")
# plt.scatter(area, mins)
# plt.xlabel("Area")
# plt.ylabel("Walk mins")


plt.show()# ヒストグラムと正規分布の比較
mu_p = np.mean(price)
var_p = np.var(price)

xx = np.linspace(min(price), max(price), 300)
x_density = st.norm.pdf(xx, loc=mu_p, scale=np.sqrt(var_p))

# 家賃を値段の順番に並び替え
price_ordered = np.sort(price)

# plt.figure(figsize=(8,6))
# plt.hist(price,bins=20)
# plt.title("Histgram of Price")
# plt.xlabel("Price")
# ax = plt.twinx()
# ax.plot(xx, x_density, "red", linewidth=2, zorder=300)
# plt.show()# 家賃
# plt.figure(figsize=(7,6))
# plt.xlim(0, 1)
# plt.ylim(5900, 19500)
# plt.title("House Price(sorted)", size=13)
# plt.scatter(np.linspace(0, 1, data_size), price_ordered)
# plt.grid(True)

# # 正規累積分布関数
# plt.figure(figsize=(7,6))
# plt.xlim(-3, 3)
# plt.ylim(0,1)
# plt.title("Cumulative Norm Dist", size=13)
# plt.scatter(np.linspace(-3, 3, data_size), st.norm.cdf(np.linspace(-3, 3, data_size)))
# plt.grid(True)

# # 累積のヒストグラムと累積正規分布の比較
# xx = np.linspace(min(price)-1000, max(price), 300)
# x_cdensity = st.norm.cdf(xx, loc=mu_p, scale=np.sqrt(var_p))

# plt.figure(figsize=(8,6))
# plt.xlim(min(price)-1000, max(price))
# plt.ylim(0, 188)
# plt.hist(price,bins=20, cumulative=True, histtype='step')
# plt.title("Histgram of Price (Cumulative)")
# plt.xlabel("Price")
# ax = plt.twinx()
# ax.set_xlim(min(price)-1000, max(price))
# ax.set_ylim(0,1)
# ax.plot(xx, x_cdensity, "red", linewidth=2, zorder=300)
# plt.show()# 家賃

# 標準正規分布の逆関数(xの定義域と粒度は0-1の間をデータサイズの数分割したもの)
inv = ndtri(np.linspace(0, 1, data_size))#float(i)/len(price)) for i in range(len(price))] 

plt.title("Q-Q Plot", size=13)
plt.xlabel("Theoretical Quantailes")
plt.ylabel("Price")
plt.ylim(5900, 20000)
plt.xlim(-3, 3)
plt.scatter(inv, price_ordered)
plt.show()

data = price_ordered
