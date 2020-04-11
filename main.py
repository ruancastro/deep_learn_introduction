import src.classificador as classf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly_express as px

data = pd.read_csv('dataset/banco_de_rosas.csv')
# print(data.head())
# print(data.shape)
# print(data.index)
# print(data.describe())

# print(data[data['variety']==1].describe())  Observando variáveis estatísticas da variedade tipo 1 
# print(data.groupby('variety').size())  Quantificando o número de amostras de cada variação. 

# # Criando um histograma para observar o comportamento das variáveis .
# data.hist()
# his = plt.gcf()
# his.set_size_inches(12, 6)
# plt.savefig('output/histograma.png')
# plt.show()

# # Usando a biblioteca seaborn para plotar a ocorrência de casos de cada variedade dada largura e comprimento da sépala
# sns.set_style('whitegrid')
# sns.FacetGrid(data, hue = 'variety', size = 6)\
# .map(plt.scatter, 'sepal.length', 'sepal.width')\
# .add_legend()
# plt.savefig('output/Sepal_Length_Width_Vs_Species.png')
# plt.show()


# # Usando a biblioteca seaborn para plotar a ocorrência de casos de cada variedade dada largura e comprimento da pétala
# pet_len_wid = data[data.variety == 1].plot(kind = 'scatter', x = 'petal.length', y = 'petal.width'
# ,color = 'blue', label = 'setosa')
# data[data.variety == 2].plot(kind = 'scatter', x = 'petal.length', y = 'petal.width', color = 'red'
# ,label = 'versicolor', ax = pet_len_wid)
# data[data.variety == 3].plot(kind = 'scatter', x = 'petal.length', y = 'petal.width', color = 'green'
# ,label = 'virginica', ax = pet_len_wid)
# pet_len_wid.set_xlabel('Petal Length')
# pet_len_wid.set_ylabel('Petal Width')
# pet_len_wid.set_title('Petal-Length-Width Vs Species')
# pet_len_wid = plt.gcf()
# pet_len_wid.set_size_inches(15, 7)
# plt.savefig('output/Petal_Length_Width_Vs_Species')
# plt.show()


# Usando a biblioteca seaborn para plotar todos os pares de atributos através de um gráfico de dispersão
plt.close()
sns.pairplot(data, hue = 'variety', height = 2, diag_kind = 'kde')
plt.savefig('output/pares_de__todos_os_atributos')
plt.show()

# plot em 3d utilizando a biblioteca plotly_express .
# fig = px.scatter_3d(data, x='sepal.length', y='sepal.width', z='petal.width',color='variety')
# fig.show()

# # Matriz de correlação , observe que a última "coluna", "variety", não deveria existir
# plt.figure(figsize=(7,5)) 
# sns.heatmap(data.corr(),annot=True,cmap='RdYlGn_r') 
# plt.savefig('output/Matriz_de_correlac.png')
# plt.show()

classe = data["variety"]
# print(classe)
# print(classe.tail)

feature = data.drop("variety",axis='columns')
# print(feature.head())

classes_np=classe.to_numpy()
features_np=feature.to_numpy()

# print(features_np)
# print(type(features_np))

# print(classes_np)
# print(type(features_np))
classf.classificadador_MPL(features_np,classes_np)
