#import src.matriz_correlac as tst
#st.teste()
#import tensorflow as ts
# print("Teste")
import src.classificador as classf
import pandas as pd
import numpy as np
data = pd.read_csv('dataset/banco_de_rosas.csv')
# data= data.drop("CHAVE",axis=1)
# print(data.head())
classe = data["variety"]
# print(classe)
feature = data.drop("variety",axis=1)
print(feature)
# print("\n\n\n%%%%%%%%%%%%%%%%%%%%%%%%\n\n\n")
classes_np=classe.to_numpy()
features_np=feature.to_numpy()

# print(features_np)
# # print(type(features_np))
# print("\n\n\n%%%%%%%%%%%%%%%%%%%%%%%%\n\n\n")
# print(classes_np)
# print(type(features_np))
classf.classificadador_MPL(features_np,classes_np)
