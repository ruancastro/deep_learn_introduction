#import src.matriz_correlac as tst
#st.teste()
#import tensorflow as ts
# print("Teste")
import src.classificador as classf
import pandas as pd
import numpy as np
data = pd.read_csv('dataset/banco.csv')
data= data.drop("CHAVE",axis=1)

print(data.head())

print("\n\n\n%%%%%%%%%%%%%%%%%%%%%%%%\n\n\n")

#x= data.columns
#print(x)
#print(type(x))

pdtonp=data.to_numpy()
x=pdtonp[0:,0:7]
#print(x)
#print("\n\n\n%%%%%%%%%%%%%%%%%%%%%%%%\n\n\n")

y=pdtonp[:,7]
#print(y)
classf.classificadador_MPL(x,y,data) 