from sklearn.neural_network import MLPClassifier
# import tensorflow as tf
#from tensorflow import feature_column
#from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


def classificadador_MPL(x,y,dataframe) :
    # train, test = train_test_split(dataframe, test_size=0.3)
    # #train, val = train_test_split(train, test_size=0.2)
    # print(len(train), 'train examples')
    # #print(len(val), 'validation examples')
    # print(len(test), 'test examples')
    # # print("TESTE")

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42,shuffle=0)

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
    print(clf.fit(x,y))
    print("Score treino:",clf.score(X_train,y_train))
    print("Score teste:",clf.score(X_test,y_test))
    print(len(X_train), 'train examples')
    #print(len(train), 'train examples')   
    
#predic ou score ->
# tenho que separar o banco de dados que eu tenho em teste e treino, antes de usar a predict ou o score dá uma sacada aqui nesse site https://www.tensorflow.org/tutorials/structured_data/feature_columns
# procure saber o que é o predict e o score, e como fatiar facilmente um banco de dados para treino.

