from sklearn.neural_network import MLPClassifier
# import tensorflow as tf
#from tensorflow import feature_column
#from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


def classificadador_MPL(x,y) :

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=None,shuffle=0)
    clf = MLPClassifier(hidden_layer_sizes=(100, 80), activation='relu', solver='adam', 
        alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, 
        power_t=0.5, max_iter=10000, shuffle=True, random_state=None, tol=0.0001, verbose=False, 
        warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, 
        validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)
# cross validation
    print(clf.fit(x,y))
    print("Score treino:",clf.score(X_train,y_train))
    print("Score teste:",clf.score(X_test,y_test))
#predic ou score

