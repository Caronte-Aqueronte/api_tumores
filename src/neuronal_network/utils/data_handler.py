
from typing import Tuple
from numpy import ndarray
from sklearn.discriminant_analysis import StandardScaler
from sklearn.utils import Bunch
from sklearn.datasets import load_breast_cancer


class DataHandler:
    def __init__(self, percentage_to_use: int, first_feature: int, second_feature: int, scaler: StandardScaler):
        cancer_cases: Bunch = load_breast_cancer()  # se carga la data de la libreria

        #  contiene los limites de cada dato
        self.__fetures: ndarray = cancer_cases.data

        # target contiene la info  1 o 0 (maligno, venigno)
        self.__labels: ndarray = cancer_cases.target

        # lo dividimos por 100 para que sea procentakje
        self.__percentage_to_use: float = percentage_to_use / 100

        # pocisiones de las columnas a tomar en cuenta en el entrenamiento
        self.__first_feature: int = first_feature
        self.__second_feature: int = second_feature

        # escalador que servira para proporcionar los datos y evitar que sigmoide se sature
        self.__scaler: StandardScaler = scaler

    def get_data_for_train(self) -> Tuple[ndarray, ndarray]:

        # seleccionamos el porcentaje de datos a usar
        elements_for_train: Tuple[ndarray,
                                  ndarray] = self.__get_percentage_of_data()
        features_for_train: ndarray = elements_for_train[0]
        labels_for_train: ndarray = elements_for_train[1]

        # seleccionamos en los datos solo las columnas que se desean, : indica que seleccionamos todas las filas
        features_for_train = features_for_train[:, [
            self.__first_feature, self.__second_feature]]

        # para evitar que la sigmoide se sature, escalamos las feteatures con la formula med-desv/desv
        features_for_train = self.__scaler.fit_transform(features_for_train)

        # en vals estan todos los valorees correspondientes a los registros es decir si ese registro es
        # reshape convierte el vector a matriz, -1 le indica que calcule cuantas filas deben haber para completar una col
        labels_for_train = labels_for_train.reshape(-1, 1)

        return features_for_train, labels_for_train

    def __get_percentage_of_data(self) -> Tuple[ndarray, ndarray]:
        # shape devuelve las dimensiones en una tupla, 0 es el numero de filas
        total_elements = self.__fetures.shape[0]
        # calculamos el numero de elementos a utilizar multiplicandolo por el porcentaje
        cout_in: int = int(total_elements * self.__percentage_to_use)

        return self.__fetures[:cout_in], self.__labels[:cout_in]
