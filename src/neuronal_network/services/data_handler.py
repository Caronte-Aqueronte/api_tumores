
from typing import Tuple
from numpy import ndarray
import numpy
from sklearn.utils import Bunch
from sklearn.datasets import load_breast_cancer


class DataHandler:
    def __init__(self, percentage_to_use: int, first_feature: int, second_feature: int):
        data: Bunch = load_breast_cancer()
        # data contiene la info  1 o 0 (maligno, venigno)
        self.__info: ndarray = data.data
        self.__vals: ndarray = data.target  # target contiene los valores de la info
        self.__percentage_to_use = int(percentage_to_use / 100)
        self.__first_feature = first_feature
        self.__second_feature = second_feature
        pass

    def get_data(self) -> Tuple[ndarray, ndarray]:

        # seleccionamos el porcentaje de datos a usar
        elements_for_train: Tuple[ndarray,
                                  ndarray] = self.__get_percentage_of_data()
        rows_for_train: ndarray = elements_for_train[0]
        vals_for_train: ndarray = elements_for_train[1]

        # seleccionamos en los datos solo las columnas que se desean, : indica que seleccionamos todas las filas
        training_features: ndarray = rows_for_train[:, [
            self.__first_feature, self.__second_feature]]

        # en vals estan todos los valorees correspondientes a los registros es decir si ese registro es
        # reshape convierte el vector a matriz, -1 le indica que calcule cuantas filas deben haber para completar una col
        training_labels: ndarray = vals_for_train.reshape(-1, 1)

        return training_features, training_labels

    def __get_percentage_of_data(self) -> Tuple[ndarray, ndarray]:
        # shape devuelve las dimensiones en una tupla, 0 es el numero de filas
        total_elements = self.__info.shape[0]
        # calculamos el numero de elementos a utilizar multiplicandolo por el porcentaje
        cout_in = int(total_elements * self.__percentage_to_use)

        return self.__info[:cout_in], self.__vals[:cout_in]
