
from typing import Tuple
from numpy import ndarray
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch
from sklearn.datasets import load_breast_cancer


class DataHandler:
    def __init__(self, percentage_to_use: int, first_feature: int, second_feature: int, scaler: StandardScaler):
        # se carga la data de la libreria
        self.__cancer_cases: Bunch = load_breast_cancer()

        #  contiene los limites de cada dato
        self.__features: ndarray = self.__cancer_cases.data

        # target contiene la info  1 o 0 (maligno, venigno)
        self.__labels: ndarray = self.__cancer_cases.target

        # lo dividimos por 100 para que sea procentakje
        self.__percentage_to_use: float = percentage_to_use / 100

        # pocisiones de las columnas a tomar en cuenta en el entrenamiento
        self.__first_feature: int = first_feature
        self.__second_feature: int = second_feature

        # escalador que servira para proporcionar los datos y evitar que sigmoide se sature
        self.__scaler: StandardScaler = scaler

    def get_data_for_plot(self) -> Tuple[ndarray, ndarray]:
        return self.__features, self.__labels

    def get_data_for_train(self) -> Tuple[ndarray, ndarray]:
        """
        Esta funcion agarra los datos y los prepara para entrenar la red neuronal.

        Primero toma solo un porcentaje de todos los datos, segun lo que le digamos, de cada registro
        agarra dos columnas especificas.

        Esas columnas las escala, o sea, las normaliza para que no tengan valores tan grandes o desbalanceados 
        que puedan hacer que la funcion sigmoide no funcione bien.

        Returns:
            Tuple[ndarray, ndarray]: las features escaladas y las etiquetas en forma de matriz.
        """
        # seleccionamos el porcentaje de datos a usar
        features_for_train, labels_for_train = self.__get_percentage_of_data()

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
        """
        Agarra solo un porcentaje de los datos originales segun lo que se haya indicado.
        Returns:
            Tuple[ndarray, ndarray]: subset de features y labels limitado al porcentaje especificado.
        """
        # shape devuelve las dimensiones en una tupla, 0 es el numero de filas
        total_elements = self.__features.shape[0]
        # calculamos el numero de elementos a utilizar multiplicandolo por el porcentaje
        cout_in: int = int(total_elements * self.__percentage_to_use)

        return self.__features[:cout_in], self.__labels[:cout_in]
