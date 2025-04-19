
from typing import Tuple
import numpy
from numpy import ndarray

from sklearn.utils import Bunch

from neuronal_network.services.data_handler import DataHandler


class NeuronalNetwork:

    def __init__(self, max_epoachs: int, learning_rate: float, first_feature: int, second_feature: int, percentage_to_use: int):
        self.__max_epoachs = max_epoachs
        self.__learning_rate = learning_rate
        self.__data_handler = DataHandler(
            percentage_to_use, first_feature, second_feature)

    def __sigmoid(self, entry: int | ndarray) -> float | ndarray:
        """
        Aplica la funcion sigmoide a un valor escalar o a cada elemento de un arreglo.

        Args:
            entry (int | ndarray): valor o arreglo de valores sobre los que se aplica la funcion sigmoide.

        Returns:
            float | ndarray: resultado de aplicar la funcion sigmoide.
        """
        return 1 / (1 + numpy.exp(-entry))

    def __sigmoid_deriv(self, input_value: int | ndarray) -> float | ndarray:
        """
        Calcula la derivada de la funcion sigmoide evaluada en un valor o arreglo.

        Args:
            input_value (int | ndarray): valor escalar o arreglo donde se evaluara la derivada.

        Returns:
            float | ndarray: resultado de la derivada de la funcion sigmoide.
        """
        sigmoid_value = self.__sigmoid(input_value)
        return sigmoid_value * (1 - sigmoid_value)

    def __mean_squared_error(self, predictions: ndarray, targets: ndarray) -> float:
        """
        Calcula el error cuadratico medio entre las predicciones del modelo y los valores reales.

        Args:
            predictions (numpy.ndarray): arreglo de valores predichos por el modelo
            targets (numpy.ndarray): arreglo de valores deseados
        Returns:
            float: valor del error cuadratico medio 
        """
        return numpy.mean((predictions - targets) ** 2)  # se eleva para convertir valores negativos a positivos

    def train(self):
        elements_for_train: Tuple[ndarray,
                                  ndarray] = self.__data_handler.get_data()
        training_features: ndarray = elements_for_train[0]
        training_labels: ndarray = elements_for_train[1]

        # inicializamos los pesos con dis normal, crea una matriz (2,1)
        weights: ndarray = numpy.random.randn(2)

        for i in range(self.__max_epoachs):
            # propagacion hacia adelante
            linear_combination: ndarray = numpy.dot(training_features, weights)
            predicted_outputs = self.__sigmoid(linear_combination)
            # calculamos los errores
            current_loss: float = self.__mean_squared_error(
                predicted_outputs, training_labels)
            prediction_error: ndarray = predicted_outputs - training_labels

            # propagacion hacia atras
            sigmoid_gradient: ndarray = self.__sigmoid_deriv(
                linear_combination)
            gradient_output_layer: ndarray = prediction_error * sigmoid_gradient

        pass
