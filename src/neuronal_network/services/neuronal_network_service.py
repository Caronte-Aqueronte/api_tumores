
import numpy


class NeuronalNetwork:

    def __init__(self):
        pass

    def __sigmoid(self, entry: int | numpy.ndarray) -> float | numpy.ndarray:
        """
        Aplica la funcion sigmoide a un valor escalar o a cada elemento de un arreglo.

        Args:
            entry (int | numpy.ndarray): valor o arreglo de valores sobre los que se aplica la funcion sigmoide.

        Returns:
            float | numpy.ndarray: resultado de aplicar la funcion sigmoide.
        """
        return 1 / (1 + numpy.exp(-entry))

    def __sigmoid_deriv(self, input_value: int | numpy.ndarray) -> float | numpy.ndarray:
        """
        Calcula la derivada de la funcion sigmoide evaluada en un valor o arreglo.

        Args:
            input_value (int | numpy.ndarray): valor escalar o arreglo donde se evaluara la derivada.

        Returns:
            float | numpy.ndarray: resultado de la derivada de la funcion sigmoide.
        """
        sigmoid_value = self.__sigmoid(input_value)
        return sigmoid_value * (1 - sigmoid_value)

    def __mean_squared_error(self, predictions: numpy.ndarray, targets: numpy.ndarray) -> float:
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
        pass
