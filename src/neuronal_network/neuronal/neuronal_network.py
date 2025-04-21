
from typing import Dict, List, Tuple
from fastapi import HTTPException, status
import numpy
from numpy import ndarray


from neuronal_network.models.entry import Entry
from neuronal_network.utils.data_handler import DataHandler
from neuronal_network.utils.entry_list_converter import EntryConverter


class NeuronalNetwork:

    def __init__(self, max_epoachs: int, learning_rate: float, first_feature: int, second_feature: int, percentage_to_use: int):
        self.__max_epoachs: int = max_epoachs
        self.__learning_rate: float = learning_rate / 100
        self.__data_handler = DataHandler(
            percentage_to_use, first_feature, second_feature)
        self.__weights: ndarray = None
        self.__training_labels: ndarray = None

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

    def train(self) -> Dict[int, float]:

        error_per_epoach: Dict[int, float] = {}
        elements_for_train: Tuple[ndarray,
                                  ndarray] = self.__data_handler.get_data_for_train()
        training_features: ndarray = elements_for_train[0]
        self.__training_labels: ndarray = elements_for_train[1]

        print("training_features shape:", training_features.shape)

        print("training_labels shape:", self.__training_labels.shape)

        # inicializamos los pesos con dis normal, crea una matriz (2,1)
        weights: ndarray = numpy.random.randn(2, 1)
        bias: float = 0.0

        for epoach in range(self.__max_epoachs):
            # propagacion hacia adelante
            # hacemos la prediccion combinando los datos con los pesos actuales
            linear_combination: ndarray = numpy.dot(
                training_features, weights) + bias

            # aplicamos la funcion sigmoide para obtener un resultado entre 0 y 1
            predicted_outputs = self.__sigmoid(linear_combination)

            # medimos el error total de esta vuelta del ciclo para ver como va mejorando
            current_loss: float = self.__mean_squared_error(
                predicted_outputs, self.__training_labels)

            # alimentamos el set que representa el numero de la epoca y su error (para grafico en front)
            error_per_epoach[epoach + 1] = current_loss

            # calculamos que tan lejos estuvo la prediccion del valor real
            prediction_error: ndarray = predicted_outputs - self.__training_labels

            # propagacion hacia atras
            # ahora vamos para atras, viendo como afecta el error a los pesos
            sigmoid_gradient: ndarray = self.__sigmoid_deriv(
                linear_combination)

            # multiplicamos el error por lo que cambia la sigmoide para cada entrada
            gradient_output_layer: ndarray = prediction_error * sigmoid_gradient

            # calculamos cuanto deberiamos ajustar los pesos segun el error
            deriv: ndarray = numpy.dot(
                training_features.T, gradient_output_layer)
            bias_deriv: float = numpy.sum(gradient_output_layer)

            # ajustmaos los pesos con lo que nos dijo la derivada, tambien las bias
            weights = weights - (deriv * self.__learning_rate)
            bias = bias - (bias_deriv * self.__learning_rate)

        self.__weights = weights
        return error_per_epoach

    def predict(self, inputs: List[Entry]):

        if self.__weights is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="La red no está entrenada aún.")

        # mandamos a convertir las entries en una lista de tuplas
        entryListConverter = EntryConverter(inputs)
        inputs_tuple_list = entryListConverter.convert_list_of_entry_to_list_of_tuple()

        # convertimos las tuplas en un ndarray
        data: ndarray = numpy.array(inputs_tuple_list, dtype=float)

        # combinamos las entradas con los pesos entrenados con un producto punto
        linear_combination = numpy.dot(data, self.__weights)

        # aplicamos la función sigmoide para convertir esos valores en probs 0 y 1
        predictions = self.__sigmoid(linear_combination)

        # self.__convert_predictions_to_binary(predictions) en binario los reultados
        binary_predictions: ndarray = self.__convert_predictions_to_binary(
            predictions)

        # mandamos a calcular la exactitud
        acurrancy_percentage: float = self.__calculate_accurancy_percentage(
            binary_predictions)

        return inputs, binary_predictions, acurrancy_percentage

    def __convert_predictions_to_binary(self, final_predictions: ndarray) -> ndarray:
        # si es mayor o igual a 0.5, es 1 (maligno), si no 0 (benigno)
        binary_predictions: ndarray = (final_predictions >= 0.5).astype(int)

        return binary_predictions

    def __calculate_accurancy_percentage(self, binary_predictions: ndarray) -> float:

        # comparamos cuántas predicciones son acertadas
        # si el valor predicho es igual al valor real, se cuenta como acierto
        # luego sacamos el promedio, que sería el porcentaje de aciertos
        final_accuracy: float = numpy.mean(
            binary_predictions == self.__training_labels)

        return round(final_accuracy * 100, 2)  # lo devolvemos en porcentaje
