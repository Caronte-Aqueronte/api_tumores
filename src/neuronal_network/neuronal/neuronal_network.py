
from typing import Dict, Tuple
import numpy
from numpy import ndarray
from sklearn.preprocessing import StandardScaler

from src.neuronal_network.dto.desicion_boundary_points import DesicionBoundaryPoints
from src.neuronal_network.dto.predict_request_dto import PredictRequestDTO
from src.neuronal_network.utils.data_handler import DataHandler


class NeuronalNetwork:

    def __init__(self, max_epoachs: int, learning_rate: float, first_feature: int, second_feature: int, percentage_to_use: int):

        # indica hasta donde iterara el entrenamiento, lo escoje el usuario en el front
        self.__max_epochs: int = max_epoachs

        # se divide por cien para transformar el valor en un porcentaje
        self.__learning_rate: float = learning_rate / 100

        # se gaurdan porque son el resultado del entrenamieto y serviran para la prediccion
        self.__weights: ndarray = None

        # se guardaran las bias porque son parte del entrenamiento asi que en las predicciones deben usarse
        self.__bias: float = None

        # son necesearias para poder transformar las predicciones en un formato binario (benigno, maligno)
        self.__training_labels: ndarray = None
        self.__training_features: ndarray = None

        # escalador que servira para proporcionar los datos y evitar que sigmoide se sature
        self.__scaler: StandardScaler = StandardScaler()

        # clase que nos va a preparar la data de entrenamiento
        self.__data_handler = DataHandler(
            percentage_to_use, first_feature, second_feature, self.__scaler)

    def train(self) -> Tuple[Dict[int, float], float, DesicionBoundaryPoints]:

        # aqui vamos a guardar el error de cada epoca, donde la clave es el numero
        # de epoca y el valor es el error de la epoca
        error_per_epoch: Dict[int, float] = {}

        # mandamos a trar la data preparada con el objeto que se encarga de preprarar la data
        self.__training_features,  self.__training_labels = self.__data_handler.get_data_for_train()

        # inicializamos los pesos con dis normal, crea una matriz (2,1)
        weights: ndarray = numpy.random.randn(2, 1)
        bias: float = 0.0

        for epoch in range(self.__max_epochs):

            # -----------------------------------------------------------------------------------propagacion hacia adelante

            # hacemos la prediccion combinando los datos con los pesos actuales
            linear_combination: ndarray = numpy.dot(
                self.__training_features, weights) + bias

            # aplicamos la funcion sigmoide para obtener un resultado entre 0 y 1
            predicted_outputs = self.__sigmoid(linear_combination)

            # ---------------------------------------------------------------------------------------calculo de errores

            # medimos el error total de esta vuelta del ciclo para ver como va mejorando
            epoch_loss: float = self.__mean_squared_error(
                predicted_outputs)

            # alimentamos el set que representa el numero de la epoca y su error (para grafico en front)
            error_per_epoch[epoch + 1] = epoch_loss

            # calculamos que tan lejos estuvo la prediccion del valor real
            individual_errors: ndarray = predicted_outputs - self.__training_labels

            # --------------------------------------------------------------------------------------- propagacion hacia atras

            # ahora vamos para atras, viendo como afecta el error a los pesos
            sigmoid_gradient: ndarray = self.__sigmoid_deriv(
                linear_combination)

            # multiplicamos el error por lo que cambia la sigmoide para cada entrada
            gradient_output_layer: ndarray = individual_errors * sigmoid_gradient

            # calculamos cuanto deberiamos ajustar los pesos segun el error
            deriv: ndarray = numpy.dot(
                self.__training_features.T, gradient_output_layer)
            bias_deriv: float = numpy.sum(gradient_output_layer)

            # ajustmaos los pesos con lo que nos dijo la derivada, tambien las bias
            weights = weights - (deriv * self.__learning_rate)
            bias = bias - (bias_deriv * self.__learning_rate)

        # guardamos los pesos y el sesgo calculado para que sirva en el proceso de
        self.__weights = weights
        self.__bias = bias

        # calcular accuracy final sobre entrenamiento
        final_predictions = self.__sigmoid(
            numpy.dot(self.__training_features, weights) + bias)

        binary_predictions = self.__convert_predictions_to_binary(
            final_predictions)
        final_accuracy = self.__calculate_accuracy_percentage(
            binary_predictions)

        desicion_boundary_points: DesicionBoundaryPoints = self.__generate_decision_boundary()

        return error_per_epoch, final_accuracy, desicion_boundary_points

    def predict(self, input: PredictRequestDTO) -> str:

        # convertimos las tuplas en un ndarray
        data: ndarray = numpy.array(
            [(input.first_feature, input.second_feature)])

        # escalamos los datos de la mismo forma en que fueroin escalados los datos de entrenamiento para que no se sature la signoide
        data = self.__scaler.transform(data)

        # combinamos las entradas con los pesos entrenados con un producto punto y le sumamos los sesgos
        linear_combination = numpy.dot(data, self.__weights) + self.__bias

        # aplicamos la función sigmoide para convertir esos valores en probs 0 y 1
        predictions = self.__sigmoid(linear_combination)

        return "Maligno" if predictions[0][0] > 0.5 else "Benigno"

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

    def __mean_squared_error(self, predictions: ndarray) -> float:
        """
        Calcula el error cuadratico medio entre las predicciones del modelo y los valores reales.

        Args:
            predictions (numpy.ndarray): arreglo de valores predichos por el modelos
        Returns:
            float: valor del error cuadratico medio
        """
        return numpy.mean((predictions - self.__training_labels) ** 2)  # se eleva para convertir valores negativos a positivos

    def __convert_predictions_to_binary(self, final_predictions: ndarray) -> ndarray:
        """
        Se explora cada uno de los elementos en el array y la condicin los convierte en true o false
        si es mayor o igual a 0.5, es 1 (maligno), si no 0 (benigno)
        luego con astype se convierten los boleanos en su representacion numerica correspondiente

        Args:
            final_predictions (ndarray): Predicciones final hechas por el modelo.

        Returns:
            ndarray: La conversion de las predicciones en binario
        """
        binary_predictions: ndarray = (final_predictions >= 0.5).astype(int)

        return binary_predictions

    def __calculate_accuracy_percentage(self, binary_predictions: ndarray) -> float:
        """
        Se compara cada elemento del arreglo de predicciones binarias con el arreglo de valores reales,
        generando un nuevo arreglo de True si coincide o False si no coincide
        cada True representa una prediccion correcta
        luego, se calcula el promedio de aciertos: cada True cuenta como 1, cada False como 0.
        el resultado es el porcentaje de predicciones correctas sobre el total.
        Args:
            predictions (numpy.ndarray): Arreglo de valores predichos por el modelo.
            targets (numpy.ndarray): Arreglo de valores reales .

        Returns:
            float: Porcentaje de aciertos del modelo
        """
        final_accuracy: float = numpy.mean(
            binary_predictions == self.__training_labels)

        return round(final_accuracy * 100, 2)  # lo devolvemos en porcentaje

    def __generate_decision_boundary(self) -> DesicionBoundaryPoints:
        # los pesos al traterse de una matriz de (2,0) debemos aplanarla para que sea un array normal
        weight_1, weight_2 = self.__weights.flatten()

        # obtenemos solamente los mayores y menores de la primera posicion de las featrues pero con : tomamos en cuenta
        # todas las features
        feature1_min = min(self.__training_features[:, 0])
        feature1_max = max(self.__training_features[:, 0])

        feature2_for_min = - (weight_1 / weight_2) * \
            feature1_min - (self.__bias / weight_2)

        feature2_for_max = - (weight_1 / weight_2) * \
            feature1_max - (self.__bias / weight_2)

        # se crea una nueva matriz con essas coordenadas para poder desescalarlas, ya que los
        # __training_features han sido escalados al momento del entranemiento
        points_scaled = numpy.array([[feature1_min, feature2_for_min],
                                     [feature1_max, feature2_for_max]])

        # desescalamos los resultados
        points_original = self.__scaler.inverse_transform(points_scaled)

        return DesicionBoundaryPoints(
            x=float(points_original[0][0]), y=float(points_original[0][1]),
            x2=float(points_original[1][0]), y2=float(points_original[1][1])
        )
