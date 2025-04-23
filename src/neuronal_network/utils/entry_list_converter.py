
from typing import List, Tuple

from numpy import ndarray

from neuronal_network.dto.prediction_response_dto import PredicionResponseDto
from neuronal_network.dto.entry_request_dto import EntryRequestDTO


class EntryConverter:

    def __init__(self):
        pass

    def convert_list_of_entry_to_list_of_tuple(self, entries: List[EntryRequestDTO]) -> List[Tuple[float, float]]:
        """
        Se itera sobre cada elemento de la lista de entradas dada par de valores de la entrada se agrupa como una tupla. 
        El resultado es una lista de tuplas.

        Args:
            entries (List[Entry]): Lista de objetos Entry.

        Returns:
            List[Tuple[float, float]]: Prepresentacion de la lista en tuplas.
        """
        return [(entrie.get_first_feature(), entrie.get_second_feature())
                for entrie in entries]

    def convert_entries_to_predictions_response_dto(self, entries: List[EntryRequestDTO], predicitons: ndarray) -> List[PredicionResponseDto]:
        """
        Utilizamos zip para emparejar en tuplaz las ocurrencias correspondientes de ambas listas 
        de forma uno a uno, e iteramos sobre cada par para crear instancias de PrediccionResponseDto,
        las cuales se agregan a la lista de respuestas.

        Args:
            entries (List[Entry]): Lista de entradas originales utilizadas para realizar la prediccion.
            predicitons (ndarray): Arreglo de predicciones binarias generadas por la red neuronal, donde cada valor indica si la predicción es benigna (0) o maligna (1).

        Returns:
            List[PredicionResponseDto]: Lista de objetos que contienen tanto la entrada original como su predicción correspondiente, en un formato estructurado.
        """
        # lista que guardara las respuestas
        responses: List[PredicionResponseDto] = []

        # con zip juntamos las courrencias de ambas listas 1 a 1 e iteramos sobre las nuevas convinacioens
        # para crear los nuevos objetos PredicionResponseDto, los adjuntamos a la lista de responses
        for entry, prediciton in zip(entries, predicitons):

            # creamos el nuevo objeto de respuesta con los atributos de las ocurrencias de la tupla
            responses.append(PredicionResponseDto(
                x_feature=entry.get_first_feature(), y_feature=entry.get_second_feature(), prediction=prediciton))

        # retonramos la lista elimentada
        return responses
