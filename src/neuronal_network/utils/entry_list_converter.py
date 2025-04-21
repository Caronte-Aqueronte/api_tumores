
from typing import List, Tuple

from neuronal_network.models.entry import Entry


class EntryConverter:

    def __init__(self, entries: List[Entry]):
        self.__entries: List[Entry] = entries

    def convert_list_of_entry_to_list_of_tuple(self) -> List[Tuple[float, float]]:

        [(entrie.get_first_feature(), entrie.get_second_feature())
         for entrie in self.__entries]
