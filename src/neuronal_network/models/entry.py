
class Entry:

    def __init__(self, first_feature: float, second_feature: float):
        self.__first_feature = first_feature
        self.__second_feature = second_feature

    def get_first_feature(self) -> float:
        return self.__first_feature

    def get_second_feature(self) -> float:
        return self.__second_feature
