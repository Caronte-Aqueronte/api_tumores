class PredicionResponseDto:

    def __init__(self, x_feature: int, y_feature: int, prediction: int):

        self.__x_feature: int = x_feature
        self.__y_featurre: int = y_feature
        self.__prediction: int = prediction
