class Target:
    '''
    运动目标
    '''
    __center_temperature = 0 #邻域平均温度
    def __init__(self ,center_temperature= 0 ):
        self.__center_temperature = center_temperature
    def getNeiborhoddTemp(self):
        return self.__center_temperature
    def setNeiborhoodTemp(self , neiborhood):
        self.__center_temperature = neiborhood
    def showContent(self):
        print(self.__center_temperature)

