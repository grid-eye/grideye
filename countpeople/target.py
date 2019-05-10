class Target:
    '''
    运动目标
    '''
    def __init__(self ,center_temperature= 0 ):
        self.__center_temperature = center_temperature
    def getNeiborhoddTemp(self):#用不上
        return self.__center_temperature
    def setNeiborhoodTemp(self , neiborhood):#用不上
        self.__center_temperature = neiborhood


