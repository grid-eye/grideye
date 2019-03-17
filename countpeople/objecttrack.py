class ObjectTrack:
    '''
    目标轨迹类,保存每个目标的在每一帧的位置
    '''
    def __init__(self):
        self.__loc_list = [] #运动轨迹队列
        self.__img_list = [] #运动所在的帧的数据，和上面的轨迹一一对应
        self.__pos = -1
        self.__size = 0
        self.__direction = 1#进入
        self.__compensation =2# 补偿值
        self.__max_age = 100
        self.__max_interval = 3
        self.__age_counter = 0
        self.__interval_counter = 0
    def put(self,point,img):
        self.__loc_list.append(point)
        self.__img_list.append(img)
        self.__size += 1
    def get(self):
        loc = self.__loc_list[self.__size -1]
        img = self.__img_list[self.__size-1]
        return loc,img
    def getAge(self):
        return self.__age_counter
    def getInterval(self):
        return self.__interval_counter
    def isEmpty(self):
        return self.__size == 0 
    def isAgeOverflow(self):
        return self.__age_counter >= self.__max_age
    def isIntervalOverflow(self):
        return self.__interval_counter >= self.__max_interval
    def incrementAge(self):#年龄自增
        self.__age_counter += 1
    def incrementInterval(self):
        self.__interval_counter += 1

    def hasPassDoor(self,frame_width = 8):
        '''
            是否已经通过

        '''
        xs,xend = self.__loc_list[0] ,self.__loc_list[self.__size-1]
        if xend[1] < xs[1] :
            self.__direction = 0 
        s =frame_width -1 
        dis =abs( xend[1] - xs[1])+self.__compensation
        return dis >= s
    def getDirection(self):
        return self.__direction
    def showContent(self):
        print(self.__loc_list,end=",interval is ")
        print(self.__interval_counter,end=" ,age is ")
        print(self.__age_counter)

