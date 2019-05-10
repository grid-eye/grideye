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
        self.__row = 8
        self.__col = 8
        self.__last_vote = -1
    def put(self,point,img):
        self.__loc_list.append(point)
        self.__img_list.append(img)
        self.__size += 1
        self.__updateWalkTrend(point)
        if self.__size == 1:
            self.__autoUpdateShape(img.shape[0],img.shape[1])
    def __autoUpdateShape(self,row,col):
        self.__row = row
        self.__col = col
    def getLastTrend(self):
        return self.__last_vote
    def __updateWalkTrend(self,point):
        if self.__size == 1:
            print("=====================update walk trend===================")
            self.__last_vote = self.__getEntranceDirection(point)
            print(self.__last_vote)
        else:
            self.__last_vote = self.__getWalkTrend()
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
    def clearInterval(self):
        self.__interval_counter =0
    def __getEntranceDirection(self,point):
        xcorr = point[1]
        if xcorr <= 1 :
            return 1
        if  xcorr >= self.__col-2:
            return 0
        print("return 1")
        return -1
    def __edgePointVote(self,point_arr):
        vote = {0:0,1:0}
        for item in point_arr:
            direction = self.getEntranceDirection(item)
            if direction == 1:
                vote[1] += 1
            elif direction == 0:
                vote[0] += 1
        if not hasattr(self,"last_vote"):
            if vote[1] > vote[0] :
                self.last_vote = 1
                return 1,vote
            elif vote[0] > vote[1] :
                self.last_vote = 0 
                return 0,vote
        else:
            if vote[1] > 0  or vote[0] > 0 :
                return self.last_vote,vote
        return -1,vote
    def __getVote(self,point_arr):
        vote ={0:0,1:0}
        p0 = point_arr[0]
        for p in point_arr[1:]:
            vector = p[1]-p0[1]
            p0 = p
            if vector > 0 :
                vote[1] += 1
            elif vector < 0:
                vote[0] += 1
            else:
                vote[self.__last_vote] += 1
        if vote[0] > vote[1]:
            self.__last_vote = 0
            return 0
        elif vote[0] < vote[1]:
            self.__last_vote = 1
            return 1
        else:
            return self.__last_vote
    def __getWalkTrend(self):#得到当前运动趋势
        length = 4
        if self.__size < 4:
            length = self.__size
        length = -length
        return self.__getVote(self.__loc_list[length:])
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

