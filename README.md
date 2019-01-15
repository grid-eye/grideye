# grideye
1. countpeople 
    ```
    核心包,countpeople.py 里面的CountPeople定义了对读取的图像的处理过程和方法
    ```
       主体方法是process ,重复读取传感器的数据，并对数据进行相应的处理
        
    ```
2. examples

    ```
     测试驱动是否连接成功的文件
    ```
3. bgtemperature_curr_diff dirs
    ```
    	
    ```
4. main.py
   ```

    1 .func: 
        收集测试数据
    2. command format:

	    python3 main.py dir sequence_num frame_num
	    sequence_num:循环次数,也表示序列数
	    frame_num:帧数
	    ex:python3 main.py images/2019-01-12-bg 10 1000
		这个命令是将保存10个序列（每个序列包含一千帧数据)保存在images/2019-01-120bgx(这个bgx后面的x表示序列号:1~1000)
	
   ```
5. countpeople/processDiff.py
   ```
        1. func:
            这个脚本是分析某个序列下的多个帧和平均温度之间的差值，包括差值的平均值，最小值和最大值
        2. command format:
            python3 processDiff.py dirpath
            例子 ：python3 processDiff.py images/2019-01-12-xx
	
   ```
6. countpeople/processDiffSets.py
   ```
    1. func:
         这个脚本和前面的那个类似，不同的是他是面向多个序列的
    2. command format:
        python3 processDiffSets.py dir num
        dir :表示起始序列
        num :表示序列数
        ex: python3 processDiffSets.py images/2019-01-12-x1 30
        这个命令是分析 images/2019-01-12-x1~images/2019-01-12-x30目录的diffdata.npy(通过imagedata.npy和avgtemp.npy求得)的数据
   ```
7. countpeople/analyseSquence.py 

    ```
     1. func:
       这个脚本是分析某个序列的diffdata.npy,imagedata.npy,和avgtemp.npy的关系,并且绘制相关图像，主要是分析是否有人通过，这会影响背景温度的测试
     2. command format:
	    python3 analyseSequence.py dir n1,n2,n3,n4,n5
	    dir:表示序列的目录	
	    nx:表示第nx帧
	    ex: python3 analyseSequence.py images/2019-01-12-bgtemp12 34,38,45,47
    ```
8. countpeople/transformdata2image
    ```
       将数据转换为图像，比较直观
    ```
9. countpeople/deleteFile.py
    ```
        删除一系列.npy数据(可以忽略这个文件)
    ```
10. countpeople/moveFile.py
   ```
   	移动一系列.npy数据(可以忽略这个文件)
   ```
