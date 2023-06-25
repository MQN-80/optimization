import matplotlib.pyplot as plt
import numpy as np
import matplotlib
class pic:
    def solutionChange(self,data):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        y = np.array(data)
        plt.plot(y,color='green',label='数据值变化')
        #plt.text(0, 400, 'P型mos的cv曲线')
        plt.legend()
        plt.show()
