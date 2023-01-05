import numpy as np
from scipy.optimize import minimize
import file
class Question:
    def __init__(self) -> None:
        self.x=[]   #x为m维度向量
        self.a=[]   #a为(n,m)维向量
        self.b=[]   #b为(n,m)维向量
        self.c=[]   #c为(n,m)维向量
        self.d=[]   #d为(n,m)维向量
        self.m=0    #m为x维度
        self.n=0    #n为abcd四个矩阵的行数
        self.h1=[]  #x和h1内积等于1
        self.h2=[]  #x和h2内积等于0
        self.upper=[]  #x的上限
        self.lower=[]  #x的下限
        self.k1=[]     #kkt条件中f1(x)的系数
        self.k2=[]     #kkt条件中f2(x)的系数
        self.v1=0      #kkt条件中h1(x)的系数
        self.v2=0      #kkt条件中h2(x)的系数
        self.file=file.File()    #用于读取数据文件
    # 设置一组数据用于测试   
    def ceshi(self):
        self.x=[1,2,3]   #x为m维度向量
        self.a=[[2,4,5],[2,-4,-3]]   #a为(n,m)维向量
        self.b=[[2,300,5],[2,-2,-3]]   #b为(n,m)维向量
        self.c=[[1,4,5],[2,-1,-3]]   #c为(n,m)维向量
        self.d=[[2,5,5],[3,-4,-3]]   #d为(n,m)维向量
        self.m=3    #m为x维度
        self.n=2    #n为abcd四个矩阵的行数
        self.h1=[0.5,0.1,0.1]  #x和h1内积等于1
        self.h2=[-1,-1,1]  #x和h2内积等于0
        self.upper=[3,4,5]  #x的上限
        self.lower=[1,1,0]  #x的下限
        self.k1=[1,1,1]     #kkt条件中f1(x)的系数
        self.k2=[1,1,1]     #kkt条件中f2(x)的系数
        self.v1=0      #kkt条件中h1(x)的系数
        self.v2=0      #kkt条件中h2(x)的系数
    # 转换为对应ndarray形式,方便进行计算    
    def changeArray(self):
        self.x=np.array(self.x,dtype="float64")
        self.a=np.array(self.a,dtype="float64")
        self.b=np.array(self.b,dtype="float64")
        self.c=np.array(self.c,dtype="float64")
        self.d=np.array(self.d,dtype="float64")
        self.h1=np.array(self.h1,dtype="float64")
        self.h2=np.array(self.h2,dtype="float64")
        self.upper=np.array(self.upper,dtype="float64")
        self.lower=np.array(self.lower,dtype="float64")
        self.k1=np.array(self.k1,dtype="float64")
        self.k2=np.array(self.k2,dtype="float64")
        #self.init_x()
    #从文件内读取
    def readData(self):
        self.m,self.n,self.a,self.b,self.c,self.d,self.h1,self.h2,self.lower,self.upper=self.file.read_all('D:\电子课本习题\最优化理论\optimization\FractionalQPdata\C3_n5000_m22')
        self.x=[1e-5]*self.m  #转换ndarry初始化
        self.k1=[1e-3]*self.m
        self.k2=[1e-3]*self.m
    #x的初始化
    def init_x(self):
        self.x[0]=self.h2[2]/(self.h1[1]*self.h2[2]-self.h1[2]*self.h2[1])
        print(self.x[0])
        self.x[1]=self.h2[1]/(self.h1[2]*self.h2[1]-self.h1[1]*self.h2[2])
    # 转换为对应ndarray形式,方便进行计算    
    # 原问题的算术形式
    def solution(self,input):
        res_0=np.dot(input,self.a.T)/np.dot(input,self.b.T)-np.dot(input,self.c.T)/np.dot(input,self.d.T)
        res_1=np.sum(np.square(res_0))
        return res_1
    #用于输入项
    def solution_min(self,args):
        def v(x):
            res_0=np.dot(x,args[0].T)/np.dot(x,args[1].T)-np.dot(x,args[2].T)/np.dot(x,args[3].T)
            res_1=np.sum(np.square(res_0))
            return res_1
        return v
    #用于限制
    def con(self,args):
    # 约束条件 分为eq 和ineq
    # eq表示 函数结果等于0 ； ineq 表示 表达式大于等于0  
        cons=[]
    #先处理上下限
        for i in range(args[0].size):
            lw=args[0][i]  #下限
            up=args[1][i]  #上限
            lower={'type': 'ineq', 'fun': lambda x: x[i] - lw}
            upper={'type': 'ineq', 'fun': lambda x: up - x[i]}
            cons.append(lower)
            cons.append(upper)
        #再处理等式
        h1=args[2]
        h2=args[3]
        h1_1={'type': 'eq', 'fun': lambda x: np.dot(x,h1.T)-1}
        h2_2={'type': 'eq', 'fun': lambda x: np.dot(x,h2.T)}
        cons.append(h1_1)
        cons.append(h2_2)
        return cons
    # 计算导数
    def differential_origin(self,step=1e-10):
        x_mid=self.x+step
        differ=(self.solution(x_mid)-self.solution(self.x))/step
        return differ
    # 对原式x求导,采用定义求导数,step代表▽x,index代表下标,即求第几项的偏导
    def differential(self,step=1e-10,index=0):
        x_mid=self.x
        x_mid[index]+=step
        differ=(self.solution(x_mid)-self.solution(self.x))/step+self.k1[index]-self.k2[index]+self.v1*self.h1[index]+self.v2*self.h2[index]
        return differ
    # 对k1求偏导
    def differential_k1(self,index=0):
        return self.x[index]-self.upper[index]
    # 对k2求偏导
    def differential_k2(self,index=0):
        return -self.x[index]+self.lower[index]
    # 对v1求偏导
    def differential_v1(self):
        return np.dot(self.x.T,self.h1)-1 
    # 对v2求偏导
    def differential_v2(self):
        return np.dot(self.x.T,self.h2) 
    # 梯度下降算法,step是选择步长,用于更新x的值,group是重复组数
    def descent(self,step=1e-5,groups=1000):
        min=1e7
        for i in range(groups):
            #对x进行梯度下降
            for k in range(self.m):
                self.x[k]-=self.differential(index=k)*step
            #再对k1,k2进行梯度下降
            for k in range(self.m):
                self.k1[k]-=self.differential_k1(index=k)*step
                self.k2[k]-=self.differential_k2(index=k)*step
            #再对v1,v2进行梯度下降
            self.v1-=self.differential_v1()*step
            self.v2-=self.differential_v2()*step
            mid=self.solution(self.x)
            if mid<min:
                min=mid
            print(mid)
            # 梯度小于0.01时候停止
        print(min)       
    def optimize(self):
        x_origin=self.x
        x_lu=[]   #获取限制
        for i in len(self.lower):
            x_lu.append(self.lower[i])
            x_lu.append(self.upper[i])
        
        minimize(self.solution(self.x),x_origin)
    def process(self):
        self.readData()
        self.changeArray()
        self.descent()
    def ce(self):
        self.readData()
        self.changeArray()
        data=[]
        data.append(self.a)
        data.append(self.b)
        data.append(self.c)
        data.append(self.d)
        cons=[]
        cons.append(self.lower)
        cons.append(self.upper)
        cons.append(self.h1)
        cons.append(self.h2)
        mid=self.con(cons)
        res = minimize(self.solution_min(data), self.x,method='Nelder-Mead',constraints=mid)
        print(res.fun)
        print(res.success)
question=Question()
question.ce()
def fun(args):
     def v(x):
        return np.dot(x,x.T)
     return v
def con(args):
    # 约束条件 分为eq 和ineq
    # eq表示 函数结果等于0 ； ineq 表示 表达式大于等于0  
    x1min, x1max = args
    cons = ({'type': 'ineq', 'fun': lambda x: x[0] - x1min},\
            {'type': 'ineq', 'fun': lambda x: -x[0] + x1max})
    
    return cons
args = [1]  #a
x0 = np.array([1])  # 初始猜测值
args=[x0,x0]
res = minimize(fun(args), x0, method='SLSQP')

