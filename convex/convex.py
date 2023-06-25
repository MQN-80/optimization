import numpy as np
from scipy.optimize import minimize
import file
import pic
import torch
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
        self.pic=pic.pic()
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
        self.x = torch.tensor(self.x, dtype=torch.float64,requires_grad=True)  #需要求梯度
        self.a = torch.tensor(np.array(self.a,dtype="float64"), dtype=torch.float64)
        self.b = torch.tensor(np.array(self.b,dtype="float64"), dtype=torch.float64)
        self.c = torch.tensor(np.array(self.c,dtype="float64"), dtype=torch.float64)
        self.d = torch.tensor(np.array(self.d,dtype="float64"), dtype=torch.float64)
        self.h1 = torch.tensor(np.array(self.h1,dtype="float64"), dtype=torch.float64)
        self.h2 = torch.tensor(np.array(self.h2,dtype="float64"), dtype=torch.float64)
        self.upper = torch.tensor(np.array(self.upper,dtype="float64"), dtype=torch.float64)
        self.lower = torch.tensor(np.array(self.lower,dtype="float64"), dtype=torch.float64)
        self.k1 = torch.tensor(np.array(self.k1,dtype="float64"), dtype=torch.float64)
        self.k2 = torch.tensor(np.array(self.k2,dtype="float64"), dtype=torch.float64)
        #self.init_x()
    #从文件内读取
    def readData(self):
        self.m,self.n,self.a,self.b,self.c,self.d,self.h1,self.h2,self.lower,self.upper=self.file.read_all('D:\study\最优化理论\optimization\FractionalQPdata\C2_n2000_m22')
        self.x=[0.1]*self.m  #转换ndarry初始化
        self.k1=[1e-3]*self.m
        self.k2=[1e-4]*self.m
    #x的初始化
    def init_x(self):
        for i in range(self.m):
            self.x[i] = (self.lower[i]+self.upper[i])/2  
    # 原问题的算术形式
    def solution(self):
        res_0=torch.matmul(self.x,self.a.T)/torch.matmul(self.x,self.b.T)-torch.matmul(self.x,self.c.T)/torch.matmul(self.x,self.d.T)
        res_1=torch.sum(torch.square(res_0))
        return res_1
    #线性约束形式,k是惩罚因子
    def equal_con(self):
        return torch.square(torch.dot(self.x.T,self.h1)-1)+torch.square(torch.dot(self.x.T,self.h2))
    #非线性约束形式
    def noequal_con(self):
        return torch.sum(torch.square(torch.relu(self.x-self.upper))+torch.square(torch.relu(self.lower-self.x)))
    # 计算导数
    # 梯度下降算法,step是选择步长,用于更新x的值,group是重复组数
    def descent(self,groups=50,k=1000):
        #optimizer = torch.optim.Adam([self.x], lr=0.1)  #定义优化器
        optimizer=torch.optim.LBFGS([self.x], lr=0.1) 
        def closure():
            optimizer.zero_grad()  # 清空梯度
            res=self.solution()+self.equal_con()*k+self.noequal_con()*k     # 计算损失
            res.backward()        # 反向传播
            return res
        for i in range(groups):
            #使用牛顿法进行迭代
            optimizer.step(closure)
            if self.equal_con()<0.01 and self.noequal_con()<0.01:
                break
        print("最优值为"+str(self.solution().item()))    
        print("等式约束误差:"+str(self.equal_con().item())) 
        print("不等式约束误差:"+str(self.noequal_con().item()))
    def process(self):
        self.readData()
        self.changeArray()
        self.descent()
        
question=Question()
question.process()

