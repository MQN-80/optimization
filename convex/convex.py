import numpy as np
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
    # 转换为对应ndarray形式,方便进行计算    
    # 原问题的算术形式
    def solution(self,input):
        res_0=np.dot(input,self.a.T)/np.dot(input,self.b.T)-np.dot(input,self.c.T)/np.dot(input,self.d.T)
        res_1=np.sum(np.square(res_0))
        return res_1
    # 计算导数
    def differential_origin(self,step=1e-6):
        x_mid=self.x+step
        differ=(self.solution(x_mid)-self.solution(self.x))/step
        return differ
    # 对原式x求导,采用定义求导数,step代表▽x,index代表下标,即求第几项的偏导
    def differential(self,step=1e-6,index=0):
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
    def descent(self,step=0.01,groups=1000):
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
            print(self.solution(self.x))
            # 梯度小于0.01时候停止
            if np.abs(self.differential_origin())<1e-5:
                break
    def end(self,judge_differ=1e-4):
        differ=self.differential()
        if judge_differ>=differ:
            return True
        else:
            return False
    def process(self):
        self.ceshi()
        self.changeArray()
        self.descent()
question=Question()
question.process()
