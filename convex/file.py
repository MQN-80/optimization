import numpy as np
import os
class File:
    def __init__(self) -> None:
        self.m=22    #x维度,初始化为22
        self.n=0
        self.a=[]
        self.b=[]
        self.c=[]
        self.d=[]
        self.h1=[]
        self.h2=[]
        self.lower=[]
        self.upper=[]
    #拆分每行数据
    def split_line(self,line):
        res=line.split()
        return res
    #读取mn维度
    def read_mn(self,file_name:str):
        res=file_name.split('_')
        self.n=int(res[1][1:])
        self.m=int(res[2][1:])
    def read_a(self,filepath:str):
        with open(filepath,'r') as f:
            for line in f:
                if(line!=None):
                    self.a.append(self.split_line(line))
    def read_b(self,filepath:str):
        with open(filepath,'r') as f:
            for line in f:
                if(line!=None):
                    self.b.append(self.split_line(line))
    def read_c(self,filepath:str):
        with open(filepath,'r') as f:
            for line in f:
                if(line!=None):
                    self.c.append(self.split_line(line))
    def read_d(self,filepath:str):
        with open(filepath,'r') as f:
            for line in f:
                if(line!=None):
                    self.d.append(self.split_line(line))
    def read_h1(self,filepath:str):
        with open(filepath,'r') as f:
            for line in f:
                if(line!=None):
                    self.h1.append(line.replace("\n",""))
    def read_h2(self,filepath:str):
        with open(filepath,'r') as f:
            for line in f:
                if(line!=None):
                    self.h2.append(line.replace("\n",""))
    def read_lower(self,filepath:str):
        with open(filepath,'r') as f:
            for line in f:
                if(line!=None):
                    self.lower.append(line.replace("\n",""))
    def read_upper(self,filepath:str):
        with open(filepath,'r') as f:
            for line in f:
                if(line!=None):
                    self.upper.append(line.replace("\n",""))
    def read_all(self,path:str):
        self.read_mn(path.split("\\")[-1])
        file_list=[]
        for file_name in os.listdir(path):
            file_list.append(path+'\\'+file_name)
        self.read_a(file_list[0])
        self.read_b(file_list[1])
        self.read_c(file_list[2])
        self.read_d(file_list[3])
        self.read_h1(file_list[4])
        self.read_h2(file_list[5])
        self.read_lower(file_list[6])
        self.read_upper(file_list[7])
        return self.m,self.n,self.a,self.b,self.c,self.d,self.h1,self.h2,self.lower,self.upper      
file=File()