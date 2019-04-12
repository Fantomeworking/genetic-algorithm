import math
import numpy as np
import matplotlib.pyplot as plt

pop=500#种群大小，种群二进制矩阵行数
poplines=20#种群二进制矩阵列数,限制偶数，前半部分为整数，后半部分为小数
muta=0.2#变异率
cross=0.6#交叉率
best_x=[]#记录最优结果
best_y=[]#记录最优结果
def translation10to2(fitness):#10进制转化为2进制,输入10进制向量，输出二进制矩阵(list)
    population=np.zeros([pop,poplines])
    len_=int(poplines/2)
    for s,i in enumerate(fitness):
        i=round(i,4)
        x=0
        F=[]
        I=[]
        int_=i//1#整数部分
        while int_/2>=0.5:
            I.append(1*(int_ % 2!=0))
            int_=int_//2
        float_=i%1#小数部分
        while float_%1!=0 and len_>x:#小数部分转化为二进制
            F.append(float_*2//1)
            float_=float_*2%1
            x+=1
        population[s,0:len(I)]=I
        population[s,len_:len_+len(F)] = F
    return population
def translation2to10(population):#2进制转化为10进制,输入二进制矩阵，输出10进制向量
    len_ = int(poplines / 2)
    fitness=np.zeros([pop])
    for s,j in enumerate(population):
        int_=j[0:len_]#整数部分
        float_=j[len_:poplines]#小数部分
        i_,f_=1,1
        I,F=0,0
        for i,f in zip(int_,float_):
            I+=i*i_#整数
            i_=i_*2
            f_ = round(f_ / 2,4)

            F+=f*f_#小数
        fitness[s]=I + F
    return fitness

fitness_x=10*np.random.rand(pop)#建立随机初始10进制种群向量,并限定取值范围
population=translation10to2(fitness_x)#转化为初始2进制种群

def fun(X):
    fitness_new=[]
    for x in X:
        x=x*10/(math.pow(2,poplines/2)-1)
        fitness_new.append(-math.pow((x-2),2)+2.3564)#计算新的适应度
    fitness_new=np.array(fitness_new)
    return fitness_new
def selection(fitness_Y,population):
    fitness_add=[]
    new_population=np.zeros([pop,poplines])
    population_=[]
    Rf=np.random.rand(pop)
    Rf.sort()
    for s,i in enumerate(fitness_Y):
        if i>0:
            fitness_add.append(sum(fitness_add)+math.pow(i,2))
            population_.append([population[s,:]])
        else:
            if np.random.rand()<0.9:
                fitness_add.append(0)
                population_.append([[0]*poplines])
    fitness_add = np.array(fitness_add)
    population_ = np.array(population_)
    if max(fitness_add)!=0:
        fitness_add=fitness_add/max(fitness_add)
    r, p = 0,0
    while r<pop and p<population_.shape[0]:
        if Rf[r]<fitness_add[p]:
            new_population[r,:]=population_[p,:]
            r+=1
        else:
            p+=1
    return new_population
def across(population):
    raw=population.shape[0]
    for i in range(raw):
        R=np.random.rand(3)#产生两个随机数，第一个为是否交叉判断，第二个为交叉位置判断
        if cross>R[0]:
            x=int(R[1]*raw//1)#行位置
            y=int(R[2]*poplines//1)#列位置
            s=population[x-1,y]
            population[x-1, y]=population[x,y]
            population[x, y]=s
    return population
def mutation(population):
    raw = population.shape[0]
    for i in range(raw):
        R=np.random.rand(3)#产生两个随机数，第一个为是否交叉判断，第二个为交叉位置判断
        if muta>R[0]:
            x=int(R[1]*raw//1)#行位置
            y=int(R[2]*poplines//1)#列位置
            population[x, y]=-population[x,y]+1
    return population
f=10#循环次数
for i in range(f):#循环200次
    fitness_Y=fun(fitness_x)#适应度函数，计算适应度
    best_y.append(np.max(fitness_Y))#记录输出最大值y
    max_=np.where(fitness_Y==np.max(fitness_Y))
    best_x.append(fitness_x[max_][0])#记录输出最大值y对应的x
    population_new=selection(fitness_Y,population)#赌轮盘选择，将优质种群放大
    population = across(population_new)  # 先交叉
    population=mutation(population)#再变异
    fitness_x = translation2to10(population)
print(max(best_y))
plt.plot(range(f),best_y,'r')
plt.show()
