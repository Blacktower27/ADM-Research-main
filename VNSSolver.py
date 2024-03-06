import random
import io
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import time
import numpy as np
import copy
import json
import networkx as nx
from NetworkGenerator import Scenario
import multiprocessing
import os

class VNSSolver:
    def __init__(self,S,seed,baseline="distance",enumFlag=False):
        self.S=S
        random.seed(seed)
        np.random.seed(seed)
        self.node=S.name2FNode#飞行节点
        self.flts=list(S.flight2dict.keys())#航班
        self.tails=list(S.tail2flights.keys())#飞机
        self.tail2cap=S.tail2capacity
        self.tail2srcdest={tail:[self.node[flts[0]].Ori,self.node[flts[-1]].Des] for tail,flts in S.tail2flights.items()}#飞机最初的起点机场和最后的到达机场
        self.crews=list(S.crew2flights.keys())
        self.crew2srcdest={crew:[self.node[flts[0]].Ori,self.node[flts[-1]].Des] for crew,flts in S.crew2flights.items()}#机组成员最初的起点机场和最后的到达机场
        self.skdPs=[[self.tail2srcdest[tail][0]]+flts+[self.tail2srcdest[tail][1]] for tail,flts in S.tail2flights.items()]#每架飞机的起点机场 中间经过的fnode 终点机场   # e.g. P=[LAX,F00,F01,F02,ORD]; Ps=[P1,P2,P3,..]
        self.skdQs=[[self.crew2srcdest[crew][0]]+flts+[self.crew2srcdest[crew][1]] for crew,flts in S.crew2flights.items()]#每个机组成员的起点机场 中间经过的fnode 终点机场
        self.k2func={0:"pass_",1:"swap",2:"cut",3:"insert"}#临域动作的编号 delete不知道去哪里了
        self.baseline=baseline
        self.enumFlag=enumFlag#是否枚举？
        # 基于三种类型的基线VNS
        # 将有向图转换为无向图
        undirectGraph = self.S.connectableGraph.to_undirected()

        # uniform：均匀概率操作
        if baseline == "uniform":
            # 为每个节点设置相同的权重
            self.node2weight = {self.S.FNode2name[node]: 1 for node in self.S.FNodes}#全部的概率都是1
        # degree：连接其他节点的数量越多权重越高（包括其他节点指向它 以及它指向其他节点）
        elif baseline == "degree":
            # 为每个节点设置与其度数相同的权重
            self.node2weight = {self.S.FNode2name[node]: deg for node, deg in undirectGraph.degree()}
        # distance：距离越小表示被操作的概率越高
        elif baseline == "distance":
            # 初始化节点权重字典为空
            self.node2weight = {}
            # 循环计算每个延误节点到其他节点到距离（这个距离不是实际飞行距离，而是延误节点到该节点的边的数量）
            for drpNode in self.S.drpFNodes:
                node2distance = nx.single_source_dijkstra_path_length(undirectGraph, drpNode, cutoff=None, weight='weight')
                # 更新节点权重字典，距离越小权重越大
                for node, dis in node2distance.items():
                    if self.S.FNode2name[node] not in self.node2weight:
                        self.node2weight[self.S.FNode2name[node]] = -1 * dis
                    elif self.node2weight[self.S.FNode2name[node]] < -1 * dis:
                        self.node2weight[self.S.FNode2name[node]] = -1 * dis
        # # three types of baseline VNS
        # undirectGraph=self.S.connectableGraph.to_undirected()
        # if baseline=="uniform": # uniform probability to be operated
        #     self.node2weight={self.S.FNode2name[node]:1 for node in self.S.FNodes}
        # elif baseline=="degree": # larger degree indicates higher probability to be operated
        #     self.node2weight={self.S.FNode2name[node]:deg for node,deg in undirectGraph.degree()}
        # elif baseline=="distance":  # smaller distance indicates higher probability to be operated
        #     self.node2weight={}
        #     for drpNode in self.S.drpFNodes:
        #         node2distance=nx.single_source_dijkstra_path_length(undirectGraph,drpNode,cutoff=None,weight='weight')
        #         for node,dis in node2distance.items():
        #             if self.S.FNode2name[node] not in self.node2weight:
        #                 self.node2weight[self.S.FNode2name[node]]=-1*dis
        #             elif self.node2weight[self.S.FNode2name[node]]<-1*dis:
        #                 self.node2weight[self.S.FNode2name[node]]=-1*dis
                        
    def checkConnections(self,pairs): #pairs=[(P1,P2)] or [(Q1,Q2)]
        flag=True
        for pair in pairs:
            if pair[0] not in self.flts and pair[1] not in self.flts:#两个都是机场无法连接
                flag=False
            elif pair[0] not in self.flts and pair[0]!=self.node[pair[1]].Ori:#0是机场且不是1航班的出发机场 无法连接
                flag=False
            elif pair[1] not in self.flts and pair[1]!=self.node[pair[0]].Des:#1是机场且不是0航班的目的机场 无法连接
                flag=False
            #0，1都是航班时，0的目的机场和1的出发机场不同,或者1的最晚出发时间小于(0的最早到达时间+连接时间) 无法连接
            elif pair[0] in self.flts and pair[1] in self.flts and (self.node[pair[0]].Des!=self.node[pair[1]].Ori or self.node[pair[1]].LDT<self.node[pair[0]].CT+self.node[pair[0]].EAT):
                flag=False
        return flag
        
    def pass_(self,X1,X2):#啥也不干
        return [(X1,X2)]
    
    def swap(self,X1,X2):#交换
        pairs=[]
        for u in range(1,len(X1)-1):
            for v in range(1,len(X2)-1):
                if self.checkConnections([(X1[u-1],X2[v]),(X2[v-1],X1[u]),(X2[v],X1[u+1]),(X1[u],X2[v+1])]):
                    pairs.append((X1[:u]+[X2[v]]+X1[u+1:],X2[:v]+[X1[u]]+X2[v+1:]))#交换中间的单个节点
        return pairs
    
    def cut(self,X1,X2):#交换中间的一坨
        pairs = []  # 初始化结果列表
        # 遍历 X1 中所有可能的切割位置
        for u1 in range(1, len(X1) - 1):
            for u2 in range(u1, len(X1) - 1):
                # 遍历 X2 中所有可能的切割位置
                for v1 in range(1, len(X2) - 1):
                    for v2 in range(v1, len(X2) - 1):
                        # 检查切割后的序列是否保持连通性
                        if self.checkConnections([(X1[u1 - 1], X2[v1]), (X2[v1 - 1], X1[u1]), (X2[v2], X1[u2 + 1]),
                                                  (X1[u2], X2[v2 + 1])]):
                            # 如果保持连通性，则将切割后的结果添加到 pairs 中
                            pairs.append((X1[:u1] + X2[v1:v2 + 1] + X1[u2 + 1:], X2[:v1] + X1[u1:u2 + 1] + X2[v2 + 1:]))
        return pairs
    
    def insert(self,X1,X2):#把X2中的一段 插入X1
        pairs = []  # 初始化结果列表
        # 遍历 X1 中所有可能的插入位置
        for u in range(1, len(X1) - 1):
            # 遍历 X2 中所有可能用于插入的中间段
            for v1 in range(1, len(X2) - 1):
                for v2 in range(v1 + 1, len(X2) - 1):
                    # 检查插入后的序列是否保持连通性
                    if self.checkConnections([(X1[u - 1], X2[v1]), (X2[v2], X1[u]), (X2[v1 - 1], X2[v2 + 1])]):
                        # 如果保持连通性，则将插入后的结果添加到 pairs 中
                        pairs.append((X1[:u] + X2[v1:v2 + 1] + X1[u:], X2[:v1] + X2[v2 + 1:]))
        return pairs


    def evaluate(self,Ps,Qs):
        flt2tail={flt:self.tails[i] for i,P in enumerate(Ps) for flt in P[1:-1]}#航班对飞机的映射
        # find the father flight for each flight based on Qs and multiple hop schedule itin
        flt2father={}
        #根据机组人与查看父航线
        for Q in Qs:
            flts=Q[1:-1]
            flt2father[flts[0]]=None
            for i in range(1,len(flts)):
                flt2father[flts[i]]=flts[i-1]
        #根据行程查看父航线
        for fltleg in self.S.fltlegs2itin.keys():
            flts=fltleg.split('-')
            if len(flts)>1:
                for i in range(1,len(flts)):
                    flt2father[flts[i]]=flts[i-1]
                            
        # greedy assignment of dt&at for each flight based on the structure of Ps
        #根据ps贪心分配航班起飞和到达时间
        timeDict={flt:[0,0] for flt in self.flts}
        for P in Ps:
            for i in range(len(P)):
                if i==1:
                    timeDict[P[1]]=[self.node[P[1]].SDT,self.node[P[1]].SAT]
                elif i>1 and i<len(P)-1:
                    timeDict[P[i]][0]=max(timeDict[P[i-1]][1]+self.S.config["ACMINCONTIME"],self.node[P[i]].SDT)
                    timeDict[P[i]][1]=timeDict[P[i]][0]+self.node[P[i]].SFT
        
        # repair timeDict based on the structure of Qs
        # 根据 Qs 的结构重构以及ps的时间机组人员飞行和到达时间
        for flt,father in flt2father.items():
            if father!=None:
                fatherAt=timeDict[father][1]
                timeDict[flt][0]=max(timeDict[flt][0],fatherAt+self.S.config["CREWMINCONTIME"])
                timeDict[flt][1]=timeDict[flt][0]+self.node[flt].SFT
                
        # feasibility check for crews: check conflict/unconnected flights based on current timeDict and Qs
        # 检查机组人员的可行性：检查冲突或未连接的航班
        for Q in Qs:
            curArrTime=self.S.config["STARTTIME"]
            for flt in Q[1:-1]:
                if curArrTime+self.node[flt].CT>timeDict[flt][0]:
                    return np.inf,None,None,None
                else:
                    curArrTime=timeDict[flt][1]
        
        # the delay of itin is the actual arrival time of itin minus the schedule arrival time of itin
        # 计算行程的延误
        itinDelay=[(itin,timeDict[self.S.itin2flights[itin][-1]][1]-self.S.itin2skdtime[itin][1]) for itin in self.S.itin2flights.keys()]
        itinDelay=sorted(itinDelay,key=lambda xx:xx[1],reverse=True)
        itin2pax=self.S.itin2pax.copy()
        bothItin2pax=defaultdict(int); itin2flowin=defaultdict(int)
        for itin1, delay1 in itinDelay:
            # Case1: not rerouted
            # 情况1：没有重新分配
            minCost = self.S.config["DELAYCOST"]*itin2pax[itin1]*delay1
            minflt2 = None

            flts1 = self.S.itin2flights[itin1]
            for flt2 in self.flts:
                # condition for replaceable flt2: current arrival time of flt2 should be earlier than current arrival time of itin1
                #可更换flt2条件:flt2当前到达时间应早于itin1当前到达时间
                if self.S.itin2origin[itin1] == self.node[flt2].Ori and self.S.itin2destination[itin1] == self.node[flt2].Des and timeDict[flt2][1] < timeDict[flts1[-1]][1]:
                    # 计算该航班当前已承载的乘客数量
                    paxFlt2 = sum([itin2pax[itin] for itin in self.S.flt2skditins[flt2]])
                    # 计算该航班剩余可容纳的乘客数量
                    remain2 = self.tail2cap[flt2tail[flt2]] - paxFlt2
                    # 计算该行程可分配给当前航班的乘客数量
                    leave = min(remain2, itin2pax[itin1])
                    # 计算该行程剩余未分配的乘客数量
                    remain1 = itin2pax[itin1] - leave
                    # Case2: reroute with a late-depart flight. The replaceable flt2 has later departure time than the schedule departure time of itin1, so the leave in itin1 can be sure to catch the flt2, but they will experience arrival delay than schedule 
                    # 情况2：重新配置延误的航班，将航班的起飞时间保持不变
                    if timeDict[flt2][0] > self.S.itin2skdtime[itin1][0]:
                        # the timing of flt2 remains unchanged
                        timeFlt2 = [timeDict[flt2][0],timeDict[flt2][1]]
                        # sign operation: if positive keep it, if negative set it to 0
                        cost = self.S.config["DELAYCOST"]*leave*max(timeFlt2[1]-self.S.itin2skdtime[itin1][1],0) \
                               + self.S.config["DELAYCOST"]*remain1*delay1
                    # Case3: reroute with an early flight: # The replaceable flt2 has earlier departure time than the schedule departure time of itin1, so we need to change the time of flight2 to schedule departure time of itin1
                    else:
                        # the timing of flt2 is aligned to the schedule departure time of itin1
                        timeFlt2 = [self.S.itin2skdtime[itin1][0],self.S.itin2skdtime[itin1][0]+self.node[flt2].SFT]
                        cost = self.S.config["DELAYCOST"]*remain1*delay1 \
                               + self.S.config["DELAYCOST"]*paxFlt2*max(timeFlt2[1]-self.node[flt2].ScheduleAT,0) \
                               + self.S.config["DELAYCOST"]*leave*max(timeFlt2[1]-self.S.itin2skdtime[itin1][1],0)

                    if cost < minCost:
                        minCost = cost
                        minflt2 = flt2
                        minTimeFlt2 = timeFlt2
                        minLeave = leave

            # 如果存在可替换的航班
            if minflt2 != None:
                # 获取替换航班对应的行程
                itin2 = self.S.fltlegs2itin[minflt2]
                # 更新航班的起飞和到达时间
                timeDict[minflt2] = minTimeFlt2
                # 更新行程流入的乘客数量
                itin2flowin[itin2] += minLeave
                # 更新行程的乘客数量
                itin2pax[itin1] -= minLeave
                itin2pax[itin2] += minLeave
                # 更新行程对应的乘客数量字典
                bothItin2pax[(itin2, itin1)] = minLeave

            # feasibility check for itin1: check conflict or unconnected flights
            #检查行程的可行性
            if itin2pax[itin1]>0:
                curAt=self.S.config["STARTTIME"]
                for flt in flts1:
                    if curAt+self.node[flt].CT>timeDict[flt][0]:
                        return np.inf,None,None,None
                    else:
                        curAt=timeDict[flt][1]

            bothItin2pax[(itin1,itin1)]=itin2pax[itin1]-itin2flowin.get(itin1,0)
        
        # compute paxDict
        # 计算每个航班承载的乘客数量，并存储到字典中
        paxDict=defaultdict(int)
        for recItin,pax in itin2pax.items():
            for flt in self.S.itin2flights[recItin]:
                paxDict[flt]+=pax         
                
        # feasibility check for tail capacity
        # 检查航班容量的可行性
        for flt,tail in flt2tail.items():
            if paxDict[flt]>self.tail2cap[tail]:
                return np.inf,None,None,None

        # 计算延误成本
        delayCost = 0
        for bothItin, pax in bothItin2pax.items():
            recItin, skdItin = bothItin
            # 计算行程的实际延误时间
            delay = timeDict[self.S.itin2flights[recItin][-1]][1] - self.S.itin2skdtime[skdItin][1]
            # 如果延误时间为负数，则置为0
            if delay < 0:
                delay = 0
            # 计算延误成本
            delayCost += self.S.config["DELAYCOST"] * delay * pax

        # 计算成本
        followCost = 0
        # 计算航班的成本
        for i, skdFlts in enumerate(self.S.tail2flights.values()):
            recFlts = Ps[i][1:-1]
            followCost += self.S.config["FOLLOWSCHEDULECOST"]
            for j in range(len(skdFlts)):
                if j < len(recFlts) and skdFlts[j] == recFlts[j]:
                    followCost += self.S.config["FOLLOWSCHEDULECOST"]
        # 计算机组人员的成本
        for i, skdFlts in enumerate(self.S.crew2flights.values()):
            recFlts = Qs[i][1:-1]
            followCost += self.S.config["FOLLOWSCHEDULECOST"]
            for j in range(len(skdFlts)):
                if j < len(recFlts) and skdFlts[j] == recFlts[j]:
                    followCost += self.S.config["FOLLOWSCHEDULECOST"]
        # 计算乘客行程的成本
        for bothItin, pax in bothItin2pax.items():
            recItin, skdItin = bothItin
            # 如果行程未重新分配，则乘客行程的跟进成本为：(行程的航班数 + 1) * 每个乘客的跟进成本
            if recItin == skdItin:
                followCost += self.S.config["FOLLOWSCHEDULECOSTPAX"] * pax * (len(self.S.itin2flights[skdItin]) + 1)
        # 计算目标函数值，为延误成本和跟进成本之和
        objective = delayCost + followCost
        return objective, timeDict, bothItin2pax, paxDict, delayCost, followCost
    
    def generateVNSRecoveryPlan(self,minPs,minQs,minRes):
        objective,timeDict,bothItin2pax,paxDict,delayCost,followCost=minRes
        flt2crew={flt:self.crews[i] for i,Q in enumerate(minQs) for flt in Q[1:-1]}
        
        resd=defaultdict(list)
        for i,P in enumerate(minPs):
            for flight in P[1:-1]:
                resd["Tail"].append(self.tails[i])
                resd["Flight"].append(flight)
                resd["Crew"].append(flt2crew[flight])
                resd["From"].append(self.node[flight].Ori)
                resd["To"].append(self.node[flight].Des)
                resd["RDT"].append(timeDict[flight][0])
                resd["RAT"].append(timeDict[flight][1])
                resd["Flight_time"].append(timeDict[flight][1]-timeDict[flight][0])
                resd["Capacity"].append(self.tail2cap[self.tails[i]])
                resd["Pax"].append(paxDict[flight])
                resd["Timestring"].append(self.S.getTimeString(timeDict[flight][0])+" -> "+self.S.getTimeString(timeDict[flight][1]))
                arrdelay=round(resd["RAT"][-1]-self.S.flight2scheduleAT[flight])
                resd["Delay"].append(self.S.getTimeString(arrdelay) if arrdelay>0 else '')
                if flight in self.S.disruptedFlights:
                    resd["DelayType"].append(1)
                elif arrdelay>0:
                    resd["DelayType"].append(2)
                else:
                    resd["DelayType"].append(0)
                    
        resdItin=defaultdict(list)
        for bothItin,pax in bothItin2pax.items():
            recItin,skdItin=bothItin
            recFlts=self.S.itin2flights[recItin]
            resdItin["Rec_itin"].append(recItin)
            resdItin["Skd_itin"].append(skdItin)
            resdItin["Rec_flights"].append('-'.join(recFlts))
            resdItin["Skd_flights"].append('-'.join(self.S.itin2flights[skdItin]))
            resdItin["Pax"].append(pax)
            resdItin["From"].append(self.S.itin2origin[skdItin])
            resdItin["To"].append(self.S.itin2destination[skdItin])
            resdItin["RDT"].append(timeDict[recFlts[0]][0])
            resdItin["RAT"].append(timeDict[recFlts[-1]][1])
            delay=timeDict[recFlts[-1]][1]-self.S.itin2skdtime[skdItin][1]
            if delay<0:
                delay=0
            resdItin["Arr_delay"].append(delay)
            resdItin["Cost"].append("%.1f"%(self.S.config["DELAYCOST"]*pax*delay))
            resdItin["Rec_string"].append(self.S.getTimeString(timeDict[recFlts[0]][0])+' -> '+self.S.getTimeString(timeDict[recFlts[-1]][1]))
                    
        dfrecovery=pd.DataFrame(resd)
        dfrecovery.to_csv("Results/"+self.S.scname+"/RecoveryVNS.csv",index=None)
        dfitin=pd.DataFrame(resdItin).sort_values(by=["Rec_itin","Skd_itin"])
        dfitin.to_csv("Results/"+self.S.scname+"/ItineraryVNS.csv",index=None)
        costDict={"DelayCost":delayCost,"ExtraFuelCost":0.,"CancelCost":0.,"FollowGain":followCost,"Objective":objective}
        with open("Results/%s/CostVNS.json"%(self.S.scname), "w") as outfile:
            json.dump(costDict, outfile, indent = 4)
    
    def getStringDistribution(self,Xs):
        Xweights=np.array([np.mean([self.node2weight[fnode] for fnode in x[1:-1]]) for x in Xs])
        if self.baseline!="uniform":
           Xweights=(Xweights-Xweights.min())/(Xweights.max()-Xweights.min())
        Xdist=np.exp(Xweights)/np.exp(Xweights).sum()
        return Xdist
    
    def transformStrings(self,k,curXs,Xdist):
        Xind1,Xind2=np.random.choice(np.arange(len(curXs)),size=2,replace=False,p=Xdist)
        Xpairs=eval("self."+self.k2func[k])(curXs[Xind1],curXs[Xind2])
        if len(Xpairs)>=1:
            nX1,nX2=random.choice(Xpairs)
            nXs=curXs.copy()
            nXs[Xind1],nXs[Xind2]=nX1,nX2
        else:
            nXs=curXs
        return nXs
        
    def exploreNeighborK(self,k,curPs,curQs,Pdist,Qdist):
        # sample the pair based on metrics, without enumeration through operating choices
        if self.enumFlag==False:            
            nPs=self.transformStrings(k,curPs,Pdist)
            nQs=self.transformStrings(k,curQs,Qdist)
            nRes=self.evaluate(nPs,nQs)
            return nPs,nQs,nRes

        # random sample the pair + full enumeration through operating choices
        else:
            curRes=self.evaluate(curPs,curQs)
            Pind1,Pind2=random.sample(range(len(curPs)),2)
            for (nP1,nP2) in eval("self."+self.k2func[k])(curPs[Pind1],curPs[Pind2]):
                nPs=curPs.copy()
                nPs[Pind1],nPs[Pind2]=nP1,nP2
                for i in range(len(curQs)):
                    Qind1,Qind2=random.sample(range(len(curQs)),2)        
                    for (nQ1,nQ2) in eval("self."+self.k2func[k])(curQs[Qind1],curQs[Qind2]):
                        nQs=curQs.copy()
                        nQs[Qind1],nQs[Qind2]=nQ1,nQ2
                        nRes=self.evaluate(nPs,nQs)
                        if nRes[0]<curRes[0]:
                            curPs,curQs,curRes=nPs,nQs,nRes
            return curPs,curQs,curRes    

    def VNS(self,trajLen):
        minPs,minQs=self.skdPs,self.skdQs
        minRes=self.evaluate(minPs,minQs)
        for i in range(trajLen):
            k=0
            while k<=3:
                Pdist=self.getStringDistribution(minPs)
                Qdist=self.getStringDistribution(minQs)
                curPs,curQs,curRes=self.exploreNeighborK(k,minPs,minQs,Pdist,Qdist)
                if curRes[0]<minRes[0]:
                    minPs,minQs,minRes=curPs,curQs,curRes
                else:
                    k+=1
        return minPs,minQs,minRes

def runVNSWithEnumSaveSolution(config):
    S=Scenario(config["DATASET"],config["SCENARIO"],"PAX")
    solver=VNSSolver(S,0,config["BASELINE"],config["ENUMFLAG"])
    res=solver.VNS(config["TRAJLEN"])
    print(res[2][0])
    solver.generateVNSRecoveryPlan(*res)
    
def runVNS(par):
    
    config={"DATASET": "ACF%d"%par[0],
            "SCENARIO": "ACF%d-SC%c"%(par[0],par[1]),
            "BASELINE": par[2],
            "TRAJLEN": 10,
            "ENUMFLAG": False,
            "EPISODES": 100
            }
    
    if not os.path.exists("Results"):
        os.makedirs("Results")
    if not os.path.exists("Results/%s"%config["SCENARIO"]):
        os.makedirs("Results/%s"%config["SCENARIO"])
        
    res=[]
    times=[]
    S=Scenario(config["DATASET"],config["SCENARIO"],"PAX")
    for seed in range(config["EPISODES"]):
        T1=time.time()
        solver=VNSSolver(S,seed,config["BASELINE"],config["ENUMFLAG"])
        objective=solver.VNS(config["TRAJLEN"])[2][0]
        T2=time.time()
        res.append(objective)
        times.append(T2-T1)
        print(config["SCENARIO"],'episode: {:>3}'.format(seed), ' objective: {:>6.1f} '.format(objective),'best:',"%.1f"%min(res))

    np.savez_compressed('Results/%s/res_%s'%(config["SCENARIO"],config["BASELINE"]),res=res)
    np.savez_compressed('Results/%s/time_%s'%(config["SCENARIO"],config["BASELINE"]),res=times)

if __name__ == '__main__':
    
    p=multiprocessing.Pool()
    todo=[(i,typ,m) for i in range(5,25,5) for typ in ['p','m'] for m in ["degree","uniform","distance"]]
    t1=time.time()
    p=multiprocessing.Pool()
    for i,res in enumerate(p.imap_unordered(runVNS,todo),1):
        t2=time.time()
        print("%0.2f%% done, time to finish: ~%0.1f minutes"%(i*100/len(todo),((t2-t1)/(60*(i+1)))*len(todo)-((t2-t1)/60)))
            
