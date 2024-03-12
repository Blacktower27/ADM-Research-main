import shutil
import time
import pandas as pd
from Scenarios.ScenarioGenerator import ScenarioGenerator
from ModelExecutor import mainModelExecutor
from ResultAnalyzer import mainResultAnalyzer
import random
import multiprocessing

def createScenario(dataset,scenario,k,seed=0):
    SC=ScenarioGenerator(dataset,scenario,seed)
    delayinfo=SC.getRandomFlightDelay(k)
    SC.setFlightDepartureDelay(delayinfo)
    SC.setDelayedReadyTime({})

def main(dataset,scenario,k,seed=0):
    SC=ScenarioGenerator(dataset,scenario,seed)
    delayinfo=SC.getRandomFlightDelay(k)
    SC.setFlightDepartureDelay(delayinfo)
    SC.setDelayedReadyTime({})
    try:
        mainModelExecutor(dataset,scenario)
        mainResultAnalyzer(dataset,scenario)
    except: # cplex no solution error
        shutil.rmtree("Scenarios/"+scenario)
        shutil.rmtree("Results/"+scenario)
        return False
    return True

def mainWithoutExcept(dataset,scenario,seed=0):
    SC=ScenarioGenerator(dataset,scenario,seed)
    delayinfo=SC.getRandomFlightDelay(2)
    SC.setFlightDepartureDelay(delayinfo)
    SC.setDelayedReadyTime({})
    mainModelExecutor(dataset,scenario)
    mainResultAnalyzer(dataset,scenario)

def runCPLEX(dataset,scenario):
    mainModelExecutor(dataset,scenario)
    mainResultAnalyzer(dataset,scenario)


#    main("ACF5","ACF5-SCm",0.1,17)
#    main("ACF25","ACF25-SCm",0.1,3)
    
if __name__ == '__main__':

    # createScenario("ACF5","ACF5-SCp",0.3,114)
    # createScenario("ACF20", "ACF20-SCp", 0.3, 114)
    # main("ACF10","ACF10-SCp",0.3,114)#no solution
    # main("ACF15","ACF15-SCp",0.3,114)#no solution
    # main("ACF20","ACF20-SCp",0.3,114)#no solution
    main("ACF25","ACF25-SCp",0.3,42)
    # todo=[(i,typ) for i in range(25,30,5) for typ in ['p','m']]
    # for i, j in todo:
    #     if j=='p':
    #         k=0.3
    #     else:
    #         k=0.1
    #     main("ACF%d"%i,"ACF%d-SC%c"%(i,j),k,42)

    # main("ACF%d"%,"ACF%d-SC%c"(),0.1,114)
    # for i,res in enumerate(p.imap_unordered(runVNS,todo),1):

#p就是论文中的DS+
#m就是论文中的DS-

