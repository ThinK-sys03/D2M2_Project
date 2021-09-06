# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 15:13:00 2021

@author: lenovo
"""

import numpy as np
import pandas as pd
import time

def Assigninfectionstate(InfectedAgent, p, k):
    AgentState = np.random.rand(len(InfectedAgent))
    AgentState[AgentState > 1-p] = 2 # 2 denotes asy
    AgentState[AgentState < (1-p)*k] = 0 # 0 denotes pre-trans
    AgentState[(AgentState != 0) & (AgentState != 2)] = 1 # normal infections
    return AgentState

def truncated_Poisson(mu, min_value, size):
    temp_size = size
    while True:
        temp_size *= 2
        temp = np.random.poisson(mu, size=temp_size)
        truncated = temp[temp >= min_value]
        if len(truncated) >= size:
            return truncated[:size]

def Assigninfectiondate(Infectedstate, Epsilon, Epsilon_asy, Gamma, mu):
    Latentperiod = np.zeros(len(Infectedstate))
    Infectiousperiod = np.random.poisson(mu, len(Infectedstate))
    Asy_latentperiod = np.random.poisson(Epsilon_asy, np.sum(Infectedstate==2))
    Latentperiod[Infectedstate==2] = Asy_latentperiod
    Sy_Latentperiod = truncated_Poisson((Epsilon+Gamma), Epsilon ,np.sum(Infectedstate!=2))
    Latentperiod[Infectedstate!=2] = Sy_Latentperiod
    Latentperiod[Infectedstate == 0] -= 2 
    return Latentperiod, Infectiousperiod

def insertunbalancedfamilycount(HouseID, AsagentID, HouseAgents):    
    nouseid, temp_loc, nouseloc = np.intersect1d(HouseAgents[:,0], AsagentID, return_indices=True, assume_unique=True)
    Temp_asyhouseID, temp_AsyFamilyactivenumber = np.unique(HouseAgents[temp_loc, 6].astype(int), return_counts = True)
    nouseid, insert, nouseind = np.intersect1d(HouseID[:,0], Temp_asyhouseID, assume_unique=True, return_indices=True)
    AsyFamilyactivenumber = np.zeros(len(HouseID))
    AsyFamilyactivenumber[insert] = temp_AsyFamilyactivenumber    
    return AsyFamilyactivenumber

def insertunbalancedgridcount(UnionIndex, AsagentID, HouseAgents):
    
    nouseid, temp_loc, nouseloc = np.intersect1d(HouseAgents[:,0], AsagentID, return_indices=True, assume_unique=True)
    Temp_asygridID, temp_Asygridactivenumber = np.unique(HouseAgents[temp_loc, 1].astype(int), return_counts = True)
    nouseid, insert, nouseind = np.intersect1d(UnionIndex, Temp_asygridID, assume_unique=True, return_indices=True)
    Asygridactivenumber = np.zeros(len(UnionIndex))
    Asygridactivenumber[insert] = temp_Asygridactivenumber
    
    return Asygridactivenumber

def getInfectiousgriddata(InfectiousAgent, Infectedstate, HouseAgents, TotalGrids, HouseWorker):
    """    
    Get the data for family grid trans

    Returns
    -------
    UsedFamilydata: int32
        0: Infectious grid ID
        1: active infectious count
        2: asymptomatic infectious count
        3: active outworker count
        4: active outstudent count
        5: active domestic helper count
        6: asymptomatic outworker count
        7: asymptomatic outstudent count
        8: asymptomatic domestic helper count
        9: total grid population
        10: total outworker count
        11: total outstudent count
        12: total domestic helper count
        13: the total rest family member count
        14: the workers who working here
        15: the active infectious workers here
        16: the asymptomatically infectious workers here
    """
    AgentID, Agent_ind, Agent_indloc = np.intersect1d(HouseAgents[:,0], InfectiousAgent, return_indices=True) # get the active infectious individuals 
    AsagentID = np.intersect1d(HouseAgents[:,0], InfectiousAgent[Infectedstate == 2]) # get the asy infectious individuals
    ActiveworkerID = np.intersect1d(HouseAgents[HouseAgents[:,7]==1,0], InfectiousAgent) # get the infectious outworkers
    ActivestudentID = np.intersect1d(HouseAgents[HouseAgents[:,8]==1,0], InfectiousAgent) # get the infectious students
    ActiveUnistudentID = np.intersect1d(Unistudent[:,0], InfectiousAgent)
    ActivestudentID = np.unique(np.r_[ActivestudentID, ActiveUnistudentID]) # UpdatedUnistudent
    ActivedhelperID = np.intersect1d(HouseAgents[HouseAgents[:,9]==1,0], InfectiousAgent) # get the infectious students
    AsyactiveworkerID = np.intersect1d(HouseAgents[HouseAgents[:,7]==1,0], InfectiousAgent[Infectedstate == 2]) # get the asy_infectious outworkers
    AsyactivestudentID = np.intersect1d(HouseAgents[HouseAgents[:,8]==1,0], InfectiousAgent[Infectedstate == 2]) # get the asy_infectious students
    AsyactivedhelperID = np.intersect1d(HouseAgents[HouseAgents[:,9]==1,0], InfectiousAgent[Infectedstate == 2]) # get the asy_infectious students
    AsyactiveUnistudentID = np.intersect1d(Unistudent[:,0], InfectiousAgent[Infectedstate == 2])
    AsyactivestudentID = np.unique(np.r_[AsyactivestudentID, AsyactiveUnistudentID])
    temp_activedesworkerID, temp_ind, nouse = np.intersect1d(HouseWorker[:,0], ActiveworkerID, return_indices=True) # get the location infectious outworkers
    activedesworkergridID, activedesworkergridcount = np.unique(HouseWorker[temp_ind, 12], return_counts = True)
    temp_asydesworkerID, temp_asyind, nouse = np.intersect1d(HouseWorker[:,0], AsyactiveworkerID, return_indices=True) # get the location infectious outworkers
    asydesworkergridID, asydesworkergridcount = np.unique(HouseWorker[temp_asyind, 12], return_counts = True)
    
    Temp_GridID, Temp_gridactivenumber = np.unique(HouseAgents[Agent_ind, 1].astype(int), return_counts=True) # get the grid ID and infectious number
    GridID = np.c_[Temp_GridID, Temp_gridactivenumber]
    GridID = GridID[GridID[:,0].argsort()]
    UnionIndex = np.union1d(activedesworkergridID, GridID[:,0]).astype(int) # the grid of working and live may not match
    nouse, forgrid, nouse = np.intersect1d(UnionIndex, GridID[:,0], assume_unique=True, return_indices=True)
    nouse, forworkinghere, nouse = np.intersect1d(UnionIndex, activedesworkergridID, assume_unique=True, return_indices=True)
    nouse, forasyworkinghere, nouse = np.intersect1d(UnionIndex, asydesworkergridID, assume_unique=True, return_indices=True)
    activeworkinghere = np.zeros(len(UnionIndex))
    activeworkinghere[forworkinghere] = activedesworkergridcount
    asyworkinghere = np.zeros(len(UnionIndex))
    asyworkinghere[forasyworkinghere] = asydesworkergridcount
    gridactivenumber = np.zeros(len(UnionIndex))
    gridactivenumber[forgrid] = Temp_gridactivenumber
    
    Gridsize = TotalGrids[UnionIndex, 3]
    Gridoutworkersize = TotalGrids[UnionIndex, 1]
    Gridoutstudentsize = TotalGrids[UnionIndex, 2]
    Griddomestic = TotalGrids[UnionIndex, 4]
    Gridhomeagents = Gridsize - Gridoutworkersize - Gridoutstudentsize - Griddomestic # total - 1,2,3,4
    Griddesworkersize = TotalGrids[UnionIndex, 5] # the count of workers working here
    Gridasyactivenumber = insertunbalancedgridcount(UnionIndex, AsagentID, HouseAgents)
    Gridactiveworkernumber = insertunbalancedgridcount(UnionIndex, ActiveworkerID, HouseAgents)
    Gridactivestudentnumber = insertunbalancedgridcount(UnionIndex, ActivestudentID, HouseAgents)
    Gridactivedhelpernumber = insertunbalancedgridcount(UnionIndex, ActivedhelperID, HouseAgents)
    Gridasyactiveworkernumber = insertunbalancedgridcount(UnionIndex, AsyactiveworkerID, HouseAgents)
    Gridasyactivestudentnumber = insertunbalancedgridcount(UnionIndex, AsyactivestudentID, HouseAgents)
    Gridasyactivedhelpernumber = insertunbalancedgridcount(UnionIndex, AsyactivedhelperID, HouseAgents)
        
    UsedGriddata = np.c_[UnionIndex,gridactivenumber,Gridasyactivenumber,Gridactiveworkernumber,Gridactivestudentnumber,Gridactivedhelpernumber,Gridasyactiveworkernumber,Gridasyactivestudentnumber,Gridasyactivedhelpernumber,Gridsize,Gridoutworkersize,Gridoutstudentsize,Griddomestic,Gridhomeagents,Griddesworkersize,activeworkinghere,asyworkinghere].astype(int)    
    return UsedGriddata

def insertunbalancedTPUcount(TdataID, AsagentID, HouseAgents, TotalGrids): 
    temp_loc = np.intersect1d(HouseAgents[:,0], AsagentID, return_indices=True, assume_unique=True)[1]
    Temp_asyGridID, temp_AsyGridactivenumber = np.unique(HouseAgents[temp_loc, 1].astype(int), return_counts = True)
    Temp_asyTPUID = TotalGrids[Temp_asyGridID, 6].astype(int)  
    TPUtempdata = pd.DataFrame([Temp_asyTPUID, temp_AsyGridactivenumber]).T
    Temp_TPUactivenumber = TPUtempdata.groupby(by=[0]).sum().values
    Temp_TdataID = TPUtempdata.groupby(by=[0]).sum().index.values
    asyTdataID = np.c_[Temp_TdataID, Temp_TPUactivenumber]
    AsyTPUactivenumber = np.zeros(len(TdataID))
    insert= np.intersect1d(TdataID[:,0], asyTdataID[:,0], return_indices=True)[1]
    AsyTPUactivenumber[insert] = asyTdataID[:,1]
    return AsyTPUactivenumber

def getInfectiousfamilydata(InfectiousAgent, Infectedstate, HouseAgents, TotalHouse):
    """    
    Get the data for family trans

    Returns
    -------
    UsedFamilydata: int32
        0: Infectious family ID
        1: family active number
        2: asymptomatic infectious count
        3: active infectious count
        4: active outworker count
        5: active outstudent count
        6: active domestic helper count
        7: asymptomatic outworker count
        8: asymptomatic outstudent count
        9: asymptomatic domestic helper count
        10: total family size
        11: total outworker count
        12: total outstudent count
        13: total domestic helper count
        14: the total rest family member count
    """
    AgentID, Agent_ind, Agent_indloc = np.intersect1d(HouseAgents[:,0], InfectiousAgent, return_indices=True) # get the active infectious individuals 
    AsagentID = np.intersect1d(HouseAgents[:,0], InfectiousAgent[Infectedstate == 2]) # get the asy infectious individuals
    ActiveworkerID = np.intersect1d(HouseAgents[HouseAgents[:,7]==1,0], InfectiousAgent) # get the infectious outworkers
    ActivestudentID = np.intersect1d(HouseAgents[HouseAgents[:,8]==1,0], InfectiousAgent) # get the infectious students
    ActivedhelperID = np.intersect1d(HouseAgents[HouseAgents[:,9]==1,0], InfectiousAgent) # get the infectious students
    AsyactiveworkerID = np.intersect1d(HouseAgents[HouseAgents[:,7]==1,0], InfectiousAgent[Infectedstate == 2]) # get the asy_infectious outworkers
    AsyactivestudentID = np.intersect1d(HouseAgents[HouseAgents[:,8]==1,0], InfectiousAgent[Infectedstate == 2]) # get the asy_infectious students
    AsyactivedhelperID = np.intersect1d(HouseAgents[HouseAgents[:,9]==1,0], InfectiousAgent[Infectedstate == 2]) # get the asy_infectious students
    Temp_HouseID, Temp_familyactivenumber = np.unique(HouseAgents[Agent_ind, 6].astype(int), return_counts=True) # get the house ID and infectious number
    HouseID = np.c_[Temp_HouseID, Temp_familyactivenumber]
    HouseID = HouseID[HouseID[:,0].argsort()] 
    Familysize = TotalHouse[HouseID[:,0], 1]
    Familyoutworkersize = TotalHouse[HouseID[:,0], 2]
    Familyoutstudentsize = TotalHouse[HouseID[:,0], 3]
    Familydomestic = TotalHouse[HouseID[:,0], 4]
    Familyhomeagents = TotalHouse[HouseID[:,0], 5] # total - 1,2,3,4
    Familyasyactivenumber = insertunbalancedfamilycount(HouseID, AsagentID, HouseAgents)
    Familyactiveworkernumber = insertunbalancedfamilycount(HouseID, ActiveworkerID, HouseAgents)
    Familyactivestudentnumber = insertunbalancedfamilycount(HouseID, ActivestudentID, HouseAgents)
    Familyactivedhelpernumber = insertunbalancedfamilycount(HouseID, ActivedhelperID, HouseAgents)
    Familyasyactiveworkernumber = insertunbalancedfamilycount(HouseID, AsyactiveworkerID, HouseAgents)
    Familyasyactivestudentnumber = insertunbalancedfamilycount(HouseID, AsyactivestudentID, HouseAgents)
    Familyasyactivedhelpernumber = insertunbalancedfamilycount(HouseID, AsyactivedhelperID, HouseAgents)
    UsedFamilydata = np.c_[HouseID,Familyasyactivenumber,Familyactiveworkernumber,Familyactivestudentnumber,Familyactivedhelpernumber,Familyasyactiveworkernumber,Familyasyactivestudentnumber,Familyasyactivedhelpernumber,Familysize,Familyoutworkersize,Familyoutstudentsize,Familydomestic,Familyhomeagents].astype(int)    
    return UsedFamilydata

def getInfectiousTPUdata(InfectiousAgent, Infectedstate, HouseAgents, TPUdata, TotalGrids):
    AgentID, Agent_ind, Agent_indloc = np.intersect1d(HouseAgents[:,0], InfectiousAgent, return_indices=True) # get the active infectious individuals 
    AsagentID = np.intersect1d(HouseAgents[:,0], InfectiousAgent[Infectedstate == 2]) # get the asy infectious individuals
    ActiveworkerID = np.intersect1d(HouseAgents[HouseAgents[:,7]==1,0], InfectiousAgent) # get the infectious outworkers
    ActivestudentID = np.intersect1d(HouseAgents[HouseAgents[:,8]==1,0], InfectiousAgent) # get the infectious students
    ActivedhelperID = np.intersect1d(HouseAgents[HouseAgents[:,9]==1,0], InfectiousAgent) # get the infectious students
    AsyactiveworkerID = np.intersect1d(HouseAgents[HouseAgents[:,7]==1,0], InfectiousAgent[Infectedstate == 2]) # get the asy_infectious outworkers
    AsyactivestudentID = np.intersect1d(HouseAgents[HouseAgents[:,8]==1,0], InfectiousAgent[Infectedstate == 2]) # get the asy_infectious students
    AsyactivedhelperID = np.intersect1d(HouseAgents[HouseAgents[:,9]==1,0], InfectiousAgent[Infectedstate == 2]) # get the asy_infectious students
    Temp_GridID, Temp_Gridactivenumber = np.unique(HouseAgents[Agent_ind, 1].astype(int), return_counts=True) # get the Grid ID and infectious number
    Temp_TPUID = TotalGrids[Temp_GridID, 6].astype(int)    
    TPUtempdata = pd.DataFrame([Temp_GridID, Temp_TPUID, Temp_Gridactivenumber]).T
    Temp_TPUactivenumber = TPUtempdata.groupby(by=[1]).sum().values
    Temp_TdataID = TPUtempdata.groupby(by=[1]).sum().index.values
    TdataID = np.c_[Temp_TdataID, Temp_TPUactivenumber]
    TdataID = np.delete(TdataID, 1, axis=1)
    TPUsize = TPUdata[np.intersect1d(TPUdata[:,0], TdataID[:,0],return_indices=True)[1], 3] # total population
    TPUoutworkersize = TPUdata[np.intersect1d(TPUdata[:,0], TdataID[:,0],return_indices=True)[1], 1] # total worker
    TPUoutstudentsize = TPUdata[np.intersect1d(TPUdata[:,0], TdataID[:,0],return_indices=True)[1], 2] # total student
    TPUdomestic = TPUdata[np.intersect1d(TPUdata[:,0], TdataID[:,0],return_indices=True)[1], 4] # total domestic helper
    TPUhomeagents = TPUsize - TPUoutworkersize - TPUoutstudentsize - TPUdomestic # total - 1,2,3,4
    Familyasyactivenumber = insertunbalancedTPUcount(TdataID, AsagentID, HouseAgents,TotalGrids)
    Familyactiveworkernumber = insertunbalancedTPUcount(TdataID, ActiveworkerID, HouseAgents,TotalGrids)
    Familyactivestudentnumber = insertunbalancedTPUcount(TdataID, ActivestudentID, HouseAgents,TotalGrids)
    Familyactivedhelpernumber = insertunbalancedTPUcount(TdataID, ActivedhelperID, HouseAgents,TotalGrids)
    Familyasyactiveworkernumber = insertunbalancedTPUcount(TdataID, AsyactiveworkerID, HouseAgents,TotalGrids)
    Familyasyactivestudentnumber = insertunbalancedTPUcount(TdataID, AsyactivestudentID, HouseAgents,TotalGrids)
    Familyasyactivedhelpernumber = insertunbalancedTPUcount(TdataID, AsyactivedhelperID, HouseAgents,TotalGrids)
    UsedTPUdata = np.c_[TdataID,Familyasyactivenumber,Familyactiveworkernumber,Familyactivestudentnumber,Familyactivedhelpernumber,Familyasyactiveworkernumber,Familyasyactivestudentnumber,Familyasyactivedhelpernumber,TPUsize,TPUoutworkersize,TPUoutstudentsize,TPUdomestic,TPUhomeagents].astype(int)    
    return UsedTPUdata

def HouseInfectiousProcess(beta, InfectiousAgent, Infectedstate, HouseAgents, TotalHouse, InfectedAgentRec, weekdayind, index,varying_beta_asy):
    st = time.time()
    HI = 0 # count the family infected agents
    TodayHouseInfected = np.zeros(1) # record the infected agents
    Infectiousfamilydata = getInfectiousfamilydata(InfectiousAgent, Infectedstate, HouseAgents, TotalHouse)  
    if weekdayind == 2: # weekends 50% people go out
        outfamilysize = Infectiousfamilydata[:,12] # Familyhomeagents
        restfamilysize = Infectiousfamilydata[:,9] - Infectiousfamilydata[:,12]
        act_outfamilysize = Infectiousfamilydata[:,5]
        act_restfamilysize = Infectiousfamilydata[:,1] - Infectiousfamilydata[:,5]
        asy_outfamilysize = Infectiousfamilydata[:,8]
        asy_restfamilysize = Infectiousfamilydata[:,2] - Infectiousfamilydata[:,8]
        HouseCaseProb = 15.4/20.2 * beta * (13*(act_outfamilysize-varying_beta_asy*asy_outfamilysize) + 22*(act_restfamilysize-varying_beta_asy*asy_restfamilysize)) / (13*outfamilysize + 22*restfamilysize) * (100+index)/100
    elif weekdayind == 1:
        outfamilysize = np.zeros(len(Infectiousfamilydata[:,0]))
        restfamilysize = Infectiousfamilydata[:,9]
        act_outfamilysize = np.zeros(len(Infectiousfamilydata[:,0]))
        act_restfamilysize = Infectiousfamilydata[:,1]
        asy_outfamilysize = np.zeros(len(Infectiousfamilydata[:,0]))
        asy_restfamilysize = Infectiousfamilydata[:,2]
        HouseCaseProb = 15.4/20.2 * beta * (13*(act_outfamilysize-varying_beta_asy*asy_outfamilysize) + 22*(act_restfamilysize-varying_beta_asy*asy_restfamilysize)) / (13*outfamilysize + 22*restfamilysize) * (100+index)/100
    elif weekdayind == 0:
        outfamilysize = Infectiousfamilydata[:,10] + Infectiousfamilydata[:,11]
        restfamilysize = Infectiousfamilydata[:,9] - outfamilysize
        act_outfamilysize = Infectiousfamilydata[:,3] + Infectiousfamilydata[:,4]
        act_restfamilysize = Infectiousfamilydata[:,1] - act_outfamilysize
        asy_outfamilysize = Infectiousfamilydata[:,6] + Infectiousfamilydata[:,7]
        asy_restfamilysize = Infectiousfamilydata[:,2] - asy_outfamilysize
        HouseCaseProb = 13.7/22.3 * beta * (13*(act_outfamilysize-varying_beta_asy*asy_outfamilysize) + 22*(act_restfamilysize-varying_beta_asy*asy_restfamilysize)) / (13*outfamilysize + 22*restfamilysize) * (100+index)/100
    if np.min(HouseCaseProb) < 0:
        print('sth wrong in the house trans prob,', np.argmin(HouseCaseProb))
    
    InfectedHouseID, InfectedFamilySize = np.unique(HouseAgents[np.intersect1d(HouseAgents[:,0], InfectedAgentRec, return_indices=True)[1], 6], return_counts=True)
    matchedInfectedfamilysize = np.zeros(len(Infectiousfamilydata))
    matchedInfectedfamilysizeind = np.intersect1d(InfectedHouseID, Infectiousfamilydata[:,0], return_indices=True)[2]
    matchedInfectedfamilysize[matchedInfectedfamilysizeind] = InfectedFamilySize[np.intersect1d(InfectedHouseID, Infectiousfamilydata[:,0], return_indices=True)[1]]
    SupectedFamilySize = (Infectiousfamilydata[:,9] - matchedInfectedfamilysize).astype(int)
    if np.min(SupectedFamilySize) < 0:
        print('sth wrong in the house suspected number,', np.argmin(SupectedFamilySize))    
    
    aboveonesize = np.sum(HouseCaseProb > 1)
    if aboveonesize > 0:
        print('House prob above one', aboveonesize)
        HouseCaseProb[HouseCaseProb>1] = 1
    Familyinfectednumber = np.random.binomial(SupectedFamilySize, HouseCaseProb)
    for i in range(len(Familyinfectednumber)):
        FamilymemberID = HouseAgents[(HouseAgents[:, 6] == Infectiousfamilydata[i,0]), 0]
        FamilysuspectedID = np.setdiff1d(FamilymemberID, InfectedAgentRec)
        if Familyinfectednumber[i] > 0:
            NewHouseinfectedID = np.random.choice(FamilysuspectedID,  Familyinfectednumber[i], replace=False)
            TodayHouseInfected = np.append(TodayHouseInfected, NewHouseinfectedID)
    HI += np.sum(Familyinfectednumber)
    TodayHouseInfected = np.delete(TodayHouseInfected, 0)
    TodayHouseInfected = np.unique(TodayHouseInfected).astype(int)
    nouse, nouse, forind= np.intersect1d(TodayHouseInfected, HouseAgents[:,0], assume_unique=True, return_indices=True)
    TodayHouseInfectedloc = HouseAgents[forind, 1]
    et = time.time()
    print("House using time:", et-st)
    return TodayHouseInfected, TodayHouseInfectedloc, HI

def CommunityInfectiousProcess(beta, InfectiousAgent, Infectedstate, HouseAgents, InfectedAgentRec, TPUdata, TotalGrids, weekdayind, index, varying_beta_asy):
    st = time.time()
    CI = 0 # count the Community infected agents
    TodayCommunityInfected = np.zeros(1) # record the infected agents
    InfectiousTPUdata = getInfectiousTPUdata(InfectiousAgent, Infectedstate, HouseAgents, TPUdata, TotalGrids)
    if weekdayind == 2: # weekends 50% people go out
        outTPUsize = InfectiousTPUdata[:,12]
        restTPUsize = InfectiousTPUdata[:,9] - InfectiousTPUdata[:,12]
        act_outTPUsize = InfectiousTPUdata[:,5]
        act_restTPUsize = InfectiousTPUdata[:,1] - InfectiousTPUdata[:,5]
        asy_outTPUsize = InfectiousTPUdata[:,8]
        asy_restTPUsize = InfectiousTPUdata[:,2] - InfectiousTPUdata[:,8]
        TPUCaseProb = (15.4-1.5+0.8)/20.2 * beta * (13*(act_outTPUsize-varying_beta_asy*asy_outTPUsize) + 22*(act_restTPUsize-varying_beta_asy*asy_restTPUsize)) / (13*outTPUsize + 22*restTPUsize) * (100+index)/100
    elif weekdayind == 1:
        outTPUsize = np.zeros(len(InfectiousTPUdata[:,0]))
        restTPUsize = InfectiousTPUdata[:,9]
        act_outTPUsize = np.zeros(len(InfectiousTPUdata[:,0]))
        act_restTPUsize = InfectiousTPUdata[:,1]
        asy_outTPUsize = np.zeros(len(InfectiousTPUdata[:,0]))
        asy_restTPUsize = InfectiousTPUdata[:,2]
        TPUCaseProb = (15.4-1.5+0.8)/20.2 * beta * (13*(act_outTPUsize-varying_beta_asy*asy_outTPUsize) + 22*(act_restTPUsize-varying_beta_asy*asy_restTPUsize)) / (13*outTPUsize + 22*restTPUsize) * (100+index)/100
    elif weekdayind == 0:
        outTPUsize = InfectiousTPUdata[:,10] + InfectiousTPUdata[:,11]
        restTPUsize = InfectiousTPUdata[:,9] - outTPUsize
        act_outTPUsize = InfectiousTPUdata[:,3] + InfectiousTPUdata[:,4]
        act_restTPUsize = InfectiousTPUdata[:,1] - act_outTPUsize
        asy_outTPUsize = InfectiousTPUdata[:,6] + InfectiousTPUdata[:,7]
        asy_restTPUsize = InfectiousTPUdata[:,2] - asy_outTPUsize
        TPUCaseProb = (13.7-1.1+0.9)/22.3 * beta * (13*(act_outTPUsize-varying_beta_asy*asy_outTPUsize) + 22*(act_restTPUsize-varying_beta_asy*asy_restTPUsize)) / (13*outTPUsize + 22*restTPUsize) * (100+index)/100
    else:
        print('wrong weekday')
    if np.min(TPUCaseProb) < 0:
        print('sth wrong in the Community trans prob,', np.argmin(TPUCaseProb)) 

    InfectedTPUID, InfectedTPUSize = np.unique(HouseAgents[np.intersect1d(HouseAgents[:,0], InfectedAgentRec, return_indices=True)[1], 10], return_counts=True)
    matchedInfectedTPUsize = np.zeros(len(InfectiousTPUdata))
    matchedInfectedTPUsizeind = np.intersect1d(InfectedTPUID, InfectiousTPUdata[:,0], return_indices=True)[2]
    matchedInfectedTPUsize[matchedInfectedTPUsizeind] = InfectedTPUSize[np.intersect1d(InfectedTPUID, InfectiousTPUdata[:,0], return_indices=True)[1]]
    SupectedTPUSize = (InfectiousTPUdata[:,9] - matchedInfectedTPUsize).astype(int)    
    if np.min(SupectedTPUSize) < 0:
        print('sth wrong in the Community suspected number,', np.argmin(SupectedTPUSize),InfectiousTPUdata[np.argmin(SupectedTPUSize),9], matchedInfectedTPUsize[np.argmin(SupectedTPUSize)])
    
    Communityinfectednumber = np.random.binomial(SupectedTPUSize, TPUCaseProb)
    for i in range(len(Communityinfectednumber)):
        TPUymemberID = HouseAgents[(HouseAgents[:, 10] == InfectiousTPUdata[i,0]), 0]
        TPUsuspectedID = np.setdiff1d(TPUymemberID, InfectedAgentRec)
        if Communityinfectednumber[i] > 0:
            NewHouseinfectedID = np.random.choice(TPUsuspectedID,  Communityinfectednumber[i], replace=False)
            TodayCommunityInfected = np.append(TodayCommunityInfected, NewHouseinfectedID)
    CI += np.sum(Communityinfectednumber)    
    TodayCommunityInfected = np.delete(TodayCommunityInfected, 0)
    TodayCommunityInfected = np.unique(TodayCommunityInfected).astype(int)
    forind= np.intersect1d(TodayCommunityInfected, HouseAgents[:,0], assume_unique=True, return_indices=True)[2]
    TodayCommunityInfectedloc = HouseAgents[forind, 1]
    et = time.time()
    print("Community using time:", et-st)
    return TodayCommunityInfected, TodayCommunityInfectedloc, CI

def OfficeInfectiousProcess(BasicProb, TotalGrids, HouseAgents, HouseWorker, InfectiousAgent, Infectedstate, InfectedAgentRec, index, varying_beta_asy):    
    st = time.time()
    OI = 0
    TodayOfficeInfected = np.zeros(1)
    WorkerinfectiousID, nouse, infectiousworker_ind = np.intersect1d(InfectiousAgent, HouseWorker[:, 0], return_indices=True)
    AsyworkerinfectedID, Inf_ind, asyworker_ind = np.intersect1d(InfectiousAgent[Infectedstate == 2], HouseWorker[:, 0], return_indices=True)
    infectedOfficeID, officeactivecase = np.unique(HouseWorker[infectiousworker_ind, 11], return_counts=True)
    asyinfectedOfficeID, officeasyactivecase = np.unique(HouseWorker[asyworker_ind, 11], return_counts=True)
    insertasy_ind = np.intersect1d(infectedOfficeID, asyinfectedOfficeID, return_indices=True,assume_unique=True)[1]
    asyofficeactivecase = np.zeros(len(officeactivecase))
    asyofficeactivecase[insertasy_ind] = officeasyactivecase
    totalOfficeID, infectedofficetotalsize = np.unique(HouseWorker[np.isin(HouseWorker[:, 11], infectedOfficeID), 11], return_counts=True)
    infectedOfficeID = np.c_[infectedOfficeID, officeactivecase, infectedofficetotalsize, asyofficeactivecase] # 1st for officeID, 2nd for active casenumber, 3rd for total size, 4th for asy infections
    OfficeCaseProb = (7.4/22.7) * BasicProb *(officeactivecase-varying_beta_asy*asyofficeactivecase)/infectedofficetotalsize * (100+index)/100
    OfficeCaseProb[OfficeCaseProb<0] = 0
    
    for i in range(len(infectedOfficeID)): # search the infection from workplace
       WorkerID = HouseWorker[HouseWorker[:, 11] == infectedOfficeID[i,0], 0]
       OfficeSuspectedID = np.setdiff1d(WorkerID, InfectedAgentRec)
       if OfficeSuspectedID.size > 0:
           NewOfficeinfected = np.random.binomial(OfficeSuspectedID.size, OfficeCaseProb[i])
           if NewOfficeinfected > 0:
               NewOfficeinfectedID = np.random.choice(OfficeSuspectedID, NewOfficeinfected, replace=False)
               TodayOfficeInfected = np.append(TodayOfficeInfected, NewOfficeinfectedID)
               OI += NewOfficeinfected    
    et = time.time()
    print("Office using time:", et-st)
    TodayOfficeInfected = np.delete(TodayOfficeInfected, 0) 
    TodayOfficeInfected = np.unique(TodayOfficeInfected).astype(int)
    nouse, nouse, forind= np.intersect1d(TodayOfficeInfected, HouseWorker[:,0], assume_unique=True, return_indices=True)
    TodayOfficeInfectedloc = HouseWorker[forind, 12]
    return TodayOfficeInfected, TodayOfficeInfectedloc, OI

def StudentInfectiousProcess(beta, HouseAgents, HouseChild, HousePristudent, HouseSecstudent, InfectedAgentRec, InfectiousAgent, Infectedstate, Unistudent,varying_beta_asy):    
    st = time.time()
    PI, SI, KI, UI = 0, 0, 0, 0
    TodayKgartenInfected = np.zeros(1)
    TodayPrischInfected = np.zeros(1)
    TodaySecschInfected = np.zeros(1)
    TodayUniInfected = np.zeros(1)
    KinderchildID = HouseAgents[(HouseAgents[:,4] == 2), 0]
    PriStudentID = HouseAgents[(HouseAgents[:,4] == 3), 0]
    SecStudentID = HouseAgents[(HouseAgents[:,4] == 4), 0]
    UniStudentID = Unistudent[:,0]
    KgarteninfectedID = np.intersect1d(KinderchildID, InfectiousAgent)
    PriStudentinfectedID = np.intersect1d(PriStudentID, InfectiousAgent)
    SecStudentinfectedID = np.intersect1d(SecStudentID, InfectiousAgent)
    UniStudentinfectedID = np.intersect1d(UniStudentID, InfectiousAgent)
    forcountind = np.intersect1d(KgarteninfectedID, HouseChild[:,0], return_indices=True)[2]
    KgartenID, Kgartenactivecount = np.unique(HouseChild[forcountind, 11], return_counts=True)
    forcountind = np.intersect1d(PriStudentinfectedID, HousePristudent[:,0], return_indices=True)[2]
    PriSchoolID, PrischoolCount = np.unique(HousePristudent[forcountind, 11], return_counts=True)
    nouseID, nouseind, forcountind = np.intersect1d(SecStudentinfectedID, HouseSecstudent[:,0], return_indices=True)
    SecSchoolID, SecschoolCount = np.unique(HouseSecstudent[forcountind, 11], return_counts=True)
    nouseID, nouseind, forcountind = np.intersect1d(UniStudentinfectedID, Unistudent[:,0], return_indices=True)
    UniversityID, UniversityCount = np.unique(Unistudent[forcountind, 10], return_counts=True)
    totalkgartenID, kgartentotalsize = np.unique(HouseChild[:, 11], return_counts=True)
    nouseID, nouseind, fortotalcountind = np.intersect1d(KgartenID, totalkgartenID, return_indices=True)
    infectedkgartentotalsize = kgartentotalsize[fortotalcountind]
    totalPrischoolID, prischooltotalsize = np.unique(HousePristudent[:, 11], return_counts=True)
    nouseID, nouseind, fortotalcountind = np.intersect1d(PriSchoolID, totalPrischoolID, return_indices=True)
    infectedprischtotalsize = prischooltotalsize[fortotalcountind]
    totalSecschoolID, secschooltotalsize = np.unique(HouseSecstudent[:, 11], return_counts=True)
    nouseID, nouseind, fortotalcountind = np.intersect1d(SecSchoolID, totalSecschoolID, return_indices=True)
    infectedsecschtotalsize = secschooltotalsize[fortotalcountind]
    totalUniversityID, Universitytotalsize = np.unique(Unistudent[:, 10], return_counts=True)
    nouseID, nouseind, fortotalcountind = np.intersect1d(UniversityID, totalUniversityID, return_indices=True)
    infecteduniversitytotalsize = Universitytotalsize[fortotalcountind]    
    
    forasycountind = np.intersect1d(InfectiousAgent[Infectedstate==2], HouseChild[:,0], return_indices=True)[2]
    asy_KgartenID, asy_kgartencount = np.unique(HouseChild[forasycountind, 11], return_counts=True)
    forasytotalcountind = np.intersect1d(asy_KgartenID, KgartenID, return_indices=True)[2]
    Kgartenasycount = np.zeros(len(KgartenID))
    Kgartenasycount[forasytotalcountind] = asy_kgartencount

    forasycountind = np.intersect1d(InfectiousAgent[Infectedstate==2], HousePristudent[:,0], return_indices=True)[2]
    asy_PrischID, asy_prischcount = np.unique(HousePristudent[forasycountind, 11], return_counts=True)
    forasytotalcountind = np.intersect1d(asy_PrischID, PriSchoolID, return_indices=True)[2]
    Prischasycount = np.zeros(len(PriSchoolID))
    Prischasycount[forasytotalcountind] = asy_prischcount

    forasycountind = np.intersect1d(InfectiousAgent[Infectedstate==2], HouseSecstudent[:,0], return_indices=True)[2]
    asy_SecschID, asy_secschcount = np.unique(HouseSecstudent[forasycountind, 11], return_counts=True)
    forasytotalcountind = np.intersect1d(asy_SecschID, SecSchoolID, return_indices=True)[2]
    Secschasycount = np.zeros(len(SecSchoolID))
    Secschasycount[forasytotalcountind] = asy_secschcount

    forasycountind = np.intersect1d(InfectiousAgent[Infectedstate==2], HouseSecstudent[:,0], return_indices=True)[2]
    asy_SecschID, asy_secschcount = np.unique(HouseSecstudent[forasycountind, 11], return_counts=True)
    forasytotalcountind = np.intersect1d(asy_SecschID, SecSchoolID, return_indices=True)[2]
    Secschasycount = np.zeros(len(SecSchoolID))
    Secschasycount[forasytotalcountind] = asy_secschcount

    forasycountind = np.intersect1d(InfectiousAgent[Infectedstate==2], Unistudent[:,0], return_indices=True)[2]
    asy_UnischID, asy_unischcount = np.unique(Unistudent[forasycountind, 10], return_counts=True)
    forasytotalcountind = np.intersect1d(asy_UnischID, UniversityID, return_indices=True)[2]
    Unischasycount = np.zeros(len(UniversityID))
    Unischasycount[forasytotalcountind] = asy_unischcount
    
    KgartenCaseProb = ((6.2+1.0)/22.3) * beta * ((Kgartenactivecount-varying_beta_asy*Kgartenasycount)/infectedkgartentotalsize)
    PriSchoolCaseProb = ((6.2+1.0)/22.3) * beta * ((PrischoolCount-varying_beta_asy*Prischasycount)/infectedprischtotalsize)
    SecSchoolProb = ((6.2+1.0)/22.3) * beta * ((SecschoolCount-varying_beta_asy*Secschasycount)/infectedsecschtotalsize)
    UniversityProb = ((6.2+1.0)/22.3) * beta * ((UniversityCount-varying_beta_asy*Unischasycount)/infecteduniversitytotalsize)
    for i in range(len(KgartenID)):
        if infectedkgartentotalsize[i] > 1:
            KgartenSuspectedID = HouseChild[HouseChild[:,11] == KgartenID[i], 0]
            KgartenSuspectedID = np.setdiff1d(KgartenSuspectedID, InfectedAgentRec)
            if KgartenSuspectedID.size > 0:
                if KgartenCaseProb[i] < 0 or KgartenCaseProb[i] > 1:
                    print('sth wrong in Kinder')
                    KgartenCaseProb[i] = 0
                else:
                    NewKgarteninfected = np.random.binomial(KgartenSuspectedID.size, KgartenCaseProb[i])
                    if NewKgarteninfected > 0:
                        NewKgarteninfectedID = np.random.choice(KgartenSuspectedID, NewKgarteninfected, replace=False)
                        TodayKgartenInfected = np.append(TodayKgartenInfected, NewKgarteninfectedID)           
                        KI += NewKgarteninfected
    for i in range(len(PriSchoolID)):
        if infectedprischtotalsize[i] > 1:
            PrischSuspectedID = HousePristudent[HousePristudent[:,11] == PriSchoolID[i], 0]
            PrischSuspectedID = np.setdiff1d(PrischSuspectedID, InfectedAgentRec)
            if PrischSuspectedID.size > 0:
                if PriSchoolCaseProb[i] < 0 or PriSchoolCaseProb[i] > 1:
                    print('sth wrong in Pri')
                    PriSchoolCaseProb[i] = 0
                else:
                    NewKPrischinfected = np.random.binomial(PrischSuspectedID.size, PriSchoolCaseProb[i])
                    if NewKPrischinfected > 0:
                        NewKPrischinfectedID = np.random.choice(PrischSuspectedID, NewKPrischinfected, replace=False)
                        TodayPrischInfected = np.append(TodayPrischInfected, NewKPrischinfectedID)           
                        PI += NewKPrischinfected
    for i in range(len(SecSchoolID)):
        if infectedsecschtotalsize[i] > 1:
            SecschSuspectedID = HouseSecstudent[HouseSecstudent[:,11] == SecSchoolID[i], 0]
            SecschSuspectedID = np.setdiff1d(SecschSuspectedID, InfectedAgentRec)
            if SecschSuspectedID.size > 0:
                if SecSchoolProb[i] < 0 or SecSchoolProb[i] > 1:
                    print('sth wrong in Sec')
                    SecSchoolProb[i] = 0
                else:
                    NewSecschinfected = np.random.binomial(SecschSuspectedID.size, SecSchoolProb[i])
                    if NewSecschinfected > 0:
                        NewSecschinfectedID = np.random.choice(SecschSuspectedID, NewSecschinfected, replace=False)
                        TodaySecschInfected = np.append(TodaySecschInfected, NewSecschinfectedID)           
                        SI += NewSecschinfected
    for i in range(len(UniversityID)):
        if infecteduniversitytotalsize[i] > 1:
            UnischSuspectedID = Unistudent[Unistudent[:,10] == UniversityID[i], 0]
            UnischSuspectedID = np.setdiff1d(UnischSuspectedID, InfectedAgentRec)
            if UnischSuspectedID.size > 0:
                if UniversityProb[i] < 0 or UniversityProb[i] > 1:
                    print('sth wrong in uni')
                    UniversityProb[i] = 0
                else:
                    NewUnischinfected = np.random.binomial(UnischSuspectedID.size, UniversityProb[i])
                    if NewUnischinfected > 0:
                        NewUnischinfectedID = np.random.choice(UnischSuspectedID, NewUnischinfected, replace=False)
                        TodayUniInfected = np.append(TodayUniInfected, NewUnischinfectedID)           
                        UI += NewUnischinfected
                    
    TodayKgartenInfected = np.delete(TodayKgartenInfected, 0) 
    TodayKgartenInfected = np.unique(TodayKgartenInfected).astype(int)
    TodayPrischInfected = np.delete(TodayPrischInfected, 0) 
    TodayPrischInfected = np.unique(TodayPrischInfected).astype(int)
    TodaySecschInfected = np.delete(TodaySecschInfected, 0) 
    TodaySecschInfected = np.unique(TodaySecschInfected).astype(int)
    TodayUniInfected = np.delete(TodayUniInfected, 0) 
    TodayUniInfected = np.unique(TodayUniInfected).astype(int)
    nouse, nouse, forind= np.intersect1d(TodayKgartenInfected, HouseChild[:,0], assume_unique=True, return_indices=True)
    TodayKgartenInfectedloc = HouseChild[forind, 12]
    nouse, nouse, forind= np.intersect1d(TodayPrischInfected, HousePristudent[:,0], assume_unique=True, return_indices=True)
    TodayPrischInfectedloc = HousePristudent[forind, 12] 
    nouse, nouse, forind= np.intersect1d(TodaySecschInfected, HouseSecstudent[:,0], assume_unique=True, return_indices=True)
    TodaySecschInfectedloc = HouseSecstudent[forind, 12]
    nouse, nouse, forind= np.intersect1d(TodayUniInfected, Unistudent[:,0], return_indices=True)
    TodayUnischInfectedloc = Unistudent[forind, 11]           
    et = time.time()
    print("School using time:", et-st)
    return TodayKgartenInfected, TodayPrischInfected, TodaySecschInfected,TodayUniInfected, TodayKgartenInfectedloc, TodayPrischInfectedloc, TodaySecschInfectedloc, TodayUnischInfectedloc, KI, PI, SI, UI

def ResturantInfectiousProcess(BasicProb, InfectiousAgent, InfectedAgentRec, Infectedstate, HouseAgents, TotalGrids, HouseWorker, weekdayhomehomecorr, weekdayhomeworkcorr, weekdayworkhomecorr, weekdayworkworkcorr, Saturdayhomehomecorr, Sundayhomehomecorr, weekdayind, index,varying_beta_asy):       
    st = time.time()
    ReI = 0 # count the resturant infected agents
    TodayResturantInfected = np.zeros(1) # record the infected agents
    TodayResturantInfectedloc = np.zeros(1) # record the infected agents
    WorkplaceReI = 0 # count the resturant infected agents by workplace
    Infectiousgriddata = getInfectiousgriddata(InfectiousAgent, Infectedstate, HouseAgents, TotalGrids, HouseWorker)
    if weekdayind == 2:
        outgridsize = Infectiousgriddata[:,12]
        restgridsize = Infectiousgriddata[:,9] - Infectiousgriddata[:,12]
        act_outgridsize = Infectiousgriddata[:,5]
        act_restgridsize = Infectiousgriddata[:,1] - Infectiousgriddata[:,5]
        asy_outgridsize = Infectiousgriddata[:,8]
        asy_restgridsize = Infectiousgriddata[:,2] - Infectiousgriddata[:,8]
        HouseGridcaseProb = BasicProb * 1.5/20.2 * (act_restgridsize - varying_beta_asy * asy_restgridsize) / restgridsize * (100+index)/100
        HouseGridcaseProb[np.isnan(HouseGridcaseProb)] = 0
        HouseGridcaseProb[np.isinf(HouseGridcaseProb)] = 0
        totalhomegridcorr = Sundayhomehomecorr       
    if weekdayind == 1: 
        outgridsize = np.zeros(len(Infectiousgriddata[:,0]))
        restgridsize = Infectiousgriddata[:,9]
        act_outgridsize = np.zeros(len(Infectiousgriddata[:,0]))
        act_restgridsize = Infectiousgriddata[:,1]
        asy_outgridsize = np.zeros(len(Infectiousgriddata[:,0]))
        asy_restgridsize = Infectiousgriddata[:,2]
        HouseGridcaseProb = BasicProb * 1.5/20.2 * (act_restgridsize - varying_beta_asy * asy_restgridsize) / restgridsize * (100+index)/100
        HouseGridcaseProb[np.isnan(HouseGridcaseProb)] = 0
        HouseGridcaseProb[np.isinf(HouseGridcaseProb)] = 0
        totalhomegridcorr = Saturdayhomehomecorr
    if weekdayind == 0: # weekends 50% people go out
        outgridsize = Infectiousgriddata[:,10] + Infectiousgriddata[:,11]
        restgridsize = Infectiousgriddata[:,9] - outgridsize
        act_outgridsize = Infectiousgriddata[:,3] + Infectiousgriddata[:,4]
        act_restgridsize = Infectiousgriddata[:,1] - act_outgridsize
        asy_outgridsize = Infectiousgriddata[:,6] + Infectiousgriddata[:,7]
        asy_restgridsize = Infectiousgriddata[:,2] - asy_outgridsize
        HouseGridcaseProb = BasicProb * 1.0/21.3 * (act_restgridsize - varying_beta_asy * asy_restgridsize) / restgridsize * (100+index)/100
        HouseGridcaseProb[np.isnan(HouseGridcaseProb)] = 0
        HouseGridcaseProb[np.isinf(HouseGridcaseProb)] = 0
        totalhomegridcorr = weekdayhomehomecorr + weekdayhomeworkcorr
        totalworkgridcorr = weekdayworkhomecorr + weekdayworkworkcorr
        workplaceGridcaseProb = BasicProb * 1.2/22.7 * (Infectiousgriddata[:,15] - varying_beta_asy * Infectiousgriddata[:,16]) / Infectiousgriddata[:,14] * (100+index)/100
        
    Probtohometrans = HouseGridcaseProb
    for i in range(len(HouseGridcaseProb)): # find resturant infections by grids
        if Probtohometrans[i] > 0:
            gridind = Infectiousgriddata[i,0].astype(int) 
            gridresinfectiouscount = np.random.binomial(np.around(totalhomegridcorr[gridind,:]).astype(int),Probtohometrans[i],size=len(totalhomegridcorr[gridind,:]))
            if np.sum(gridresinfectiouscount) > 0:
                temp_gridresinfectiouscase = gridresinfectiouscount[np.nonzero(gridresinfectiouscount)]
                tem_loc = np.nonzero(gridresinfectiouscount)[0]
                for j in range(len(tem_loc)):
                    if weekdayind != 0:
                        GridSuspectedID = np.setdiff1d(HouseAgents[HouseAgents[:,1] == tem_loc[j], 0], InfectedAgentRec)
                        if GridSuspectedID.size > 0:
                            if temp_gridresinfectiouscase[j] <= GridSuspectedID.size:
                                NewHousecominfectedID = np.random.choice(GridSuspectedID, temp_gridresinfectiouscase[j], replace=False)
                                NewHousecominfectedloc = np.ones(len(NewHousecominfectedID )) * tem_loc[j]
                                TodayResturantInfected = np.append(TodayResturantInfected, NewHousecominfectedID)
                                TodayResturantInfectedloc = np.append(TodayResturantInfectedloc, NewHousecominfectedloc)
                            else:
                                TodayResturantInfected = np.append(TodayResturantInfected, GridSuspectedID)
                                TodayResturantInfectedloc = np.append(TodayResturantInfectedloc, np.ones(len(GridSuspectedID )) * tem_loc[j])
                    else:
                        TotalGridSuspectedID = np.union1d(HouseAgents[HouseAgents[:,1] == tem_loc[j], 0], HouseWorker[HouseWorker[:,12] == tem_loc[j],0])
                        GridSuspectedID = np.setdiff1d(TotalGridSuspectedID, InfectedAgentRec)
                        GridSuspectedID = np.setdiff1d(GridSuspectedID, HouseAgents[HouseAgents[:,7] == 1, 1])
                        GridSuspectedID = np.setdiff1d(GridSuspectedID, HouseAgents[HouseAgents[:,8] == 1, 1])
                        if GridSuspectedID.size > 0:
                            if temp_gridresinfectiouscase[j] <= GridSuspectedID.size:
                                NewHousecominfectedID = np.random.choice(GridSuspectedID, temp_gridresinfectiouscase[j], replace=False)
                                NewHousecominfectedloc = np.ones(len(NewHousecominfectedID )) * tem_loc[j]
                                TodayResturantInfected = np.append(TodayResturantInfected, NewHousecominfectedID)
                                TodayResturantInfectedloc = np.append(TodayResturantInfectedloc, NewHousecominfectedloc)
                            else:
                                TodayResturantInfected = np.append(TodayResturantInfected, GridSuspectedID)
                                TodayResturantInfectedloc = np.append(TodayResturantInfectedloc, np.ones(len(GridSuspectedID )) * tem_loc[j])  
                ReI += np.sum(gridresinfectiouscount)
                
    if weekdayind == 0:
        Probtoworktrans = workplaceGridcaseProb
        for i in range(len(workplaceGridcaseProb)):
            if Probtoworktrans[i] > 0:
                workgridind = Infectiousgriddata[i,0].astype(int)
                workgridresinfectiouscount = np.random.binomial(np.around(totalworkgridcorr[workgridind,:]).astype(int),Probtoworktrans[i],size=len(totalworkgridcorr[workgridind,:]))
                if np.sum(workgridresinfectiouscount) > 0:
                    temp_gridresinfectiouscase = workgridresinfectiouscount[np.nonzero(workgridresinfectiouscount)]
                    tem_loc = np.nonzero(workgridresinfectiouscount)[0]
                    for j in range(len(tem_loc)):
                        TotalGridSuspectedID = np.union1d(HouseAgents[HouseAgents[:,1] == tem_loc[j], 0], HouseWorker[HouseWorker[:,12] == tem_loc[j],0])
                        GridSuspectedID = np.setdiff1d(TotalGridSuspectedID, InfectedAgentRec)
                        GridSuspectedID = np.setdiff1d(GridSuspectedID, HouseAgents[HouseAgents[:,7] == 1, 1])
                        GridSuspectedID = np.setdiff1d(GridSuspectedID, HouseAgents[HouseAgents[:,8] == 1, 1])
                        if GridSuspectedID.size > 0:                          
                            if temp_gridresinfectiouscase[j] <= GridSuspectedID.size:
                                NewRestcominfectedID = np.random.choice(GridSuspectedID, temp_gridresinfectiouscase[j], replace=False)
                                NewHousecominfectedloc = np.ones(len(NewRestcominfectedID)) * tem_loc[j]
                                TodayResturantInfected = np.append(TodayResturantInfected, NewRestcominfectedID)
                                TodayResturantInfectedloc = np.append(TodayResturantInfectedloc, NewHousecominfectedloc)
                            else:
                                TodayResturantInfected = np.append(TodayResturantInfected, GridSuspectedID)
                                TodayResturantInfectedloc = np.append(TodayResturantInfectedloc, np.ones(len(GridSuspectedID )) * tem_loc[j])
                    WorkplaceReI += np.sum(workgridresinfectiouscount)

    TodayResturantInfected = np.delete(TodayResturantInfected, 0)
    TodayResturantInfected = np.unique(TodayResturantInfected).astype(int)
    TodayResturantInfectedloc = np.delete(TodayResturantInfectedloc, 0)
    TodayResturantInfectedloc = np.unique(TodayResturantInfectedloc).astype(int)       
    et = time.time()
    print("Resturant using time:", et-st)
    return TodayResturantInfected, TodayResturantInfectedloc, ReI, WorkplaceReI

def OtherInfectiousProcess(BasicProb, InfectiousAgent, InfectedAgentRec, HouseAgents, TotalGrids, HouseWorker, poptranprob, CumInfectiouSeedsIndexs, weekdayind, togetherhour, index, Infectedstate):
    st = time.time()
    Dparty = 0 # count the Dhelper together infected agents
    TodayDhelperInfectedloc = np.zeros(1) # record the infected locations
    TodayDhelperInfected = np.zeros(1) # record the infected agents
    Infectiousgriddata = getInfectiousgriddata(InfectiousAgent, Infectedstate, HouseAgents, TotalGrids, HouseWorker)
    if weekdayind > 0:
        Probtranscase = (togetherhour/22.3) * BasicProb * (Infectiousgriddata[:,1]-0.5 * Infectiousgriddata[:,2])/(Infectiousgriddata[:,9]) * (100+index)/100
    else:
        Probtranscase = (togetherhour/20.2) * BasicProb * (Infectiousgriddata[:,1]-0.5 * Infectiousgriddata[:,2])/(Infectiousgriddata[:,9]) * (100+index)/100
    Probtranscase[np.isnan(Probtranscase)]=0
    Probtranscase[np.isinf(Probtranscase)]=0
    Probtohometrans = Probtranscase
    for i in range(len(Probtohometrans)): # find resturant infections by grids
        if Probtohometrans[i] > 0:
            gridind = Infectiousgriddata[i,0].astype(int)
            if weekdayind == 2:
                discount_TotalGrids = TotalGrids[:,3] - TotalGrids[:,4]
                gridresinfectiouscount = np.random.binomial(np.around(discount_TotalGrids * poptranprob[gridind,:]).astype(int), Probtohometrans[i])
            elif weekdayind == 1:
                gridresinfectiouscount = np.random.binomial(np.around(TotalGrids[:,3]* poptranprob[gridind,:]).astype(int), Probtohometrans[i])
            else:
                gridresinfectiouscount = np.random.binomial(np.around((TotalGrids[:,3]- TotalGrids[:,2]-TotalGrids[:,1])* poptranprob[gridind,:]).astype(int), Probtohometrans[i])
            if np.sum(gridresinfectiouscount) > 0:
                temp_gridresinfectiouscase = gridresinfectiouscount[np.nonzero(gridresinfectiouscount)]
                tem_loc = np.nonzero(gridresinfectiouscount)[0]
                for j in range(len(tem_loc)):
                    TotalGridSuspectedID = HouseAgents[HouseAgents[:,1] == tem_loc[j], 0]
                    GridSuspectedID = np.setdiff1d(TotalGridSuspectedID, InfectedAgentRec)
                    if GridSuspectedID.size != 0:
                        NewHousecominfectedID = np.random.choice(GridSuspectedID, temp_gridresinfectiouscase[j], replace=False)
                        NewHousecominfectedloc = np.ones(len(NewHousecominfectedID)) * tem_loc[j]
                        TodayDhelperInfected = np.append(TodayDhelperInfected, NewHousecominfectedID)
                        TodayDhelperInfectedloc = np.append(TodayDhelperInfectedloc,NewHousecominfectedloc )
            Dparty += np.sum(gridresinfectiouscount)    
    TodayDhelperInfected = np.delete(TodayDhelperInfected, 0) 
    TodayDhelperInfected = np.unique(TodayDhelperInfected).astype(int)
    TodayDhelperInfectedloc = np.delete(TodayDhelperInfectedloc, 0) 
    TodayDhelperInfectedloc = np.unique(TodayDhelperInfectedloc).astype(int)
    et = time.time()
    print("Other using time:", et-st)    
    return TodayDhelperInfected, TodayDhelperInfectedloc, Dparty

def contact_tracing(Usedorinal, InfectiousAgent, HouseAgents, HouseWorker):
    Usedresidents = InfectiousAgent[np.isin(InfectiousAgent, Usedorinal)].astype(int)
    Familymemberloc = np.unique(HouseAgents[Usedresidents, 1].astype(int))
    for i in range(len(Familymemberloc)): # get the family members
        oneFamilymemberloc = np.where(HouseAgents[:,1] == Familymemberloc[i])[0]
        Familymemberloc = np.append(Familymemberloc, oneFamilymemberloc)
    if len(Familymemberloc) > 0:
        Familymemberloc = np.delete(Familymemberloc, 0) 
    worker = np.intersect1d(Usedresidents, HouseWorker[:,0], assume_unique=True, return_indices=True)[2]
    Officeid = HouseWorker[worker, 11]
    Officecolleagueloc = 0
    OfficecolleagueID = np.zeros(1)
    if worker.size > 0:
        for i in range(len(worker)):
            oneWorkplaceloc = np.where(HouseWorker[:, 11] == Officeid[i])
            Officecolleagueloc = np.append(Officecolleagueloc, oneWorkplaceloc)
        Officecolleagueloc = np.delete(Officecolleagueloc, 0)
        OfficecolleagueID = HouseWorker[Officecolleagueloc, 0]
    else:
        pass
    return Familymemberloc, OfficecolleagueID

def iterations(period, beta, InfectiousAgent, Infectedstate, HouseAgents, TotalHouse, InfectedAgentRec, InfectiondateRec, InfectionstateRec, p, k, LatentperiodRec, InfectiousperiodRec, Epsilon, Epsilon_asy, Gamma, beta_asy, beta_pre, mu, shoppingpoptranprob, pharmacypoptranprob, varying_mobility):
    SumInfectRec = np.zeros([period, 13])
    Homeindex, Communityindex, workplaceindex, resturantindex, mallindex, marketindex = 0, 0, 0, 0, 0, 0    
    presymptomatic_count = np.sum(Infectedstate==0)
    varying_beta = beta
    varying_beta_asy = 0.5
    infectlocation = []
    gatherlocdata = np.zeros(1)
    testedordinal = np.zeros(1)
    Onedaytracingdelay = np.zeros(1)
    TrackedagentRec = np.zeros(1)
    NewInfectiousAgent = InfectiousAgent
    NewInfectedstate = Infectedstate
    for t in range(period):
        stit = time.time()
        print('Day',t,'Active infectious cases', len(NewInfectiousAgent))
        if t >= 16:
            Homeindex, Communityindex, workplaceindex, resturantindex, mallindex, marketindex = varying_mobility[t,5], 0, varying_mobility[t,4], varying_mobility[t,0], varying_mobility[t,0], varying_mobility[t,1]
        if t >= 26:
            Homeindex, Communityindex, workplaceindex, resturantindex, mallindex, marketindex = varying_mobility[t,5], (varying_mobility[t,5] - 50), (varying_mobility[t,4]-50), (varying_mobility[t,0]-50), (varying_mobility[t,0]-50), (varying_mobility[t,1]-50)       
        if t >= 34:
            Homeindex, Communityindex, workplaceindex, resturantindex, mallindex, marketindex = varying_mobility[t,5], (varying_mobility[t,5] - 85), (varying_mobility[t,4]-90), (varying_mobility[t,0]-90), (varying_mobility[t,0]-90), (varying_mobility[t,1]-90)
            # Homeindex, Communityindex, workplaceindex, resturantindex, mallindex, marketindex = (varying_mobility[t,5]-75), (varying_mobility[t,5] - 90), (varying_mobility[t,4]-90), (varying_mobility[t,0]-90), (varying_mobility[t,0]-90), (varying_mobility[t,1]-90)
        if len(NewInfectiousAgent) == 0:
            NewinfectedID, HI = [], 0
        elif t % 7 == 3:
            weekdayind = 2 # Sunday
            TodayHouseInfected, TodayHouseInfectedloc, HI = HouseInfectiousProcess(varying_beta, NewInfectiousAgent, NewInfectedstate, HouseAgents, TotalHouse, InfectedAgentRec, weekdayind, Homeindex,varying_beta_asy)
            TodayCommunityInfected, TodayCommunityInfectedloc, CI = CommunityInfectiousProcess(varying_beta, NewInfectiousAgent, NewInfectedstate, HouseAgents, InfectedAgentRec, TPUdata, TotalGrids, weekdayind, Communityindex,varying_beta_asy)
            TodayOfficeInfected, TodayKgartenInfected, TodayPrischInfected, TodaySecschInfected, TodayUniInfected = [], [], [], [], []
            TodayResturantInfected, TodayResturantInfectedloc, ReI, WorkplaceReI = ResturantInfectiousProcess(varying_beta, NewInfectiousAgent, InfectedAgentRec, NewInfectedstate, HouseAgents, TotalGrids, HouseWorker, weekdayhomehomecorr, weekdayhomeworkcorr, weekdayworkhomecorr, weekdayworkworkcorr, Saturdayhomehomecorr, Sundayhomehomecorr, weekdayind, resturantindex,varying_beta_asy)
            shopping, market = 1.5, 0.8
            TodayshoppingmallInfected,TodayshoppingmallInfectedloc, mallI = OtherInfectiousProcess(varying_beta, NewInfectiousAgent, InfectedAgentRec, HouseAgents, TotalGrids, HouseWorker, shoppingpoptranprob, InfectedAgentRec, weekdayind, shopping, mallindex, NewInfectedstate)
            TodaymarketInfected,TodaymarketInfectedloc, marketI = OtherInfectiousProcess(varying_beta, NewInfectiousAgent, InfectedAgentRec, HouseAgents, TotalGrids, HouseWorker, pharmacypoptranprob, InfectedAgentRec, weekdayind, market, marketindex, NewInfectedstate)
            OI, PI, SI, KI, UI = 0, 0, 0, 0 ,0
            infectlocation.append(TodayHouseInfectedloc)
            infectlocation.append(TodayCommunityInfectedloc)
            infectlocation.append(TodayResturantInfectedloc)
            infectlocation.append(TodayshoppingmallInfectedloc)
            infectlocation.append(TodaymarketInfectedloc)
        elif t % 7 == 2:
            weekdayind = 1 # Saturday
            TodayHouseInfected, TodayHouseInfectedloc, HI = HouseInfectiousProcess(varying_beta, NewInfectiousAgent, NewInfectedstate, HouseAgents, TotalHouse, InfectedAgentRec, weekdayind, Homeindex,varying_beta_asy)
            TodayCommunityInfected, TodayCommunityInfectedloc, CI = CommunityInfectiousProcess(varying_beta, NewInfectiousAgent, NewInfectedstate, HouseAgents, InfectedAgentRec, TPUdata, TotalGrids, weekdayind, Communityindex,varying_beta_asy)
            TodayOfficeInfected, TodayKgartenInfected, TodayPrischInfected, TodaySecschInfected, TodayUniInfected = [], [], [], [], []
            TodayResturantInfected, TodayResturantInfectedloc, ReI, WorkplaceReI = ResturantInfectiousProcess(varying_beta, NewInfectiousAgent, InfectedAgentRec, NewInfectedstate, HouseAgents, TotalGrids, HouseWorker, weekdayhomehomecorr, weekdayhomeworkcorr, weekdayworkhomecorr, weekdayworkworkcorr, Saturdayhomehomecorr, Sundayhomehomecorr, weekdayind, resturantindex,varying_beta_asy)
            shopping, market = 1.5, 0.8
            TodayshoppingmallInfected,TodayshoppingmallInfectedloc, mallI = OtherInfectiousProcess(varying_beta, NewInfectiousAgent, InfectedAgentRec, HouseAgents, TotalGrids, HouseWorker, shoppingpoptranprob, InfectedAgentRec, weekdayind, shopping, mallindex, NewInfectedstate)
            TodaymarketInfected,TodaymarketInfectedloc, marketI = OtherInfectiousProcess(varying_beta, NewInfectiousAgent, InfectedAgentRec, HouseAgents, TotalGrids, HouseWorker, pharmacypoptranprob, InfectedAgentRec, weekdayind, market, marketindex, NewInfectedstate)
            OI, PI, SI, KI, UI = 0, 0, 0, 0 ,0
            infectlocation.append(TodayHouseInfectedloc)
            infectlocation.append(TodayCommunityInfectedloc)
            infectlocation.append(TodayResturantInfectedloc)
            infectlocation.append(TodayshoppingmallInfectedloc)
            infectlocation.append(TodaymarketInfectedloc)
        else:
            weekdayind = 0 # weekday
            TodayHouseInfected, TodayHouseInfectedloc, HI = HouseInfectiousProcess(varying_beta, NewInfectiousAgent, NewInfectedstate, HouseAgents, TotalHouse, InfectedAgentRec, weekdayind, Homeindex,varying_beta_asy)
            TodayCommunityInfected, TodayCommunityInfectedloc, CI = CommunityInfectiousProcess(varying_beta, NewInfectiousAgent, NewInfectedstate, HouseAgents, InfectedAgentRec, TPUdata, TotalGrids, weekdayind, Communityindex,varying_beta_asy)
            TodayOfficeInfected, TodayOfficeInfectedloc, OI = OfficeInfectiousProcess(varying_beta, TotalGrids, HouseAgents, HouseWorker, NewInfectiousAgent, NewInfectedstate, InfectedAgentRec, workplaceindex,varying_beta_asy)
            TodayResturantInfected, TodayResturantInfectedloc, ReI, WorkplaceReI = ResturantInfectiousProcess(varying_beta, NewInfectiousAgent, InfectedAgentRec, NewInfectedstate, HouseAgents, TotalGrids, HouseWorker, weekdayhomehomecorr, weekdayhomeworkcorr, weekdayworkhomecorr, weekdayworkworkcorr, Saturdayhomehomecorr, Sundayhomehomecorr, weekdayind, resturantindex,varying_beta_asy)
            shopping, market = 1.0, 0.7
            TodayshoppingmallInfected,TodayshoppingmallInfectedloc, mallI = OtherInfectiousProcess(varying_beta, NewInfectiousAgent, InfectedAgentRec, HouseAgents, TotalGrids, HouseWorker, shoppingpoptranprob, InfectedAgentRec, weekdayind, shopping, mallindex, NewInfectedstate)
            TodaymarketInfected,TodaymarketInfectedloc, marketI = OtherInfectiousProcess(varying_beta, NewInfectiousAgent, InfectedAgentRec, HouseAgents, TotalGrids, HouseWorker, pharmacypoptranprob, InfectedAgentRec, weekdayind, market, marketindex, NewInfectedstate)
            if t <= 16:
                TodayKgartenInfected, TodayPrischInfected, TodaySecschInfected,TodayUniInfected, TodayKgartenInfectedloc, TodayPrischInfectedloc, TodaySecschInfectedloc, TodayUnischInfectedloc, KI, PI, SI, UI = StudentInfectiousProcess(varying_beta, HouseAgents, HouseChild, HousePristudent, HouseSecstudent, InfectedAgentRec, NewInfectiousAgent, NewInfectedstate, Unistudent, varying_beta_asy)
                infectlocation.append(TodayKgartenInfectedloc)
                infectlocation.append(TodayPrischInfectedloc)
                infectlocation.append(TodaySecschInfectedloc)
                infectlocation.append(TodayUnischInfectedloc) 
            else:
                TodayKgartenInfected, TodayPrischInfected, TodaySecschInfected, TodayUniInfected = [], [], [], []
                PI, SI, KI, UI = 0, 0, 0, 0
                print('school closed')
            if t <= 34:
                TodayResturantInfected, TodayResturantInfectedloc, ReI, WorkplaceReI = ResturantInfectiousProcess(varying_beta, NewInfectiousAgent, InfectedAgentRec, NewInfectedstate, HouseAgents, TotalGrids, HouseWorker, weekdayhomehomecorr, weekdayhomeworkcorr, weekdayworkhomecorr, weekdayworkworkcorr, Saturdayhomehomecorr, Sundayhomehomecorr, weekdayind, resturantindex,varying_beta_asy)
                infectlocation.append(TodayResturantInfectedloc)
            else:
                ReI, WorkplaceReI = 0, 0
                print('restaurant closed')
            infectlocation.append(TodayHouseInfectedloc)
            infectlocation.append(TodayCommunityInfectedloc)
            infectlocation.append(TodayshoppingmallInfectedloc)
            infectlocation.append(TodaymarketInfectedloc)
            infectlocation.append(TodayOfficeInfectedloc)
           
        NewinfectedID = np.union1d(TodayHouseInfected, TodayCommunityInfected)
        NewinfectedID = np.union1d(NewinfectedID, TodayOfficeInfected)
        NewinfectedID = np.union1d(NewinfectedID, TodayKgartenInfected)
        NewinfectedID = np.union1d(NewinfectedID, TodayPrischInfected)
        NewinfectedID = np.union1d(NewinfectedID, TodaySecschInfected)
        NewinfectedID = np.union1d(NewinfectedID, TodayUniInfected)
        NewinfectedID = np.union1d(NewinfectedID, TodayResturantInfected)
        NewinfectedID = np.union1d(NewinfectedID, TodayshoppingmallInfected)
        NewinfectedID = np.union1d(NewinfectedID, TodaymarketInfected)
        
        InfectedAgentRec = np.append(InfectedAgentRec, NewinfectedID)
        Newinfectionstate = Assigninfectionstate(NewinfectedID, p, k)
        InfectedAgentRec, uniind = np.unique(InfectedAgentRec, return_index=True)
        InfectiondateRec = np.append(InfectiondateRec, np.ones(len(NewinfectedID))*(t+1))
        # Newinfectionstate = Assigninfectionstate(NewinfectedID, p, k)
        NewLatentperiod, NewInfectiousperiod = Assigninfectiondate(Newinfectionstate, Epsilon, Epsilon_asy, Gamma, mu)
        InfectionstateRec = np.append(InfectionstateRec, Newinfectionstate)
        LatentperiodRec = np.append(LatentperiodRec, NewLatentperiod)
        InfectiousperiodRec = np.append(InfectiousperiodRec, NewInfectiousperiod)
        InfectiondateRec = InfectiondateRec[uniind]
        InfectionstateRec = InfectionstateRec[uniind]
        LatentperiodRec = LatentperiodRec[uniind]
        InfectiousperiodRec = InfectiousperiodRec[uniind]
               
        infectious_period = ((t-InfectiondateRec) >= LatentperiodRec) & ((t-InfectiondateRec) <= (LatentperiodRec+InfectiousperiodRec)) # Cases may be infectious
        presymptomatic_period = ((t-InfectiondateRec) >= LatentperiodRec-Gamma) & ((t-InfectiondateRec) <= LatentperiodRec) & (InfectionstateRec==0)
        
        InfectiousAgent = InfectedAgentRec[infectious_period] # infectious period: 3 days
        Infectedstate = InfectionstateRec[infectious_period]
        RemovedAgent = InfectedAgentRec[(t-InfectiondateRec) > (LatentperiodRec+InfectiousperiodRec)]
        InfectiousAgent, locind = np.unique(InfectiousAgent, return_index=True)
        Infectedstate = Infectedstate[locind] # follow the filtering of infected agents
        RemovedAgent = np.unique(RemovedAgent)
        presymptomatic_count = np.sum(presymptomatic_period)
        if len(InfectiousAgent) == 0 or (len(InfectiousAgent) - presymptomatic_count) < 0:
            varying_beta = beta
        else:
            varying_beta = 0.874*beta + 0.126*beta_pre
            varying_beta_asy = (0.5-(1-varying_beta/beta))
        
        NewInfectiousAgent = np.setdiff1d(InfectiousAgent, Onedaytracingdelay)
        NewInfectiousAgent = np.setdiff1d(NewInfectiousAgent, TrackedagentRec)
        Trackedagent = np.intersect1d(InfectiousAgent, Onedaytracingdelay)
        TrackedagentRec = np.append(TrackedagentRec, Trackedagent)
        NewInfectedstate = Infectedstate[np.isin(InfectiousAgent, NewInfectiousAgent)]
        
        if t >= 26:
            if len(InfectiousAgent) <= 15000 and len(InfectiousAgent) > 0: 
                Usedorinal = np.random.choice(InfectiousAgent[Infectedstate==1], np.around(0.5*len(InfectiousAgent[Infectedstate==1])).astype(int), replace=False)
                print("today tested number:", len(Usedorinal))
                testedordinal = np.append(testedordinal, Usedorinal)
                FamilymemberID, OfficecolleagueID = contact_tracing(Usedorinal, InfectiousAgent, HouseAgents, HouseWorker)
                Onedaytracingdelay = np.union1d(FamilymemberID, OfficecolleagueID)
                print("cumlative tested number:", len(testedordinal))
            else:
                print('too many people to test')
        
        etit = time.time()
        print('Main script time consuming:', etit-stit, 's')
        print('Day',t,'House Infectious number is', HI)
        print('Day',t,'Community Infectious number is', CI)
        print('Day',t,'Office Infectious number is', OI)
        print('Day',t,'Kindergarten Infectious number is', KI)
        print('Day',t,'Pri school Infectious number is', PI)
        print('Day',t,'Sec school Infectious number is', SI)
        print('Day',t,'Uni school Infectious number is', UI)
        print('Day',t,'Home restaurant Infectious number is', ReI)
        print('Day',t,'Workplace restaurant Infectious number is', WorkplaceReI)
        print('Day',t,'Shopping mall Infectious number is', mallI)
        print('Day',t,'Market Infectious number is', marketI)
        print('Day',t,'Cumulative Infectious number is', len(InfectedAgentRec))
        SumInfectRec[t, 0] = HI
        SumInfectRec[t, 1] = CI
        SumInfectRec[t, 2] = OI
        SumInfectRec[t, 3] = KI
        SumInfectRec[t, 4] = PI
        SumInfectRec[t, 5] = SI
        SumInfectRec[t, 6] = UI
        SumInfectRec[t, 7] = ReI
        SumInfectRec[t, 8] = WorkplaceReI
        SumInfectRec[t, 9] = mallI
        SumInfectRec[t, 10] = marketI
        SumInfectRec[t, 11] = len(InfectedAgentRec)
        SumInfectRec[t, 12] = len(InfectionstateRec[InfectionstateRec == 1])
        
    for x in infectlocation:
        locdata = np.array(x)
        gatherlocdata = np.append(gatherlocdata, locdata)
    gatherlocdatafinal = np.delete(gatherlocdata, 0, axis=0)
    return SumInfectRec, gatherlocdatafinal
        
if __name__ == '__main__':
    
    # import tranportation network and agents data 
    st = time.time()
    HouseAgentsInput = pd.read_csv('D:/DataUsed/HouseAgentsupdatev3.csv', encoding='UTF-8', header=None)
    HouseAgents = HouseAgentsInput.values
    HouseWorkerInput = pd.read_csv('D:/DataUsed/HouseWorkerupdatev2.csv', encoding='UTF-8', header=None, delimiter=',')
    HouseWorker = HouseWorkerInput.values
    HouseChildInput = pd.read_csv('D:/DataUsed/HouseChildupdate.csv', encoding='UTF-8', header=None)
    HouseChild = HouseChildInput.values
    HousePristudentInput = pd.read_csv('D:/DataUsed/HousePristudentupdate.csv', encoding='UTF-8', header=None)
    HousePristudent = HousePristudentInput.values
    HouseSecstudentInput = pd.read_csv('D:/DataUsed/HouseSecstudentupdate.csv', encoding='UTF-8', header=None)
    HouseSecstudent = HouseSecstudentInput.values
    TotalHouseInput = pd.read_csv('D:/DataUsed/House.csv', encoding='UTF-8', header=None)
    TotalHouse = TotalHouseInput.values
    TotalGridsInput = pd.read_csv('D:/DataUsed/Gridsupdate.csv', encoding='UTF-8', header=None)  
    TotalGrids = TotalGridsInput.values #TotalGridID,TotalGridoutworker,TotalGridoutstudent,TotalGridPopulation,TotalGridDomestichelper, workerdes, TPU
    varying_mobility = np.loadtxt('D:/DataUsed/Mobilityupdate.csv', encoding='UTF-8',delimiter=',')
    UnistudentInput = pd.read_csv('D:/DataUsed/Unistudentupdate.csv', encoding='UTF-8', header=None)
    Unistudent = UnistudentInput.values 
    TPUdata = np.loadtxt('D:/DataUsed/TPUdata.csv', encoding='UTF-8',delimiter=',').astype(int) # TotalTPUID,TotalGridoutworker,TotalGridoutstudent,TotalGridPopulation,TotalGridDomestichelper, workerdes, TPU

    # np.savetxt('D:/DataUsed/HouseWorkerupdatev2.csv', HouseWorkerupdate, delimiter=',', encoding='UTF-8')
    weekdayhomehomecorr = np.loadtxt('D:/DataUsed/ResturantUsed/weekdayhomehomecorr.csv')
    weekdayhomeworkcorr = np.loadtxt('D:/DataUsed/ResturantUsed/weekdayhomeworkcorr.csv')
    weekdayworkhomecorr = np.loadtxt('D:/DataUsed/ResturantUsed/weekdayworkhomecorr.csv')
    weekdayworkworkcorr = np.loadtxt('D:/DataUsed/ResturantUsed/weekdayworkworkcorr.csv')
    Saturdayhomehomecorr = np.loadtxt('D:/DataUsed/ResturantUsed/Saturdayhomehomecorr.csv')
    Sundayhomehomecorr = np.loadtxt('D:/DataUsed/ResturantUsed/Sundayhomehomecorr.csv')
    shoppingpoptranprob = np.loadtxt('D:/DataUsed/EntertainmentUsed/shoppingmall_corr.csv', encoding='UTF-8',delimiter=',')
    poptranprob = np.loadtxt('D:/DataUsed/EntertainmentUsed/poptranprob.csv', encoding='UTF-8') # for Dhelperhappy
    pharmacypoptranprob = np.loadtxt('D:/DataUsed/EntertainmentUsed/market_corr.csv', delimiter=',')
    et = time.time()
    print('Input data using:', et-st)
    
    """
    setting the transmission model
    """
    st = time.time()
    period = 92  # Simulation period
    simulationround = 5 # number of simulations
    ini_seednumber = np.random.randint(low=10,high=16) # Initial infectious inidividual
    R0 = 2.5 # Basic reproduction number 
    Epsilon = 2.9 # latent period for symptomatic cases
    Epsilon_asy = 6.2 # latent period for asymptomatic cases
    Gamma = 3.3 # pre-symptomatic period, i.e. Epsilon + Gamma = Epsilon_asy
    p = 595/3361 # https://chp-dashboard.geodata.gov.hk/covid-19/en.html
    r = 0.5 # relative infectiousness of asymptomatic individuals
    k = 0.126 # proportion of presymptomatic transmission
    mu = 2.9 # time to removed, i.e. infectious period
    beta = (R0*1/mu)/(p*r+(1-p)/(1-k)) # transmission for symptomatic individuals
    beta_pre = beta*(1/Gamma)*k/(1/mu*(1-k)) # transmission for pre-symptomatic individuals
    beta_asy = r*beta # transmission for asymptomatic individuals
    
    duration_period = [13.7, 1.1, 1.0, 0.7, 0.9, 7.4, 6.2 + 1] # Residence, Restaurant, Shopping centre, Market, Public transport, Workplace, Place of study (Canteen)
    et = time.time()
    print('Assigning parameters using:', et-st) 
    
    """
    offically start the simulations
    """
    TotalExport = []
    TotallocExport = []
    allst = time.time()
    for i in range(simulationround):
        tst = time.time()
        ini_Agent_ID = np.random.choice(len(HouseAgents), ini_seednumber, replace=False) # get the initial seeds
        InfectiousAgent = ini_Agent_ID # ini_ Used infected ID
        RemovedAgent = [] # ini_ Removed infected ID
        infectiondate = np.zeros(len(InfectiousAgent)) # ini_ Removed infected ID
        Infectedstate = Assigninfectionstate(InfectiousAgent, p, k) # ini_ Removed infected ID
        Latentperiod, Infectiousperiod = Assigninfectiondate(Infectedstate, Epsilon, Epsilon_asy, Gamma, mu) # random assign date
        InfectedAgentRec = np.sort(InfectiousAgent)
        InfectiondateRec = infectiondate        
        InfectionstateRec = Infectedstate
        LatentperiodRec = Latentperiod
        InfectiousperiodRec = Infectiousperiod
        LatentAgent = []
        Datarecord = []     
        SumInfectRec, SumInfectlocRec = iterations(period, beta, InfectiousAgent, Infectedstate, HouseAgents, TotalHouse, InfectedAgentRec, InfectiondateRec, InfectionstateRec, p, k, LatentperiodRec, InfectiousperiodRec, Epsilon, Epsilon_asy, Gamma, beta_asy, beta_pre, mu, shoppingpoptranprob, pharmacypoptranprob, varying_mobility)
        # np.savetxt('fitting'+str(i+1)+'consle1.csv', SumInfectRec, delimiter=',')
        TotalExport.append(SumInfectRec)
        TotallocExport.append(SumInfectlocRec)
        tet = time.time()
        print('time using:', tet-tst)
    print('total time using', time.time()-allst)
    Totalcumulativenumber = np.array(TotalExport)
    i = 0
    for fp in Totalcumulativenumber:
        np.savetxt('Result'+str(i)+'.csv', fp, delimiter=',')
        i += 1
    # Totalcumulativeloc = np.array(TotallocExport)
    # i = 0
    # for fp in Totalcumulativeloc:
    #     np.savetxt('0331FinalLoclaptop2consle2v2'+str(i)+'.csv', fp, delimiter=',')
    #     i += 1
