import numpy as np 
import copy
from RL_Aux import *

class CLF():
    def __init__(self,apx,env,
                 rewardDiscount,epsilon,epsilonDecay,
                 maxEpisodes,printoutEps,featurize):
        self.Q_Apx = copy.deepcopy(apx)
        self.env = env
        self.nA = env.nA
        self.nS = env.nS
        self.rewardDiscount = rewardDiscount
        self.epsilon0 = epsilon
        self.epsilonDecay = epsilonDecay
        self.maxEpisodes = maxEpisodes
        self.episodeStepsList = []
        self.episodeRewardList = [] 
        self.printoutEps = printoutEps
        if featurize == None:
            self.featurize = lambda x: x
        else:
            self.featurize = featurize
        self.t = 0
    def Train(self,env):
        for episode in range(self.maxEpisodes):
            state = env.reset()
            state = self.featurize(state).reshape([-1,1])
            episodeSteps = 0
            episodeReward = 0
            self.epsilon = np.maximum(0.01,self.epsilon0*self.epsilonDecay**episode)
            while True:
                action = self.PickAction(state)
                nextState,reward, done = env.step(action)
                nextState = self.featurize(nextState).reshape([-1,1])
                self.Optimize(state,nextState,reward,action)
                state = nextState
                episodeSteps += 1
                episodeReward += reward
                if done:
                    self.episodeStepsList.append(episodeSteps)
                    self.episodeRewardList.append(episodeReward)
                    if not episode % self.printoutEps and episode>0:
                        totalSteps = sum(self.episodeStepsList[-self.printoutEps:])
                        totalReward = sum(self.episodeRewardList[-self.printoutEps:])
                        #print('W ',np.sqrt(np.mean(self.Q_Apx.wv[-1]**2)))
                        print('Episode {}/{} ; Steps {} ; Reward {:.4}'
                              .format(episode,self.maxEpisodes, totalSteps/self.printoutEps,totalReward/self.printoutEps))
                        if episode % (self.printoutEps*5)==0 and episode>0:
                            Pickler('pickled.dat',[self.Q_Apx.wv,self.Q_Apx.bv])
                    break
    def Optimize(self,state,nextState,reward,action):
        a,z,Qnow = self.GetQ(state)
        y = Qnow.copy()
        _,_,Qnext = self.GetQ(nextState)
        y[action] =  reward + self.rewardDiscount * np.max(Qnext)
        dz,dw,db = self.Q_Apx.BackProp(y,a,z,dzFunc = 'Linear/L2') # dzFunc is dL/dz = dL/da*da/dz=self.actuators[-1](z[-1],1)
        self.t+=1
        self.Q_Apx.OptimizationStep(dw,db,self.t)
    def GetQ(self,state):
        a,z = self.Q_Apx.ForwardProp(state)
        prediction = a[-1]
        return a,z,prediction                
    def EpsilonPolicy(self,state):
        _,_,Q = self.GetQ(state)
        Q = Q.squeeze()
        bestAction = np.argwhere(Q == np.amax(Q)) # This gives ALL indices where Q == max(Q)
        actionProbablities = np.ones(self.nA)*self.epsilon/self.nA
        actionProbablities[bestAction]+=(1-self.epsilon)/len(bestAction)
        return actionProbablities
    def PickAction(self,state):
        actionP = self.EpsilonPolicy(state)
        action = np.random.choice(self.nA,p = actionP)
        return action
    
