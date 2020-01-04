import matplotlib.pyplot as plt
import copy
import sys

sys.path.append('.\\Envs\\')
import Pong,TrackRunner,GridWorld,Jumper
from TrackBuilder import *
sys.path.append('.\\RL_Algorithms\\')
import TDL,TDL_Linear,DQN,QL
from RL_Aux import *
from Evo_Aux import *

sys.path.append('.\\DecoupledNN\\')
import DecoupledNN



## Build Env        
track = TrackRunner.BuildRoundTrack(7,14)
trackName = 'third'
track = pickle.load(open(".\envs\Tracks\\"  + trackName + ".dat","rb"))
env = TrackRunner.TrackRunnerEnv(runVelocity = 0.02,turnDegrees = 15,track = track)

## Create Approximators
saveFile = None
# Approximators
np.random.seed(48)
linApx = TDL_Linear.LinearApproximator(nS= env.nS,nA = 3,learningRate = 1e-3,featurize= None,saveFile = None)
QnetApx = SetupNeuralNetApx(nS= env.nS,nA = 3,learningRate=1e-3,featurize= None,saveFile = saveFile)
dcNN = DecoupledNN.DecoupledNN(learningRate=5e-4,batchSize = 500,batches=20,maxEpochs=100,
                            netLanes = env.nA, layerSizes = [200],inputSize =env.nS,activationFunctions = [[],ReLU2,Softmax])
if saveFile==None:
  NullifyQs(QnetApx,env)

## Evo Optimization
print('\nGA Optimization')
EvoNet = copy.deepcopy(QnetApx)

def EvoAgent(state):
    a,_ = EvoNet.ForwardProp(state)
    action = np.argmax(a[-1])
    return action
evoAgent= EvoAgent
fitness = EvoFitnessFunction(EvoNet,env,evoAgent)
gao = GAOptimizer(specimenCount = 100,survivorCount=20,tol = 0,maxIterations = 5000,
                  mutationRate = 0.1,generationMethod = "Random Splice",smoothing =1)
gao.Optimize(EvoNet.wv + EvoNet.bv,fitness)

## RL Optimization
maxEpisodes = 5000
# List of classifiers to train

clfs =[
##        TDL.CLF(QnetApx,env, rewardDiscount = 0.95,lam = 0.95, epsilon = 0.3, epsilonDecay = 0.95,
##            maxEpisodes = maxEpisodes , printoutEps = 100, featurize= None),
        
##        TDL_Linear.CLF(linApx,env,rewardDiscount = 0.95,lam = 0,epsilon = 0.3,epsilonDecay = 0.95,
##            maxEpisodes = maxEpisodes , printoutEps = 100),
        
##        DQN.CLF(QnetApx,env,rewardDiscount = 0.95, epsilon = 0.3, epsilonDecay = 0.95,
##            maxEpisodes = maxEpisodes , printoutEps = 100, featurize = None,
##                experienceCacheSize=100, experienceBatchSize=10, QCopyEpochs=50),
        
        QL.CLF(QnetApx,env,rewardDiscount = 0.95, epsilon = 0.3, epsilonDecay = 0.95,
            maxEpisodes = maxEpisodes , printoutEps = 100,featurize = None)
        ]

## Training
print('\nRL Optimization')
for i in range(len(clfs)):
    print('\nTraining Classifier #',i+1)
    clfs[i].Train(env)
    
## Plot
plt.close('all')

episodes = len(clfs[0].episodeStepsList)
windowSize = int( episodes*0.02 )

xVector = np.arange(episodes-windowSize+1)
# Plot steps over episodes
plt.close('all')
plt.figure(1)
for c in clfs:
    plt.semilogy(xVector, MovingAverage(c.episodeStepsList,windowSize))
plt.xlabel('Episode #')
plt.ylabel('Number of Steps')
plt.legend(['lam = 0','lam = 0.95'])

# Plot rewards over episodes
plt.figure(2)
for c in clfs:
    plt.plot(xVector, MovingAverage(c.episodeRewardList,windowSize))
plt.xlabel('Episode #')
plt.ylabel('Total Reward')
plt.show(block=False)

## Example
ce = 0
clfs[ce].stateHistory=[]
clfs[ce].QHistory=[]
clfs[ce].steps=0
def RLagent(state):
    clfs[ce].steps+=1
    action = clfs[ce].PickAction(state)
    QNow = clfs[ce].Q_Apx.Predict(state)
    clfs[ce].stateHistory.append(state)
    clfs[ce].QHistory.append(QNow)
    return action

agent = RLagent
##agent = evoAgent
env.score = np.array([0,0])
RunEnv(runs = 1,env = env,agent = agent,frameRate=30)     

