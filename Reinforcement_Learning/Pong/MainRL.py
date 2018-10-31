import matplotlib.pyplot as plt

from AuxRL import *

# Setup game
pongGame = SetupGame(controllerSpeed = 0.01,ballSpeed = 0.04)
pongGame.RunFrame()
  
# Setup neural network policy approximator
y = np.zeros([3,1]) # Required to intialize weights
state = pongGame.state # Required to intialize weights
NeuralNet = InitializeNetwork(state,y)

wv0,bv0 = NeuralNet.initialize()
wv = None
bv = None

# Unpickle -  Data = [wv,bv]
##wv,bv = Unpickler("wSaved.dat")

if wv == None:
    wv = wv0.copy()
if bv == None:
    bv = bv0.copy()
    
# Insert NN into RL controller 
pongGame.player.net = NeuralNet
pongGame.player.wv = wv
pongGame.player.bv = bv

# Setup RL object
alpha = 5e-4 # This is the actual learning rate, it is assigned to the network by the learning algorithm, this is convinient for learning decay
maxBatches = 2
rewardDecay = 0.99
RLC = ReinforcementLearner(pongGame, NeuralNet,wv,bv,alpha,maxBatches,rewardDecay,learningDecay = False)

# Initialize loop variables
gamesPlayed = 0
scoreTemp = np.array([0,0])

gamesList = []
winList = []


print("Start Learning")
while gamesPlayed<30000:
    # Run one game frame
    pongGame.RunFrame()

    state = pongGame.state# State = [pad1.y ,ball.x,ball.y,ball.vx,ball.vy,vy/vx]
    deltaScore = pongGame.deltaScore
    # Reinforcement Learning Step - updates weights for both RL class and the player controller
    RLC.update(state,deltaScore)
    if pongGame.ball.goal:
        gamesPlayed +=1
        scoreTemp += deltaScore
        frameCounter = 0
        if not gamesPlayed % 100:
            winP = scoreTemp[0]/100 # win percentage of last k games
            gamesList.append(gamesPlayed)
            winList.append(winP) 
            print('Games Played', gamesPlayed)
            print('Win %%  %5.4f' % winP ) 
            print('Weights mean %7.4f' % (np.mean(RLC.wv[-1]**2))) # This track the weights' magnitude
            scoreTemp = np.array([0,0])
        if not gamesPlayed % 500:
            Pickler('pickled.dat',[wv,bv])

# Plot progress - win percentage as a function of games played
plt.figure(1)
plt.plot(gamesList,winList)
plt.show()
