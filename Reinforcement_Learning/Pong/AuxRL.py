import Pong
import numpy as np
import pygame

from MyNN import *

def InitializeNetwork(x,y):
    ## Define Neural Network policy approximator
    # Hyper parameters
    epochs = 10 # Irrelevant to RL
    tolerance = 1e-5 # Irrelevant to RL
    layer_parameters = [[200],
                        [50],
                        ]
    layer_types = ['fc','fc']
    actuators = [[0] ,ReLU2,ReLU2,Softmax]

    alpha = 0.001 # Learning Rate, this is just a temporary placeholder, the actual value is defined in the main loop
    beta1 = 0.9 # Step weighted average parameter
    beta2 = 0.99 # Step normalization parameter
    gamma = 1 # Irrelevant to RL
    epsilon = 1e-8 # Addition to denominator to prevent div by 0
    lam = 1e-5 # Regularization parameter
    NeuralNet = network(epochs,tolerance,actuators,layer_parameters,layer_types,alpha,beta1,beta2,epsilon,gamma,lam)
    NeuralNet.setupLayerSizes(x,y)
    return NeuralNet

def SetupGame(controllerSpeed,ballSpeed):
    # Setup game parameters, objects and controllers
    pi = 3.1415
    
    wallY = 0.1
    vel = Pong.InitiateVelocity(ballSpeed)
    ball = Pong.BallClass([0.5,0.5], vel , wallY)
    
    playerPaddle = Pong.PaddleClass([0.05,0.5],0.1)
    rivalPaddle = Pong.PaddleClass([0.95,0.5],0.1)
    paddles = [playerPaddle,rivalPaddle]
    
    AI = ReinforcedAI(playerPaddle, controllerSpeed*2 , wallY)
    hardCodedAI = Pong.AIController(rivalPaddle,ball, controllerSpeed)
    score = np.array([0,0])
    counter = 0
    pongGame = Pong.PongGame(counter,paddles,ball,AI,hardCodedAI,score,wallY)
    return pongGame

class ReinforcedAI():
    # This is the RL agent,
    # it uses a NN policy approximation to decide it's next action for each state
    def __init__(self,paddle, moveSpeed,wallY):
        self.net = None
        self.state = None
        self.wv = None
        self.bv = None
        self.a = None
        self.z = None
        self.y = None
        self.action = np.zeros([3,1])
        self.paddle = paddle
        self.moveSpeed = moveSpeed
        self.wallY = wallY
    def DecideAction(self,state):
        # Use NN policy approximation to predict the best action for current state
        self.a,self.z = self.net.Forward_prop(state,self.wv,self.bv)
        actionProbabilities = list(self.a[-1].squeeze())
        chosenAction = np.random.choice ([0,1,2], p = actionProbabilities )
        self.y = np.zeros([3,1])
        self.y[chosenAction] = 1
        self.action = self.y
    def Update(self):
        # Implement action for the current game step
        if self.action[0] ==1 and self.paddle.pos[1] < (1-self.wallY)  :
            self.paddle.pos[1] += self.moveSpeed
        elif self.action[2] == 1 and self.paddle.pos[1] > self.wallY:
            self.paddle.pos[1] -= self.moveSpeed
            
def SimpleAI(state,paddleWidth):
    # Simple agent algorithm - go after the ball
    # State : [pad1.y, pad2.y,ball.x,ball.y,ball.vx,ball.vy,vy/vx]
    paddleY = state[0]
    ballY = state[3]
    y = np.zeros([3,1])
    if paddleY - ballY > paddleWidth/8:
        y[2] = 1
    elif ballY - paddleY > paddleWidth/8:
        y[0] = 1
    else:
        y[1] = 1
    return y
    
def PlayGames(gamesToPlay,pongGame):
    # This allows viewing games in real time to see the current RL agent performance
    pygame.init()
    exitGame = False
    gamesPlayed = 0
    displayWidth = 800
    displayHeight = 600
    pygame.font.init() 
    myfont = pygame.font.SysFont('Comic Sans MS', 30)
    clock = pygame.time.Clock()
    pongGame.score = np.array([0,0])
    gameDisplay = pygame.display.set_mode((displayWidth,displayHeight))
    frameCounter = 0
    while not exitGame:
        frameCounter +=1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exitGame = True
        pongGame.player.DecideAction (pongGame.state)
        pongGame.RunFrame()
        if (pongGame.ball.deflect) and (pongGame.ball.pos[0]<0.5):
            print('deflect')
        if pongGame.ball.goal:
            pygame.time.wait(100)
            gamesPlayed += 1
         #Rendering
        pongGame.Render(gameDisplay,displayWidth,displayHeight)
        clock.tick(60)
        if (gamesPlayed >= gamesToPlay):
            exitGame = True
    pygame.quit()


def Pickler(fileName,data):
        print('pickled')
        PIK = fileName
        with open(PIK, "wb") as f:
            pickle.dump(data, f)
            
def Unpickler(PIK):
    with open(PIK, "rb") as f:
        data = pickle.load(f)
    return data

def InitializeCaches(wv,bv):
    dwCache = []
    dbCache = []
    for index,_ in enumerate(wv):
        dwCache.append(np.zeros_like(wv[index]))
        dbCache.append(np.zeros_like(bv[index]))
    return dwCache,dbCache

def FactorCaches(dwCache,dbCache,factor):
    for index,_ in enumerate(wv):
        if index > 0:
            dwCache[index] *= factor
            dbCache[index] *= factor
            
def GetReward(deltaScore,ball,paddle):
    if ball.deflect and ball.pos[0]<0.5 :
        reward = 1.0
    elif deltaScore[1] == 1:
        reward = -0.1
    else:
        reward = None
    return reward

def DirectLearner(pongGame,state,wv,bv,alpha):
    # This function teaches the NN based on the decisions of some hardcoded AI
    # Currently disabled
    global t
    pongGame.player.DecideAction(state)
    a,z,y = pongGame.player.a,pongGame.player.z,pongGame.player.y
    paddleWidth = pongGame.paddles[0].width
    y = SimpleAI(state,paddleWidth)
    dz,dw,db = NeuralNet.Back_prop(y,a,z,wv)
    t += 1
    decay = 1 # No decay
    NeuralNet.alpha = alpha
    wv,bv = NeuralNet.Optimization_step(wv,bv,decay,t)
    return wv,bv,Vw,Vb,Sw,Sb,y


class ReinforcementLearner():
    # This is the main learning class
    def __init__(self,pongGame, NeuralNet,wv,bv,alpha,maxBatches,rewardDecay,learningDecay = False):
        
        self.wv,self.bv = [wv,bv]
        self.rewardDecay = rewardDecay # Reward decay over
        self.maxBatches = maxBatches # Number of batches before weights are updated
        self.t = 0
        self.batchCounter = 0
        self.frameCounter = 0
        self.pongGame = pongGame
        self.NN = NeuralNet
        self.dwBatch,self.dbBatch = InitializeCaches(wv,bv)
        self.dwTotal,self.dbTotal = InitializeCaches(wv,bv)
        self.alpha = alpha
        self.NN.alpha = self.alpha
        # decay parameters
        self.decay = 1 # intial learning rate multiplier
        self.learningDecay = learningDecay
        self.I0 = 1e5
        self.A0 = 0.01

    def update(self,state,deltaScore):
        reward = GetReward(deltaScore,self.pongGame.ball,self.pongGame.player.paddle)
        if reward:
            for index,_ in enumerate(self.wv):
                if index > 0:
                    self.dwTotal[index] +=  self.dwBatch[index]*reward/self.maxBatches
                    self.dbTotal[index] +=  self.dbBatch[index]*reward/self.maxBatches
            self.dwBatch,self.dbBatch = InitializeCaches(self.wv,self.bv)
            self.batchCounter += 1
            if self.batchCounter >= self.maxBatches:
                self.t += 1 # Time counter for ADAM
                if self.learningDecay:
                    self.decay = np.exp(np.log(self.A0)*self.t/I0) # decay = A0 (@ t = I0)
                # ADAM optimization step
                self.wv,self.bv = self.NN.Optimization_step(self.wv,self.bv,self.dwTotal,self.dbTotal,
                                                                self.decay,self.t)
                self.dwTotal,self.dbTotal = InitializeCaches(self.wv,self.bv)
                self.batchCounter = 0
        
        if self.pongGame.ball.vel[0]<0:
            # This condition allows the agent to decide and learn only when the ball goes in its direction (left)
            # This improve learning rate, but might harm overall optimization
            self.pongGame.player.DecideAction(state)
            a,z,y = self.pongGame.player.a,self.pongGame.player.z,self.pongGame.player.y
            dz,dw,db = self.NN.Back_prop(y,a,z,self.wv)
            for index,_ in enumerate(self.wv):
                self.dwBatch[index] = self.rewardDecay * self.dwBatch[index] + dw[index]
                self.dbBatch[index] = self.rewardDecay * self.dbBatch[index] + db[index]
        # Update player weights
        self.pongGame.player.wv = self.wv
        self.pongGame.player.bv = self.bv
