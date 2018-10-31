import pygame
import time
import random
import os
import math
import numpy as np
from itertools import cycle
pygame.init()
 
    
def xy_trans(x,y):
    # Transfer pixel coordinates to normalize Cartesian
    x_t = int( x*displayWidth )
    y_t = int( (1-y)*displayHeight )
    return (x_t,y_t)
    

# Game object classes 
class PaddleClass():
    def __init__(self,pos,width):
        self.pos = pos
        self.vel = np.array([0,0])
        self.width = width
        
class BallClass():
    def __init__(self,pos,vel,wallY):
        self.pos = np.array(pos)
        self.vel = np.float64(vel)
        self.totalVel = float(np.sqrt(self.vel[0]**2+self.vel[1]**2))
        self.deflect = False
        self.goal = False
        pi = 3.1415
        self.maxAngle = 60*pi/180
        self.wallY = wallY
    def PadCollision(self,pad):
        # Check for collision with a pad
        if (np.abs(self.pos[1] - pad.pos[1])<(pad.width/2)):
            if (np.abs(self.pos[0] - pad.pos[0])<np.abs(self.vel[0])):
                self.pos[0] = pad.pos[0]
                self.deflect = True
    def WallCollision(self):
        # Check for collision with a pad
        if self.pos[1] <= self.wallY:
            self.pos[1] = self.wallY 
        elif self.pos[1]>= (1-self.wallY):
            self.pos[1] = (1-self.wallY)
        else:
            return
        self.vel[1] = -self.vel[1]
    def GoalCheck(self):
        # Check if a goal is scored
        if self.pos[0] <=0 or self.pos[0]>=1:
            self.goal = True
        else:
            self.goal = False
    def Update(self,paddles):
        # Ball updates every frame
        self.deflect = False
        self.pos += self.vel
        self.WallCollision()
        self.GoalCheck()
        for pad in paddles:
            self.PadCollision(pad)
            if self.deflect:
                self.angle = float((self.pos[1] - pad.pos[1])/(pad.width/2)*self.maxAngle)      
                self.vel[0] = self.totalVel*(np.cos(self.angle))*float(-np.sign(self.vel[0]))
                self.vel[1] = self.totalVel*(np.sin(self.angle))
                self.pos[0] += self.vel[0]*2
                break
                

class AIController():
    def __init__(self,paddle,ball,moveSpeed):
        self.paddle = paddle
        if self.paddle.pos[0] < 0.5:
            self.side = -1
        else:
            self.side = 1
        self.ball = ball
        self.moveSpeed = moveSpeed
        
    def Update(self):
        if self.side*self.ball.vel[0] > 0:
            destination = self.ball.pos[1]
        else:
            destination = 0.5
        distance = (destination-self.paddle.pos[1])
        if abs(distance)>=self.paddle.width/4:
            self.paddle.pos[1] += np.sign(distance) * self.moveSpeed

class AIControllerTrajectory():
    def __init__(self,paddle,ball,moveSpeed,wallY):
        self.paddle = paddle
        if self.paddle.pos[0] < 0.5:
            self.side = -1
        else:
            self.side = 1
        self.ball = ball
        self.moveSpeed = moveSpeed
        self.ballDestination = 0.5
        self.wallY = wallY
    def CalcTrajectory(self,ball):
        Y = (ball.pos[1])+ball.vel[1]/ball.vel[0]*(1-ball.pos[0]-(1-self.paddle.pos[0]))
        fieldHeight = 1-self.wallY*2
        delta = (Y-self.wallY)% ( fieldHeight)
        modder = (Y-self.wallY)// ( fieldHeight)
        if modder%2 == 0:
            destination = delta
        else:
            destination = fieldHeight -delta
        self.ballDestination = destination + self.wallY 
    def Update(self):
        if self.side*self.ball.vel[0] > 0:
            self.CalcTrajectory(self.ball)
        else:
            self.ballDestination = None
        if self.ballDestination == None:
            self.ballDestination = 0.5
        distance = (self.ballDestination-self.paddle.pos[1])
        if abs(distance)>=self.paddle.width/4:
            self.paddle.pos[1] += np.sign(distance) *self.moveSpeed
       
def InitiateVelocity(totalVel):
    pi = 3.1415
    angleArc = 10
    angleOptions = np.squeeze([angleArc*(np.random.rand(1)-0.5),180+angleArc*(np.random.rand(1)-0.5)])
    ngleOptions = np.squeeze([180+angleArc*(np.random.rand(1)-0.5)])
    angle = pi/180*np.random.choice(angleOptions)
    #print(angle*180/pi)
    
    velX = float(totalVel*(np.cos(angle)))
    velY = float(totalVel*(np.sin(angle)))
    vel = np.array([velX,velY])
    return vel 



def ScoreUpdate(ball,score):
    if ball.pos[0] <=0:
        score[1] += 1
    if ball.pos[0] >=1:
        score[0]  += 1
    return score

def InitializeState(ball,paddles):
    vel = InitiateVelocity(ball.totalVel)
    ball.pos = np.array([0.50,0.50])
    ball.vel = vel
    for pad in paddles:
        pad.pos[1] = 0.5





class PongGame():
    def __init__(self,counter,paddles,ball,player,AI,score,wallY):
        self.counter = counter
        self.paddles = paddles
        self.ball = ball
        self.player = player
        self.AI = AI
        self.score = score
        self.wallY = wallY
        self.deltaScore = np.array([0,0])
    def RunFrame(self):
        self.deltaScore = np.array([0,0])
        maxFrames = 5000
        if self.ball.goal or self.counter >= maxFrames:
            InitializeState(self.ball,self.paddles)
            self.ball.deflect = False
            self.ball.goal = False
            self.StateUpdate()
            return
        self.counter += 1
        self.ball.Update(self.paddles)
        self.AI.Update()
        self.player.Update()
        if self.ball.goal or self.counter >= maxFrames:
            if self.counter >= maxFrames:
                print('Counter Expired')
            oldScore = np.copy(self.score)
            self.score = ScoreUpdate(self.ball,self.score)
            self.deltaScore = self.score - oldScore
            #pygame.time.wait(500)
            self.counter = 0
        self.StateUpdate()
    def StateUpdate(self):
        state = []
        for pad in self.paddles:
    ##        state += list(pad.pos)
            state += [pad.pos[1]]
        state += list(self.ball.pos)
        state += list(self.ball.vel)
        state += [self.ball.vel[1]/self.ball.vel[0]]
        state = np.array(state)
        state = state[:,None]
        self.state = state
    def Render(self,gameDisplay,displayWidth,displayHeight):
        gameDisplay.fill(black)
        pygame.draw.line(gameDisplay,white,xy_trans(0,self.wallY),xy_trans(1,self.wallY),2)
        pygame.draw.line(gameDisplay,white,xy_trans(0,1-self.wallY),xy_trans(1,1-self.wallY),2)   
        
        # Score
        sString = myfont.render(str(self.score[0]),False,white)
        gameDisplay.blit(sString,(10,10))
        sString = myfont.render(str(self.score[1]),False,white)
        gameDisplay.blit(sString,(displayWidth-50,10))

        for pad in self.paddles:
            pygame.draw.line(gameDisplay,white,xy_trans(pad.pos[0],pad.pos[1]-pad.width/2),xy_trans(pad.pos[0],pad.pos[1]+pad.width/2),5)
        pygame.draw.circle(gameDisplay,dark_red,xy_trans(self.ball.pos[0],self.ball.pos[1]),6)
        pygame.draw.circle(gameDisplay,red,xy_trans(self.ball.pos[0],self.ball.pos[1]),4)
        pygame.display.update()
            
def game_loop(displayWidth,displayHeight):
    exitgame=False  
    pi = 3.1415
    gameState = 'RUNNING'
    ground_level = 0.2
    playerPaddle = PaddleClass([0.05,0.5],0.1)
    rivalPaddle = PaddleClass([0.95,0.5],0.1)
    moveSpeed = 0.01
    wallY = 0.15
    ballStartVel = 0.01
    vel = InitiateVelocity(ballStartVel)
    ball = BallClass([0.5,0.5], vel , wallY)
    AI = AIControllerTrajectory(rivalPaddle,ball, moveSpeed/3.3,wallY )
    hardCodedAI = AIController(playerPaddle,ball, moveSpeed )
    paddles = [playerPaddle,rivalPaddle]
    

    
    score = np.array([0,0])
    
    counter = 0
    pongGame = PongGame(counter,paddles,ball,AI,hardCodedAI,score,wallY)
    while not exitgame:        
        # In case quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p: gameState = 'PAUSED'
            if event.key == pygame.K_o: gameState = 'RUNNING'
            if event.key == pygame.K_DOWN:
                playerPaddle.pos[1] -= 0.005
            if event.key == pygame.K_UP:
                playerPaddle.pos[1] += 0.005
        
        if gameState == 'RUNNING':
            
            pongGame.RunFrame()
            if pongGame.ball.deflect:
                print('deflect')
            pongGame.Render(gameDisplay,displayWidth,displayHeight)
        elif gameState == 'PAUSED':
            pass
        
        clock.tick(60)



# Colors:
black = (0,0,0)
white = (255,255,255)
red = (200,0,0)
green = (0,200,0)
blue = (0,0,255)
nokia_background = (100,160,120)
nokia_background_org = (136,192,157)
bright_red = (255,0,0)
bright_green = (0,255,0)
dark_red = (160,0,0)
displayWidth = 800
displayHeight = 600
pygame.font.init() 
myfont = pygame.font.SysFont('Comic Sans MS', 30)
        
if __name__ == "__main__":
    # Display
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (200, 50)
    modes = pygame.display.list_modes(32)
    gameDisplay = pygame.display.set_mode((displayWidth,displayHeight))
    pygame.display.set_caption("Evo")
    clock = pygame.time.Clock()
    # Run game
    game_loop(displayWidth,displayHeight)
