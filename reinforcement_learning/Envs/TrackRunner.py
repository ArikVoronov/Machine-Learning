import pygame
import numpy as np
import os
import time
import pickle

from TrackBuilder import *

def GetLineParameters(pos,angle):
    # Get line eq parameters y = m*x + n from point coordinates and angle
    if angle != 90 and angle!=-90:
        mi = np.tan(angle*np.pi/180)
        ni = pos[1] - mi * pos[0]
    return mi,ni

def LinesIntersect(m1,n1,m2,n2):
    # Get calculated point of intersection between 2 lines
    x = - (n2-n1)/(m2-m1+1e-20)
    y = m2*x + n2
    return x,y
    
def FixTo180(angle):
    # Fix angle to be in range between -180 and 180
    newAngle = angle
    while newAngle > 180: newAngle-=360
    while newAngle <-180: newAngle+=360
    return newAngle

def CheckInLine(point,vertices):
    # Check if point is inside of line defined by 2 vertices
    inLine = False
    v1 = vertices[0]-point
    v2 = vertices[1]-point
    dotProduct = np.dot(v1,v2)/(np.linalg.norm(v1,2)*np.linalg.norm(v2,2)+1e-20)
    if np.abs(dotProduct+1) < 0.01:
        inLine = True
    return inLine

def DistToPoint(p1,p2):
    # Distance between 2 points
    d = np.sqrt( (p2[1]-p1[1])**2 + (p2[0]-p1[0])**2 )
    return d

def DistToLine(point,m,n):
    d = np.abs(point[1] - m*point[0] - n)/(np.sqrt(1+m**2))
    return d
                                            
class Player():
    def __init__(self,startingPos,startingDirection,track,runVelocity,turnDegrees):
        self.vertLists,self.lineLists = track.vertLists, track.lineLists
        self.pos = np.array(startingPos)
        self.dir = startingDirection
        self.initialSpeed = runVelocity
        self.speed = runVelocity
        self.acel = self.speed * 0.01
        self.vel = (0,0)
        self.turnDegrees = turnDegrees
        self.collide = False
        self.sensedPointList =[]
        self.senseAngles = [-90, -45, 0, 45, 90]
        self.allLines = []
        for l in self.lineLists:
            self.allLines += l
        parms = [[line.m,line.n] for line in self.allLines]
        parms = np.array(parms)
        self.mLines = parms[:,0];self.nLines = parms[:,1]
    def Update(self,action):
        # Actions : 0 - Nothing; 1 - Left ; 2 - Right ;  3 - Accelerate ; 4 - Decelerate ; 
        if action == 1: self.dir += self.turnDegrees
        if action == 2: self.dir -= self.turnDegrees
        if action == 3: self.speed += self.acel
        if action == 4: self.speed -= self.acel
        self.vel = (np.cos(self.dir*np.pi/180),np.sin(self.dir*np.pi/180))
        self.pos+=np.array([self.vel[0]*self.speed,self.vel[1]*self.speed])
        self.sensedPointList = self.Sensors()
        self.CollisionDetection()
    def Sensors(self):
        pointList = []
        for a in self.senseAngles:
            pClose = [0.5,0.5]
            d = 1
            angle = FixTo180(self.dir + a) # Get absolute angle of sensor beam
            # This is a cheat, to make sure we don't get vertical lines
            # it works ok because screen coordinates are limited to values [0,1]
            if angle == 90:
                angle = 90.1
            if angle == -90:
                angle = -90.1
            mi,ni = GetLineParameters(self.pos,angle) # Get line parameters for sensor beam

            # Find closest intesection between beam and track lines
            xii,yii = LinesIntersect(self.mLines,self.nLines,mi,ni)
            dii = DistToPoint(self.pos,[xii,yii])
            cArray = np.array([xii,yii,dii]).T
            lineIndex = cArray[:,-1].argsort()
            cArray = cArray[lineIndex,:]
            for i in range(cArray.shape[0]):
                p = cArray[i,0],cArray[i,1]
                di = cArray[i,2]
                line = self.allLines[lineIndex[i]]
                pos = self.pos
                pDir = 180/np.pi*np.arctan2(p[1]-pos[1],p[0]-pos[0]) # Actual direction (compare to sensor angle)
                if (pDir-angle)**2<0.01:
                    check = CheckInLine(np.array(p),[line.v1.pos,line.v2.pos]) # Is intersection point inside of line
                    if check:
                        d = di
                        pClose = p
                        break
            pointList.append([pClose,d])
        return pointList
    def CollisionDetection(self):
        self.collide = False
        # Collide with screen edges
        if (self.pos <=0).any() or (self.pos>=1).any(): self.collide = True
        # Collide with lines
        senseArray = np.array(self.sensedPointList)[:,1]
        if senseArray[2]< self.speed: # if almost hitting a wall with fwd sensor
            self.collide = True
        if np.min(senseArray)< 0.005:
            self.collide = True
    def Render(self,gameDisplay,Tr):
        pygame.draw.circle(gameDisplay, red,
                           Tr(self.pos+0.01*np.array([np.cos(self.dir*np.pi/180),np.sin(self.dir*np.pi/180)])),
                           3)
        pygame.draw.circle(gameDisplay, green, Tr(self.pos), 5)

class TrackRunnerEnv():
    def __init__(self,runVelocity,turnDegrees,track):
        self.track = track
        self.turnDegrees = turnDegrees
        self.runVelocity = runVelocity
        self.nA = 3
        self.nS = 5
        self.reset()
    def reset(self): 
        self.player = Player(self.track.spPos,self.track.spDir,self.track,self.runVelocity,self.turnDegrees)
        self.done = False
        self.steps = 0
        self.player.sensedPointList = self.player.Sensors()
        self.state = self.GetState()
        return self.state
    def step(self,action=-1):
        self.steps += 1
        self.player.Update(action)
        self.GetState()
        self.reward = self.GetReward()
        if self.player.collide or self.steps >1000:
            if self.steps >1000:
                print('Timed out')
            self.done = True
        return self.state, self.reward, self.done
    def GetReward(self):
        factor = self.player.speed/self.player.initialSpeed # factor=1 for constant speed
        reward = 0.1 * factor
        if self.player.collide:
            reward = -10
        return reward     
    def GetState(self):
        self.state = np.zeros([self.nS,1])
        for i,s in enumerate(self.player.sensedPointList):
            self.state[i] =  s[1]
##        self.state[-5] = self.player.speed/self.turnDegrees
##        self.state[-4],self.state[-3] = self.player.vel[0],self.player.vel[1]
##        self.state[-2],self.state[-1] = self.player.pos
        return self.state
    def Render(self,gameDisplay):
        displayWidth, displayHeight = pygame.display.get_surface().get_size()
        def Tr(point):
        # Translate cartesian normalized coordinates to pygame screen coordinates
            x = int(point[0] * displayWidth)
            y = int((1-point[1]) * displayHeight)
            return [x,y]
        gameDisplay.fill((0,0,0))
        self.track.Render(gameDisplay,Tr)
        self.player.Render(gameDisplay,Tr)
        # Sensed points
        if len(self.player.sensedPointList)>0:
            for s in self.player.sensedPointList:
                if s[0] != None: pygame.draw.circle(gameDisplay, (150,150,150), Tr(s[0]), 5)





# Colors:
blue = (0,0,200)
green = (0,200,0)
red = (200,0,0)

if __name__ == "__main__":
    def HumanPlayer(state):
        # Must run pygame.event.get() previously to execute:
        left = pygame.key.get_pressed() [pygame.K_LEFT]
        right = pygame.key.get_pressed() [pygame.K_RIGHT]
        up = pygame.key.get_pressed() [pygame.K_UP]
        down = pygame.key.get_pressed() [pygame.K_DOWN]
        action = 0 # do nothing
        if left: action = 1
        if right: action = 2
        if up: action = 3
        if down: action = 4
        return action


    track = BuildRoundTrack(9,14)
    trackName = "roundish"
    track = pickle.load(open(".\Tracks\\"  + trackName + ".dat","rb"))
    runVelocity = 0.01
    turnDegrees = 10

    agent = HumanPlayer
    ##agent = lambda x: 2
    env = TrackRunnerEnv(runVelocity,turnDegrees,track)

    RunEnv(runs = 2,env = env,agent = agent,frameRate=30)        
