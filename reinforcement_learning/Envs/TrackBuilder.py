import os
import pygame
import numpy as np
import pickle
#Runner Track Builder


class Vertex():
    def __init__(self,pos):
        self.pos = pos
        self.lines = []

class Line():
    def __init__(self,v1,v2,ID):
        self.ID = ID
        v1.lines.append(self)
        v2.lines.append(self)
        self.v1 = v1
        self.v2 = v2
        self.m = (v2.pos[1]-v1.pos[1])/(v2.pos[0]-v1.pos[0]+1e-20)
        self.n = v1.pos[1] - self.m * v1.pos[0]

class Loop():
    def __init__(self,startID):
        self.startID = startID
        self.vertList = []
        self.lineList = []
        self.done = False
        self.ID = startID
    def Update(self,mouse,UnTr,Tr):
        if mouse[0]:
            self.ID+=1
            vi = Vertex(UnTr(pygame.mouse.get_pos()))
            self.vertList.append(vi)
            if len(self.vertList) > 1:
                self.lineList.append(Line(self.vertList[-2],self.vertList[-1],self.ID))
        if mouse[2]: # connect last point to the first, finish the loop
            self.ID+=1
            self.lineList.append(Line(self.vertList[-1],self.vertList[0],self.ID))
            self.done = True

class Track():
    def __init__(self,spPos,spDir,vertLists,lineLists):
        self.spPos = spPos
        self.spDir = spDir
        self.vertLists = vertLists
        self.lineLists = lineLists
    def Render(self,gameDisplay,Tr):
        if len(self.vertLists) > 0:
            for loop in range(len(self.vertLists)):
                for v in self.vertLists[loop]:
                    pygame.draw.circle(gameDisplay, blue, Tr(v.pos), 6)
                for l in self.lineLists[loop]:
                        pygame.draw.line(gameDisplay, blue, Tr(l.v1.pos), Tr(l.v2.pos), 4)

class StartingPoint():
    def __init__(self,pos = None, direction = None):
        self.pos = pos
        self.dir = direction
    def Pick(self,mouse,UnTr,Tr):
        mousePos = pygame.mouse.get_pos()
        pygame.draw.circle(gameDisplay, (0,100,0), mousePos, 6)
        if self.pos == None:
            if mouse[0]:
                self.pos = UnTr(pygame.mouse.get_pos())
        else:
            pygame.draw.circle(gameDisplay, green, Tr(self.pos), 6)
            if mouse[0]:
                p = UnTr(pygame.mouse.get_pos())
                self.dir = 180/np.pi*np.arctan2(p[1]-self.pos[1],p[0]-self.pos[0])

    def Render(self,gameDisplay,Tr):
        if self.dir != None:
            pygame.draw.circle(gameDisplay, red, Tr(self.pos+0.01*np.array([np.cos(self.dir*np.pi/180),np.sin(self.dir*np.pi/180)])), 4)
        if self.pos != None:    
            pygame.draw.circle(gameDisplay, green, Tr(self.pos), 6)

def BuildTrack(gameDisplay):
    displayWidth, displayHeight = pygame.display.get_surface().get_size()
    def Tr(point):
    # Translate cartesian normalized coordinates to pygame screen coordinates
        x = int(point[0] * displayWidth)
        y = int((1-point[1]) * displayHeight)
        return [x,y]
    
    def UnTr(point):
        x = float(point[0])/displayWidth
        y = 1 - float(point[1])/displayHeight
        return [x,y]

    
    loopCount = 2
    loops = [Loop(10**l) for l in range(loopCount)]
    startingPoint = StartingPoint()
    track = Track(None,None,[],[])
    delay = 0
    l = 0
    exitGame=False
    while not exitGame:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exitGame = True
        if delay < 10:
            mouse = (0,0,0)
            delay+=1
        else:
            mouse = pygame.mouse.get_pressed()
            if sum(mouse)>0:
                delay = 0

        # Create inner and outer loops for the track
        if l < loopCount:
            loops[l].Update(mouse,UnTr,Tr)
            if loops[l].done:
                l+=1
        # Create starting point and direction
        else:
            if startingPoint.dir == None:
                startingPoint.Pick(mouse,UnTr,Tr)
            else:
                exitGame = True
        track.vertLists = [l.vertList for l in loops]
        track.lineLists = [l.lineList for l in loops]
        # Render
        gameDisplay.fill((0,0,0))
        track.Render(gameDisplay,Tr)
        startingPoint.Render(gameDisplay,Tr)
        pygame.display.update()
        clock.tick(60)
        
    track.spPos,track.spDir = startingPoint.pos,startingPoint.dir

    # Save track as inputed name
    trackName = input("Track name:")
    pygame.image.save(gameDisplay, ".\Tracks\\"  + trackName + ".jpg")
    pygame.quit()

    pickle.dump(track,open(".\Tracks\\"  + trackName + ".dat","wb"))
    print("Saved track object as: ", trackName)

def LoadAndRender(trackName):
    [[spPos,spDir],track] = pickle.load(open(".\Tracks\\"  + trackName + ".dat","rb"))
    startingPoint = StartingPoint(spPos,spDir)
    pygame.init()
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (200, 50)
    modes = pygame.display.list_modes(32)
    gameDisplay = pygame.display.set_mode((displayWidth,displayHeight))
    pygame.display.set_caption("RunnerTrack")

    exitGame = False
    while not exitGame:
    # In case quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exitGame = True
        startingPoint.Render(gameDisplay)
        track.Render(gameDisplay)
        pygame.display.update()
    pygame.quit()


def BuildRoundTrack(vertIn,vertOut):
        # inside bound
        dAngle = (360/vertIn)*np.pi/180
        ri = 0.25
        vertInList = []
        lineInList = []
        for i in range(vertIn):
            xi = 0.5 + ri*np.cos(dAngle*(i))
            yi = 0.5 + ri*np.sin(dAngle*(i))
            vi = Vertex([xi,yi])
            vertInList.append(vi)
            if i >0:
                lineInList.append( Line(vertInList[i-1],vertInList[i],i-1) )
        lineInList.append( Line(vertInList[-1],vertInList[0],i) ) # close loop
        # outside bound
        ro = 0.4
        dAngle = (360/vertOut)*np.pi/180
        vertOutList = []
        lineOutList = []
        for i in range(vertOut):
            xi = 0.5 + ro*np.cos(dAngle*(i))
            yi = 0.5 + ro*np.sin(dAngle*(i))
            vi = Vertex([xi,yi])
            vertOutList.append(vi)
            if i >0:
                lineOutList.append(Line(vertOutList[i-1],vertOutList[i],100+i-1))
        lineOutList.append(Line(vertOutList[-1],vertOutList[0],100+i)) # close loop
        
        vertLists = [vertInList,vertOutList]
        lineLists = [lineInList,lineOutList]
        spPos = [0.5-0.25-0.15/2,0.5]; spDir = 90
        roundTrack = Track(spPos,spDir,vertLists,lineLists)
        return roundTrack

# Colors:
blue = (0,0,200)
green = (0,200,0)
red = (200,0,0)

if __name__ == "__main__":
    # Display
    pygame.init()
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (200, 50)
    modes = pygame.display.list_modes(32)
    displayWidth = 1200
    displayHeight = 750
    gameDisplay = pygame.display.set_mode((displayWidth,displayHeight))
    pygame.display.set_caption("RunnerTrack")
    clock = pygame.time.Clock()

    pygame.font.init()
    myfont = pygame.font.SysFont('Comic Sans MS', 30)
    # Run game
    BuildTrack(gameDisplay)
