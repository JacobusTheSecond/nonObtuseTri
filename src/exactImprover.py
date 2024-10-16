# El Grande
import copy

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from cgshop2025_pyutils import InstanceDatabase, ZipSolutionIterator, ZipWriter, Cgshop2025Solution, verify, \
    Cgshop2025Instance
from cgshop2025_pyutils.geometry import FieldNumber, Point, Segment

from exact_geometry import isBadTriangle, badAngle, badVertex, badness,innerIntersect,circumcenter,altitudePoint, inCircle
from hacky_internal_visualization_stuff import plotExact, plot, plot_solution #internal ugly functions I dont want you to see


import triangle as tr #https://rufat.be/triangle/

import math

def convert(data):

    #convert to triangulation type
    points = np.column_stack((data.points_x, data.points_y))
    constraints = np.column_stack((data.region_boundary, np.roll(data.region_boundary, -1)))
    if (len(data.additional_constraints) != 0):
        constraints = np.concatenate((constraints, data.additional_constraints))
    A = dict(vertices=points, segments=constraints)
    return A

class Triangulation:
    def __init__(self,instance:Cgshop2025Instance):
        self.instanceSize = len(instance.points_x)
        self.exactVerts = []
        self.numericVerts = []
        for x, y in zip(instance.points_x, instance.points_y):
            self.exactVerts.append(Point(x, y))
            self.numericVerts.append([x,y])
        self.exactVerts = np.array(self.exactVerts,dtype=Point)
        self.numericVerts = np.array(self.numericVerts)
        Ain = tr.triangulate(convert(instance), 'p')
        self.segments = Ain['segments']
        self.triangles = Ain['triangles']


        self.voronoiEdges = [[] for tri in self.triangles]
        self.constrainedMask = [[] for tri in self.triangles]

        #temporary edgeSet for triangleNeighbourhoodMap
        fullMap = [[[] for j in self.exactVerts] for i in self.exactVerts]
        for i in range(len(self.triangles)):
            tri = self.triangles[i]
            for edge in [[tri[0],tri[1]],[tri[1],tri[2]],[tri[2],tri[0]]]:
                fullMap[edge[0]][edge[1]].append(i)
                fullMap[edge[1]][edge[0]].append(i)


        for i in range(len(self.triangles)):
            tri = self.triangles[i]
            for edge in [[tri[1],tri[2]],[tri[2],tri[0]],[tri[0],tri[1]]]:
                added = False
                for j in fullMap[edge[0]][edge[1]]:
                    if i != j:
                        self.voronoiEdges[i].append(j)
                        self.constrainedMask[i].append(self.getSegmentIdx(edge))
                        added = True
                if not added:
                    self.voronoiEdges[i].append("dummy")
                    self.constrainedMask[i].append("dummy")

        self.circumCenters = [circumcenter(*self.exactVerts[tri]) for tri in self.triangles]
        self.circumRadiiSqr = [Segment(Point(*self.exactVerts[self.triangles[i][0]]),self.circumCenters[i]).squared_length() for i in range(len(self.triangles))]

    def getSegmentIdx(self,querySeg):
        revseg = [querySeg[1], querySeg[0]]
        for i in range(len(self.segments)):
            seg = self.segments[i]
            if np.all(seg == querySeg) or np.all(seg == revseg):
                return i
        return None

    def flipTrianglePair(self,triAIdx,triBIdx):

        # As perspective
        assert(triBIdx in self.voronoiEdges[triAIdx])
        neighbourAIdx = None
        for nAIdx in range(len(self.voronoiEdges[triAIdx])):
            if triBIdx == self.voronoiEdges[triAIdx][nAIdx]:
                neighbourAIdx = nAIdx
        assert(neighbourAIdx != None)
        assert(self.constrainedMask[triAIdx][neighbourAIdx] is None)

        #Bs perspective
        assert(triAIdx in self.voronoiEdges[triBIdx])
        neighbourBIdx = None
        for nBIdx in range(len(self.voronoiEdges[triBIdx])):
            if triAIdx == self.voronoiEdges[triBIdx][nBIdx]:
                neighbourBIdx = nBIdx
        assert(neighbourBIdx != None)
        assert(self.constrainedMask[triBIdx][neighbourBIdx] is None)

        triA = self.triangles[triAIdx]
        triB = self.triangles[triBIdx]

        if triA[(neighbourAIdx + 1)%3] == triB[(neighbourBIdx+1)%3]:
            #same orientation of shared edge
            newA = [triA[neighbourAIdx],triB[neighbourBIdx],triA[(neighbourAIdx+2)%3]]
            newAVedge = [self.voronoiEdges[triBIdx][(neighbourBIdx+1)%3],self.voronoiEdges[triAIdx][(neighbourAIdx+1)%3],triBIdx]
            newAVedgeMask = [self.constrainedMask[triBIdx][(neighbourBIdx+1)%3],self.constrainedMask[triAIdx][(neighbourAIdx+1)%3],None]

            newB = [triA[neighbourAIdx],triB[neighbourBIdx],triA[(neighbourAIdx+1)%3]]
            newBVedge = [self.voronoiEdges[triBIdx][(neighbourBIdx+2)%3],self.voronoiEdges[triAIdx][(neighbourAIdx+2)%3],triAIdx]
            newBVedgeMask = [self.constrainedMask[triBIdx][(neighbourBIdx+2)%3],self.constrainedMask[triAIdx][(neighbourAIdx+2)%3],None]

            if self.voronoiEdges[triAIdx][(neighbourAIdx + 2) % 3] != "dummy":
                modifiedOutsideEdge = []
                for i in range(3):
                    if self.voronoiEdges[self.voronoiEdges[triAIdx][(neighbourAIdx + 2) % 3]][i] != triAIdx:
                        modifiedOutsideEdge.append(self.voronoiEdges[self.voronoiEdges[triAIdx][(neighbourAIdx + 2) % 3]][i])
                    else:
                        modifiedOutsideEdge.append(triBIdx)
                self.voronoiEdges[self.voronoiEdges[triAIdx][(neighbourAIdx + 2) % 3]] = modifiedOutsideEdge

            if self.voronoiEdges[triBIdx][(neighbourBIdx + 1) % 3] != "dummy":
                modifiedOutsideEdge = []
                for i in range(3):
                    if self.voronoiEdges[self.voronoiEdges[triBIdx][(neighbourBIdx + 1) % 3]][i] != triBIdx:
                        modifiedOutsideEdge.append(
                            self.voronoiEdges[self.voronoiEdges[triBIdx][(neighbourBIdx + 1) % 3]][i])
                    else:
                        modifiedOutsideEdge.append(triAIdx)
                self.voronoiEdges[self.voronoiEdges[triBIdx][(neighbourBIdx + 1) % 3]] = modifiedOutsideEdge

            self.triangles[triAIdx] = newA
            self.voronoiEdges[triAIdx] = newAVedge
            self.constrainedMask[triAIdx] = newAVedgeMask

            self.triangles[triBIdx] = newB
            self.voronoiEdges[triBIdx] = newBVedge
            self.constrainedMask[triBIdx] = newBVedgeMask


        else:
            #different orientation of shared edge
            newA = [triA[neighbourAIdx],triB[neighbourBIdx],triA[(neighbourAIdx+1)%3]]
            newAVedge = [self.voronoiEdges[triBIdx][(neighbourBIdx+1)%3],self.voronoiEdges[triAIdx][(neighbourAIdx+2)%3],triBIdx]
            newAVedgeMask = [self.constrainedMask[triBIdx][(neighbourBIdx+1)%3],self.constrainedMask[triAIdx][(neighbourAIdx+2)%3],None]

            newB = [triA[neighbourAIdx],triB[neighbourBIdx],triA[(neighbourAIdx+2)%3]]
            newBVedge = [self.voronoiEdges[triBIdx][(neighbourBIdx+2)%3],self.voronoiEdges[triAIdx][(neighbourAIdx+1)%3],triAIdx]
            newBVedgeMask = [self.constrainedMask[triBIdx][(neighbourBIdx+2)%3],self.constrainedMask[triAIdx][(neighbourAIdx+1)%3],None]

            if self.voronoiEdges[triAIdx][(neighbourAIdx + 1) % 3] != "dummy":
                modifiedOutsideEdge = []
                for i in range(3):
                    if self.voronoiEdges[self.voronoiEdges[triAIdx][(neighbourAIdx + 1) % 3]][i] != triAIdx:
                        modifiedOutsideEdge.append(self.voronoiEdges[self.voronoiEdges[triAIdx][(neighbourAIdx + 1) % 3]][i])
                    else:
                        modifiedOutsideEdge.append(triBIdx)
                self.voronoiEdges[self.voronoiEdges[triAIdx][(neighbourAIdx + 1) % 3]] = modifiedOutsideEdge

            if self.voronoiEdges[triBIdx][(neighbourBIdx + 1) % 3] != "dummy":
                modifiedOutsideEdge = []
                for i in range(3):
                    if self.voronoiEdges[self.voronoiEdges[triBIdx][(neighbourBIdx + 1) % 3]][i] != triBIdx:
                        modifiedOutsideEdge.append(
                            self.voronoiEdges[self.voronoiEdges[triBIdx][(neighbourBIdx + 1) % 3]][i])
                    else:
                        modifiedOutsideEdge.append(triAIdx)
                self.voronoiEdges[self.voronoiEdges[triBIdx][(neighbourBIdx + 1) % 3]] = modifiedOutsideEdge

            self.triangles[triAIdx] = newA
            self.voronoiEdges[triAIdx] = newAVedge
            self.constrainedMask[triAIdx] = newAVedgeMask

            self.triangles[triBIdx] = newB
            self.voronoiEdges[triBIdx] = newBVedge
            self.constrainedMask[triBIdx] = newBVedgeMask


        self.circumCenters[triAIdx] = circumcenter(*self.exactVerts[newA])
        self.circumCenters[triBIdx] = circumcenter(*self.exactVerts[newB])


        self.circumRadiiSqr[triAIdx] = Segment(Point(*self.exactVerts[self.triangles[triAIdx][0]]),self.circumCenters[triAIdx]).squared_length()
        self.circumRadiiSqr[triBIdx] = Segment(Point(*self.exactVerts[self.triangles[triBIdx][0]]),self.circumCenters[triBIdx]).squared_length()

    def _isInHorribleEdgeStack(self,edgestack,edge):
        for e in edgestack:
            for dire in e:
                if np.all(dire == edge[0]) or np.all(dire == edge[1]):
                    return True
        return False

    def _isNotBanned(self,bannedList,edge):
        e = [self.triangles[edge[0][0]][(edge[0][1]+1)%3],self.triangles[edge[0][0]][(edge[0][1]+2)%3]]
        reve = [e[1],e[0]]
        if (e in bannedList) or (reve in bannedList):
            return True
        return False

    def ensureDelauney(self):
        #they are stored as [triangleindex, inducing index]
        badEdgesInTriangleLand = []
        bannedEdges =[]
        for i in range(len(self.triangles)):
            cc = self.circumCenters[i]
            cr = self.circumRadiiSqr[i]
            for jIdx in range(3):
                j = self.voronoiEdges[i][jIdx]
                jMask = self.constrainedMask[i][jIdx]
                oppositeIndexInJ = None
                if jMask is None:
                    onlyOn = True
                    for v in range(3):
                        if self.triangles[j][v] not in self.triangles[i]:
                            oppositeIndexInJ = v
                        inCirc = inCircle(cc,cr,self.exactVerts[self.triangles[j][v]])
                        if inCirc == "inside":
                            edge = [[i,jIdx],[j,oppositeIndexInJ]]
                            onlyOn = False
                            if not self._isInHorribleEdgeStack(badEdgesInTriangleLand,edge):
                                badEdgesInTriangleLand.append(edge)
                        if inCirc == "outside":
                            onlyOn = False
                    if onlyOn == True:
                        newTriangleA = [self.triangles[i][jIdx],self.triangles[i][(jIdx+1)%3],self.triangles[j][oppositeIndexInJ]]
                        newTriangleB = [self.triangles[i][jIdx],self.triangles[i][(jIdx+2)%3],self.triangles[j][oppositeIndexInJ]]
                        if not isBadTriangle(*self.exactVerts[newTriangleA]) and not isBadTriangle(*self.exactVerts[newTriangleB]):
                            edge = [[i,jIdx],[j,oppositeIndexInJ]]
                            if (not self._isInHorribleEdgeStack(badEdgesInTriangleLand,edge)) and (not self._isNotBanned(bannedEdges,edge)):
                                badEdgesInTriangleLand.append(edge)
                                bannedEdges.append([self.triangles[edge[0][0]][(edge[0][1]+1)%3],self.triangles[edge[0][0]][(edge[0][1]+2)%3]])


        while len(badEdgesInTriangleLand)>0:
            edge = badEdgesInTriangleLand[-1]
            badEdgesInTriangleLand = badEdgesInTriangleLand[:-1]
            assert(len(edge)>0)
            i,jIdx = edge[0]
            j = self.voronoiEdges[i][jIdx]
            opposingIdx = None
            for v in range(3):
                if self.triangles[j][v] not in self.triangles[i]:
                    opposingIdx = v
            assert(opposingIdx != None)

            #remove all unnecessary mentions from stack
            for e in badEdgesInTriangleLand:
                for it in reversed(range(len(e))):
                    if np.all(e[it] == [it,jIdx]) or np.all(e[it] == [j,opposingIdx]):
                        e.pop(it)

            self.flipTrianglePair(i,j)
            triIdx = i
            cc = self.circumCenters[triIdx]
            cr = self.circumRadiiSqr[triIdx]
            for otherIdx in range(3):
                otherTri = self.voronoiEdges[triIdx][otherIdx]
                otherMask = self.constrainedMask[triIdx][otherIdx]
                opposingIdxInJ = None
                if otherMask is None:
                    onlyOn = True
                    for v in range(3):
                        if self.triangles[otherTri][v] not in self.triangles[triIdx]:
                            oppositeIndexInJ = v
                        inCirc = inCircle(cc, cr, self.exactVerts[self.triangles[otherTri][v]])
                        if inCirc == "inside":
                            edge = [[triIdx, otherIdx], [otherTri, oppositeIndexInJ]]
                            onlyOn = False
                            if not self._isInHorribleEdgeStack(badEdgesInTriangleLand, edge):
                                badEdgesInTriangleLand.append(edge)
                        if inCirc == "outside":
                            onlyOn = False
                    if onlyOn == True:
                        newTriangleA = [self.triangles[i][jIdx],self.triangles[i][(jIdx+1)%3],self.triangles[otherTri][oppositeIndexInJ]]
                        newTriangleB = [self.triangles[i][jIdx],self.triangles[i][(jIdx+2)%3],self.triangles[otherTri][oppositeIndexInJ]]
                        if not isBadTriangle(*self.exactVerts[newTriangleA]) and not isBadTriangle(
                                *self.exactVerts[newTriangleB]):
                            edge = [[triIdx, otherIdx], [otherTri, oppositeIndexInJ]]
                            if (not self._isInHorribleEdgeStack(badEdgesInTriangleLand,edge)) and (not self._isNotBanned(bannedEdges,edge)):
                                badEdgesInTriangleLand.append(edge)
                                bannedEdges.append([self.triangles[edge[0][0]][(edge[0][1]+1)%3],self.triangles[edge[0][0]][(edge[0][1]+2)%3]])

            triIdx = j
            cc = self.circumCenters[triIdx]
            cr = self.circumRadiiSqr[triIdx]
            for otherIdx in range(3):
                otherTri = self.voronoiEdges[triIdx][otherIdx]
                otherMask = self.constrainedMask[triIdx][otherIdx]
                opposingIdxInJ = None
                if otherMask is None:
                    onlyOn = True
                    for v in range(3):
                        if self.triangles[otherTri][v] not in self.triangles[triIdx]:
                            oppositeIndexInJ = v
                        inCirc = inCircle(cc, cr, self.exactVerts[self.triangles[otherTri][v]])
                        if inCirc == "inside":
                            edge = [[triIdx, otherIdx], [otherTri, oppositeIndexInJ]]
                            onlyOn = False
                            if not self._isInHorribleEdgeStack(badEdgesInTriangleLand, edge):
                                badEdgesInTriangleLand.append(edge)
                        if inCirc == "outside":
                            onlyOn = False
                    if onlyOn == True:
                        newTriangleA = [self.triangles[i][jIdx],self.triangles[i][(jIdx+1)%3],self.triangles[otherTri][oppositeIndexInJ]]
                        newTriangleB = [self.triangles[i][jIdx],self.triangles[i][(jIdx+2)%3],self.triangles[otherTri][oppositeIndexInJ]]
                        if not isBadTriangle(*self.exactVerts[newTriangleA]) and not isBadTriangle(
                                *self.exactVerts[newTriangleB]):
                            edge = [[triIdx, otherIdx], [otherTri, oppositeIndexInJ]]
                            if (not self._isInHorribleEdgeStack(badEdgesInTriangleLand,edge)) and (not self._isNotBanned(bannedEdges,edge)):
                                badEdgesInTriangleLand.append(edge)
                                bannedEdges.append([self.triangles[edge[0][0]][(edge[0][1]+1)%3],self.triangles[edge[0][0]][(edge[0][1]+2)%3]])

    def dropAltitude(self,idx):
        tri = self.triangles[idx]
        if not isBadTriangle(*self.exactVerts[tri]):
            #print("nowhere to drop from!")
            return
            #assert(False)
        badIdx = badAngle(*self.exactVerts[tri])
        if self.constrainedMask[idx][badIdx] is None:
            #print("nowhere to drop to!")
            return
            #assert(False)
        otherIdx = self.voronoiEdges[idx][badIdx]

        #first things first, split the segment
        segment = [tri[(badIdx+1)%3],tri[(badIdx+2)%3]]
        segIdx = self.getSegmentIdx(segment)
        assert(segIdx != None)

        #the point to be inserted on the segment
        ap = altitudePoint(Segment(Point(*self.exactVerts[segment[0]]),Point(*self.exactVerts[segment[1]])), self.exactVerts[tri[badIdx]])

        newPointIndex = len(self.exactVerts)

        self.segments[segIdx] = [segment[0],newPointIndex]
        self.segments = np.vstack((self.segments,[newPointIndex,segment[1]]))

        self.exactVerts = np.vstack((self.exactVerts,[ap]))
        self.numericVerts = np.vstack((self.exactVerts,[float(ap.x()),float(ap.y())]))

        #now split both triangles attached to the split segment
        if otherIdx == "dummy":
            newTriIndex = len(self.triangles)
            #easy case!
            newA = [tri[badIdx],tri[(badIdx+1)%3],newPointIndex]
            newAVedge = ["dummy",newTriIndex,self.voronoiEdges[idx][(badIdx+2)%3]]
            newAVedgeMask = ["dummy",None,self.constrainedMask[idx][(badIdx+2)%3]]

            newTri = [tri[badIdx],newPointIndex,tri[(badIdx+2)%3]]
            newVedge = ["dummy",self.voronoiEdges[idx][(badIdx+1)%3],idx]
            newVedgeMask = ["dummy",self.constrainedMask[idx][(badIdx+1)%3],None]

            if self.voronoiEdges[idx][(badIdx + 1) % 3] != "dummy":
                modifiedOutsideEdge = []
                for i in range(3):
                     if self.voronoiEdges[self.voronoiEdges[idx][(badIdx + 1) % 3]][i] != idx:
                         modifiedOutsideEdge.append(self.voronoiEdges[self.voronoiEdges[idx][(badIdx + 1) % 3]][i])
                     else:
                         modifiedOutsideEdge.append(newTriIndex)
                self.voronoiEdges[self.voronoiEdges[idx][(badIdx + 1) % 3]] = modifiedOutsideEdge

            self.triangles[idx] = newA
            self.voronoiEdges[idx] = newAVedge
            self.constrainedMask[idx] = newAVedgeMask

            self.triangles = np.vstack((self.triangles,[newTri]))
            self.voronoiEdges.append(newVedge)
            self.constrainedMask.append(newVedgeMask)

            self.circumCenters[idx] = circumcenter(*self.exactVerts[newA])
            self.circumCenters.append(circumcenter(*self.exactVerts[newTri]))

            self.circumRadiiSqr[idx] = Segment(Point(*self.exactVerts[self.triangles[idx][0]]),
                                                   self.circumCenters[idx]).squared_length()
            self.circumRadiiSqr.append(Segment(Point(*self.exactVerts[newTri[0]]),
                                                   self.circumCenters[-1]).squared_length())
        else:
            #phew fuck me...
            newInsideIdx = len(self.triangles)
            newOutsideIdx = newInsideIdx + 1
            otherTri = self.triangles[otherIdx]
            opposingIdx = None
            for v in range(3):
                if not self.triangles[otherIdx][v] in tri:
                    opposingIdx = v
            assert(opposingIdx != None)

            #if shared edge is oriented the same way
            if otherTri[(opposingIdx+1)%3] == tri[(badIdx+1)%3]:
                newA = [tri[badIdx],tri[(badIdx+1)%3],newPointIndex]
                newAVedge = [otherIdx,newInsideIdx,self.voronoiEdges[idx][(badIdx+2)%3]]
                newAMask = [self.getSegmentIdx([tri[(badIdx+1)%3],newPointIndex]),None,self.constrainedMask[idx][(badIdx+2)%3]]

                newInside = [tri[badIdx],newPointIndex,tri[(badIdx+2)%3]]
                newInsideVedge = [newOutsideIdx,self.voronoiEdges[idx][(badIdx+1)%3],idx]
                newInsideMask = [self.getSegmentIdx([newPointIndex,tri[(badIdx+2)%3]]),self.constrainedMask[idx][(badIdx+1)%3],None]

                newB = [otherTri[opposingIdx],otherTri[(opposingIdx+1)%3],newPointIndex]
                newBVedge = [idx,newOutsideIdx,self.voronoiEdges[otherIdx][(opposingIdx+2)%3]]
                newBMask = [newAMask[0],None,self.constrainedMask[otherIdx][(opposingIdx+2)%3]]

                newOutside = [otherTri[opposingIdx],newPointIndex,otherTri[(opposingIdx+2)%3]]
                newOutsideVedge = [newInsideIdx,self.voronoiEdges[otherIdx][(opposingIdx + 1)%3],otherIdx]
                newOutsideMask = [newInsideMask[0],self.constrainedMask[otherIdx][(opposingIdx+1)%3],None]

                if self.voronoiEdges[idx][(badIdx + 1) % 3] != "dummy":
                    modifiedOutsideEdge = []
                    for i in range(3):
                        if self.voronoiEdges[self.voronoiEdges[idx][(badIdx + 1) % 3]][i] != idx:
                            modifiedOutsideEdge.append(self.voronoiEdges[self.voronoiEdges[idx][(badIdx + 1) % 3]][i])
                        else:
                            modifiedOutsideEdge.append(newInsideIdx)
                    self.voronoiEdges[self.voronoiEdges[idx][(badIdx + 1) % 3]] = modifiedOutsideEdge

                if self.voronoiEdges[otherIdx][(opposingIdx + 1) % 3] != "dummy":
                    modifiedOutsideEdge = []
                    for i in range(3):
                        if self.voronoiEdges[self.voronoiEdges[otherIdx][(opposingIdx + 1) % 3]][i] != otherIdx:
                            modifiedOutsideEdge.append(self.voronoiEdges[self.voronoiEdges[otherIdx][(opposingIdx + 1) % 3]][i])
                        else:
                            modifiedOutsideEdge.append(newOutsideIdx)
                    self.voronoiEdges[self.voronoiEdges[otherIdx][(opposingIdx + 1) % 3]] = modifiedOutsideEdge

                self.triangles[idx] = newA
                self.voronoiEdges[idx] = newAVedge
                self.constrainedMask[idx] = newAMask

                self.triangles = np.vstack((self.triangles, [newInside]))
                self.voronoiEdges.append(newInsideVedge)
                self.constrainedMask.append(newInsideMask)

                self.circumCenters[idx] = circumcenter(*self.exactVerts[newA])
                self.circumCenters.append(circumcenter(*self.exactVerts[newInside]))

                self.circumRadiiSqr[idx] = Segment(Point(*self.exactVerts[self.triangles[idx][0]]),
                                                   self.circumCenters[idx]).squared_length()
                self.circumRadiiSqr.append(Segment(Point(*self.exactVerts[newInside[0]]),
                                                   self.circumCenters[-1]).squared_length())

                self.triangles[otherIdx] = newB
                self.voronoiEdges[otherIdx] = newBVedge
                self.constrainedMask[otherIdx] = newBMask

                self.triangles = np.vstack((self.triangles, [newOutside]))
                self.voronoiEdges.append(newOutsideVedge)
                self.constrainedMask.append(newOutsideMask)

                self.circumCenters[otherIdx] = circumcenter(*self.exactVerts[newB])
                self.circumCenters.append(circumcenter(*self.exactVerts[newOutside]))

                self.circumRadiiSqr[otherIdx] = Segment(Point(*self.exactVerts[self.triangles[otherIdx][0]]),
                                                   self.circumCenters[otherIdx]).squared_length()
                self.circumRadiiSqr.append(Segment(Point(*self.exactVerts[newOutside[0]]),
                                                   self.circumCenters[-1]).squared_length())

            else:
                newA = [tri[badIdx],tri[(badIdx+1)%3],newPointIndex]
                newAVedge = [otherIdx,newInsideIdx,self.voronoiEdges[idx][(badIdx+2)%3]]
                newAMask = [self.getSegmentIdx([tri[(badIdx+1)%3],newPointIndex]),None,self.constrainedMask[idx][(badIdx+2)%3]]

                newInside = [tri[badIdx],newPointIndex,tri[(badIdx+2)%3]]
                newInsideVedge = [newOutsideIdx,self.voronoiEdges[idx][(badIdx+1)%3],idx]
                newInsideMask = [self.getSegmentIdx([newPointIndex,tri[(badIdx+2)%3]]),self.constrainedMask[idx][(badIdx+1)%3],None]

                newB = [otherTri[opposingIdx],otherTri[(opposingIdx+2)%3],newPointIndex]
                newBVedge = [idx,newOutsideIdx,self.voronoiEdges[otherIdx][(opposingIdx+1)%3]]
                newBMask = [newAMask[0],None,self.constrainedMask[otherIdx][(opposingIdx+1)%3]]

                newOutside = [otherTri[opposingIdx],newPointIndex,otherTri[(opposingIdx+1)%3]]
                newOutsideVedge = [newInsideIdx,self.voronoiEdges[otherIdx][(opposingIdx + 2)%3],otherIdx]
                newOutsideMask = [newInsideMask[0],self.constrainedMask[otherIdx][(opposingIdx+2)%3],None]

                if self.voronoiEdges[idx][(badIdx + 1) % 3] != "dummy":
                    modifiedOutsideEdge = []
                    for i in range(3):
                        if self.voronoiEdges[self.voronoiEdges[idx][(badIdx + 1) % 3]][i] != idx:
                            modifiedOutsideEdge.append(self.voronoiEdges[self.voronoiEdges[idx][(badIdx + 1) % 3]][i])
                        else:
                            modifiedOutsideEdge.append(newInsideIdx)
                    self.voronoiEdges[self.voronoiEdges[idx][(badIdx + 1) % 3]] = modifiedOutsideEdge

                if self.voronoiEdges[otherIdx][(opposingIdx + 2) % 3] != "dummy":
                    modifiedOutsideEdge = []
                    for i in range(3):
                        if self.voronoiEdges[self.voronoiEdges[otherIdx][(opposingIdx + 2) % 3]][i] != otherIdx:
                            modifiedOutsideEdge.append(
                                self.voronoiEdges[self.voronoiEdges[otherIdx][(opposingIdx + 2) % 3]][i])
                        else:
                            modifiedOutsideEdge.append(newOutsideIdx)
                    self.voronoiEdges[self.voronoiEdges[otherIdx][(opposingIdx + 2) % 3]] = modifiedOutsideEdge

                self.triangles[idx] = newA
                self.voronoiEdges[idx] = newAVedge
                self.constrainedMask[idx] = newAMask

                self.triangles = np.vstack((self.triangles, [newInside]))
                self.voronoiEdges.append(newInsideVedge)
                self.constrainedMask.append(newInsideMask)

                self.circumCenters[idx] = circumcenter(*self.exactVerts[newA])
                self.circumCenters.append(circumcenter(*self.exactVerts[newInside]))

                self.circumRadiiSqr[idx] = Segment(Point(*self.exactVerts[self.triangles[idx][0]]),
                                                   self.circumCenters[idx]).squared_length()
                self.circumRadiiSqr.append(Segment(Point(*self.exactVerts[newInside[0]]),
                                                   self.circumCenters[-1]).squared_length())

                self.triangles[otherIdx] = newB
                self.voronoiEdges[otherIdx] = newBVedge
                self.constrainedMask[otherIdx] = newBMask

                self.triangles = np.vstack((self.triangles, [newOutside]))
                self.voronoiEdges.append(newOutsideVedge)
                self.constrainedMask.append(newOutsideMask)

                self.circumCenters[otherIdx] = circumcenter(*self.exactVerts[newB])
                self.circumCenters.append(circumcenter(*self.exactVerts[newOutside]))

                self.circumRadiiSqr[otherIdx] = Segment(Point(*self.exactVerts[self.triangles[otherIdx][0]]),
                                                   self.circumCenters[otherIdx]).squared_length()
                self.circumRadiiSqr.append(Segment(Point(*self.exactVerts[newOutside[0]]),
                                                   self.circumCenters[-1]).squared_length())
        self.ensureDelauney()

    def createSteinerpoint(self):
        pass

    def moveSteinerpoint(self):
        pass



    def markTriangle(self,mark,axs,withNeighbor = True):
        t = plt.Polygon(self.numericVerts[self.triangles[mark]], color='g', zorder=97)
        axs.add_patch(t)

        axs.scatter([float(self.circumCenters[mark].x())], [float(self.circumCenters[mark].y())], marker='.',
                    color='yellow', zorder=1000)
        circle = plt.Circle((float(self.circumCenters[mark].x()), float(self.circumCenters[mark].y())),
                            np.sqrt(float(self.circumRadiiSqr[mark])), color="yellow", fill=False, zorder=1000)
        axs.add_patch(circle)

        if withNeighbor:
            for i in self.voronoiEdges[mark]:
                if i == "dummy":
                    continue
                axs.plot([float(self.circumCenters[mark].x()),float(self.circumCenters[i].x())],[float(self.circumCenters[mark].y()),float(self.circumCenters[i].y())],zorder=1000,color="yellow")

    def plotTriangulation(self,axs,mark=None):
        SC = len(self.exactVerts) - self.instanceSize
        name = ""
        badCount = 0
        if SC > 0:
            name += " [SC:" + str(SC) + "]"

        #axs.scatter([p[0] for p in self.numericVerts],[p[1] for p in self.numericVerts],marker=".")

        for tri in self.triangles:
            cords = self.numericVerts[tri]
            cords = np.concatenate((cords, [cords[0]]))
            axs.plot(*(cords.T), color='black', linewidth=1,zorder=98)
            if isBadTriangle(*self.exactVerts[tri]):
                badCount += 1
                t = plt.Polygon(self.numericVerts[tri], color='b')
                axs.add_patch(t)
        name += " (>90°: " + str(badCount) + ")"

        if mark != None:
            self.markTriangle(mark,axs)

        for e in self.segments:
            axs.plot(*(self.numericVerts[e].T), color='red', linewidth=2,zorder=99)
        axs.scatter(*(self.numericVerts[:self.instanceSize].T), marker='.', color='black', zorder=100)
        axs.scatter(*(self.numericVerts[self.instanceSize:].T), marker='.', color='green', zorder=100)

        axs.set_aspect('equal')
        axs.title.set_text(name)

def improveQuality(instance:Cgshop2025Instance,withShow=True,axs=None,verbosity=0):
    print("WORK IN PROGRESS. PROCEED WITH CARE.")
    triangulation = Triangulation(instance)
    l = len(triangulation.triangles)
    if(withShow):
        plt.ion()

        axs.clear()
        triangulation.plotTriangulation(axs)
        plt.draw()
        plt.pause(0.01)
        changed = True
        while changed:
            print(".",end="")
            changed = False
            for i in range(len(triangulation.triangles)):
                triangulation.dropAltitude(i)
                if len(triangulation.triangles) != l:
                    l = len(triangulation.triangles)
                    changed = True

                    axs.clear()
                    triangulation.plotTriangulation(axs)
                    plt.draw()
                    plt.pause(0.01)

        axs.clear()
        triangulation.plotTriangulation(axs)
        plt.draw()
        plt.pause(2)