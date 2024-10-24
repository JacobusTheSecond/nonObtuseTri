import copy
import time

import matplotlib.pyplot as plt
import numpy as np
from cgshop2025_pyutils import Cgshop2025Solution, Cgshop2025Instance
from cgshop2025_pyutils.geometry import FieldNumber, Point, Segment
from dask.array import outer

import exact_geometry as eg

import triangle as tr  # https://rufat.be/triangle/

import logging

from src.primitiveTester import onWhichSide

logging.basicConfig(format="%(asctime)s %(message)s",datefmt="%y-%m-%d %H:%M%S",level=logging.INFO)

#some useful constants
outerFace = np.iinfo(int).max
noneFace = outerFace - 1
noneEdge = outerFace - 2
noneVertex = outerFace - 3
noneIntervalVertex = outerFace - 4

class abstractTriangulation():
    def __init__(self,exactVerts,numericVerts,triangles,segments):
        self.triangles = triangles
        self.segments = segments
        self.triangleMap = None
        self.exactVerts = np.array(exactVerts,dtype=Point)
        self.numericVerts = np.array(numericVerts)

        self.vertexMap = [[] for v in range(len(self.exactVerts))]
        for triIdx in range(len(self.triangles)):
            for i in range(3):
                vIdx = self.triangles[triIdx,i]
                self.vertexMap[vIdx].append([triIdx,i])

    def setTriangleMap(self,triangleMap):
        self.triangleMap = triangleMap

    def getSegmentIdx(self, querySeg):
        locs = np.concatenate(
            (np.argwhere(np.all((self.segments == querySeg), -1)),
             np.argwhere(np.all((self.segments == querySeg[::-1]), -1))))
        if len(locs) == 1:
            return locs[0, 0]
        else:
            return noneEdge

    def getInternalIdx(self,triIdx,vIdx):
        locs = np.argwhere(self.triangles[triIdx] == vIdx)
        if len(locs) == 1:
            return locs[0][0]
        else:
            return noneIntervalVertex

    def getOppositeInternalIdx(self,triIdx,vIdx):
        iVIdx = self.getInternalIdx(triIdx,vIdx)
        if iVIdx == noneIntervalVertex:
            return noneIntervalVertex
        else:
            return self.triangleMap[triIdx,iVIdx,1]

    def unsetVertexMap(self, triIdx):
        for vIdx in self.triangles[triIdx]:
            self.vertexMap[vIdx] = [vm for vm in self.vertexMap[vIdx] if vm[0] != triIdx]
            #self.vertexMap[idx].remove(triIdx)

    def setVertexMap(self, triIdx):
        for iVIdx in range(3):
            self.vertexMap[self.triangles[triIdx,iVIdx]].append([triIdx,iVIdx])

    def setInvalidTriangle(self,triIdx,tri,triMap):
        self.triangles[triIdx] = tri
        self.triangleMap[triIdx] = triMap

        self.setVertexMap(triIdx)

        for iVIdx in range(3):
            neighbour, oppIVIdx, constraint = self.triangleMap[triIdx,iVIdx]
            if neighbour != outerFace and neighbour != noneFace and neighbour < len(self.triangles):
                self.triangleMap[neighbour,oppIVIdx,:2] = [triIdx,iVIdx]


    def isBad(self, triIdx):
        return eg.isBadTriangle(*self.exactVerts[self.triangles[triIdx]])

    def splitSegment(self, segIdx, vIdx):
        seg = [self.segments[segIdx][0], self.segments[segIdx][1]]
        self.segments[segIdx] = [seg[0], vIdx]
        self.segments = np.vstack((self.segments, [vIdx, seg[1]]))

    def unlinkTriangle(self, triIdx):
        tri = self.triangles[triIdx]

        #unlink from neighbouring triangles
        for neighbour, oppIVIdx, constraint in self.triangleMap[triIdx]:
            if neighbour != outerFace and neighbour != noneFace:
                self.triangleMap[neighbour,oppIVIdx,:2] = [noneFace,noneIntervalVertex]

        #unset map
        for i in range(3):
            self.triangleMap[triIdx,i] = [noneFace,noneIntervalVertex,noneEdge]

        #unset vertexMap:
        self.unsetVertexMap(triIdx)

        self.triangles[triIdx] = [noneVertex,noneVertex,noneVertex]

#envelope class for a collection of faces
class SubTriangulation(abstractTriangulation):
    def __init__(self, vIdxs, boundary, exactPoints, numericPoints, tris, triMap, innerCons, boundaryCons, boundaryType,
                 steinercutoff, axs):
        #external map
        self.vIdxs = vIdxs
        self.axs = axs

        self.localMap = self.vIdxs + list(set(boundary))
        inside = np.array([True for v in self.vIdxs] + [False for v in list(set(boundary))])
        args = np.argsort(np.array(self.localMap))
        self.localInside = inside[args]
        self.localMap = np.array(self.localMap)[args]
        self.localSteiner = [False if v < steinercutoff else True for v in self.localMap]

        #these should under no circumstances be changed
        exactCoords = exactPoints[args]
        numericCoords = numericPoints[args]

        self.localInnerCons = [[np.argwhere(self.localMap==idx)[0][0] for idx in e] for e in innerCons]
        self.localOuterCons = [[np.argwhere(self.localMap==idx)[0][0] for idx in e] for e in boundaryCons]

        segments = self.localInnerCons + self.localOuterCons
        self.localSegmentType = ["in" for con in self.localInnerCons] + ["boundary" if localtype else "halfin" for localtype in boundaryType]
        #self.localOuterType = boundaryType


        self.localBoundary = np.array([np.where(self.localMap == b)[0][0] for b in boundary])
        localTris = np.array([[np.where(self.localMap == triV)[0][0] for triV in tri] for tri in tris])

        super().__init__(exactCoords,numericCoords,localTris,segments)

        localTriMap = []
        for triIdx in range(len(tris)):
            localTriM = []
            tri = tris[triIdx]
            localTri = self.triangles[triIdx]
            triM = triMap[triIdx]
            for i in range(3):
                faceId, opp, segmentId = triM[i]
                if faceId in localTris:
                    localTriM.append([np.where(localTris == faceId)[0][0],opp,self.getSegmentIdx([localTri[(i+1)%3],localTri[(i+2)%3]])])
                elif faceId == outerFace:
                    localTriM.append([outerFace,noneIntervalVertex,self.getSegmentIdx([localTri[(i+1)%3],localTri[(i+2)%3]])])
                else:
                    localTriM.append([noneFace,noneIntervalVertex,self.getSegmentIdx([localTri[(i+1)%3],localTri[(i+2)%3]])])
            localTriMap.append(localTriM)

        super().setTriangleMap(localTriMap)

        self.plotMe()

    def plotMe(self):
        self.axs.clear()

        for triIdx in range(len(self.triangles)):
            tri = self.triangles[triIdx]
            cords = self.numericVerts[tri]
            cords = np.concatenate((cords, [cords[0]]))
            for i in range(3):
                    self.axs.plot(*(cords[[(i+1)%3,(i+2)%3]].T), color='black', linewidth=1, zorder=98)
            if self.isBad(triIdx):
                t = plt.Polygon(self.numericVerts[tri], color='mediumorchid')
                self.axs.add_patch(t)
            else:
                t = plt.Polygon(self.numericVerts[tri], color='palegoldenrod')
                self.axs.add_patch(t)

        for edgeId in range(len(self.segments)):
            e = self.segments[edgeId]
            t = self.localSegmentType[edgeId]
            color = 'blue'
            if t == "in":
                color = 'lime'
            elif t == "halfin":
                color = 'forestgreen'
            self.axs.plot(*(self.numericVerts[e].T), color=color, linewidth=2, zorder=99)
        for vIdx in range(len(self.localMap)):
            color = 'red' if self.localSteiner[vIdx] else "black"
            self.axs.scatter(*(self.numericVerts[[vIdx]].T), s=20, color=color, zorder=100)

        self.axs.set_aspect('equal')
        plt.draw()
        plt.pause(0.01)

    def solveWithOnePointInInteriorOrVertex(self):
        #check if all inside vertices are steiner
        for vIdx in self.localMap:
            if self.localInside[vIdx] and not self.localSteiner[vIdx]:
                return False

        #check that there is either no inside constraint, or just a single inside constraint
        onBoundary = []
        for con in self.localInnerCons:
            for c in con:
                for idx in np.where(self.localBoundary == c):
                    onBoundary.append(idx)

        points = self.exactCoords[self.localBoundary]

        solType,sol = None,None
        if len(onBoundary) == 0:
            solType,sol = eg.findCenterOfLink(points)
        if len(onBoundary) == 2:
            solType, sol = eg.findCenterOfLinkConstrained(points,onBoundary[0],onBoundary[1])
        else:
            return False

class GeometricSubproblem:
    def __init__(self, vIdxs, triIdxs, boundary, exactPoints, numericPoints, innerCons, boundaryCons, boundaryType, steinercutoff, axs):
        #external map
        self.vIdxs = vIdxs
        self.triIdxs = triIdxs
        self.axs = axs

        self.localMap = self.vIdxs + list(set(boundary))
        inside = np.array([True for v in self.vIdxs] + [False for v in list(set(boundary))])
        args = np.argsort(np.array(self.localMap))
        self.localInside = inside[args]
        self.localMap = np.array(self.localMap)[args]
        self.localSteiner = [False if v < steinercutoff else True for v in self.localMap]

        #these should under no circumstances be changed
        self.exactVerts = exactPoints[args]
        self.numericVerts = numericPoints[args]

        self.localInnerCons = [[np.argwhere(self.localMap==idx)[0][0] for idx in e] for e in innerCons]
        self.localOuterCons = [[np.argwhere(self.localMap==idx)[0][0] for idx in e] for e in boundaryCons]

        self.segments = self.localInnerCons + self.localOuterCons
        self.localSegmentType = ["in" for con in self.localInnerCons] + ["boundary" if localtype else "halfin" for localtype in boundaryType]

        self.localBoundary = np.array([np.where(self.localMap == b)[0][0] for b in boundary])

        self.plotMe()

    def plotMe(self):
        self.axs.clear()

        for i in range(len(self.localBoundary)):
            e = [self.localBoundary[i],self.localBoundary[(i+1)%len(self.localBoundary)]]
            self.axs.plot(*(self.numericVerts[e].T), color="black", linewidth=2, zorder=98)

        for edgeId in range(len(self.segments)):
            e = self.segments[edgeId]
            t = self.localSegmentType[edgeId]
            color = 'blue'
            if t == "in":
                color = 'lime'
            elif t == "halfin":
                color = 'forestgreen'
            self.axs.plot(*(self.numericVerts[e].T), color=color, linewidth=2, zorder=99)
        for vIdx in range(len(self.localMap)):
            color = 'red' if self.localSteiner[vIdx] else "black"
            self.axs.scatter(*(self.numericVerts[[vIdx]].T), s=20, color=color, zorder=100)

        self.axs.set_aspect('equal')
        plt.draw()
        plt.pause(0.01)

class Triangulation(abstractTriangulation):
    def __init__(self, instance: Cgshop2025Instance, withValidate=False, seed=None,axs=None):

        self.withValidate = withValidate
        self.seed = seed
        self.plotTime = 0.005
        self.axs = axs

        def convert(data: Cgshop2025Instance):
            # convert to triangulation type
            points = np.column_stack((data.points_x, data.points_y))
            constraints = np.column_stack((data.region_boundary, np.roll(data.region_boundary, -1)))
            if len(data.additional_constraints) != 0:
                constraints = np.concatenate((constraints, data.additional_constraints))
            A = dict(vertices=points, segments=constraints)
            return A

        ####
        # instanciate vertexset, segmentset and triangleset
        ####
        self.instanceSize = len(instance.points_x)
        exactVerts = []
        numericVerts = []
        for x, y in zip(instance.points_x, instance.points_y):
        #for x, y in zip(xs, ys):
            exactVerts.append(Point(x, y))
            numericVerts.append([x, y])
        self.instance_uid = instance.instance_uid

        Ain = tr.triangulate(convert(instance), 'p')
        segments = Ain['segments']
        triangles = Ain['triangles']

        super().__init__(exactVerts,numericVerts,triangles,segments)

        ####
        # construct all maps
        ####

        ####
        # vertex maps
        ####

        self.pointTopologyChanged = [True for v in self.exactVerts]

        ####
        # triangle maps
        ####

        self.badTris = [idx for idx in range(len(self.triangles)) if self.isBad(idx)]

        self.triangleMap = [[] for v in self.triangles]

        # temporary edgeSet for triangleNeighbourhoodMap
        fullMap = [[[] for j in self.exactVerts] for i in self.exactVerts]
        for i in range(len(self.triangles)):
            tri = self.triangles[i]
            for edge in [[tri[0], tri[1]], [tri[1], tri[2]], [tri[2], tri[0]]]:
                fullMap[edge[0]][edge[1]].append(i)
                fullMap[edge[1]][edge[0]].append(i)

        for i in range(len(self.triangles)):
            triMap = []
            tri = self.triangles[i]
            for edge in [[tri[1], tri[2]], [tri[2], tri[0]], [tri[0], tri[1]]]:
                added = False
                for j in fullMap[edge[0]][edge[1]]:
                    if i != j:
                        opp = None
                        for k in range(3):
                            if self.triangles[j][k] not in tri:
                                opp = k
                        triMap.append([j,opp,self.getSegmentIdx(np.array(edge))])
                        added = True
                if not added:
                    triMap.append([outerFace,noneIntervalVertex,self.getSegmentIdx(np.array(edge))])
            self.triangleMap[i] = triMap
        self.triangleMap = np.array(self.triangleMap)

        self.edgeTopologyChanged = np.full(self.triangles.shape,True)

        self.circumCenters = [eg.circumcenter(*self.exactVerts[tri]) for tri in self.triangles]
        self.circumRadiiSqr = [eg.distsq(self.point(self.triangles[i][0]), self.circumCenters[i]) for i in
                               range(len(self.triangles))]

        self.closestToCC = []
        self.closestDist = []
        for triIdx in range(len(self.triangles)):
            closest = None
            closestdist = None
            for vIdx in range(len(self.exactVerts)):
                dist = eg.distsq(self.circumCenters[triIdx], self.point(vIdx))
                if (closest is None) or (dist < closestdist):
                    closest = vIdx
                    closestdist = dist
            self.closestToCC.append(closest)
            self.closestDist.append(closestdist)
        self.closestToCC = np.array(self.closestToCC)
        self.closestDist = np.array(self.closestDist)

        ####
        # segment maps
        ####

        self.segmentType = [False for seg in self.segments]
        for triIdx in range(len(self.triangles)):
            for i in range(3):
                if (edgeId := self.triangleMap[triIdx, i,2]) != noneEdge:
                    if self.triangleMap[triIdx, i,0] == outerFace:
                        self.segmentType[edgeId] = True
        self.segmentType = np.array(self.segmentType)

    def plotTriangulation(self):
        self.axs.clear()
        SC = len(self.exactVerts) - self.instanceSize
        name = ""
        badCount = 0
        if SC > 0:
            name += " [SC:" + str(SC) + "]"

        # axs.scatter([p[0] for p in self.numericVerts],[p[1] for p in self.numericVerts],marker=".")

        for triIdx in range(len(self.triangles)):
            tri = self.triangles[triIdx]
            cords = self.numericVerts[tri]
            cords = np.concatenate((cords, [cords[0]]))
            for i in range(3):
                if ((self.triangles[triIdx][(i + 1) % 3] >= self.instanceSize) and (
                            self.triangles[triIdx][(i + 2) % 3] >= self.instanceSize) and
                        (self.edgeTopologyChanged[triIdx,i] or (self.triangleMap[triIdx,i,0] != outerFace and self.edgeTopologyChanged[self.triangleMap[triIdx,i,0],self.triangleMap[triIdx,i,1]]))):
                    self.axs.plot(*(cords[[(i+1)%3,(i+2)%3]].T), color='black', linewidth=1, zorder=98, linestyle="dotted")
                else:
                    self.axs.plot(*(cords[[(i+1)%3,(i+2)%3]].T), color='black', linewidth=1, zorder=98)
            if self.isBad(triIdx):
                badCount += 1
                t = plt.Polygon(self.numericVerts[tri], color='mediumorchid')
                self.axs.add_patch(t)
            else:
                t = plt.Polygon(self.numericVerts[tri], color='palegoldenrod')
                self.axs.add_patch(t)
        name += " (>90°: " + str(badCount) + ")"

        for edgeId in range(len(self.segments)):
            e = self.segments[edgeId]
            t = self.segmentType[edgeId]
            color = 'blue' if t else 'forestgreen'
            self.axs.plot(*(self.numericVerts[e].T), color=color, linewidth=2, zorder=99)
        min = 12
        max = 30
        sizes = np.array(self.pointTopologyChanged, dtype=int) * (max - min) + min
        self.axs.scatter(*(self.numericVerts[:self.instanceSize].T), s=min, color='black', zorder=100)
        self.axs.scatter(*(self.numericVerts[self.instanceSize:].T), s=sizes[self.instanceSize:], color='red', zorder=100)

        self.axs.set_aspect('equal')
        self.axs.title.set_text(name)
        plt.draw()
        plt.pause(self.plotTime)

    def validateTriangleMap(self):
        for triIdx in range(len(self.triangles)):
            for i in range(3):
                other,opp,constraint = self.triangleMap[triIdx,i]
                actualSegIdx = self.getSegmentIdx([self.triangles[triIdx,(i+1)%3],self.triangles[triIdx,(i+2)%3]])
                assert(constraint == actualSegIdx)

                oppEdge = [self.triangles[triIdx,(i+1)%3],self.triangles[triIdx,(i+2)%3]]
                if other != noneFace and other != outerFace:
                    for vIdx in oppEdge:
                        assert(vIdx in self.triangles[other])
                    assert(self.triangleMap[other,opp,0] == triIdx)

    def plotSubinstance(self,vIdxs,triangleIdxs,segmentIdxs=[]):
        self.axs.clear()
        for triIdx in triangleIdxs:
            tri = self.triangles[triIdx]
            cords = self.numericVerts[tri]
            cords = np.concatenate((cords, [cords[0]]))
            for i in range(3):
                if ((self.triangles[triIdx][(i + 1) % 3] >= self.instanceSize) and (
                            self.triangles[triIdx][(i + 2) % 3] >= self.instanceSize) and
                        (self.edgeTopologyChanged[triIdx,i] or (self.triangleMap[triIdx,i,0] != outerFace and self.edgeTopologyChanged[self.triangleMap[triIdx,i,0],self.triangleMap[triIdx,i,1]]))):
                    self.axs.plot(*(cords[[(i+1)%3,(i+2)%3]].T), color='black', linewidth=1, zorder=98, linestyle="dotted")
                else:
                    self.axs.plot(*(cords[[(i+1)%3,(i+2)%3]].T), color='black', linewidth=1, zorder=98)
            if self.isBad(triIdx):
                t = plt.Polygon(self.numericVerts[tri], color='mediumorchid')
                self.axs.add_patch(t)
            else:
                t = plt.Polygon(self.numericVerts[tri], color='palegoldenrod')
                self.axs.add_patch(t)

        for edgeId in segmentIdxs:
            e = self.segments[edgeId]
            t = self.segmentType[edgeId]
            color = 'blue' if t else 'forestgreen'
            self.axs.plot(*(self.numericVerts[e].T), color=color, linewidth=2, zorder=99)

        self.axs.scatter(*(self.numericVerts[vIdxs].T), s=25, color='red', zorder=100)

        self.axs.set_aspect('equal')
        plt.draw()
        plt.pause(self.plotTime)
    ####
    # getters
    ####

    def point(self, i: int):
        return Point(*self.exactVerts[i])

    ####
    # lowest level modifiers. here be dragons
    ####

    def setCircumCenter(self, triIdx):
        self.circumCenters[triIdx] = eg.circumcenter(*self.exactVerts[self.triangles[triIdx]])
        self.circumRadiiSqr[triIdx] = eg.distsq(self.point(self.triangles[triIdx][0]), self.circumCenters[triIdx])

    def unsetBadness(self, triIdx):
        if triIdx in self.badTris:
            self.badTris.remove(triIdx)

    def setBadness(self, triIdx):
        if self.isBad(triIdx):
            self.badTris = self.badTris + [triIdx]
            #dists = self.closestDist[self.badTris]
            #args = dists.argsort()
            # self.badTris = list(np.array(self.badTris)[args[::-1]])
            #if self.seed != None:
            #    np.random.seed(self.seed)
            #    np.random.shuffle(self.badTris)

    def updateEdgeTopology(self,triIdx):
        for otherIdx,oppVIdx,constraint in self.triangleMap[triIdx]:
            if otherIdx != outerFace and otherIdx != noneFace:
                for i in range(3):
                    self.edgeTopologyChanged[otherIdx,i] = True

    def setVertexMap(self, triIdx):
        super().setVertexMap(triIdx)
        for vIdx in self.triangles[triIdx]:
            self.pointTopologyChanged[vIdx] = True
        for vIdx in self.triangles[triIdx]:
            for otherIdx,oppIVIdx in self.vertexMap[vIdx]:
                self.updateEdgeTopology(otherIdx)

    def deleteTriangle(self, triIdx):
        pass

    def splitSegment(self, segIdx, vIdx):
        super().splitSegment(segIdx,vIdx)
        self.segmentType = np.hstack((self.segmentType, self.segmentType[segIdx]))
        #pass

    def unlinkVertexFromConstraint(self, vIdx):
        pass

    def createPoint(self, p:Point):
        self.exactVerts = np.vstack((self.exactVerts, [p]))
        self.numericVerts = np.vstack((self.numericVerts, [float(p.x()), float(p.y())]))
        self.vertexMap.append([])
        self.pointTopologyChanged = np.hstack((self.pointTopologyChanged, True))

        for triIdx in range(len(self.triangles)):
            dist = eg.distsq(self.circumCenters[triIdx], p)
            if (dist < self.closestDist[triIdx]):
                self.closestToCC[triIdx] = len(self.exactVerts) - 1
                self.closestDist[triIdx] = dist

    def deletePoint(self, vIdx):
        pass

    def createTriangles(self, tris, triMaps):
        myIdxs = []
        for i in range(len(tris)):
            tri = tris[i]
            triMap = triMaps[i]

            self.triangles = np.vstack((self.triangles, [tri]))
            self.triangleMap = np.vstack((self.triangleMap, [triMap]))
            self.edgeTopologyChanged = np.vstack((self.edgeTopologyChanged, [True, True, True]))
            self.closestToCC = np.hstack((self.closestToCC, [noneVertex]))
            self.closestDist = np.hstack((self.closestDist, [FieldNumber(0)]))
            self.circumCenters.append(Point(FieldNumber(0), FieldNumber(0)))
            self.circumRadiiSqr.append(FieldNumber(0))

            myIdxs.append(len(self.triangles) - 1)

        for myIdx in myIdxs:

            self.setVertexMap(myIdx)
            self.setCircumCenter(myIdx)
            self.setBadness(myIdx)

            for iVIdx in range(3):
                neighbour, oppIVIdx, constraint = self.triangleMap[myIdx, iVIdx]
                if neighbour != outerFace and neighbour != noneFace and neighbour < len(self.triangles):
                    self.triangleMap[neighbour, oppIVIdx, :2] = [myIdx, iVIdx]

    def setInvalidTriangle(self,triIdx,tri,triMap):
        super().setInvalidTriangle(triIdx,tri,triMap)

        self.setCircumCenter(triIdx)
        self.setBadness(triIdx)

    ####
    # high level modifiers, that are reasonably safe to use
    ####

    def flipEdge(self, triIdx, iVIdx):
        triA = self.triangles[triIdx]
        triAIdx = triIdx
        triAIVIdx = iVIdx
        assert(self.triangleMap[triAIdx,triAIVIdx,2] == noneEdge)
        triBIdx = self.triangleMap[triAIdx,triAIVIdx,0]
        triB = self.triangles[triBIdx]
        triBIVIdx = self.triangleMap[triAIdx,triAIVIdx,1]
        leftAIVIdx = (triAIVIdx+1)%3
        rightAIVIdx = (triAIVIdx+2)%3
        leftBIVIdx = (triBIVIdx + 1)%3 if triA[leftAIVIdx] == triB[(triBIVIdx + 1)%3] else (triBIVIdx + 2)%3
        rightBIVIdx = (triBIVIdx + 1)%3 if triA[rightAIVIdx] == triB[(triBIVIdx + 1)%3] else (triBIVIdx + 2)%3

        newA = [triA[triAIVIdx],triA[leftAIVIdx],triB[triBIVIdx]]
        newAMap = [
            list(self.triangleMap[triBIdx,rightBIVIdx]),
            [triBIdx,2,noneEdge],
            list(self.triangleMap[triAIdx,rightAIVIdx])]

        newB = [triB[triBIVIdx],triA[triAIVIdx],triB[rightBIVIdx]]
        newBMap = [
            list(self.triangleMap[triAIdx,leftAIVIdx]),
            list(self.triangleMap[triBIdx,leftBIVIdx]),
            [triAIdx,1,noneEdge]]

        self.unlinkTriangle(triAIdx)
        self.unlinkTriangle(triBIdx)

        self.setInvalidTriangle(triAIdx,newA,newAMap)
        self.setInvalidTriangle(triBIdx,newB,newBMap)

        #self.plotTriangulation()

        #self.validateTriangleMap()

    def ensureDelauney(self, modifiedTriangles):

        def _isInHorribleEdgeStack(edgestack, edge):
            for e in edgestack:
                for dire in e:
                    if np.all(dire == edge[0]) or np.all(dire == edge[1]):
                        return True
            return False

        def _isNotBanned(bannedList, edge):
            e = [self.triangles[edge[0][0]][(edge[0][1] + 1) % 3], self.triangles[edge[0][0]][(edge[0][1] + 2) % 3]]
            reve = [e[1], e[0]]
            if (e in bannedList) or (reve in bannedList):
                return True
            return False

        def _addEdgeToStack(i, jIdx):
            cc = self.circumCenters[i]
            cr = self.circumRadiiSqr[i]
            #j = self.voronoiEdges[i][jIdx]
            #jMask = self.constrainedMask[i][jIdx]
            #oppositeIndexInJ = None
            j,oppositeIndexInJ,jMask = self.triangleMap[i,jIdx]
            if jMask == noneEdge:
                onlyOn = True
                for v in range(3):
                    inCirc = eg.inCircle(cc, cr, self.point(self.triangles[j][v]))
                    if inCirc == "inside":
                        edge = [[i, jIdx], [j, oppositeIndexInJ]]
                        onlyOn = False
                        if not _isInHorribleEdgeStack(badEdgesInTriangleLand, edge):
                            # add to stack, but not to banned
                            badEdgesInTriangleLand.append(edge)
                    if inCirc == "outside":
                        onlyOn = False
                if onlyOn == True:
                    newTriangleA = [self.triangles[i][jIdx], self.triangles[i][(jIdx + 1) % 3],
                                    self.triangles[j][oppositeIndexInJ]]
                    newTriangleB = [self.triangles[i][jIdx], self.triangles[i][(jIdx + 2) % 3],
                                    self.triangles[j][oppositeIndexInJ]]
                    if not eg.isBadTriangle(*self.exactVerts[newTriangleA]) and not eg.isBadTriangle(
                            *self.exactVerts[newTriangleB]):
                        edge = [[i, jIdx], [j, oppositeIndexInJ]]
                        if (not _isInHorribleEdgeStack(badEdgesInTriangleLand, edge)) and (
                                not _isNotBanned(bannedEdges, edge)):
                            badEdgesInTriangleLand.append(edge)
                            bannedEdges.append([self.triangles[edge[0][0]][(edge[0][1] + 1) % 3],
                                                self.triangles[edge[0][0]][(edge[0][1] + 2) % 3]])

        #self.validate()
        # they are stored as [triangleindex, inducing index]
        badEdgesInTriangleLand = []
        bannedEdges = []
        if modifiedTriangles is None:
            for i in range(len(self.triangles)):
                for jIdx in range(3):
                    _addEdgeToStack(i, jIdx)
        else:
            for i in modifiedTriangles:
                for jIdx in range(3):
                    _addEdgeToStack(i, jIdx)

        while len(badEdgesInTriangleLand) > 0:
            edge = badEdgesInTriangleLand[-1]
            # print(edge)
            badEdgesInTriangleLand = badEdgesInTriangleLand[:-1]
            assert (len(edge) > 0)
            i, jIdx = edge[0]
            j = edge[1][0]
            opposingIdx = edge[1][1]

            self.flipEdge(i, jIdx)

            # remove all mentions of i and j from stack
            for e in badEdgesInTriangleLand:
                for it in reversed(range(len(e))):
                    if i == e[it][0] or j == e[it][0]:
                        e.pop(it)
            # revalidate edgestack
            for it in range(len(badEdgesInTriangleLand)):
                if len(badEdgesInTriangleLand[it]) == 2:
                    continue
                elif len(badEdgesInTriangleLand[it]) == 1:
                    triIdx = badEdgesInTriangleLand[it][0][0]
                    iVIdx = badEdgesInTriangleLand[it][0][1]
                    otherTriIdx,oppositeIt,_ = self.triangleMap[triIdx,iVIdx]
                    badEdgesInTriangleLand[it].append([otherTriIdx, oppositeIt])
                else:
                    assert (False)

            for jIdx in range(3):
                _addEdgeToStack(i, jIdx)
            for iIdx in range(3):
                _addEdgeToStack(j, iIdx)
        #self.validate()

    def addPoint(self, p:Point):

        # first figure out, in which triangle the point lies. If inside, split into three, if on, split adjacent
        # faces into two each
        hitTriIdxs = []
        grazedTriIdxs = []
        for triIdx in range(len(self.triangles)):
            tri = self.triangles[triIdx]
            sides = np.array([eg.onWhichSide(Segment(self.point(self.triangles[triIdx,(i+1)%3]),self.point(self.triangles[triIdx,(i+2)%3])),p) for i in range(3)])
            if np.all((sides == "left")) or np.all((sides == "right")):
                hitTriIdxs.append([triIdx])
            elif np.all((sides == "left") | (sides == "colinear")) or np.all((sides == "right") | (sides == "colinear")):
                grazedTriIdxs.append([triIdx,np.argwhere(sides == "colinear")])
        if len(hitTriIdxs) == 1:
            #inside
            assert(len(grazedTriIdxs) == 0)
            hitTriIdx = hitTriIdxs[0][0]
            hitTri = self.triangles[hitTriIdx]

            newPointIdx = len(self.exactVerts)
            newLeftIdx = len(self.triangles)
            newRightIdx = len(self.triangles)+1

            newLeft = [hitTri[0],newPointIdx,hitTri[2]]
            newLeftMap = [
                [hitTriIdx,1,noneEdge],
                list(self.triangleMap[hitTriIdx,1]),
                [newRightIdx,1,noneEdge]]

            newRight = [hitTri[0],hitTri[1],newPointIdx]
            newRightMap = [
                [hitTriIdx,2,noneEdge],
                [newLeftIdx,2,noneEdge],
                list(self.triangleMap[hitTriIdx,2])]

            newSelf = [newPointIdx,hitTri[1],hitTri[2]]
            newSelfMap = [
                list(self.triangleMap[hitTriIdx,0]),
                [newLeftIdx,0,noneEdge],
                [newRightIdx,0,noneEdge]]


            self.createPoint(p)
            self.unlinkTriangle(hitTriIdx)

            self.createTriangles([newLeft,newRight],[newLeftMap,newRightMap])
            self.setInvalidTriangle(hitTriIdx,newSelf,newSelfMap)

            self.validateTriangleMap()


            self.ensureDelauney([hitTriIdx,newLeftIdx,newRightIdx])

            self.plotTriangulation()

            return True
        elif len(grazedTriIdxs) == 0:
            #outside
            return False
        elif len(grazedTriIdxs) == 1:
            #boundary
            assert(len(grazedTriIdxs[0][1]) == 1)
            grazedIdx = grazedTriIdxs[0][0]
            grazed = self.triangles[grazedIdx]
            grazedIVIdx = grazedTriIdxs[0][1][0][0]

            newPointIdx = len(self.exactVerts)
            newTriIdx = len(self.triangles)

            self.createPoint(p)

            #this MUST be a boundary!
            assert(self.triangleMap[grazedIdx,grazedIVIdx,2] != noneEdge)
            assert(self.triangleMap[grazedIdx,grazedIVIdx,0] == outerFace)

            segIdx = self.triangleMap[grazedIdx,grazedIVIdx,2]
            self.splitSegment(segIdx,newPointIdx)

            newTri = [grazed[grazedIVIdx],grazed[(grazedIVIdx+1)%3],newPointIdx]
            newTriMap = [
                [outerFace,noneIntervalVertex,self.getSegmentIdx([grazed[(grazedIVIdx+1)%3],newPointIdx])],
                [grazedIdx,2,noneEdge],
                list(self.triangleMap[grazedIdx,(grazedIVIdx+2)%3])]

            newSelf = [grazed[grazedIVIdx],newPointIdx,grazed[(grazedIVIdx+2)%3]]
            newSelfMap = [
                [outerFace,noneIntervalVertex,self.getSegmentIdx([grazed[(grazedIVIdx+2)%3],newPointIdx])],
                list(self.triangleMap[grazedIdx,(grazedIVIdx+1)%3]),
                [newTriIdx,1,noneEdge]]


            self.unlinkTriangle(grazedIdx)

            self.createTriangles([newTri],[newTriMap])
            self.setInvalidTriangle(grazedIdx,newSelf,newSelfMap)

            self.validateTriangleMap()

            self.ensureDelauney([grazedIdx,newTriIdx])

            self.plotTriangulation()

            return True
        elif len(grazedTriIdxs) == 2:
            #constraint or unlucky
            assert(len(grazedTriIdxs[0][1]) == 1)
            assert(len(grazedTriIdxs[1][1]) == 1)

            grazedAIdx = grazedTriIdxs[0][0]
            grazedA = self.triangles[grazedAIdx]
            grazedAIVIdx = grazedTriIdxs[0][1][0][0]
            grazedBIdx = grazedTriIdxs[1][0]
            grazedB = self.triangles[grazedBIdx]
            grazedBIVIdx = grazedTriIdxs[1][1][0][0]

            newPointIdx = len(self.exactVerts)
            newTriByAIdx = len(self.triangles)
            newTriByBIdx = len(self.triangles)+1

            self.createPoint(p)

            segIdx = self.triangleMap[grazedAIdx,grazedAIVIdx,2]
            if segIdx != noneEdge:
                self.splitSegment(segIdx,newPointIdx)

            diff = False
            if grazedA[(grazedAIVIdx+1)%3] != grazedB[(grazedBIVIdx+1)%3]:
                diff = True


            newTriByA = [grazedA[grazedAIVIdx],grazedA[(grazedAIVIdx+1)%3],newPointIdx]
            newTriByAMap = [
                [grazedBIdx if diff else newTriByBIdx,0,self.getSegmentIdx([grazedA[(grazedAIVIdx+1)%3],newPointIdx])],
                [grazedAIdx,2,noneEdge],
                list(self.triangleMap[grazedAIdx,(grazedAIVIdx+2)%3])]

            newASelf = [grazedA[grazedAIVIdx],newPointIdx,grazedA[(grazedAIVIdx+2)%3]]
            newASelfMap = [
                [newTriByBIdx if diff else grazedBIdx,0,self.getSegmentIdx([newPointIdx,grazedA[(grazedAIVIdx+2)%3]])],
                list(self.triangleMap[grazedAIdx,(grazedAIVIdx+1)%3]),
                [newTriByAIdx,1,noneEdge]]


            newTriByB = [grazedB[grazedBIVIdx],grazedB[(grazedBIVIdx+1)%3],newPointIdx]
            newTriByBMap = [
                [grazedAIdx if diff else newTriByAIdx,0,self.getSegmentIdx([grazedB[(grazedBIVIdx+1)%3],newPointIdx])],
                [grazedBIdx,2,noneEdge],
                list(self.triangleMap[grazedBIdx,(grazedBIVIdx+2)%3])]

            newBSelf = [grazedB[grazedBIVIdx],newPointIdx,grazedB[(grazedBIVIdx+2)%3]]
            newBSelfMap = [
                [newTriByAIdx if diff else grazedAIdx,0,self.getSegmentIdx([newPointIdx,grazedB[(grazedBIVIdx+2)%3]])],
                list(self.triangleMap[grazedBIdx,(grazedBIVIdx+1)%3]),
                [newTriByBIdx,1,noneEdge]]

            self.unlinkTriangle(grazedAIdx)
            self.unlinkTriangle(grazedBIdx)

            self.createTriangles([newTriByA,newTriByB],[newTriByAMap,newTriByBMap])
            self.setInvalidTriangle(grazedAIdx,newASelf,newASelfMap)
            self.setInvalidTriangle(grazedBIdx,newBSelf,newBSelfMap)

            self.validateTriangleMap()

            self.ensureDelauney([grazedAIdx,grazedBIdx,newTriByBIdx,newTriByAIdx])

            self.plotTriangulation()

            return True
        else:
            #vertex
            return False

    def removePoint(self, vIdx):
        pass

    def movePoint(self, vIdx, target: Point):

        #check if the new point has the same local topology as vIdx
        for triIdx,iVIdx in self.vertexMap[vIdx]:
            oppEdge = [self.triangles[triIdx,(iVIdx+1)%3],self.triangles[triIdx,(iVIdx+2)%3]]
            iPoint = self.point(self.triangles[triIdx,iVIdx])
            seg = Segment(self.point(oppEdge[0]),self.point(oppEdge[1]))
            if onWhichSide(seg,iPoint) != onWhichSide(seg,target):
                return False


        self.exactVerts[vIdx] = target
        self.numericVerts[vIdx] = [float(target.x()), float(target.y())]

        self.pointTopologyChanged[vIdx] = True
        for triIdx,iVIdx in self.vertexMap[vIdx]:
            self.pointTopologyChanged[self.triangles[triIdx,(iVIdx+1)%3]] = True
            self.pointTopologyChanged[self.triangles[triIdx,(iVIdx+2)%3]] = True
            self.updateEdgeTopology(triIdx)

        return True

    def getEnclosement(self, vIdxs):
        connections = [[vIdxs[i],vIdxs[i+1]] for i in range(len(vIdxs)-1)]

        def inConnections(edge):
            if edge in connections or edge[::-1] in connections:
                return True
            return False

        anchorInternalIdx = 0
        anchorAdvancer = 1
        curAnchor = vIdxs[anchorInternalIdx]
        curFace = None
        leftIVIdx = None
        curIVIdx = None
        rightIVIdx = None
        for faceIdx,internal in self.vertexMap[curAnchor]:
            curFace = faceIdx
            leftIVIdx = internal
            curIVIdx = (leftIVIdx+1)%3
            rightIVIdx = (leftIVIdx+2)%3
            if self.triangles[faceIdx,curIVIdx] in vIdxs:
                assert(inConnections([curAnchor,self.triangles[faceIdx,curIVIdx]]))
                curIVIdx,rightIVIdx = rightIVIdx,curIVIdx

                if self.triangles[faceIdx,curIVIdx] in vIdxs:
                    assert(inConnections([curAnchor,self.triangles[faceIdx,curIVIdx]]))
                else:
                    break
            else:
                break

        link = []
        insideFaces = []
        curIdx = self.triangles[curFace,curIVIdx]
        while len(link) == 0 or (link[0] != curIdx or insideFaces[0] != curFace):
            assert(self.triangles[curFace,leftIVIdx] == curAnchor)
            if curIdx not in vIdxs:
                link.append(curIdx)
            if len(insideFaces)==0 or (curFace != insideFaces[-1] and curFace != insideFaces[0]):
                insideFaces.append(curFace)

            self.plotSubinstance(link,insideFaces)

            if self.triangles[curFace,rightIVIdx] in vIdxs:
                if inConnections([self.triangles[curFace,leftIVIdx],self.triangles[curFace,rightIVIdx]]):

                    #to stay consistent, we need to remove the last added vertex. it will be added lateron again
                    assert(link[-1] == curIdx)
                    link = link[:-1]

                    #advance anchor and advance advancor if necessary
                    anchorInternalIdx += anchorAdvancer
                    if anchorInternalIdx == len(vIdxs)-1:
                        anchorAdvancer = -1
                    curAnchor = vIdxs[anchorInternalIdx]
                    #rotate
                    curIVIdx,leftIVIdx,rightIVIdx = leftIVIdx,rightIVIdx,curIVIdx
                    curIdx = self.triangles[curFace,curIVIdx]
                else:
                    assert(False)

            # advance
            elif self.triangleMap[curFace,curIVIdx,0] == outerFace:

                link.append(self.triangles[curFace, rightIVIdx])

                keepGoing = True
                while keepGoing:
                    link.append(self.triangles[curFace, leftIVIdx])


                    for faceIdx,internalIdx in self.vertexMap[curAnchor]:
                        if faceIdx == curFace:
                            continue
                        leftInternal = (internalIdx +1) %3
                        rightInternal = (internalIdx +2) %3
                        if self.triangleMap[faceIdx,leftInternal,0] != outerFace:
                            leftInternal = (internalIdx + 2) % 3
                            rightInternal = (internalIdx + 1) % 3
                        if self.triangleMap[faceIdx,leftInternal,0] != outerFace:
                            #not the other face
                            continue
                        if self.triangles[faceIdx,rightInternal] in vIdxs:
                            if inConnections([self.triangles[faceIdx, internalIdx], self.triangles[faceIdx, rightInternal]]):
                                # advance anchor and advance advancor if necessary
                                anchorInternalIdx += anchorAdvancer
                                if anchorInternalIdx == len(vIdxs) - 1:
                                    anchorAdvancer = -1
                                curAnchor = vIdxs[anchorInternalIdx]
                                # rotate
                                curFace = faceIdx
                                curIVIdx = leftInternal
                                leftIVIdx = rightInternal
                                rightIVIdx = internalIdx

                            else:
                                assert (False)
                            break
                        else:
                            keepGoing = False
                            curFace = faceIdx
                            curIVIdx = rightInternal
                            leftIVIdx = internalIdx
                            rightIVIdx = leftInternal
                            #if self.triangles[curFace, leftIVIdx] != curAnchor:
                            #    leftIVIdx, rightIVIdx = rightIVIdx, leftIVIdx
                            curIdx = self.triangles[curFace,curIVIdx]
                            #safeguard, because we will stop next iteration but the last face has not been added yet
                            if len(insideFaces)==0 or (curFace != insideFaces[-1] and curFace != insideFaces[0]):
                                insideFaces.append(curFace)
                            break
            else:
                nextFace = self.triangleMap[curFace,curIVIdx,0]
                for i in range(3):
                    if self.triangles[nextFace,i] == self.triangles[curFace,rightIVIdx]:
                        curIVIdx = i
                        curFace = nextFace
                        curIdx = self.triangles[curFace,curIVIdx]
                        break
                leftIVIdx = (curIVIdx +1) %3
                rightIVIdx = (curIVIdx+2) %3
                if self.triangles[curFace,leftIVIdx] != curAnchor:
                    leftIVIdx,rightIVIdx = rightIVIdx,leftIVIdx

                if len(insideFaces)==0 or (curFace != insideFaces[-1] and curFace != insideFaces[0]):
                    insideFaces.append(curFace)


        constraints = []
        insideConstraints = []
        boundaryConstraints = []
        for triIdx in insideFaces:
            for i in range(3):
                if self.triangleMap[triIdx,i,2] != noneEdge:
                    constraints.append(self.triangleMap[triIdx,i,2])
        for con in constraints:
            if con in boundaryConstraints:
                boundaryConstraints.remove(con)
                insideConstraints.append(con)
            else:
                boundaryConstraints.append(con)

        boundaryConstraintTypes = self.segmentType[boundaryConstraints]

        #def __init__(self, vIdxs, triIdxs, boundary, exactPoints, numericPoints, innerCons, boundaryCons, boundaryType,
        #             steinercutoff, axs):

        return GeometricSubproblem(vIdxs,insideFaces,link,self.exactVerts[vIdxs+list(set(link))],self.numericVerts[vIdxs+list(set(link))],self.segments[insideConstraints],self.segments[boundaryConstraints],boundaryConstraintTypes,self.instanceSize,self.axs)

    def replaceEnclosement(self, insideVIdxs, boundaryVIdxs, newPoints, newTriangles):
        pass