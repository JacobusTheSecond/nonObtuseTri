import copy
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from cgshop2025_pyutils import Cgshop2025Solution, Cgshop2025Instance
from cgshop2025_pyutils.geometry import FieldNumber, Point, Segment
from dask.array import outer
from numpy import ndarray, dtype

import exact_geometry as eg

import triangle as tr  # https://rufat.be/triangle/

import logging

#from src.primitiveTester import onWhichSide, isBadTriangle

logging.basicConfig(format="%(asctime)s %(message)s",datefmt="%y-%m-%d %H:%M:%S",level=logging.INFO)

#some useful constants
outerFace = np.iinfo(int).max
noneFace = outerFace - 1
noneEdge = outerFace - 2
noneVertex = outerFace - 3
noneIntervalVertex = outerFace - 4

class GeometricSubproblem:
    def __init__(self, vIdxs, triIdxs, boundary, exactPoints, numericPoints, innerCons, boundaryCons, boundaryType, steinercutoff, numBadTris, axs=None):
        #some housekeeping to have some handle on what to delete later

        #lets make this better
        self.axs = axs
        self.triIdxs = triIdxs
        self.exactVerts = exactPoints
        self.numericVerts = numericPoints
        self.numBadTris = numBadTris

        self.localMap = np.array(list(vIdxs) + list(boundary))
        self.isSteiner = [False if v < steinercutoff else True for v in self.localMap]

        insideIdxs = []
        boundaryIdxs = []

        for v in vIdxs:
            positions = np.where(self.localMap == v)[0]
            insideIdxs.append(positions[0])
        for v in boundary:
            positions = np.where(self.localMap == v)[0]
            boundaryIdxs.append(positions[0])

        boundaryIdxs = np.array(boundaryIdxs)
        insideIdxs = np.array(insideIdxs)

        insideCons = []
        for s,t in innerCons:
            insideCons.append([np.where(self.localMap == s)[0][0],np.where(self.localMap == t)[0][0]])

        outsideCons = []
        for s,t in boundaryCons:
            outsideCons.append([np.where(self.localMap == s)[0][0],np.where(self.localMap == t)[0][0]])

        insideCons = np.array(insideCons)
        outsideCons = np.array(outsideCons)
        outsideType = boundaryType

        #move points from boundary to inside
        deleteList = None
        moveInside = []
        while deleteList is None or len(deleteList) > 0:
            deleteList = []
            for i in range(len(boundaryIdxs)):
                if boundaryIdxs[i] == boundaryIdxs[(i+2)%len(boundaryIdxs)]:
                    deleteList.append(i)
                    deleteList.append((i+1)%len(boundaryIdxs))
                    moveInside.append(boundaryIdxs[(i+1)%len(boundaryIdxs)])
            boundaryIdxs = np.delete(boundaryIdxs, deleteList)
        insideIdxs = np.hstack((insideIdxs, np.array(moveInside,dtype=int)))

        #now boundary should be free of repitions. assert so (otherwise we have a loop somehwere which sucks)
        assert(len(boundaryIdxs) == len(list(set(boundaryIdxs))))

        #we still have steinerpoints on the boundary, and duplicates in insidevertices and boundary vertices. remove
        #these
        deleteList = []
        moveInside = []
        for i in range(len(boundaryIdxs)):
            bIdx = boundaryIdxs[i]
            if self.isSteiner[bIdx]:
                if len(outsideCons) > 0:
                    edgeIds = np.where((outsideCons[:,0] == bIdx)|(outsideCons[:,1]==bIdx))[0]
                    edges = outsideCons[edgeIds]
                    assert(len(edges) <= 2)
                    if len(edges) == 2:
                        if outsideType[edgeIds[0]] and outsideType[edgeIds[1]]:
                            newSeg = []
                            for e in edges:
                                newSeg.append(e[np.where(e != bIdx)][0])
                            outsideCons[edgeIds[0]] = newSeg
                            outsideCons = np.delete(outsideCons, [edgeIds[1]], axis=0)
                            outsideType = np.delete(outsideType, [edgeIds[1]], axis=0)
                            deleteList.append(i)
                            moveInside.append(bIdx)
        boundaryIdxs = np.delete(boundaryIdxs, deleteList)
        insideIdxs = np.hstack((insideIdxs, np.array([v for v in moveInside if v not in insideIdxs],dtype=int)))

        #boundary should now be free of steinerpoints on proper boundaries and as such should also be intersectionfree with insideIdxs. assert so
        for inVIdx in insideIdxs:
            assert(inVIdx not in boundaryIdxs)

        #we are left with removing unnecessary steinerpoints in the middle
        deleteList = []
        insideSteiners = []
        for i in range(len(insideIdxs)):
            iIdx = insideIdxs[i]
            if self.isSteiner[iIdx]:
                insideSteiners.append(iIdx)
                deleteList.append(i)
                if len(insideCons) > 0:
                    edgeIds = np.where((insideCons[:,0] == iIdx)|(insideCons[:,1]==iIdx))[0]
                    edges = insideCons[edgeIds]
                    assert(len(edges) == 2 or len(edges) == 0)
                    if len(edges) == 2:
                        newSeg = []
                        for e in edges:
                            newSeg.append(e[np.where(e != iIdx)][0])
                        insideCons[edgeIds[0]] = newSeg
                        insideCons = np.delete(insideCons, [edgeIds[1]], axis=0)

        assert(len(outsideCons) == len(outsideType))

        self.insideVertices = np.delete(insideIdxs, deleteList)
        self.insideSteiners = np.array(insideSteiners)
        self.boundaryVertices = boundaryIdxs
        self.boundarySegs = np.vstack((boundaryIdxs, np.roll(boundaryIdxs,1))).T
        self.segments = np.array(list(outsideCons) + list(insideCons))
        boundaryType = []
        for e in self.boundarySegs:
            eIdx = self.getSegmentIdx(e)
            if eIdx == noneEdge:
                boundaryType.append("None")
            elif outsideType[eIdx]:
                boundaryType.append("boundary")
            else:
                boundaryType.append("halfin")
        self.boundaryType = np.array(boundaryType)
        self.insideSegs = insideCons

        #self.plotMe()

    def plotMe(self):
        self.axs.clear()

        for edgeId in range(len(self.boundarySegs)):
            e = self.boundarySegs[edgeId]
            color = "black"
            if self.boundaryType[edgeId] == "halfin":
                color = "forestgreen"
            elif self.boundaryType[edgeId] == "boundary":
                color = "blue"
            self.axs.plot(*(self.numericVerts[e].T), color=color, linewidth=2, zorder=98)

        for edgeId in range(len(self.insideSegs)):
            e = self.insideSegs[edgeId]
            color = 'lime'
            self.axs.plot(*(self.numericVerts[e].T), color=color, linewidth=2, zorder=99)

        for vIdx in list(self.boundaryVertices):
            color = 'red' if self.isSteiner[vIdx] else "black"
            self.axs.scatter(*(self.numericVerts[[vIdx]].T), s=20, color=color, zorder=100)

        for vIdx in list(self.insideSteiners) + list(self.insideVertices):
            color = 'red' if self.isSteiner[vIdx] else "black"
            self.axs.scatter(*(self.numericVerts[[vIdx]].T), s=30, marker = "*", color=color, zorder=100)

        self.axs.set_aspect('equal')
        plt.draw()
        plt.pause(0.001)

    def getSegmentIdx(self, querySeg):
        if len(self.segments) == 0:
            return noneEdge
        locs = np.concatenate(
            (np.argwhere(np.all((self.segments == querySeg), -1)),
             np.argwhere(np.all((self.segments == querySeg[::-1]), -1))))
        if len(locs) == 1:
            return locs[0, 0]
        else:
            return noneEdge

    def getInsideSteiners(self):
        return self.localMap[self.insideSteiners]

    def solve(self,k=1):
        #eval estimates how good this solution is compared to what we had. Overestimates how had a bad triangle is, to weight fixing more
        eval = - len(self.insideSteiners) - (2*self.numBadTris)
        if k == 1:
            if len(self.insideSegs) == 0:
                points = [Point(*v) for v in self.exactVerts[self.boundaryVertices]]
                soltype,sol = eg.findCenterOfLink(points)
                if soltype == "None":
                    bestEval,bestSol = None,None
                    for bIdx in range(len(self.boundarySegs)):
                        con = self.boundarySegs[bIdx]
                        con = [np.where(self.boundaryVertices == con[0])[0][0],np.where(self.boundaryVertices == con[1])[0][0]]
                        if (con[0] >= len(points) or con[1] >= len(points)):
                            print("wtf")
                        constrainedSolType,constrainedSol = eg.findCenterOfLinkConstrained(points,con[0],con[1])
                        if constrainedSolType == "None":
                            continue
                        elif constrainedSolType == "vertex":
                            newEval = eval
                            newSol = [[constrainedSolType,self.localMap[self.boundaryVertices[constrainedSol]]]]
                            if bestEval == None or newEval < bestEval:
                                bestEval = newEval
                                bestSol = newSol
                        else:
                            newEval = eval
                            if self.boundaryType[bIdx] == "boundary":
                                newEval += 1
                            elif self.boundaryType[bIdx] == "halfin":
                                newEval += 1.25
                            else:
                                newEval += 2
                            newSol = [[constrainedSolType,constrainedSol[0]]]
                            if bestEval == None or newEval < bestEval:
                                bestEval = newEval
                                bestSol = newSol
                    if bestEval != None:
                        return bestEval, bestSol
                    else:
                        return None, None
                elif soltype == "vertex":
                    return eval, [[soltype,self.localMap[self.boundaryVertices[sol]]]]
                else:
                    return eval + 1, [[soltype,sol[0]]]
            elif len(self.insideSegs) == 1 and len(self.insideVertices) == 0:
                con = self.insideSegs[0]
                con = [np.where(self.boundaryVertices == con[0])[0][0],np.where(self.boundaryVertices == con[1])[0][0]]
                points = [Point(*v) for v in self.exactVerts[self.boundaryVertices]]
                solType,sol = eg.findCenterOfLinkConstrained(points,con[0],con[1])
                if solType == "None":
                    return None,None
                elif solType == "vertex":
                    return eval, [[solType,self.localMap[self.boundaryVertices[sol]]]]
                else:
                    return eval + 1, [[solType,sol[0]]]

            else:
                return None, None
        else:
            return None, None

class Triangulation:
    def __init__(self, instance: Cgshop2025Instance, withValidate=False, seed=None,axs=None):

        #_, gpaxs = plt.subplots(1,1)
        self.gpaxs = None#gpaxs

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

        #self.badTris = [idx for idx in range(len(self.triangles)) if self.isBad(idx)]
        self.badTris = np.array([False for idx in range(len(self.triangles))])
        for idx in range(len(self.triangles)):
            self.badTris[idx] = self.isBad(idx)

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

        self.circumCenters = np.array([eg.circumcenter(*self.exactVerts[tri]) for tri in self.triangles])
        self.circumRadiiSqr = np.array([eg.distsq(self.point(self.triangles[i][0]), Point(*self.circumCenters[i])) for i in
                               range(len(self.triangles))])

        self.closestToCC = []
        self.closestDist = []
        for triIdx in range(len(self.triangles)):
            closest = None
            closestdist = None
            for vIdx in range(len(self.exactVerts)):
                dist = eg.distsq(Point(*self.circumCenters[triIdx]), self.point(vIdx))
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

    def validateTriangleMap(self,ignoreList=[]):
        if not self.withValidate:
            return
        for triIdx in range(len(self.triangles)):
            if triIdx in ignoreList:
                continue
            for i in range(3):
                other,opp,constraint = self.triangleMap[triIdx,i]
                actualSegIdx = self.getSegmentIdx([self.triangles[triIdx,(i+1)%3],self.triangles[triIdx,(i+2)%3]])
                assert(constraint == actualSegIdx)

                oppEdge = [self.triangles[triIdx,(i+1)%3],self.triangles[triIdx,(i+2)%3]]

                assert other != noneFace
                if other != outerFace:
                    for vIdx in oppEdge:
                        assert(vIdx in self.triangles[other])
                    assert(self.triangleMap[other,opp,0] == triIdx)

    def validateVertexMap(self,withUnlinkedTris=False):
        if not self.withValidate:
            return
        for i in range(len(self.exactVerts)):
            for triIdx,selfInternal in self.vertexMap[i]:
                if not (self.triangles[triIdx,selfInternal] == i):
                    print("wtf")
        for triIdx in range(len(self.triangles)):
            for i in range(3):
                if withUnlinkedTris:
                    if self.triangles[triIdx,i] != noneVertex:
                        assert(triIdx in [tI for tI,_ in self.vertexMap[self.triangles[triIdx,i]]])
                else:
                    assert(triIdx in [tI for tI,_ in self.vertexMap[self.triangles[triIdx,i]]])

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

    def solutionParse(self):
        inneredges = []
        for tri in self.triangles:
            edges = ([tri[0], tri[1]], [tri[1], tri[2]], [tri[2], tri[0]])
            # check if edge is already added or are in segments
            for e in edges:
                exists = False
                e = np.sort(e)
                for seg in inneredges:
                    if np.all(e == seg):
                        exists = True
                        break
                if not exists:
                    inneredges.append(e)
        sx = []
        sy = []
        for i in range(self.instanceSize, len(self.exactVerts)):
            sx.append(self.point(i).x().exact())
            sy.append(self.point(i).y().exact())
        return Cgshop2025Solution(instance_uid=self.instance_uid, steiner_points_x=sx, steiner_points_y=sy,
                                  edges=inneredges)
    ####
    # getters
    ####

    def point(self, i: int):
        return Point(*self.exactVerts[i])

    ####
    # lowest level modifiers. here be dragons
    ####

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

    def setCircumCenter(self, triIdx):
        self.circumCenters[triIdx] = eg.circumcenter(*self.exactVerts[self.triangles[triIdx]])
        self.circumRadiiSqr[triIdx] = eg.distsq(self.point(self.triangles[triIdx][0]), Point(*self.circumCenters[triIdx]))

    def unsetBadness(self, triIdx):
        self.badTris[triIdx] = False

    def setBadness(self, triIdx):
        self.badTris[triIdx] = self.isBad(triIdx)
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
        for iVIdx in range(3):
            self.vertexMap[self.triangles[triIdx,iVIdx]].append([triIdx,iVIdx])
        for vIdx in self.triangles[triIdx]:
            self.pointTopologyChanged[vIdx] = True
        for vIdx in self.triangles[triIdx]:
            for otherIdx,oppIVIdx in self.vertexMap[vIdx]:
                self.updateEdgeTopology(otherIdx)

    def deleteTriangle(self, triIdx):
        pass

    def isBad(self, triIdx):
        return eg.isBadTriangle(*self.exactVerts[self.triangles[triIdx]])

    def splitSegment(self, segIdx, vIdx):
        seg = [self.segments[segIdx][0], self.segments[segIdx][1]]
        self.segments[segIdx] = [seg[0], vIdx]
        self.segments = np.vstack((self.segments, [vIdx, seg[1]]))
        self.segmentType = np.hstack((self.segmentType, self.segmentType[segIdx]))
        #pass

    #def unlinkVertexFromConstraint(self, vIdx):
    #    constraints = self.segments[np.where((self.segments[:,0] == vIdx) | (self.segments[:,1] == vIdx))]


    def createPoint(self, p:Point):
        self.exactVerts = np.vstack((self.exactVerts, [p]))
        self.numericVerts = np.vstack((self.numericVerts, [float(p.x()), float(p.y())]))
        self.vertexMap.append([])
        self.pointTopologyChanged = np.hstack((self.pointTopologyChanged, True))

        for triIdx in range(len(self.triangles)):
            dist = eg.distsq(Point(*self.circumCenters[triIdx]), p)
            if (dist < self.closestDist[triIdx]):
                self.closestToCC[triIdx] = len(self.exactVerts) - 1
                self.closestDist[triIdx] = dist

    def deletePoint(self, vIdx):
        assert len(self.vertexMap[vIdx]) == 0
        assert len(np.where((self.segments[:,0] == vIdx) | (self.segments[:,1] == vIdx))[0]) == 0

        self.exactVerts = np.delete(self.exactVerts, [vIdx], axis = 0)
        self.numericVerts = np.delete(self.numericVerts, [vIdx], axis = 0)
        self.vertexMap.pop(vIdx)
        self.pointTopologyChanged = np.delete(self.pointTopologyChanged, [vIdx], axis = 0)

        for triIdx in range(len(self.triangles)):
            for i in range(3):
                if self.triangles[triIdx,i] > vIdx:
                    self.triangles[triIdx,i] -= 1

        for seg in range(len(self.segments)):
            for i in range(2):
                if self.segments[seg,i] > vIdx:
                    self.segments[seg,i] -= 1

        updateDists = []
        for triIdx in range(len(self.triangles)):
            if self.closestToCC[triIdx] == vIdx:
                updateDists.append(triIdx)

        for triIdx in updateDists:
            closest = None
            closestdist = None
            for vIdx in range(len(self.exactVerts)):
                dist = eg.distsq(Point(*self.circumCenters[triIdx]), self.point(vIdx))
                if (closest is None) or (dist < closestdist):
                    closest = vIdx
                    closestdist = dist
            self.closestToCC[triIdx] = closest
            self.closestDist[triIdx] = closestdist

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
            self.badTris = np.hstack((self.badTris, [False]))
            self.circumCenters = np.vstack((self.circumCenters,[Point(FieldNumber(0), FieldNumber(0))]))
            self.circumRadiiSqr = np.hstack((self.circumRadiiSqr,[FieldNumber(0)]))

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
        self.triangles[triIdx] = tri
        self.triangleMap[triIdx] = triMap

        self.setVertexMap(triIdx)
        self.setCircumCenter(triIdx)

        for iVIdx in range(3):
            neighbour, oppIVIdx, constraint = self.triangleMap[triIdx,iVIdx]
            if neighbour != outerFace and neighbour != noneFace and neighbour < len(self.triangles):
                self.triangleMap[neighbour,oppIVIdx,:2] = [triIdx,iVIdx]

        self.setCircumCenter(triIdx)
        self.setBadness(triIdx)

    def unlinkTriangle(self, triIdx):
        self.unsetBadness(triIdx)
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

    def deleteUnlinkedTriangles(self,triIdxs):
        self.triangles = np.delete(self.triangles, triIdxs, axis = 0)
        self.triangleMap = np.delete(self.triangleMap, triIdxs, axis = 0)
        self.closestToCC = np.delete(self.closestToCC, triIdxs, axis = 0)
        self.closestDist = np.delete(self.closestDist, triIdxs, axis = 0)
        self.badTris = np.delete(self.badTris, triIdxs, axis = 0)
        self.circumCenters = np.delete(self.circumCenters, triIdxs, axis = 0)
        self.circumRadiiSqr = np.delete(self.circumRadiiSqr, triIdxs, axis = 0)

        for triIdx in range(len(self.triangles)):
            for i in range(3):
                if self.triangleMap[triIdx,i,0] != outerFace and self.triangleMap[triIdx,i,0] != noneFace:
                    self.triangleMap[triIdx,i,0] -= len(np.where(triIdxs < self.triangleMap[triIdx,i,0])[0])

        for i in range(len(self.vertexMap)):
            newVMap = []
            for face,selfId in self.vertexMap[i]:
                newVMap.append([face - len(np.where(triIdxs < face)[0]),selfId])
            self.vertexMap[i] = newVMap


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
            cc = Point(*self.circumCenters[i])
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
            self.validateVertexMap()

            for triIdx in range(len(self.triangles)):
                for i in range(3):
                    if not (self.triangleMap[triIdx,i,0] != noneFace):
                        print("och man")
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
            self.validateVertexMap()

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
            self.validateVertexMap()

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
            if eg.onWhichSide(seg,iPoint) != eg.onWhichSide(seg,target):
                return False


        self.exactVerts[vIdx] = target
        self.numericVerts[vIdx] = [float(target.x()), float(target.y())]

        self.pointTopologyChanged[vIdx] = True
        for triIdx,iVIdx in self.vertexMap[vIdx]:
            self.pointTopologyChanged[self.triangles[triIdx,(iVIdx+1)%3]] = True
            self.pointTopologyChanged[self.triangles[triIdx,(iVIdx+2)%3]] = True
            self.updateEdgeTopology(triIdx)

        return True

    def trianglesOnEdge(self, pIdx, qIdx):
        return np.array([triIdx for triIdx,_ in self.vertexMap[pIdx] if qIdx in self.triangles[triIdx]])

    def internalMergePoints(self, source,target):

        sharedTris = self.trianglesOnEdge(source, target)
        specialSourceTris = []
        specialSourceOpps = []
        specialTargetTris = []
        specialTargetOpps = []
        newConstraint = []

        for triIdx in sharedTris:
            tri = self.triangles[triIdx]
            sourceInt = np.where(tri == source)[0][0]
            targetInt = np.where(tri == target)[0][0]

            possibleSpecial = self.triangleMap[triIdx, targetInt, 0]
            specialSourceTris.append(possibleSpecial)
            specialSourceOpps.append(self.triangleMap[triIdx, targetInt, 1])

            possibleSpecial = self.triangleMap[triIdx, sourceInt, 0]
            specialTargetTris.append(possibleSpecial)
            specialTargetOpps.append(self.triangleMap[triIdx, sourceInt, 1])
            newConstraint.append(self.triangleMap[triIdx, sourceInt, 2])

        sourceTris = np.array(
            [[triIdx, internal] for triIdx, internal in self.vertexMap[source] if (triIdx not in sharedTris)])

        conIds = np.where((self.segments[:,0] == source) | (self.segments[:,1] == source))
        sourceConstraints = self.segments[conIds]
        if len(sourceConstraints) > 0:
            assert(len(sourceConstraints) == 2)
            sharedIdx = conIds[0][0]
            newIdx = conIds[0][1]
            if target not in self.segments[sharedIdx]:
                sharedIdx,newIdx = newIdx,sharedIdx
                if target not in self.segments[sharedIdx]:
                    return "source and target do not share edge!"

            #delete shared segment
            otherSource = self.segments[newIdx][np.where(self.segments[newIdx] != source)][0]
            self.segments[newIdx] = [otherSource,target]

            for triIdx in range(len(self.triangles)):
                for i in range(3):
                    if (self.triangleMap[triIdx,i,2] != noneEdge) and (self.triangleMap[triIdx,i,2] > sharedIdx):
                        self.triangleMap[triIdx,i,2] -= 1
            for i in range(len(newConstraint)):
                if newConstraint[i] > sharedIdx and newConstraint[i] != noneEdge:
                    newConstraint[i] -= 1
            self.segments = np.delete(self.segments,(sharedIdx),axis=0)
            self.segmentType = np.delete(self.segmentType,(sharedIdx),axis=0)

        for triIdx in sharedTris:
            self.unlinkTriangle(triIdx)

        isVeryBad = []
        for triIdx,internal in sourceTris:
            self.unsetBadness(triIdx)
            self.unsetVertexMap(triIdx)
            self.triangles[triIdx,internal] = target
            self.setVertexMap(triIdx)
            a,b,c = self.triangles[triIdx]
            if eg.onWhichSide(Segment(self.point(a),self.point(b)),self.point(c)) == "colinear":
                isVeryBad.append(triIdx)

        for i in range(len(specialSourceTris)):
            specialSourceTri,sourceInternal = specialSourceTris[i],specialSourceOpps[i]
            specialTargetTri,targetInternal = specialTargetTris[i],specialTargetOpps[i]

            if specialSourceTri != outerFace:
                self.triangleMap[specialSourceTri,sourceInternal] = [specialTargetTri,targetInternal,newConstraint[i]]
            if specialTargetTri != outerFace:
                self.triangleMap[specialTargetTri,targetInternal,:2] = [specialSourceTri,sourceInternal]

        while len(isVeryBad) > 0:
            broken = False
            for i in range(len(isVeryBad)):
                zeroVolTri = isVeryBad[i]
                #if it has a neighbour that is NOT verybad, and not on the other side of a constraint, flip the edge
                for j in range(3):
                    if self.triangleMap[zeroVolTri,j,0] not in isVeryBad and self.triangleMap[zeroVolTri,j,2] == noneEdge:
                        if self.triangleMap[zeroVolTri,j,0] != outerFace and self.triangleMap[zeroVolTri,j,0] != noneFace:
                            otherId = self.triangleMap[zeroVolTri,j,0]
                            self.flipEdge(zeroVolTri,j)
                            a, b, c = self.triangles[zeroVolTri]
                            assert(eg.onWhichSide(Segment(self.point(a), self.point(b)), self.point(c)) != "colinear")
                            a, b, c = self.triangles[otherId]
                            assert(eg.onWhichSide(Segment(self.point(a), self.point(b)), self.point(c)) != "colinear")
                            isVeryBad.pop(i)
                            broken = True
                            break
                if broken:
                    break
            if not broken:
                print("oh no")

        for triIdx, internal in sourceTris:
            self.setBadness(triIdx)
            self.setCircumCenter(triIdx)


        for triIdx,_ in sourceTris:
            a,b,c = self.triangles[triIdx]
            if eg.onWhichSide(Segment(self.point(a),self.point(b)),self.point(c)) == "colinear":
                print("oh no")

        return source,sharedTris


    def getEnclosementOfLink(self, vIdxs):
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
                if not inConnections([curAnchor,self.triangles[faceIdx,curIVIdx]]):
                    return "Link straddles Face "+ str(faceIdx)+"!"
                #assert()
                curIVIdx,rightIVIdx = rightIVIdx,curIVIdx

                if self.triangles[faceIdx,curIVIdx] in vIdxs:
                    if not inConnections([curAnchor,self.triangles[faceIdx,curIVIdx]]):
                        return "Link straddles Face "+ str(faceIdx)+"!"
                else:
                    break
            else:
                break

        link = []
        insideFaces = []
        curIdx = self.triangles[curFace,curIVIdx]
        while len(link) == 0 or (link[0] != curIdx or insideFaces[0] != curFace):
            if not (self.triangles[curFace,leftIVIdx] == curAnchor):
                print("wtf")
            if curIdx not in vIdxs:
                link.append(curIdx)
            if len(insideFaces)==0 or (curFace != insideFaces[-1] and curFace != insideFaces[0]):
                insideFaces.append(curFace)

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
                    return "Link straddles Face "+ str(curFace)+"!"

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
        numBad = len(np.where(self.badTris[list(set(insideFaces))] == True)[0])
        return GeometricSubproblem(vIdxs,insideFaces,link,self.exactVerts[vIdxs+link],self.numericVerts[vIdxs+link],self.segments[insideConstraints],self.segments[boundaryConstraints],boundaryConstraintTypes,self.instanceSize,numBad,self.gpaxs)

    def getFaceAsEnclosement(self,triIdx):
        tri = self.triangles[triIdx]
        segmentIds = []
        for _,_,segmentId in self.triangleMap[triIdx]:
            if segmentId != noneEdge:
                segmentIds.append(segmentId)
        return GeometricSubproblem([],[triIdx],tri,self.exactVerts[tri],self.numericVerts[tri],[],self.segments[segmentIds],self.segmentType[segmentIds],self.instanceSize,1 if self.isBad(triIdx) else 0,self.gpaxs)

    def replaceEnclosement(self, gs: GeometricSubproblem, solution):
        if len(gs.insideSteiners) == 0 and len(solution) == 1:
            self.addPoint(solution[0][1])
        elif len(gs.insideSteiners) > 0:

            trianglePool = []
            vertexPool = []

            for vIdx in reversed(sorted(gs.getInsideSteiners())):
                target = None
                segIds = np.where((self.segments[:,0] == vIdx) | (self.segments[:,1] == vIdx))[0]
                assert(len(segIds) == 2 or len(segIds) == 0)
                if len(segIds) > 0:
                    seg = self.segments[segIds[0]]
                    if seg[0] == vIdx:
                        target = seg[1]
                    else:
                        target = seg[0]
                else:
                    triIdx,opp = self.vertexMap[vIdx][0]
                    target = self.triangles[triIdx,(opp+1)%3]

                logging.info("merging "+str(vIdx)+" into "+str(target))
                unlinkedVertex,unlinkedTris = self.internalMergePoints(vIdx,target)
                trianglePool += list(unlinkedTris)
                vertexPool += [unlinkedVertex]
                self.validateTriangleMap(trianglePool)

            self.validateTriangleMap(trianglePool)
            self.validateVertexMap(True)
            #TODO: reuse trianglepool and vertexpool
            logging.info("deleting unlinked triangles: "+str(trianglePool))
            self.deleteUnlinkedTriangles(np.array(trianglePool))
            self.validateTriangleMap()
            self.validateVertexMap()
            for vIdx in reversed(sorted(vertexPool)):
                logging.info("deleting unlinked point: "+str(vIdx))
                self.deletePoint(vIdx)

            self.validateTriangleMap()
            self.validateVertexMap()

            for type,point in solution:
                if type == "vertex":
                    logging.info("vertex-solution: i.e. just remove the inside steinerpoints")
                else:
                    self.addPoint(point)
                    logging.info("inside-solution: the solving point will be added")


            pass
        #1. unlink all steinerpoints in the inside
        #2. reuse unlinked steinerpoints and triangles to add the points from the solution
        pass

class QualityImprover:
    def __init__(self,tri:Triangulation):
        self.tri = tri

    def improve(self):
        keepGoing = True
        lastEdit = "None"
        while keepGoing:
            curEdit = "None"
            bestEval,bestSol,replacer = 0,None,None
            self.tri.plotTriangulation()
            self.tri.validateTriangleMap()
            badTris = np.where(self.tri.badTris == True)[0]
            for triIdx in badTris:
                gp = self.tri.getFaceAsEnclosement(triIdx)
                eval,sol = gp.solve()
                if eval != None and (bestEval == None or eval < bestEval):
                    logging.info("subproblem with eval "+str(eval)+" found with bad triangle generator "+str(triIdx))
                    bestEval = eval
                    bestSol = sol
                    replacer = gp
                    curEdit = "altitude"

            for vIdx in range(self.tri.instanceSize,len(self.tri.exactVerts)):
                #vIdx is steiner
                gp = self.tri.getEnclosementOfLink([vIdx])
                #gp.plotMe()
                eval,sol = gp.solve()
                if eval != None and (bestEval == None or eval < bestEval):
                    logging.info("subproblem with eval "+str(eval)+" found with steiner generator "+str([vIdx]))
                    if vIdx == len(self.tri.exactVerts) - 1 and lastEdit == "circumcenter" and sol[0][0] =="vertex":
                        logging.info("deleting the last added steinerpoint was blocked")
                        continue
                    bestEval = eval
                    bestSol = sol
                    replacer = gp
                    curEdit = "singleSteiner"

            for triIdx in range(len(self.tri.triangles)):
                for i in range(3):
                    if self.tri.triangleMap[triIdx,i,0] > triIdx:
                        if self.tri.edgeTopologyChanged[triIdx,i]:
                            if self.tri.triangles[triIdx,(i+1)%3] >= self.tri.instanceSize and self.tri.triangles[triIdx,(i+2)%3] >= self.tri.instanceSize:
                                #vIdx is steiner
                                gp = self.tri.getEnclosementOfLink([self.tri.triangles[triIdx,(i+1)%3],self.tri.triangles[triIdx,(i+2)%3]])
                                #gp.plotMe()
                                eval,sol = gp.solve()
                                if eval != None and (bestEval == None or eval < bestEval):

                                    logging.info(str(self.tri.triangles[triIdx,(i+1)%3]) + " " + str(self.tri.triangles[triIdx,(i+2)%3]) + " " + lastEdit + " " + str(len(sol)))
                                    if ((self.tri.triangles[triIdx,(i+1)%3] == len(self.tri.exactVerts) - 1) or (self.tri.triangles[triIdx,(i+2)%3] == len(self.tri.exactVerts) - 1)) and lastEdit == "circumcenter" and len(sol) == 1:
                                        logging.info("deleting the last added steinerpoint was blocked")
                                        continue
                                    logging.info("subproblem with eval "+str(eval)+" found with steiner generator "+str([self.tri.triangles[triIdx,(i+1)%3],self.tri.triangles[triIdx,(i+2)%3]]))
                                    bestEval = eval
                                    bestSol = sol
                                    replacer = gp
                                    curEdit = "doubleSteiner"

            if bestEval == 0:
                #we add a circumcenter i suppose
                if len(badTris) == 0:
                    keepGoing = False
                    continue
                added = False
                for triIdx in badTris:
                    logging.info("attempting to add circumcenter of triangle "+str(triIdx))
                    if self.tri.addPoint(Point(*self.tri.circumCenters[triIdx])):
                        lastEdit = "circumcenter"
                        added = True
                        break
                if not added:
                    for triIdx in badTris:
                        logging.info("attempting to add centroid of triangle " + str(triIdx))
                        avg = Point(FieldNumber(0), FieldNumber(0))
                        for v in range(3):
                            avg += self.tri.point(self.tri.triangles[triIdx,v])
                        avg = avg.scale(FieldNumber(1) / FieldNumber(3))
                        if self.tri.addPoint(avg):
                            lastEdit = "centroid"
                            added = True
                            break
                if not added:
                    logging.error("No solution computable...")
                    return None
            else:
                #replacer.plotMe()
                lastEdit = curEdit
                logging.info("replacing identified subproblem")
                self.tri.replaceEnclosement(replacer,bestSol)
                self.tri.validateTriangleMap()
        return self.tri.solutionParse()