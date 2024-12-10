import copy
import itertools
import time

import numpy as np
import matplotlib.ticker
import matplotlib.pyplot as plt
from cgshop2025_pyutils import Cgshop2025Solution, Cgshop2025Instance
from cgshop2025_pyutils.geometry import FieldNumber, Point, Segment

import exact_geometry as eg
from constants import *
from GeometricSubproblem import GeometricSubproblem, StarSolver#
from scipy.spatial import KDTree

import triangle as tr  # https://rufat.be/triangle/

import logging


class Triangulation:
    def __init__(self, instance: Cgshop2025Instance, withValidate=False, seed=0, axs=None):
        if axs == None:
            axs = [None,None,None,None,None]

        #_, gpaxs = plt.subplots(1,1)
        self.histoaxs = axs[1]
        self.histoaxtwin = axs[2]
        self.internalaxs = axs[3]
        self.gpaxs = axs[4] #None  # gpaxs

        self.withValidate = withValidate
        self.seed = seed
        self.plotTime = 0.005
        self.axs = axs[0]
        self.plotWithIds = self.withValidate

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
            # for x, y in zip(xs, ys):
            exactVerts.append(Point(x, y))
            numericVerts.append([x, y])
        self.instance_uid = instance.instance_uid

        Ain = tr.triangulate(convert(instance), 'p')
        segments = Ain['segments']
        triangles = Ain['triangles']

        self.triangles = np.array(triangles,dtype=int)
        self.segments = np.array(segments,dtype=int)
        self.triangleMap = None
        self.exactVerts = np.array(exactVerts, dtype=Point)
        self.numericVerts = np.array(numericVerts)
        self.isValidVertex = np.array([True for _ in self.numericVerts], dtype=bool)

        self.inputPointTree = KDTree(numericVerts)

        self.vertexMap = [[] for _ in range(len(self.exactVerts))]
        for triIdx in range(len(self.triangles)):
            for i in range(3):
                vIdx = self.triangles[triIdx, i]
                self.vertexMap[vIdx].append([triIdx, i])

        ####
        # construct all maps
        ####

        ####
        # vertex maps
        ####

        self.pointTopologyChanged = [True for _ in self.exactVerts]

        ####
        # triangle maps
        ####

        # self.badTris = [idx for idx in range(len(self.triangles)) if self.isBad(idx)]
        self.badTris = np.array([False for _ in range(len(self.triangles))])
        self.isValidTriangle = np.array([True for _ in range(len(self.triangles))])
        for idx in range(len(self.triangles)):
            self.badTris[idx] = self.isBad(idx)

        self.triangleMap = [[] for _ in self.triangles]

        # temporary edgeSet for triangleNeighbourhoodMap
        fullMap = [[[] for _ in self.exactVerts] for _ in self.exactVerts]
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
                        triMap.append([j, opp, self.getSegmentIdx(np.array(edge))])
                        added = True
                if not added:
                    triMap.append([outerFace, noneIntervalVertex, self.getSegmentIdx(np.array(edge))])
            self.triangleMap[i] = triMap
        self.triangleMap = np.array(self.triangleMap)

        self.edgeTopologyChanged = np.full(self.triangles.shape, True)
        self.triangleChanged = np.array([True for _ in range(len(self.triangles))])

        self.circumCenters = np.array([eg.circumcenter(*self.exactVerts[tri]) for tri in self.triangles])
        self.circumRadiiSqr = np.array(
            [eg.distsq(self.point(self.triangles[i][0]), Point(*self.circumCenters[i])) for i in
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

        self.segmentType = [False for _ in self.segments]
        for triIdx in range(len(self.triangles)):
            for i in range(3):
                if (edgeId := self.triangleMap[triIdx, i, 2]) != noneEdge:
                    if self.triangleMap[triIdx, i, 0] == outerFace:
                        self.segmentType[edgeId] = True
        self.segmentType = np.array(self.segmentType)

        #self.geometricProblems = []
        self.geometricFaceProblems = []
        self.geometricLinkProblems = []
        self.geometricCircleProblems = []
        self.geometricSegmentProblems = []
        self.updateGeometricProblems()
        self.watch = 0

    ####
    # visualization and parsing
    ####

    def plotTriangulation(self):
        if self.axs == None:
            return
        self.axs.clear()
        self.axs.set_facecolor('lightgray')
        SC = self.getNumSteiner()
        name = ""
        badCount = 0
        if SC > 0:
            name += " [SC:" + str(SC) + "]"

        # axs.scatter([p[0] for p in self.numericVerts],[p[1] for p in self.numericVerts],marker=".")

        nonsuperseeded = self.getNonSuperseededBadTris()

        for triIdx in self.validTriIdxs():
            tri = self.triangles[triIdx]
            cords = self.numericVerts[tri]
            cords = np.concatenate((cords, [cords[0]]))
            for i in range(3):
                if ((self.triangles[triIdx][(i + 1) % 3] >= self.instanceSize) and (
                        self.triangles[triIdx][(i + 2) % 3] >= self.instanceSize) and
                        (self.edgeTopologyChanged[triIdx, i] or (
                                self.triangleMap[triIdx, i, 0] != outerFace and self.edgeTopologyChanged[
                            self.triangleMap[triIdx, i, 0], self.triangleMap[triIdx, i, 1]]))):
                    self.axs.plot(*cords[[(i + 1) % 3, (i + 2) % 3]].T, color='black', linewidth=1, zorder=98,
                                  linestyle="dotted")
                else:
                    self.axs.plot(*cords[[(i + 1) % 3, (i + 2) % 3]].T, color='black', linewidth=1, zorder=98)
            if self.isBad(triIdx):
                badCount += 1
                if triIdx in nonsuperseeded:
                    t = plt.Polygon(self.numericVerts[tri], color='mediumorchid')
                    self.axs.add_patch(t)
                else:
                    t = plt.Polygon(self.numericVerts[tri], color='orchid')
                    self.axs.add_patch(t)

            else:
                t = plt.Polygon(self.numericVerts[tri], color='palegoldenrod')
                self.axs.add_patch(t)
            midX = (self.numericVerts[tri[0]][0] + self.numericVerts[tri[1]][0] + self.numericVerts[tri[2]][0]) / 3
            midY = (self.numericVerts[tri[0]][1] + self.numericVerts[tri[1]][1] + self.numericVerts[tri[2]][1]) / 3
            if self.plotWithIds:
                if self.triangleChanged[triIdx]:
                    self.axs.text(midX, midY, str(triIdx), ha="center", va="center", fontsize=6, color="red")
                else:
                    self.axs.text(midX, midY, str(triIdx), ha="center", va="center", fontsize=6, color="black")
        name += " (>90°: " + str(badCount) + ")"

        for edgeId in range(len(self.segments)):
            e = self.segments[edgeId]
            t = self.segmentType[edgeId]
            color = 'blue' if t else 'forestgreen'
            self.axs.plot(*(self.numericVerts[e].T), color=color, linewidth=2, zorder=99)
        minSize = 20
        maxSize = 40
        sizes = np.array(self.pointTopologyChanged, dtype=int) * (maxSize - minSize) + minSize
        self.axs.scatter(*self.numericVerts[:self.instanceSize].T, s=sizes[:self.instanceSize], color='black',
                         zorder=100)
        for i in self.validVertIdxs():
            if i >= self.instanceSize:
                self.axs.scatter(*self.numericVerts[i].T, s=sizes[i], color='red', zorder=100)
        if self.plotWithIds:
            for idx in self.validVertIdxs():
                if idx < self.instanceSize:
                    self.axs.text(self.numericVerts[idx, 0], self.numericVerts[idx, 1], str(idx), ha="center",
                                  va="center", fontsize=6, color="white", zorder=100)
                else:
                    self.axs.text(self.numericVerts[idx, 0], self.numericVerts[idx, 1], str(idx), ha="center",
                                  va="center", fontsize=6, color="black", zorder=100)

        self.axs.set_aspect('equal')
        self.axs.title.set_text(name)

        #for vIdx in self.validVertIdxs():
        #    _,segs,_ = self.getEdgeClippedConstrainedVoronoiFaceAsSegmentSet(vIdx)
        #    for seg in segs:
        #        p = eg.numericPoint(seg.source())
        #        q = eg.numericPoint(seg.target())
        #        self.axs.plot([p[0], q[0]], [p[1], q[1]], color="blue",zorder=1000000)
        #        self.axs.scatter([p[0], q[0]], [p[1], q[1]], color="blue",marker="*",s=minSize,zorder=1000000)

        plt.draw()
        plt.pause(self.plotTime)

    def solutionParse(self):
        inneredges = []
        idxs = self.validVertIdxs()
        for tri in self.validTris():
            edges = [[np.where(idxs == tri[i])[0][0] for i in internal] for internal in [[0,1],[1,2],[2,0]]]
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
        for i in self.validVertIdxs():
            if i < self.instanceSize:
                continue
            sx.append(self.point(i).x().exact())
            sy.append(self.point(i).y().exact())
        return Cgshop2025Solution(instance_uid=self.instance_uid, steiner_points_x=sx, steiner_points_y=sy,
                                  edges=inneredges)

    ####
    # validation
    ####

    def validateCircumcenters(self):
        if not self.withValidate:
            return
        for triIdx in self.validTriIdxs():
            actualCC = eg.circumcenter(self.point(self.triangles[triIdx, 0]), self.point(self.triangles[triIdx, 1]),
                                       self.point(self.triangles[triIdx, 2]))
            actualDist = eg.distsq(actualCC, self.point(self.triangles[triIdx, 0]))
            if np.all(self.circumCenters[triIdx] != actualCC):
                assert False
            if self.circumRadiiSqr[triIdx] != actualDist:
                assert False

    def validateTriangleMap(self):
        if not self.withValidate:
            return
        for triIdx in self.validTriIdxs():
            for i in range(3):
                other, opp, constraint = self.triangleMap[triIdx, i]
                actualSegIdx = self.getSegmentIdx(
                    [self.triangles[triIdx, (i + 1) % 3], self.triangles[triIdx, (i + 2) % 3]])
                assert (constraint == actualSegIdx)

                oppEdge = [self.triangles[triIdx, (i + 1) % 3], self.triangles[triIdx, (i + 2) % 3]]

                assert other != noneFace
                if other != outerFace:
                    for vIdx in oppEdge:
                        assert (vIdx in self.triangles[other])
                    assert (self.triangleMap[other, opp, 0] == triIdx)

    def validateVertexMap(self):
        if not self.withValidate:
            return
        for i in self.validVertIdxs():
            for triIdx, selfInternal in self.vertexMap[i]:
                if not (self.triangles[triIdx, selfInternal] == i):
                    print("wtf")
        for triIdx in self.validTriIdxs():
            for i in range(3):
                assert (triIdx in [tI for tI, _ in self.vertexMap[self.triangles[triIdx, i]]])

    ####
    # low level getters and setters
    ####

    # valid/invalid iterators

    def validTriIdxs(self):
        return np.where(self.isValidTriangle)[0]

    def invalidTriIdxs(self):
        locs = np.where(~self.isValidTriangle)[0]
        return locs

    def validTris(self):
        return self.triangles[self.validTriIdxs()]

    def validTriMaps(self):
        return self.triangleMap[self.validTriIdxs()]

    def validVertIdxs(self):
        return np.where(self.isValidVertex)[0]

    def invalidVertIdxs(self):
        locs = np.where(~self.isValidVertex)[0]
        return locs

    def validExactVerts(self):
        return self.exactVerts[self.validVertIdxs()]

    def validNumericVerts(self):
        return self.numericVerts[self.validVertIdxs()]

    def getNumSteiner(self):
        return len(self.validVertIdxs()) - self.instanceSize

    # point/id getters

    def point(self, i: int):
        if not (i in self.validVertIdxs()):
            assert (False)
        return Point(*self.exactVerts[i])

    def circumcenter(self,triIdx:int):
        return Point(*self.circumCenters[triIdx])

    def isBad(self, triIdx):
        return eg.isBadTriangle(*self.exactVerts[self.triangles[triIdx]])

    def getSegmentIdx(self, querySeg):
        locs = np.concatenate(
            (np.argwhere(np.all((self.segments == querySeg), -1)),
             np.argwhere(np.all((self.segments == querySeg[::-1]), -1))))
        if len(locs) == 1:
            return locs[0, 0]
        else:
            return noneEdge

    def getInternalIdx(self, triIdx, vIdx):
        locs = np.argwhere(self.triangles[triIdx] == vIdx)
        if len(locs) == 1:
            return locs[0][0]
        else:
            return noneIntervalVertex

    def getOppositeInternalIdx(self, triIdx, vIdx):
        iVIdx = self.getInternalIdx(triIdx, vIdx)
        if iVIdx == noneIntervalVertex:
            return noneIntervalVertex
        else:
            return self.triangleMap[triIdx, iVIdx, 1]

    def trianglesOnEdge(self, pIdx, qIdx):
        return np.array([triIdx for triIdx, _ in self.vertexMap[pIdx] if qIdx in self.triangles[triIdx]])

    # very low level modifiers

    def unsetVertexMap(self, triIdx):
        for vIdx in self.triangles[triIdx]:
            self.vertexMap[vIdx] = [vm for vm in self.vertexMap[vIdx] if vm[0] != triIdx]
            # self.vertexMap[idx].remove(triIdx)

    def setCircumCenter(self, triIdx):
        self.circumCenters[triIdx] = eg.circumcenter(*self.exactVerts[self.triangles[triIdx]])
        self.circumRadiiSqr[triIdx] = eg.distsq(self.point(self.triangles[triIdx][0]),
                                                Point(*self.circumCenters[triIdx]))

    def unsetBadness(self, triIdx):
        self.badTris[triIdx] = False

    def setBadness(self, triIdx):
        self.badTris[triIdx] = self.isBad(triIdx)
        # dists = self.closestDist[self.badTris]
        # args = dists.argsort()
        # self.badTris = list(np.array(self.badTris)[args[::-1]])
        # if self.seed != None:
        #    np.random.seed(self.seed)
        #    np.random.shuffle(self.badTris)

    def rebaseTriangleState(self):
        self.triangleChanged = np.full(self.triangleChanged.shape, False)

    def setValidTri(self, triIdx):
        self.triangleChanged[triIdx] = True
        self.isValidTriangle[triIdx] = True

    def unsetValidTri(self, triIdx):
        self.triangleChanged[triIdx] = True
        self.isValidTriangle[triIdx] = False

    def setValidVert(self, vIdx):
        self.isValidVertex[vIdx] = True

    def unsetValidVert(self, vIdx):
        self.isValidVertex[vIdx] = False

    def setVertexMap(self, triIdx):
        for iVIdx in range(3):
            if self.triangles[triIdx,iVIdx] >= len(self.vertexMap):
                print("oh no")
            self.vertexMap[self.triangles[triIdx, iVIdx]].append([triIdx, iVIdx])
        for vIdx in self.triangles[triIdx]:
            self.pointTopologyChanged[vIdx] = True
        for vIdx in self.triangles[triIdx]:
            for otherIdx, oppIVIdx in self.vertexMap[vIdx]:
                self.updateEdgeTopology(otherIdx)

    def getNonSuperseededBadTris(self):
        #return np.where(self.badTris)[0]
        result = []
        for badTri in np.where(self.badTris)[0]:
            isNotSuperseeded = True
            bA = eg.badAngle(self.point(self.triangles[badTri,0]),self.point(self.triangles[badTri,1]),self.point(self.triangles[badTri,2]))
            assert bA != -1
            internalQueries = [(bA+1)%3,(bA+2)%3]
            for iq in internalQueries:
                nId = self.triangleMap[badTri,iq,0]
                if nId != outerFace and self.badTris[nId]:
                    if self.triangleMap[badTri,iq,1] == eg.badAngle(self.point(self.triangles[nId,0]),self.point(self.triangles[nId,1]),self.point(self.triangles[nId,2])):
                        isNotSuperseeded = False
                        break
            if isNotSuperseeded:
                result.append(badTri)
        if len(result) == 0:
            result = np.where(self.badTris)[0]
        return np.array(result)


    # slightly less low level modifiers

    def updateEdgeTopology(self, triIdx):
        for otherIdx, oppVIdx, constraint in self.triangleMap[triIdx]:
            if otherIdx != outerFace and otherIdx != noneFace:
                for i in range(3):
                    self.edgeTopologyChanged[otherIdx, i] = True

    def getCoordinateQuality(self):
        maxQuality = 0
        for vIdx in self.validVertIdxs():
            if vIdx < self.instanceSize:
                continue
            for coord in self.exactVerts[vIdx]:
                if (q:= len(coord.exact())) > maxQuality:
                    maxQuality = q
        return maxQuality

    def plotCoordinateQuality(self):
        if self.histoaxs == None:
            return
        self.histoaxs.clear()
        qualities = []
        for vIdx in self.validVertIdxs():
            if vIdx < self.instanceSize:
                continue
            for coord in self.exactVerts[vIdx]:
                qualities.append(len(coord.exact()))

        hist, bins = np.histogram(qualities, bins=20)
        logbins = np.logspace(min(1,np.log10(bins[0])), max(2,np.log10(bins[-1])), len(bins))
        self.histoaxs.hist(qualities, bins=logbins)
        self.histoaxs.set_xscale('log')
        self.histoaxs.set_yscale('log')
        self.histoaxs.get_xaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.0f'))
        self.histoaxs.get_yaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.0f'))
        self.histoaxs.set_xlim((min(10,self.histoaxs.get_xlim()[0]),max(101,self.histoaxs.get_xlim()[1])))
        self.histoaxs.set_ylim((1,max(11,self.histoaxs.get_ylim()[1])))
        #self.histoaxs.get_xlim
        #self.histoaxs.hist(qualities,bins=min(30,max(qualities)))

    ####
    # internal medium level modifiers with internal logic. HERE BE DRAGONS!
    ####

    def splitSegment(self, segIdx, vIdx):
        seg = [self.segments[segIdx][0], self.segments[segIdx][1]]
        self.segments[segIdx] = [seg[0], vIdx]
        self.segments = np.vstack((self.segments, [vIdx, seg[1]]))
        self.segmentType = np.hstack((self.segmentType, self.segmentType[segIdx]))

    def createPoint(self, p: Point, preferedId = None):
        invalids = self.invalidVertIdxs()
        if len(invalids) == 0:
            if preferedId != None:
                assert(False)
            self.exactVerts = np.vstack((self.exactVerts, [p]))
            self.numericVerts = np.vstack((self.numericVerts, [float(p.x()), float(p.y())]))
            self.vertexMap.append([])
            self.pointTopologyChanged = np.hstack((self.pointTopologyChanged, True))
            self.isValidVertex = np.hstack((self.isValidVertex, True))

            for triIdx in self.validTriIdxs():
                dist = eg.distsq(Point(*self.circumCenters[triIdx]), p)
                if dist < self.closestDist[triIdx]:
                    self.closestToCC[triIdx] = len(self.exactVerts) - 1
                    self.closestDist[triIdx] = dist
            return len(self.exactVerts) - 1
        else:
            myIdx = invalids[0]
            if preferedId != None:
                if preferedId in invalids:
                    myIdx = preferedId
                else:
                    assert(False)
            self.exactVerts[myIdx] = p
            self.numericVerts[myIdx] = [float(p.x()), float(p.y())]
            self.pointTopologyChanged[myIdx] = True
            self.setValidVert(myIdx)

            for triIdx in self.validTriIdxs():
                dist = eg.distsq(Point(*self.circumCenters[triIdx]), p)
                if dist < self.closestDist[triIdx]:
                    self.closestToCC[triIdx] = myIdx
                    self.closestDist[triIdx] = dist
            return myIdx

    def createTriangles(self, tris, triMaps, doNotUse=[]):
        myIdxs = []
        for i in range(len(tris)):
            invalids = self.invalidTriIdxs()
            myIdx = None
            for idx in invalids:
                if idx not in doNotUse:
                    myIdx = idx
                    break
            if myIdx is None:
                tri = tris[i]
                triMap = triMaps[i]

                self.triangles = np.vstack((self.triangles, [tri]))
                self.triangleMap = np.vstack((self.triangleMap, [triMap]))
                self.edgeTopologyChanged = np.vstack((self.edgeTopologyChanged, [True, True, True]))
                self.isValidTriangle = np.hstack((self.isValidTriangle, True))
                self.triangleChanged = np.hstack((self.triangleChanged, True))
                self.closestToCC = np.hstack((self.closestToCC, [noneVertex]))
                self.closestDist = np.hstack((self.closestDist, [FieldNumber(0)]))
                self.badTris = np.hstack((self.badTris, [False]))
                self.circumCenters = np.vstack((self.circumCenters, [Point(FieldNumber(0), FieldNumber(0))]))
                self.circumRadiiSqr = np.hstack((self.circumRadiiSqr, [FieldNumber(0)]))

                myIdxs.append(len(self.triangles) - 1)
            else:
                tri = tris[i]
                triMap = triMaps[i]

                self.triangles[myIdx] = tri
                self.triangleMap[myIdx] = triMap
                self.edgeTopologyChanged[myIdx] = [True, True, True]
                self.triangleChanged[myIdx] = True
                self.setValidTri(myIdx)

                myIdxs.append(myIdx)

        for myIdx in myIdxs:

            self.setVertexMap(myIdx)
            self.setCircumCenter(myIdx)
            self.setBadness(myIdx)

            for iVIdx in range(3):
                neighbour, oppIVIdx, constraint = self.triangleMap[myIdx, iVIdx]
                if neighbour != outerFace and neighbour != noneFace and neighbour < len(self.triangles):
                    self.triangleMap[neighbour, oppIVIdx, :2] = [myIdx, iVIdx]
        return myIdxs

    def copyOfCombinatorialState(self):
        state = dict()
        state["triangles"] = np.copy(self.triangles)
        state["triangleMap"] = np.copy(self.triangleMap)
        state["edgeTopologyChanged"] = np.copy(self.edgeTopologyChanged)
        state["isValidTriangle"] = np.copy(self.isValidTriangle)
        state["triangleChanged"] = np.copy(self.triangleChanged)
        state["closestToCC"] = np.copy(self.closestToCC)
        state["closestDist"] = np.copy(self.closestDist)
        state["badTris"] = np.copy(self.badTris)
        state["circumCenters"] = np.copy(self.circumCenters)
        state["circumRadiiSqr"] = np.copy(self.circumRadiiSqr)
        state["vertexMap"] = copy.deepcopy(self.vertexMap)
        state["segments"] = np.copy(self.segments)
        state["isValidVertex"] = np.copy(self.isValidVertex)
        state["segmentType"] = np.copy(self.segmentType)
        state["pointTopologyChanged"] = np.copy(self.pointTopologyChanged)
        return state

    def applyCombinatorialState(self,state):
        self.triangles = np.copy(state["triangles"])
        self.triangleMap = np.copy(state["triangleMap"])
        self.edgeTopologyChanged = np.copy(state["edgeTopologyChanged"])
        self.isValidTriangle = np.copy(state["isValidTriangle"])
        self.triangleChanged = np.copy(state["triangleChanged"])
        self.closestToCC = np.copy(state["closestToCC"])
        self.closestDist = np.copy(state["closestDist"])
        self.badTris = np.copy(state["badTris"])
        self.circumCenters = np.copy(state["circumCenters"])
        self.circumRadiiSqr = np.copy(state["circumRadiiSqr"])
        self.vertexMap = copy.deepcopy(state["vertexMap"])
        self.segments = np.copy(state["segments"])
        self.isValidVertex = np.copy(state["isValidVertex"])
        self.segmentType = np.copy(state["segmentType"])
        self.pointTopologyChanged = np.copy(state["pointTopologyChanged"])


        # need to repopulate empty rows to stay consistent. maybe in future change combinatorial state as well...
        #hopefuly we only need to update vertex based rows...
        for idx in range(len(self.vertexMap),len(self.exactVerts)):
            #create the combinatorial side of an invalid vertex
            self.vertexMap.append([])
            self.pointTopologyChanged = np.hstack((self.pointTopologyChanged, False))
            self.isValidVertex = np.hstack((self.isValidVertex, False))

    def setInvalidTriangle(self, triIdx, tri, triMap):
        self.triangles[triIdx] = tri
        self.triangleMap[triIdx] = triMap

        self.setVertexMap(triIdx)
        self.setCircumCenter(triIdx)

        for iVIdx in range(3):
            neighbour, oppIVIdx, constraint = self.triangleMap[triIdx, iVIdx]
            if neighbour != outerFace and neighbour != noneFace and neighbour < len(self.triangles):
                self.triangleMap[neighbour, oppIVIdx, :2] = [triIdx, iVIdx]

        self.setCircumCenter(triIdx)
        self.setBadness(triIdx)
        self.setValidTri(triIdx)

    def unlinkTriangle(self, triIdx):
        self.unsetBadness(triIdx)
        self.unsetValidTri(triIdx)

        tri = self.triangles[triIdx]

        # unlink from neighbouring triangles
        for neighbour, oppIVIdx, constraint in self.triangleMap[triIdx]:
            if neighbour != outerFace and neighbour != noneFace:
                self.triangleMap[neighbour, oppIVIdx, :2] = [noneFace, noneIntervalVertex]

        # unset map
        for i in range(3):
            self.triangleMap[triIdx, i] = [noneFace, noneIntervalVertex, noneEdge]

        # unset vertexMap:
        self.unsetVertexMap(triIdx)

        self.triangles[triIdx] = [noneVertex, noneVertex, noneVertex]

    ####
    # high level modifiers, that are reasonably safe to use
    ####

    def updateGeometricFaceProblems(self):
        pass

    def updateGeometricSegmentProblems(self):        # remove all geometric problems, whose face set has experienced a change
        for gpiIdx in reversed(range(len(self.geometricSegmentProblems))):
            hasToBeRemoved = False
            for triIdx in self.geometricSegmentProblems[gpiIdx].triIdxs:
                if self.triangleChanged[triIdx]:
                    hasToBeRemoved = True
                    break
            if hasToBeRemoved:
                self.geometricSegmentProblems.pop(gpiIdx)

        topoDisks = set()
        for gp in self.geometricCircleProblems:
            topoDisks.add(tuple(list(sorted(gp.triIdxs))))

        nonsuperseeded = self.getNonSuperseededBadTris()

        #TODO: this generator is not exhaustive i think...
        for triIdx in self.validTriIdxs():
            for internal in range(3):
                if self.triangleMap[triIdx,internal,2] != noneEdge and self.edgeTopologyChanged[triIdx,internal]:
                    #prevent doublecounting
                    if self.triangleMap[triIdx,internal,0] > triIdx:
                        segTopoDisks = self.getAllSegmentDisks(triIdx, internal)
                        for disk in segTopoDisks:
                            if disk not in topoDisks:
                                gp = self.getGeometricSubproblemFromTopoDisk(disk)
                                if gp is not None:
                                    gp.gpType = "segmentTopoDisk"
                                    self.geometricCircleProblems.append(gp)
                                topoDisks.add(disk)
        logging.info("Number of topological segment disk problems: " + str(len(topoDisks)))

                        #lDisks = self.getAllTruncatedCirclesIntersecting(triIdx)
                        #rDisk = None
                        #if self.triangleMap[triIdx,internal,0] != outerFace:
                        #    rDisk = self.getAllTruncatedCirclesIntersecting(self.triangleMap[triIdx,internal,0])

        # add all changed bad triangles
        #for triIdx in np.where(self.triangleChanged)[0]:
        #    if self.isValidTriangle[triIdx] and self.badTris[triIdx]:
        #        triTopoDisks = self.getAllTriangleDisks(triIdx)
        #        for disk in triTopoDisks:
        #            if disk not in topoDisks:
        #                gp = self.getGeometricSubproblemFromTopoDisk(disk)
        #                if gp is not None:
        #                    self.geometricCircleProblems.append(gp)
        #                topoDisks.add(disk)
        #    logging.info("Number of topological disk problems: " + str(len(topoDisks)))

        #self.rebaseTriangleState()

    def updateGeometricLinkProblems(self):
        # remove all geometric problems, whose face set has experienced a change
        for gpiIdx in reversed(range(len(self.geometricLinkProblems))):
            hasToBeRemoved = False
            for triIdx in self.geometricLinkProblems[gpiIdx].triIdxs:
                if self.triangleChanged[triIdx]:
                    hasToBeRemoved = True
                    break
            if hasToBeRemoved:
                self.geometricLinkProblems.pop(gpiIdx)

        # add all faces around a single steinerpoint if it changed
        for vIdx in self.validVertIdxs():
            if self.pointTopologyChanged[vIdx] and vIdx >= self.instanceSize:
                if (gp := self.getEnclosementOfLink([vIdx])) is not None:
                    self.geometricLinkProblems.append(gp)
            self.pointTopologyChanged[vIdx] = False

        for triIdx in self.validTriIdxs():
            for i in range(3):
                if self.edgeTopologyChanged[triIdx, i]:
                    if self.triangles[triIdx, (i + 1) % 3] >= self.instanceSize and self.triangles[
                        triIdx, (i + 2) % 3] >= self.instanceSize:
                        # prevent doublecounting
                        if self.triangleMap[triIdx, i, 0] > triIdx:
                            if (gp := self.getEnclosementOfLink([self.triangles[triIdx, (i + 1) % 3], self.triangles[triIdx, (i + 2) % 3]])) is not None:
                                self.geometricLinkProblems.append(gp)
                self.edgeTopologyChanged[triIdx, i] = False
        #np.random.shuffle(self.geometricProblems)

    def updateGeometricCircleProblems(self):
        # remove all geometric problems, whose face set has experienced a change
        for gpiIdx in reversed(range(len(self.geometricCircleProblems))):
            hasToBeRemoved = False
            for triIdx in self.geometricCircleProblems[gpiIdx].triIdxs:
                if self.triangleChanged[triIdx]:
                    hasToBeRemoved = True
                    break
            if hasToBeRemoved:
                self.geometricCircleProblems.pop(gpiIdx)

        topoDisks = set()
        for gp in self.geometricCircleProblems:
            topoDisks.add(tuple(list(sorted(gp.triIdxs))))

        # add all changed bad triangles
        # add all changed bad triangles
        #nonSuperseeded = self.getNonSuperseededBadTris()
        for triIdx in np.where(self.triangleChanged)[0]:
            if self.isValidTriangle[triIdx] and self.badTris[triIdx]:# and triIdx in nonSuperseeded:
                triTopoDisks = self.getAllTriangleDisks(triIdx)
                for disk in triTopoDisks:
                    if disk not in topoDisks:
                        gp = self.getGeometricSubproblemFromTopoDisk(disk)
                        if gp is not None:
                            self.geometricCircleProblems.append(gp)
                        topoDisks.add(disk)
            #logging.info("Number of topological disk problems: " + str(len(topoDisks)))

        self.rebaseTriangleState()


    def updateGeometricProblems(self,fastAndGreedy=False):
        self.updateGeometricFaceProblems()
        #self.updateGeometricSegmentProblems()
        self.updateGeometricLinkProblems()
        if not fastAndGreedy:
            self.updateGeometricCircleProblems()

    def geometricSubproblemIterator(self):
        return itertools.chain(self.geometricCircleProblems,self.geometricSegmentProblems,self.geometricFaceProblems,self.geometricLinkProblems)

    def flipEdge(self, triIdx, iVIdx):
        triA = self.triangles[triIdx]
        triAIdx = triIdx
        triAIVIdx = iVIdx
        assert (self.triangleMap[triAIdx, triAIVIdx, 2] == noneEdge)
        triBIdx = self.triangleMap[triAIdx, triAIVIdx, 0]
        triB = self.triangles[triBIdx]
        triBIVIdx = self.triangleMap[triAIdx, triAIVIdx, 1]
        leftAIVIdx = (triAIVIdx + 1) % 3
        rightAIVIdx = (triAIVIdx + 2) % 3
        leftBIVIdx = (triBIVIdx + 1) % 3 if triA[leftAIVIdx] == triB[(triBIVIdx + 1) % 3] else (triBIVIdx + 2) % 3
        rightBIVIdx = (triBIVIdx + 1) % 3 if triA[rightAIVIdx] == triB[(triBIVIdx + 1) % 3] else (triBIVIdx + 2) % 3

        newA = [triA[triAIVIdx], triA[leftAIVIdx], triB[triBIVIdx]]
        newAMap = [
            list(self.triangleMap[triBIdx, rightBIVIdx]),
            [triBIdx, 2, noneEdge],
            list(self.triangleMap[triAIdx, rightAIVIdx])]

        newB = [triB[triBIVIdx], triA[triAIVIdx], triB[rightBIVIdx]]
        newBMap = [
            list(self.triangleMap[triAIdx, leftAIVIdx]),
            list(self.triangleMap[triBIdx, leftBIVIdx]),
            [triAIdx, 1, noneEdge]]

        self.unlinkTriangle(triAIdx)
        self.unlinkTriangle(triBIdx)

        self.setInvalidTriangle(triAIdx, newA, newAMap)
        self.setInvalidTriangle(triBIdx, newB, newBMap)

        # self.plotTriangulation()

        # self.validateTriangleMap()

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
            # j = self.voronoiEdges[i][jIdx]
            # jMask = self.constrainedMask[i][jIdx]
            # oppositeIndexInJ = None
            j, oppositeIndexInJ, jMask = self.triangleMap[i, jIdx]
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
                if onlyOn:
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

        # self.validate()
        # they are stored as [triangleindex, inducing index]
        badEdgesInTriangleLand = []
        bannedEdges = []
        if modifiedTriangles is None:
            for i in self.validTriIdxs():
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
                    otherTriIdx, oppositeIt, _ = self.triangleMap[triIdx, iVIdx]
                    badEdgesInTriangleLand[it].append([otherTriIdx, oppositeIt])
                else:
                    assert (False)

            for jIdx in range(3):
                _addEdgeToStack(i, jIdx)
            for iIdx in range(3):
                _addEdgeToStack(j, iIdx)
        # self.validate()

    def uninformedGetHitTriIdxs(self,p:Point):
        hitTriIdxs = []
        grazedTriIdxs = []
        for triIdx in self.validTriIdxs():
            hit,colinearIndex = self.pointHitsTriangle(triIdx, p)
            if hit == "inside":
                hitTriIdxs.append([triIdx])
            elif hit == "on":
                grazedTriIdxs.append([triIdx,colinearIndex])
        return hitTriIdxs,grazedTriIdxs

    def informedGetHitTriIdxs(self,p:Point):
        numericalPoint = [float(p.x()),float(p.y())]
        _,nn = self.inputPointTree.query(numericalPoint)
        curTris = [triIdx for triIdx,_ in self.vertexMap[nn]]
        #breadth first search until we hit a triangle the contains p.
        touchedTris = set(curTris)
        while len(curTris) > 0:
            curTri = curTris.pop(0)
            touchedTris.add(curTri)
            #expand:
            for i in range(3):
                nIdx = self.triangleMap[curTri,i,0]
                if nIdx == outerFace:
                    continue
                if nIdx in touchedTris:
                    continue
                curTris.append(nIdx)
                touchedTris.add(nIdx)

            hit, colinearIndex = self.pointHitsTriangle(curTri, p)
            if hit == "inside":
                return [[curTri]],[]
            elif hit == "on":
                if colinearIndex is None:
                    return [],[]
                if self.triangleMap[curTri,colinearIndex[0][0],0] == outerFace:
                    return [],[[curTri,colinearIndex]]
                elif colinearIndex != None:
                    return [],[[curTri,colinearIndex],[self.triangleMap[curTri,colinearIndex[0][0],0],[[self.triangleMap[curTri,colinearIndex[0][0],1]]]]]
                #grazedTriIdxs.append([triIdx, colinearIndex])


    def insertPoint(self,p:Point,hitTriIdxs,grazedTriIdxs,preferedId = None):
        if len(hitTriIdxs) == 1:
            # inside
            assert (len(grazedTriIdxs) == 0)
            hitTriIdx = hitTriIdxs[0][0]
            hitTri = self.triangles[hitTriIdx]

            newPointIdx = self.createPoint(p,preferedId)
            newIds = list(self.invalidTriIdxs()) + [len(self.triangles), len(self.triangles) + 1]
            newLeftIdx = newIds[0]
            newRightIdx = newIds[1]

            newLeft = [hitTri[0], newPointIdx, hitTri[2]]
            newLeftMap = [
                [hitTriIdx, 1, noneEdge],
                list(self.triangleMap[hitTriIdx, 1]),
                [newRightIdx, 1, noneEdge]]

            newRight = [hitTri[0], hitTri[1], newPointIdx]
            newRightMap = [
                [hitTriIdx, 2, noneEdge],
                [newLeftIdx, 2, noneEdge],
                list(self.triangleMap[hitTriIdx, 2])]

            newSelf = [newPointIdx, hitTri[1], hitTri[2]]
            newSelfMap = [
                list(self.triangleMap[hitTriIdx, 0]),
                [newLeftIdx, 0, noneEdge],
                [newRightIdx, 0, noneEdge]]

            self.unlinkTriangle(hitTriIdx)

            self.createTriangles([newLeft, newRight], [newLeftMap, newRightMap], doNotUse=[hitTriIdx])
            self.setInvalidTriangle(hitTriIdx, newSelf, newSelfMap)

            self.validateCircumcenters()
            self.validateTriangleMap()
            self.validateVertexMap()

            #for triIdx in self.validTriIdxs():
            #    for i in range(3):
            #        if not (self.triangleMap[triIdx, i, 0] != noneFace):
            #            print("och man")
            #self.ensureDelauney()
            # self.ensureDelauney(None)

            # self.plotTriangulation()

            return True,newPointIdx,[hitTriIdx, newLeftIdx, newRightIdx]
        elif len(grazedTriIdxs) == 0:
            # outside
            return False,None,[]
        elif len(grazedTriIdxs) == 1:
            # boundary
            assert (len(grazedTriIdxs[0][1]) == 1)
            grazedIdx = grazedTriIdxs[0][0]
            grazed = self.triangles[grazedIdx]
            grazedIVIdx = grazedTriIdxs[0][1][0][0]

            newPointIdx = self.createPoint(p,preferedId)
            newIds = list(self.invalidTriIdxs()) + [len(self.triangles)]
            newTriIdx = newIds[0]

            # this MUST be a boundary!
            assert (self.triangleMap[grazedIdx, grazedIVIdx, 2] != noneEdge)
            assert (self.triangleMap[grazedIdx, grazedIVIdx, 0] == outerFace)

            segIdx = self.triangleMap[grazedIdx, grazedIVIdx, 2]
            self.splitSegment(segIdx, newPointIdx)

            newTri = [grazed[grazedIVIdx], grazed[(grazedIVIdx + 1) % 3], newPointIdx]
            newTriMap = [
                [outerFace, noneIntervalVertex, self.getSegmentIdx([grazed[(grazedIVIdx + 1) % 3], newPointIdx])],
                [grazedIdx, 2, noneEdge],
                list(self.triangleMap[grazedIdx, (grazedIVIdx + 2) % 3])]

            newSelf = [grazed[grazedIVIdx], newPointIdx, grazed[(grazedIVIdx + 2) % 3]]
            newSelfMap = [
                [outerFace, noneIntervalVertex, self.getSegmentIdx([grazed[(grazedIVIdx + 2) % 3], newPointIdx])],
                list(self.triangleMap[grazedIdx, (grazedIVIdx + 1) % 3]),
                [newTriIdx, 1, noneEdge]]

            self.unlinkTriangle(grazedIdx)

            self.createTriangles([newTri], [newTriMap], doNotUse=[grazedIdx])
            self.setInvalidTriangle(grazedIdx, newSelf, newSelfMap)

            self.validateCircumcenters()
            self.validateTriangleMap()
            self.validateVertexMap()

            #self.ensureDelauney()
            # self.ensureDelauney(None)

            # self.plotTriangulation()

            return True,newPointIdx,[grazedIdx, newTriIdx]
        elif len(grazedTriIdxs) == 2:
            # constraint or unlucky
            assert (len(grazedTriIdxs[0][1]) == 1)
            assert (len(grazedTriIdxs[1][1]) == 1)

            grazedAIdx = grazedTriIdxs[0][0]
            grazedA = self.triangles[grazedAIdx]
            grazedAIVIdx = grazedTriIdxs[0][1][0][0]
            grazedBIdx = grazedTriIdxs[1][0]
            grazedB = self.triangles[grazedBIdx]
            grazedBIVIdx = grazedTriIdxs[1][1][0][0]

            newPointIdx = self.createPoint(p,preferedId)
            newIds = list(self.invalidTriIdxs()) + [len(self.triangles), len(self.triangles) + 1]
            newTriByAIdx = newIds[0]
            newTriByBIdx = newIds[1]

            segIdx = self.triangleMap[grazedAIdx, grazedAIVIdx, 2]
            if segIdx != noneEdge:
                self.splitSegment(segIdx, newPointIdx)

            diff = False
            if grazedA[(grazedAIVIdx + 1) % 3] != grazedB[(grazedBIVIdx + 1) % 3]:
                diff = True

            newTriByA = [grazedA[grazedAIVIdx], grazedA[(grazedAIVIdx + 1) % 3], newPointIdx]
            newTriByAMap = [
                [grazedBIdx if diff else newTriByBIdx, 0,
                 self.getSegmentIdx([grazedA[(grazedAIVIdx + 1) % 3], newPointIdx])],
                [grazedAIdx, 2, noneEdge],
                list(self.triangleMap[grazedAIdx, (grazedAIVIdx + 2) % 3])]

            newASelf = [grazedA[grazedAIVIdx], newPointIdx, grazedA[(grazedAIVIdx + 2) % 3]]
            newASelfMap = [
                [newTriByBIdx if diff else grazedBIdx, 0,
                 self.getSegmentIdx([newPointIdx, grazedA[(grazedAIVIdx + 2) % 3]])],
                list(self.triangleMap[grazedAIdx, (grazedAIVIdx + 1) % 3]),
                [newTriByAIdx, 1, noneEdge]]

            newTriByB = [grazedB[grazedBIVIdx], grazedB[(grazedBIVIdx + 1) % 3], newPointIdx]
            newTriByBMap = [
                [grazedAIdx if diff else newTriByAIdx, 0,
                 self.getSegmentIdx([grazedB[(grazedBIVIdx + 1) % 3], newPointIdx])],
                [grazedBIdx, 2, noneEdge],
                list(self.triangleMap[grazedBIdx, (grazedBIVIdx + 2) % 3])]

            newBSelf = [grazedB[grazedBIVIdx], newPointIdx, grazedB[(grazedBIVIdx + 2) % 3]]
            newBSelfMap = [
                [newTriByAIdx if diff else grazedAIdx, 0,
                 self.getSegmentIdx([newPointIdx, grazedB[(grazedBIVIdx + 2) % 3]])],
                list(self.triangleMap[grazedBIdx, (grazedBIVIdx + 1) % 3]),
                [newTriByBIdx, 1, noneEdge]]

            self.unlinkTriangle(grazedAIdx)
            self.unlinkTriangle(grazedBIdx)

            self.createTriangles([newTriByA, newTriByB], [newTriByAMap, newTriByBMap],
                                 doNotUse=[grazedAIdx, grazedBIdx])
            self.setInvalidTriangle(grazedAIdx, newASelf, newASelfMap)
            self.setInvalidTriangle(grazedBIdx, newBSelf, newBSelfMap)

            self.validateCircumcenters()
            self.validateTriangleMap()
            self.validateVertexMap()

            #self.ensureDelauney()
            # self.ensureDelauney(None)

            # self.plotTriangulation()

            return True,newPointIdx,[grazedAIdx, grazedBIdx, newTriByBIdx, newTriByAIdx]
        else:
            # vertex
            return False,None,[]


    def addPoint(self, p: Point, preferedId = None):

        start = time.time()
        #representation quality guard
        if len(p.x().exact()) > 50000 or len(p.y().exact()) > 50000:
            logging.error(str(self.seed) + " DANGEROUS LEVELS OF REPRESENTATION QUALITY FOR NEW POINT!!!")
            return False

        hitTriIdxs,grazedTriIdxs = self.informedGetHitTriIdxs(p)
        added,newIdx,touchedTriangles = self.insertPoint(p,hitTriIdxs, grazedTriIdxs, preferedId)
        self.ensureDelauney(touchedTriangles)
        self.watch += time.time() - start
        return added,newIdx

    def addPoints(self,ps):
        touchedTris = []
        for p in ps:
            hits,grazeds = self.informedGetHitTriIdxs(p)
            added,newIdx,touchedTriangles = self.insertPoint(p,hits, grazeds)
            if added:
                touchedTris += touchedTriangles
        self.ensureDelauney(list(set(touchedTris)))
        #return added,newIdx

    def internalMergePoints(self, source, target):

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

        conIds = np.where((self.segments[:, 0] == source) | (self.segments[:, 1] == source))
        sourceConstraints = self.segments[conIds]
        if len(sourceConstraints) > 0:
            assert (len(sourceConstraints) == 2)
            sharedIdx = conIds[0][0]
            newIdx = conIds[0][1]
            if target not in self.segments[sharedIdx]:
                sharedIdx, newIdx = newIdx, sharedIdx
                if target not in self.segments[sharedIdx]:
                    return "source and target do not share edge!"

            # delete shared segment
            otherSource = self.segments[newIdx][np.where(self.segments[newIdx] != source)][0]
            self.segments[newIdx] = [otherSource, target]

            for triIdx in self.validTriIdxs():
                for i in range(3):
                    if (self.triangleMap[triIdx, i, 2] != noneEdge) and (self.triangleMap[triIdx, i, 2] > sharedIdx):
                        self.triangleMap[triIdx, i, 2] -= 1
            for i in range(len(newConstraint)):
                if newConstraint[i] > sharedIdx and newConstraint[i] != noneEdge:
                    newConstraint[i] -= 1
            self.segments = np.delete(self.segments, sharedIdx, axis=0)
            self.segmentType = np.delete(self.segmentType, sharedIdx, axis=0)

        for triIdx in sharedTris:
            self.unlinkTriangle(triIdx)

        self.unsetValidVert(source)

        isVeryBad = []
        for triIdx, internal in sourceTris:
            self.unsetBadness(triIdx)
            self.unsetVertexMap(triIdx)
            self.triangles[triIdx, internal] = target
            self.setVertexMap(triIdx)
            a, b, c = self.triangles[triIdx]
            if eg.onWhichSide(Segment(self.point(a), self.point(b)), self.point(c)) == "colinear":
                isVeryBad.append(triIdx)

        for i in range(len(specialSourceTris)):
            specialSourceTri, sourceInternal = specialSourceTris[i], specialSourceOpps[i]
            specialTargetTri, targetInternal = specialTargetTris[i], specialTargetOpps[i]

            if specialSourceTri != outerFace:
                self.triangleMap[specialSourceTri, sourceInternal] = [specialTargetTri, targetInternal,
                                                                      newConstraint[i]]
            if specialTargetTri != outerFace:
                self.triangleMap[specialTargetTri, targetInternal, :2] = [specialSourceTri, sourceInternal]

        while len(isVeryBad) > 0:
            broken = False
            for i in range(len(isVeryBad)):
                zeroVolTri = isVeryBad[i]
                # if it has a neighbour that is NOT verybad, and not on the other side of a constraint, flip the edge
                for j in range(3):
                    # j MUST lie opposite the longest edge, otherwise the flip results in a bad triangle
                    if eg.distsq(self.point(self.triangles[zeroVolTri, (j + 1) % 3]),
                                 self.point(self.triangles[zeroVolTri, (j + 2) % 3])) > eg.distsq(
                            self.point(self.triangles[zeroVolTri, (j + 0) % 3]),
                            self.point(self.triangles[zeroVolTri, (j + 1) % 3])) and eg.distsq(
                            self.point(self.triangles[zeroVolTri, (j + 1) % 3]),
                            self.point(self.triangles[zeroVolTri, (j + 2) % 3])) > eg.distsq(
                            self.point(self.triangles[zeroVolTri, (j + 0) % 3]),
                            self.point(self.triangles[zeroVolTri, (j + 2) % 3])):

                        if self.triangleMap[zeroVolTri, j, 0] not in isVeryBad and self.triangleMap[
                            zeroVolTri, j, 2] == noneEdge:
                            if self.triangleMap[zeroVolTri, j, 0] != outerFace and self.triangleMap[
                                zeroVolTri, j, 0] != noneFace:
                                otherId = self.triangleMap[zeroVolTri, j, 0]
                                self.flipEdge(zeroVolTri, j)
                                a, b, c = self.triangles[zeroVolTri]
                                assert (eg.onWhichSide(Segment(self.point(a), self.point(b)),
                                                       self.point(c)) != "colinear")
                                a, b, c = self.triangles[otherId]
                                assert (eg.onWhichSide(Segment(self.point(a), self.point(b)),
                                                       self.point(c)) != "colinear")
                                isVeryBad.pop(i)
                                broken = True
                                break
                    # else:
                    #    if self.triangleMap[zeroVolTri,j,0] not in isVeryBad and self.triangleMap[zeroVolTri,j,2] == noneEdge:
                    #        if self.triangleMap[zeroVolTri,j,0] != outerFace and self.triangleMap[zeroVolTri,j,0] != noneFace:
                    #            logging.info("dodged a bullet there")

                if broken:
                    break
            if not broken:
                print("oh no")

        for triIdx, internal in sourceTris:
            self.setBadness(triIdx)
            self.setCircumCenter(triIdx)

        for triIdx, _ in sourceTris:
            a, b, c = self.triangles[triIdx]
            if eg.onWhichSide(Segment(self.point(a), self.point(b)), self.point(c)) == "colinear":
                print("oh no")

        return source, sharedTris

    def internalGetEnclosementOfLink(self, vIdxs):
        connections = [[vIdxs[i], vIdxs[i + 1]] for i in range(len(vIdxs) - 1)]

        def inConnections(edge):
            if edge in connections or edge[::-1] in connections:
                return True
            return False

        assert (np.all(self.isValidVertex[vIdxs]))

        anchorInternalIdx = 0
        anchorAdvancer = 1
        curAnchor = vIdxs[anchorInternalIdx]
        curFace = None
        leftIVIdx = None
        curIVIdx = None
        rightIVIdx = None
        for faceIdx, internal in self.vertexMap[curAnchor]:
            curFace = faceIdx
            leftIVIdx = internal
            curIVIdx = (leftIVIdx + 1) % 3
            rightIVIdx = (leftIVIdx + 2) % 3
            if self.triangles[faceIdx, curIVIdx] in vIdxs:
                if not inConnections([curAnchor, self.triangles[faceIdx, curIVIdx]]):
                    return "Link straddles Face " + str(faceIdx) + "!"
                # assert()
                curIVIdx, rightIVIdx = rightIVIdx, curIVIdx

                if self.triangles[faceIdx, curIVIdx] in vIdxs:
                    if not inConnections([curAnchor, self.triangles[faceIdx, curIVIdx]]):
                        return "Link straddles Face " + str(faceIdx) + "!"
                else:
                    break
            else:
                break

        link = []
        insideFaces = []
        curIdx = self.triangles[curFace, curIVIdx]
        while len(link) == 0 or (link[0] != curIdx or insideFaces[0] != curFace):
            if not (self.triangles[curFace, leftIVIdx] == curAnchor):
                print("wtf")
            if curIdx not in vIdxs:
                link.append(curIdx)
            if len(insideFaces) == 0 or (curFace != insideFaces[-1] and curFace != insideFaces[0]):
                insideFaces.append(curFace)

            if self.triangles[curFace, rightIVIdx] in vIdxs:
                if inConnections([self.triangles[curFace, leftIVIdx], self.triangles[curFace, rightIVIdx]]):

                    # to stay consistent, we need to remove the last added vertex. it will be added lateron again
                    assert (link[-1] == curIdx)
                    link = link[:-1]

                    # advance anchor and advance advancor if necessary
                    anchorInternalIdx += anchorAdvancer
                    if anchorInternalIdx == len(vIdxs) - 1:
                        anchorAdvancer = -1
                    curAnchor = vIdxs[anchorInternalIdx]
                    # rotate
                    curIVIdx, leftIVIdx, rightIVIdx = leftIVIdx, rightIVIdx, curIVIdx
                    curIdx = self.triangles[curFace, curIVIdx]
                else:
                    return "Link straddles Face " + str(curFace) + "!"

            # advance
            elif self.triangleMap[curFace, curIVIdx, 0] == outerFace:

                link.append(self.triangles[curFace, rightIVIdx])

                keepGoing = True
                while keepGoing:
                    link.append(self.triangles[curFace, leftIVIdx])

                    for faceIdx, internalIdx in self.vertexMap[curAnchor]:
                        if faceIdx == curFace:
                            continue
                        leftInternal = (internalIdx + 1) % 3
                        rightInternal = (internalIdx + 2) % 3
                        if self.triangleMap[faceIdx, leftInternal, 0] != outerFace:
                            leftInternal = (internalIdx + 2) % 3
                            rightInternal = (internalIdx + 1) % 3
                        if self.triangleMap[faceIdx, leftInternal, 0] != outerFace:
                            # not the other face
                            continue
                        if self.triangles[faceIdx, rightInternal] in vIdxs:
                            if inConnections(
                                    [self.triangles[faceIdx, internalIdx], self.triangles[faceIdx, rightInternal]]):
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
                                assert False
                            break
                        else:
                            keepGoing = False
                            curFace = faceIdx
                            curIVIdx = rightInternal
                            leftIVIdx = internalIdx
                            rightIVIdx = leftInternal
                            # if self.triangles[curFace, leftIVIdx] != curAnchor:
                            #    leftIVIdx, rightIVIdx = rightIVIdx, leftIVIdx
                            curIdx = self.triangles[curFace, curIVIdx]
                            # safeguard, because we will stop next iteration but the last face has not been added yet
                            if len(insideFaces) == 0 or (curFace != insideFaces[-1] and curFace != insideFaces[0]):
                                insideFaces.append(curFace)
                            break
            else:
                nextFace = self.triangleMap[curFace, curIVIdx, 0]
                for i in range(3):
                    if self.triangles[nextFace, i] == self.triangles[curFace, rightIVIdx]:
                        curIVIdx = i
                        curFace = nextFace
                        curIdx = self.triangles[curFace, curIVIdx]
                        break
                leftIVIdx = (curIVIdx + 1) % 3
                rightIVIdx = (curIVIdx + 2) % 3
                if self.triangles[curFace, leftIVIdx] != curAnchor:
                    leftIVIdx, rightIVIdx = rightIVIdx, leftIVIdx

                if len(insideFaces) == 0 or (curFace != insideFaces[-1] and curFace != insideFaces[0]):
                    insideFaces.append(curFace)

        constraints = []
        insideConstraints = []
        boundaryConstraints = []
        for triIdx in insideFaces:
            for i in range(3):
                if self.triangleMap[triIdx, i, 2] != noneEdge:
                    constraints.append(self.triangleMap[triIdx, i, 2])
        for con in constraints:
            if con in boundaryConstraints:
                boundaryConstraints.remove(con)
                insideConstraints.append(con)
            else:
                boundaryConstraints.append(con)

        return vIdxs,insideFaces,link,insideConstraints, boundaryConstraints

    def getEnclosementOfLink(self,vIdxs):
        vIdxs,insideFaces,link, insideConstraints,boundaryConstraints = self.internalGetEnclosementOfLink(vIdxs)
        boundaryConstraintTypes = self.segmentType[boundaryConstraints]
        numBad = len(np.where(self.badTris[list(set(insideFaces))] == True)[0])
        #clean up link and move their inbetween vertex to the inside. if the homotopy type is non-trivial, we return None instead

        deleteList = None
        moveInside = []
        while deleteList is None or len(deleteList) > 0:
            deleteList = []
            for i in range(len(link)):
                if link[i] == link[(i + 2) % len(link)]:
                    deleteList.append(i)
                    deleteList.append((i + 1) % len(link))
                    if (newInside := link[(i + 1) % len(link)]) not in moveInside and newInside not in vIdxs:
                        moveInside.append(link[(i + 1) % len(link)])
            link = np.delete(link, deleteList)
        vIdxs = np.hstack((vIdxs, np.array(moveInside, dtype=int)))
        numBoundaryDroppers=0
        for face in insideFaces:
            if self.badTris[face]:
                i = eg.badAngle(self.point(self.triangles[face,0]),self.point(self.triangles[face,1]),self.point(self.triangles[face,2]))
                if self.triangleMap[face,i,0] == outerFace:
                    numBoundaryDroppers += 1
                    numBad -= 1

        if len(link) == len(list(set(link))):
            return GeometricSubproblem(vIdxs, insideFaces, link, self.exactVerts[list(vIdxs) + list(link)],
                                   self.numericVerts[list(vIdxs) + list(link)], self.segments[insideConstraints],
                                   self.segments[boundaryConstraints], boundaryConstraintTypes, self.instanceSize,
                                   numBad,numBoundaryDroppers,None,"enclosement",None, self.gpaxs)
        else:
            logging.debug("Enclosement with seed "+str(vIdxs)+" produced non-trivial homotopytype...")
            return None

    def getFaceAsEnclosement(self, triIdx):
        tri = self.triangles[triIdx]
        segmentIds = []
        for _, _, segmentId in self.triangleMap[triIdx]:
            if segmentId != noneEdge:
                segmentIds.append(segmentId)
        insideFaces = [triIdx]
        numBad = 1 if self.isBad(triIdx) else 0
        numBoundaryDroppers=0
        for face in insideFaces:
            if self.badTris[face]:
                i = eg.badAngle(self.point(self.triangles[face,0]),self.point(self.triangles[face,1]),self.point(self.triangles[face,2]))
                if self.triangleMap[face,i,0] == outerFace:
                    numBoundaryDroppers += 1
                    numBad -= 1
        return GeometricSubproblem([], [triIdx], tri, self.exactVerts[tri], self.numericVerts[tri], [],
                                   self.segments[segmentIds], self.segmentType[segmentIds], self.instanceSize,
                                   numBad,numBoundaryDroppers,None,"face",None, self.gpaxs)

    def removePoint(self,vIdx):
        #safely unlinks
        touchedTriangles = []
        for triIdx,_ in self.vertexMap[vIdx]:
            touchedTriangles.append(triIdx)

        #first ensure that the surrounding link is in convex position.
        keepGoing = True
        while keepGoing:
            keepGoing = False
            for triIdx,internal in self.vertexMap[vIdx]:
                for i in range(1,3):
                    triangleInternalRoot = (internal + i)%3
                    if self.triangleMap[triIdx, triangleInternalRoot, 0] != outerFace and self.triangleMap[triIdx, triangleInternalRoot, 2] == noneEdge:
                        #the edge can be flipped. now we check if we are in convex position
                        oppInternal = self.triangleMap[triIdx, triangleInternalRoot, 1]
                        oppV = self.triangles[self.triangleMap[triIdx, triangleInternalRoot, 0],oppInternal]

                        sideA = eg.onWhichSide(Segment(self.point(self.triangles[triIdx,triangleInternalRoot]),self.point(self.triangles[triIdx,(triangleInternalRoot+1)%3])),self.point(oppV))
                        sideB = eg.onWhichSide(Segment(self.point(self.triangles[triIdx,triangleInternalRoot]),self.point(self.triangles[triIdx,(triangleInternalRoot+2)%3])),self.point(oppV))

                        if sideA == "colinear" or sideB == "colinear":
                            continue
                        elif sideA != sideB:
                            #we can flip!
                            self.flipEdge(triIdx,triangleInternalRoot)
                            keepGoing = True
                            break
                if keepGoing:
                    break

        #the surrounding link should now be in convex position (up to colinearity)
        target = None
        segIds = np.where((self.segments[:, 0] == vIdx) | (self.segments[:, 1] == vIdx))[0]
        assert (len(segIds) == 2 or len(segIds) == 0)
        if len(segIds) > 0:
            seg = self.segments[segIds[0]]
            if seg[0] == vIdx:
                target = seg[1]
            else:
                target = seg[0]
        else:
            if len(self.vertexMap[vIdx]) == 3:
                triIdx, opp = self.vertexMap[vIdx][0]
                target = self.triangles[triIdx, (opp + 1) % 3]
            elif len(self.vertexMap[vIdx]) == 4:
                #there are two points that are colinear, and one of them is not on the convex hull. find this point!
                #this is SUUUUPER rare
                link = []
                for triIdx, internal in self.vertexMap[vIdx]:
                    for i in range(1,3):
                        link.append(self.triangles[triIdx, (internal + i)%3])
                link = list(set(link))
                assert(len(link) == 4)

                #for the VERY special case, that they are even in convex position (no three are even colinear), then any point works
                target = link[0]

                #if we are not in convex position (or three are colinear) this will select the right target
                for i in range(4):
                    qP = self.point(link[i])
                    tA = self.point(link[(i+1)%4])
                    tB = self.point(link[(i+2)%4])
                    tC = self.point(link[(i+3)%4])
                    tri = [tA,tB,tC]
                    sides = []
                    for j in range(3):
                        sides.append(eg.onWhichSide(Segment(tri[j],tri[(j+1)%3]),qP))
                    if "left" in sides and "right" in sides:
                        continue
                    else:
                        target = link[i]
                        break
            else:
                logging.error(str(self.seed) + ": there is a point, that is adjacent to more than 4 faces, and no two of them can be flipped...")
                assert False

        logging.debug("merging " + str(vIdx) + " into " + str(target))
        self.internalMergePoints(vIdx, target)
        self.ensureDelauney([tT for tT in touchedTriangles if self.isValidTriangle[tT]])



    def replaceEnclosement(self, gs: GeometricSubproblem, solution):
        addedPoints = []
        addedIds = []
        removedPoints = []
        removedIds = []
        if len(gs.insideSteiners) == 0 and len(solution) == 1:
            added,newId = self.addPoint(solution[0])
            if added:
                addedPoints.append(solution[0])
                addedIds.append(newId)
                return True,TriangulationAction(addedPoints,addedIds,removedPoints,removedIds)
            else:
                return False,None
        elif len(gs.insideSteiners) > 0:

            for vIdx in reversed(sorted(gs.getInsideSteiners())):
                removedPoints.append(self.point(vIdx))
                removedIds.append(vIdx)
                self.removePoint(vIdx)

            #trianglePool = []
            #vertexPool = []

            #for vIdx in reversed(sorted(gs.getInsideSteiners())):
            #    target = None
            #    segIds = np.where((self.segments[:, 0] == vIdx) | (self.segments[:, 1] == vIdx))[0]
            #    assert (len(segIds) == 2 or len(segIds) == 0)
            #    if len(segIds) > 0:
            #        seg = self.segments[segIds[0]]
            #        if seg[0] == vIdx:
            #            target = seg[1]
            #        else:
            #            target = seg[0]
            #    else:
            #        triIdx, opp = self.vertexMap[vIdx][0]
            #        target = self.triangles[triIdx, (opp + 1) % 3]

            #    logging.debug("merging " + str(vIdx) + " into " + str(target))
            #    unlinkedVertex, unlinkedTris = self.internalMergePoints(vIdx, target)
            #    #trianglePool += list(unlinkedTris)
            #    #vertexPool += [unlinkedVertex]

            self.validateCircumcenters()
            self.validateTriangleMap()
            self.validateVertexMap()

            for point in solution:
                added,newId = self.addPoint(point)
                if added:
                    addedPoints.append(point)
                    addedIds.append(newId)
            if len(addedPoints) > 0 or len(removedPoints) > 0:
                return True,TriangulationAction(addedPoints,addedIds,removedPoints,removedIds)
            else:
                return False,None

    def getVoronoiFacetEdgeSet(self,vIdx):
        edgeList = []

        def inEdgeList(edge):
            for e in edgeList:
                if np.all(edge == e):
                    return True
            return False

        for triIdx,internal in self.vertexMap[vIdx]:
            for i in range(1,3):
                testInternal = (internal + i)%3
                if self.triangleMap[triIdx,testInternal,0] > triIdx:
                    if not inEdgeList([[triIdx,testInternal],self.triangleMap[triIdx,testInternal,:-1]]):
                        edgeList.append([[triIdx,testInternal],self.triangleMap[triIdx,testInternal,:-1]])

        voronoiSegments = []
        for aAll,bAll in edgeList:
            a, aInt = aAll
            b, bInt = bAll
            if b != outerFace:
                voronoiSegments.append(Segment(self.circumcenter(a),self.circumcenter(b)))
            else:
                diff = self.point(self.triangles[a,(aInt+1)%3]) - self.point(self.triangles[a,(aInt+2)%3])
                orth = Point(FieldNumber(0)-diff.y(),diff.x())
                if eg.dot(orth,self.point(self.triangles[a,(aInt+1)%3])-self.circumcenter(a)) < eg.zero:
                    orth = orth.scale(FieldNumber(-1))
                target = self.circumcenter(a)+orth
                if (inter:=eg.supportingRayIntersectSegment(Segment(self.circumcenter(a),target),Segment(self.point(self.triangles[a,(aInt+1)%3]) , self.point(self.triangles[a,(aInt+2)%3]) ))) is not None:
                    target = inter
                else:
                    logging.error(str(self.seed) + " intersection of voronoiray with its boundary was None... that aint good")
                    assert(False)
                voronoiSegments.append(Segment(self.circumcenter(a),target))
        return edgeList,voronoiSegments

    def getVoronoiSegmentsIntersectingCircumCircle(self, triIdx):
        #TODO: i really just gotta bite the bullet, and implement rays and segments as different types...

        cc = self.circumcenter(triIdx)
        cr = self.circumRadiiSqr[triIdx]

        voronoiFaceStack = list(self.triangles[triIdx])
        hasBeenHandled = []
        edgeList = []
        voronoiSegments = []
        edgeRounders = []

        def inEdgeList(edge):
            for e in edgeList:
                if np.all(np.array(edge) == e):
                    return True
            return False

        def inSegmentList(seg):
            for vseg in voronoiSegments:
                if vseg.source() == seg.source() and vseg.target() == seg.target():
                    return True
                elif vseg.source() == seg.target() and vseg.target() == seg.source():
                    return True
            return False

        while len(voronoiFaceStack) > 0:
            voronoiFace = voronoiFaceStack.pop()
            hasBeenHandled.append(voronoiFace)

            faceEdgeList,faceEdgeSegments,faceEdgeRounders = self.getClippedConstrainedVoronoiFaceAsSegmentSet(voronoiFace,triIdx)
            for edge,segment,rounder in zip(faceEdgeList,faceEdgeSegments,faceEdgeRounders):
                if inSegmentList(segment):
                    continue
                if eg.segmentIntersectsCircle(cc,cr,segment):
                    edgeList.append(edge)
                    voronoiSegments.append(segment)
                    edgeRounders.append(rounder)
                    faceId = edge[0][0]
                    internal = edge[0][1]
                    for i in range(1,3):
                        newInternal = (internal + i)%3
                        if self.triangles[faceId,newInternal] == faceId:
                            continue
                        newVoronoiFace = self.triangles[faceId,newInternal]
                        if not (newVoronoiFace in hasBeenHandled or newVoronoiFace in voronoiFaceStack):
                            voronoiFaceStack.append(newVoronoiFace)
        return edgeList,voronoiSegments,edgeRounders

    def getSegmentsIntersectingCircumCircle(self,triIdx):
        cc = self.circumcenter(triIdx)
        cr = self.circumRadiiSqr[triIdx]

        triangleStack = [triIdx]
        hasBeenHandled = []
        edgeList = []
        segments = []

        def inEdgeList(edge):
            for e in edgeList:
                if np.all(np.array(edge) == e) or np.all(np.array(edge)[::-1,:] == e):
                    return True
            return False

        while len(triangleStack) > 0:
            triIdx = triangleStack.pop()
            hasBeenHandled.append(triIdx)

            for i in range(3):

                #if (newFace := self.triangleMap[triIdx,i,0]) != outerFace:
                segment = Segment(self.point(self.triangles[triIdx,(i+1)%3]),self.point(self.triangles[triIdx,(i+2)%3]))
                if eg.segmentIntersectsCircle(cc,cr,segment):

                    if self.triangleMap[triIdx,i,0] != outerFace:
                        if not (self.triangleMap[triIdx,i,0] in hasBeenHandled or self.triangleMap[triIdx,i,0] in triangleStack):
                            triangleStack.append(self.triangleMap[triIdx,i,0])

                    if self.triangleMap[triIdx,i,2] != noneEdge:
                        if not inEdgeList([[triIdx,i],self.triangleMap[triIdx,i,:-1]]):
                            edgeList.append([[triIdx,i],self.triangleMap[triIdx,i,:-1]])
                            segments.append(segment)
                            p = eg.numericPoint(segment.source())
                            q = eg.numericPoint(segment.target())
                            #if self.internalaxs != None:
                            #    self.internalaxs.plot([p[0],q[0]],[p[1],q[1]],color="black")



        return edgeList, segments

    def getUnclippedConstrainedVoronoiFace(self,vIdx):
        #get voronoi edges correctly oriented
        voronoiEdgeSet = []
        for triIdx,internal in self.vertexMap[vIdx]:
            for i in range(1,3):
                actualInternal = (internal + i)%3
                if triIdx < self.triangleMap[triIdx,actualInternal,0]:
                    voronoiEdgeSet.append([[triIdx,actualInternal],self.triangleMap[triIdx,actualInternal,:-1]])

        #now orient the edges properly...
        voronoiFace = [voronoiEdgeSet[0]]
        voronoiEdgeSet = voronoiEdgeSet[1:]
        curFace = voronoiFace[0][1][0]
        while len(voronoiEdgeSet) > 0:
            for i in range(len(voronoiEdgeSet)):
                vEdge = voronoiEdgeSet[i]
                if vEdge[0][0] == curFace:
                    curFace = vEdge[1][0]
                    voronoiFace.append(vEdge)
                    voronoiEdgeSet.pop(i)
                    break
                elif vEdge[1][0] == curFace:
                    curFace = vEdge[0][0]
                    voronoiFace.append(vEdge)
                    voronoiEdgeSet.pop(i)
                    break
                else:
                    continue
        #now segment into subfacets
        constrainedSubfacets = []
        while len(voronoiFace) > 0:
            nextEdge = voronoiFace.pop(0)
            if len(constrainedSubfacets) == 0:
                constrainedSubfacets.append([nextEdge])
            else:
                constrainedSubfacets[-1] =constrainedSubfacets[-1] + [nextEdge]

            if self.triangleMap[nextEdge[0][0],nextEdge[0][1],2] != noneEdge:
                constrainedSubfacets.append([nextEdge])
        if len(constrainedSubfacets) > 1:
            firstSubfacet = constrainedSubfacets[0]
            constrainedSubfacets = constrainedSubfacets[1:]
            constrainedSubfacets[-1] = constrainedSubfacets[-1] + firstSubfacet

        for i in reversed(range(len(constrainedSubfacets))):
            conFacet = constrainedSubfacets[i]
            if len(conFacet) == 2:
                if outerFace == conFacet[0][0][0] or outerFace == conFacet[0][1][0]:
                    if min(conFacet[0][0][0],conFacet[0][1][0]) != min(conFacet[1][0][0],conFacet[1][1][0]):
                        constrainedSubfacets.pop(i)

        #now there is no subfacet for outside. We may still have a corner vertex with a single adjoining face... this will be weird. maybe it just works out

        #these are not yet clipped to any segment. maybe include this functionality...
        edges,segments,roundingObjects = [],[],[]
        for subfacet in constrainedSubfacets:
            firstEdge = subfacet[0]

            def traverseFacet(first,last):
                sfedges,sfsegments,sfrounders = [],[],[]
                if first == last:
                    assert(False)
                step = 1 if last > first else -1

                es = self.point(self.triangles[subfacet[first][0][0], (subfacet[first][0][1] + 1) % 3])
                et = self.point(self.triangles[subfacet[first][0][0], (subfacet[first][0][1] + 2) % 3])
                curP = (es + et).scale(eg.onehalf)
                curF, curInt = subfacet[first][0]
                if curF != subfacet[first+step][0][0] and curF != subfacet[first+step][1][0]:
                    curF, curInt = subfacet[first][1]
                oldEdge = subfacet[first]
                sfrounders.append(Segment(es,et))

                les = self.point(self.triangles[subfacet[last][0][0], (subfacet[last][0][1] + 1) % 3])
                let = self.point(self.triangles[subfacet[last][0][0], (subfacet[last][0][1] + 2) % 3])
                lastEdge = Segment(les, let)

                for curEdgeIdx in range(first+step, last+step, step):
                    curEdge = subfacet[curEdgeIdx]

                    nextF, nextInt = None, None
                    if curF == curEdge[0][0]:
                        nextF, nextInt = curEdge[1]
                    else:
                        nextF, nextInt = curEdge[0]

                    nextP = self.circumcenter(curF)
                    nextSegment = Segment(curP, nextP)

                    # this ends everything
                    if (inter := eg.innerIntersect(lastEdge.source(), lastEdge.target(), nextSegment.source(),
                                                   nextSegment.target())) is not None:
                        sfedges.append(oldEdge)
                        sfsegments.append(Segment(curP, inter))
                        sfrounders.append(lastEdge)

                        break

                    elif nextF == outerFace:
                        sfedges.append(oldEdge)
                        sfsegments.append(nextSegment)
                        sfrounders.append(None)
                        es = self.point(self.triangles[curEdge[0][0], (curEdge[0][1] + 1) % 3])
                        et = self.point(self.triangles[curEdge[0][0], (curEdge[0][1] + 2) % 3])
                        nextP = (es + et).scale(eg.onehalf)
                        sfedges.append(curEdge)
                        sfsegments.append(Segment(nextSegment.target(), nextP))
                        sfrounders.append(Segment(es,et))
                        break
                    elif curEdgeIdx == last:
                        sfedges.append(oldEdge)
                        sfsegments.append(nextSegment)
                        sfrounders.append(None)
                        es = self.point(self.triangles[nextF, (nextInt + 1) % 3])
                        et = self.point(self.triangles[nextF, (nextInt + 2) % 3])
                        nextP = (es + et).scale(eg.onehalf)
                        sfedges.append(curEdge)
                        sfsegments.append(Segment(nextSegment.target(), nextP))
                        sfrounders.append(Segment(es,et))
                        break


                    else:
                        sfedges.append(oldEdge)
                        sfsegments.append(nextSegment)
                        sfrounders.append(None)
                        curP = nextP
                        oldEdge = curEdge
                        curF = nextF
                        curInt = nextInt
                return sfedges,sfsegments,[ [sfrounders[i],sfrounders[i+1]] for i in range(len(sfrounders)-1) ]



            if self.triangleMap[firstEdge[0][0],firstEdge[0][1],2] == noneEdge:
                #unconstrained point, and we just return the normal voronoi region. We still need to clip it to the segment set later on however...
                sfedges = subfacet
                sfsegments = []
                sfRounders = []
                for sfedge in sfedges:
                    sfsegments.append(Segment(self.circumcenter(sfedge[0][0]),self.circumcenter(sfedge[1][0])))
                    #roundingObjets
                    sfRounders.append([None,None])
                edges.append(sfedges)
                segments.append(sfsegments)
                roundingObjects.append(sfRounders)
            else:
                closestIdx = 0
                closestPoint = None
                dist = None
                for i in range(1,3):
                    if self.triangles[subfacet[0][0][0],(subfacet[0][0][1]+i)%3]== vIdx:
                        continue
                    closestPoint = self.point(self.triangles[subfacet[0][0][0],(subfacet[0][0][1]+i)%3])
                    dist= eg.distsq(self.point(vIdx),closestPoint)

                for i in range(len(subfacet)):
                    for j in range(1, 3):
                        if self.triangles[subfacet[i][0][0], (subfacet[i][0][1] + j) % 3] == vIdx:
                            continue
                        p = self.point(self.triangles[subfacet[i][0][0],(subfacet[i][0][1]+j)%3])
                        if eg.distsq(self.point(vIdx),p) < dist:
                            closestIdx = i
                            closestPoint = p
                            dist = eg.distsq(self.point(vIdx),closestPoint)

                #now we step forward and backward and build up sfedges and sgsegments
                if closestIdx == 0:
                    sfedges,sfsegments,sfRounders = traverseFacet(0,len(subfacet)-1)

                    edges.append(sfedges)
                    segments.append(sfsegments)
                    roundingObjects.append(sfRounders)

                elif closestIdx == len(subfacet)-1:

                    sfedges, sfsegments,sfRounders = traverseFacet(len(subfacet) - 1,0)

                    edges.append(sfedges)
                    segments.append(sfsegments)
                    roundingObjects.append(sfRounders)
                else:

                    forwardsfEdges,forwardsfSegments, forwardsfRounders = traverseFacet(closestIdx,len(subfacet)-1)

                    backwardsfEdges,backwardsfSegments, backwardsfRounders = traverseFacet(closestIdx,0)

                    edges.append(backwardsfEdges[::-1] + forwardsfEdges)
                    segments.append(backwardsfSegments[::-1] + forwardsfSegments)
                    roundingObjects.append(backwardsfRounders[::-1] + forwardsfRounders)
        return edges, segments, roundingObjects

    def getEdgeClippedConstrainedVoronoiFaceAsSegmentSet(self,vIdx):


        #first get the unclipped voronoiFace
        edges,segments,rounders = self.getUnclippedConstrainedVoronoiFace(vIdx)

        def closestEdge(v,p,segments):
            qSeg = Segment(v,p)
            closest = None
            dist = None
            for i in range(len(segments)):
                seg = segments[i]
                if (inter := eg.innerIntersect(qSeg.source(),qSeg.target(),seg.source(),seg.target())) is not None:
                    param = eg.getParamOfPointOnSegment(qSeg, inter)
                    if dist == None or param < dist:
                        closest = i
                        dist = param
            return closest

        edgeClippedSegs = []
        edgeClippedEdges = []
        edgeClippedRounders = []

        for subfacet,facetEdges,rounder in zip(segments,edges,rounders):

            #get the set of all edges that we might be interested in, for intersection
            segIds = []
            boundingSegs = []
            for edge in facetEdges:
                for face,_ in edge:
                    if face == outerFace:
                        continue
                    faceEdges,faceSegments = self.getSegmentsIntersectingCircumCircle(face)
                    for faceEdge,faceSegment in zip(faceEdges,faceSegments):
                        segId = self.triangleMap[faceEdge[0][0],faceEdge[0][1],2]
                        if vIdx in self.segments[segId]:
                            continue
                        if segId in segIds:
                            continue
                        segIds.append(segId)
                        boundingSegs.append(faceSegment)

            v = self.point(vIdx)
            for segment,edge,r in zip(subfacet,facetEdges,rounder):

                cS = closestEdge(v,segment.source(),boundingSegs)
                cT = closestEdge(v,segment.target(),boundingSegs)

                if cS == None and cT == None:
                    edgeClippedSegs.append(segment)
                    edgeClippedEdges.append(edge)
                    edgeClippedRounders.append(r)
                if cS == None and cT != None:
                    edgeClippedSegs.append(Segment(segment.source(),eg.innerIntersect(segment.source(),segment.target(),boundingSegs[cT].source(),boundingSegs[cT].target())))
                    edgeClippedEdges.append(edge)
                    edgeClippedRounders.append([r[0],boundingSegs[cT]])
                if cS != None and cT == None:
                    edgeClippedSegs.append(Segment(eg.innerIntersect(segment.source(),segment.target(),boundingSegs[cS].source(),boundingSegs[cS].target()),segment.target()))
                    edgeClippedEdges.append(edge)
                    edgeClippedRounders.append([boundingSegs[cS],r[1]])
                if cS != None and cT != None and cS != cT:
                    edgeClippedSegs.append(Segment(eg.innerIntersect(segment.source(),segment.target(),boundingSegs[cS].source(),boundingSegs[cS].target()),eg.innerIntersect(segment.source(),segment.target(),boundingSegs[cT].source(),boundingSegs[cT].target())))
                    edgeClippedEdges.append(edge)
                    edgeClippedRounders.append([boundingSegs[cS],boundingSegs[cT]])
                if cS != None and cT != None and cS == cT:
                    pass
                    #logging.info("intersting case! we dont do anything tho, as clipping is correct this way")

        return edgeClippedEdges,edgeClippedSegs, edgeClippedRounders

    def getClippedConstrainedVoronoiFaceAsSegmentSet(self,vIdx,circumIdx):
        cc = self.circumcenter(circumIdx)
        cr = self.circumRadiiSqr[circumIdx]

        edgeClippedEdges,edgeClippedSegs,edgeClippedRounders = self.getEdgeClippedConstrainedVoronoiFaceAsSegmentSet(vIdx)

        #lastly clip the edges to the circle
        resultSegs = []
        resultEdges = []
        resultRounders = []
        for edge,seg,rounder in zip(edgeClippedEdges,edgeClippedSegs,edgeClippedRounders):
            if eg.segmentIntersectsCircle(cc,cr,seg):
                clip = eg.outsideClipSegmentToCircle(cc,cr,seg)
                if len(clip) == 0:
                    pass
                    eg.outsideClipSegmentToCircle(cc, cr, seg)
                    logging.error("WTF??")
                    assert(False)
                if len(clip) == 1:
                    resultSegs.append(Segment(clip[0],clip[0]))
                    resultEdges.append(edge)
                    resultRounders.append([rounder[0] if eg.inCircle(cc,cr,seg.source()) != "outside" else None,rounder[1] if eg.inCircle(cc,cr,seg.target()) != "outside" else None])

                if len(clip) == 2:
                    resultSegs.append(Segment(*clip))
                    resultEdges.append(edge)
                    resultRounders.append([rounder[0] if eg.inCircle(cc,cr,seg.source()) != "outside" else None,rounder[1] if eg.inCircle(cc,cr,seg.target()) != "outside" else None])

        return resultEdges,resultSegs,resultRounders


    def findComplicatedCenter(self, triIdx):

        withPlot = (self.internalaxs != None)

        if withPlot:
            self.internalaxs.clear()
            self.internalaxs.set_aspect("equal")
        self.validateCircumcenters()
        assert (self.badTris[triIdx])
        tri = self.triangles[triIdx]
        myCC = Point(*self.circumCenters[triIdx])
        myCRsq = self.circumRadiiSqr[triIdx]
        # figure out the shortest side of the triangle
        baseId = 0
        otherId = 1
        farId = 2
        while eg.distsq(self.point(tri[baseId]), self.point(tri[otherId])) > eg.distsq(
                self.point(tri[(baseId + 1) % 3]), self.point(tri[(otherId + 1) % 3])) or eg.distsq(
                self.point(tri[baseId]), self.point(tri[otherId])) > eg.distsq(self.point(tri[(baseId + 2) % 3]),
                                                                               self.point(tri[(otherId + 2) % 3])):
            baseId, otherId, farId = otherId, farId, baseId

        base = self.point(tri[baseId])
        other = self.point(tri[otherId])

        baseNumeric = self.numericVerts[tri[baseId]]
        otherNumeric = self.numericVerts[tri[otherId]]
        farNumeric = self.numericVerts[tri[farId]]

        if withPlot:
            self.internalaxs.scatter(*baseNumeric.T)
            self.internalaxs.scatter(*otherNumeric.T)
            self.internalaxs.scatter(*farNumeric.T)
            self.internalaxs.plot([baseNumeric[0], otherNumeric[0], farNumeric[0], baseNumeric[0]],
                          [baseNumeric[1], otherNumeric[1], farNumeric[1], baseNumeric[1]])

        # now baseId and otherId are at the short edge and farId is at the far distance. now to figure out the direction
        diff = other - base
        orth = Point(FieldNumber(0) - diff.y(), diff.x())
        if eg.dot(orth, self.point(tri[farId]) - base) < FieldNumber(0):
            orth = Point(diff.y(), FieldNumber(0) - diff.x())

        # we now scale orth to be precisely the length of the segment
        mid = eg.altitudePoint(Segment(Point(FieldNumber(0), FieldNumber(0)), orth), myCC - base)
        orth = mid.scale(FieldNumber(2))

        if withPlot:
            self.internalaxs.scatter([float(myCC[0])], [float(myCC[1])], marker='.', color='yellow', zorder=1000)
            circle = plt.Circle((float(myCC[0]), float(myCC[1])), np.sqrt(float(self.circumRadiiSqr[triIdx])),
                                color="yellow", fill=False, zorder=1000)
            self.internalaxs.add_patch(circle)

            self.internalaxs.plot([baseNumeric[0], float((base + orth).x())], [baseNumeric[1], float((base + orth).y())])
            self.internalaxs.plot([otherNumeric[0], float((other + orth).x())], [otherNumeric[1], float((other + orth).y())])

        boundingSegments = [Segment(base,other),Segment(base,base+orth),Segment(other,other+orth)]

        #vEdges, vSegments = self.getUnclippedConstrainedVoronoiFace()
        edges, segments = self.getSegmentsIntersectingCircumCircle(triIdx)
        vEdges, vSegments, vRounders = self.getVoronoiSegmentsIntersectingCircumCircle(triIdx)

        #for a,_ in vEdges:
        #    tri,internal = a
        #    for i in range(1,3):#

        #        testSegments = self.getClippedConstrainedVoronoiFaceAsSegmentSet(self.triangles[tri,(internal+i)%3],triIdx)

        #        for vseg in testSegments:
        #            p = eg.numericPoint(vseg.source())
        #            q = eg.numericPoint(vseg.target())
        #            if withPlot:
        #                self.internalaxs.plot([p[0], q[0]], [p[1], q[1]], color="green")
        #                self.internalaxs.scatter([p[0], q[0]], [p[1], q[1]], color="green")


        for vseg in vSegments:
            p = eg.numericPoint(vseg.source())
            q = eg.numericPoint(vseg.target())
            if withPlot:
                self.internalaxs.plot([p[0],q[0]],[p[1],q[1]],color="green")


        for seg in segments:
            p = eg.numericPoint(seg.source())
            q = eg.numericPoint(seg.target())
            if withPlot:
                self.internalaxs.plot([p[0],q[0]],[p[1],q[1]],color="black")

        candidateIntersections = []
        spawningObjects = []
        rounderObjects = []
        for vedge,vseg,rounder in zip(vEdges,vSegments,vRounders):
            closestPointId = self.triangles[vedge[0][0],(vedge[0][1]+1)%3]
            closestPoint = self.point(closestPointId)

            candidateIntersections.append([eg.distsq(closestPoint, vseg.source()), vseg.source()])
            rounderObjects.append(rounder[0])
            spawningObjects.append(vseg)
            candidateIntersections.append([eg.distsq(closestPoint, vseg.target()), vseg.target()])
            rounderObjects.append(rounder[1])
            spawningObjects.append(vseg)


            #first intersect with boundingbox
            for bbseg in boundingSegments:
                inter = eg.innerIntersect(vseg.source(),vseg.target(),bbseg.source(),bbseg.target())
                if inter != None:
                    candidateIntersections.append([eg.distsq(closestPoint,inter),inter])
                    spawningObjects.append(vseg)
                    rounderObjects.append(bbseg)
                    if withPlot:
                        self.internalaxs.scatter([float(inter[0])], [float(inter[1])], marker="*", color='blue', zorder=100)

            #next intersect with encountered segments
            for encseg in segments:
                inter = eg.innerIntersect(encseg.source(),encseg.target(),vseg.source(),vseg.target())
                if inter != None:
                    inter = eg.roundExactOnSegment(encseg,inter)
                    candidateIntersections.append([eg.distsq(closestPoint,inter),inter])
                    spawningObjects.append(vseg)
                    rounderObjects.append(encseg)
                    if withPlot:
                        self.internalaxs.scatter([float(inter[0])], [float(inter[1])], marker="*", color='blue', zorder=100)

            #find intersections with circle. this is somehow really annoying
            inters = eg.insideIntersectionsSegmentCircle(myCC,myCRsq,vseg)
            for inter in inters:
                candidateIntersections.append([eg.distsq(closestPoint,inter),inter])
                spawningObjects.append(vseg)
                #probably not best, but whatever
                rounderObjects.append(None)
                if withPlot:
                    self.internalaxs.scatter([float(inter[0])], [float(inter[1])], marker="*", color='blue', zorder=100)
                    pass

        def verifyPointInRegion(p:Point):
            if eg.distsq(myCC, p) > myCRsq:
                return False

            # outside slab
            if eg.dot(other - base, p - base) < FieldNumber(0):
                return False
            if eg.dot(base - other, p - other) < FieldNumber(0):
                return False
            if eg.dot(orth, p - base) < FieldNumber(0):
                return False
            return True


        candidatePoints = []
        candidateSpawners = []
        candidateRounders = []
        for i in range(len(candidateIntersections)):
            d, p = candidateIntersections[i]
            candidateRounder = rounderObjects[i]
            if withPlot:
                self.internalaxs.scatter([float(p[0])], [float(p[1])], marker="*", color='yellow', zorder=101)

            if verifyPointInRegion(p):
                candidatePoints.append([d, p])
                candidateSpawners.append(spawningObjects[i])
                candidateRounders.append(candidateRounder)
            if withPlot:
                self.internalaxs.scatter([float(p[0])], [float(p[1])], marker="*", color='red', zorder=1000)

        def verifyPointDoesntCross(p):
            for i in range(3):
                triP = self.point(tri[i])
                for seg in self.segments:
                    if tri[i] in seg:
                        continue
                    segSource = self.point(seg[0])
                    segTarget = self.point(seg[1])
                    if (inter := eg.innerIntersect(segSource, segTarget, triP, p,False,False)) is not None:
                        return False
            return True

        # now all candidatePoints are guaranteed to lie inside the region. we need to check finally, if the three rays from base, other and far to the point intersect some segment
        actualInside = None
        actualDist = None
        actualRoundable = None
        for i in range(len(candidatePoints)):
            d,p = candidatePoints[i]
            rounder = candidateRounders[i]
            if actualDist is None or d > actualDist:
                rounded = eg.roundExactBoundor(p) if candidateRounders[i] is None else eg.roundExactOnSegmentBounder(candidateRounders[i],p)

                verified = verifyPointDoesntCross((candidateSpawners[i].source() + candidateSpawners[i].target()).scale(eg.onehalf))

                if verified:
                    addor = None
                    for r in rounded:
                        if not verifyPointInRegion(r):
                            continue
                        if addor is not None:
                            break
                        addor = r

                    if addor is None:
                        addor = p

                    if addor is not None:
                        actualInside = addor
                        actualDist = d
        return actualInside

    def pointHitsTriangle(self,triIdx,p:Point):
        tri = self.triangles[triIdx]
        sides = np.array([eg.onWhichSide(Segment(self.point(self.triangles[triIdx, (i + 1) % 3]),
                                                 self.point(self.triangles[triIdx, (i + 2) % 3])), p) for i in
                          range(3)])
        if np.all((sides == "left")) or np.all((sides == "right")):
            return "inside",noneIntervalVertex
        elif np.all((sides == "left") | (sides == "colinear")) or np.all(
                (sides == "right") | (sides == "colinear")):
            if len(np.where(sides == "colinear")[0])== 1:
                return "on",np.argwhere(sides == "colinear")
            else:
                return "on",None
        return "outside",noneIntervalVertex

    def getAllTruncatedCirclesIntersecting(self,triIdx):
        #first figure out all triangles that intersect the circumcenter of triIdx
        myCC = self.circumcenter(triIdx)
        myCRsqr = self.circumRadiiSqr[triIdx]
        mIds = self.triangles[triIdx]
        limitingEdges,limitingSegments = self.getSegmentsIntersectingCircumCircle(triIdx)

        tried = []
        intersecting = []
        actionstack = [triIdx]
        while len(actionstack) > 0:
            cur = actionstack.pop(0)
            tried.append(cur)
            if eg.circleIntersectsCircle(myCC,myCRsqr,self.circumcenter(cur),self.circumRadiiSqr[cur]):
                intersecting.append(cur)
                for i in range(3):
                    if self.triangleMap[cur,i,2] == noneEdge:
                        if (next:=self.triangleMap[cur,i,0]) != outerFace:
                            #check if we are on the other side of a segment anyways
                            intersectsEdge = False
                            for baseI in range(3):
                                for otherI in range(3):
                                    if self.triangles[triIdx,baseI] == self.triangles[cur,otherI]:
                                        continue
                                    for e,s in zip(limitingEdges,limitingSegments):
                                        if self.triangles[triIdx,baseI] in self.segments[self.triangleMap[e[0][0],e[0][1],2]]:
                                            continue
                                        if self.triangles[cur,otherI] in self.segments[self.triangleMap[e[0][0],e[0][1],2]]:
                                            continue
                                        if eg.innerIntersect(self.point(self.triangles[triIdx,baseI]),self.point(self.triangles[cur,otherI]),s.source(),s.target(),False,False):
                                            intersectsEdge=True
                                            break
                                    if intersectsEdge:
                                        break
                                if intersectsEdge:
                                    break
                            if (not intersectsEdge) and (next not in tried):
                                actionstack.append(next)
        return intersecting

    def getAllSegmentDisks(self,triIdx,internal):
        lDisks = self.getAllTruncatedCirclesIntersecting(triIdx)
        llimitingEdges,llimitingSegments = self.getSegmentsIntersectingCircumCircle(triIdx)
        rDisks = None
        rlimitingEdges,rlimitingSegments = None,None
        if (oppF := self.triangleMap[triIdx,internal,0]) != outerFace:
            rDisks = self.getAllTruncatedCirclesIntersecting(oppF)
            rlimitingEdges,rlimitingSegments = self.getSegmentsIntersectingCircumCircle(oppF)
        disks = list(set(lDisks+rDisks)) if rDisks != None else lDisks
        limitingEdges = llimitingEdges+rlimitingEdges if rDisks != None else llimitingEdges
        limitingSegments = llimitingSegments+rlimitingSegments if rDisks != None else llimitingSegments

        querySeg = Segment(self.point(self.triangles[triIdx,(internal+1)%3]),self.point(self.triangles[triIdx,(internal+2)%3]))

        for i in reversed(range(len(limitingSegments))):
            if limitingSegments[i].source() == querySeg.source() and limitingSegments[i].target() == querySeg.target():
                limitingSegments.pop(i)
                limitingEdges.pop(i)

            elif limitingSegments[i].target() == querySeg.source() and limitingSegments[i].source() == querySeg.target():
                limitingSegments.pop(i)
                limitingEdges.pop(i)

        queryPoints = [(querySeg.source()+querySeg.target()).scale(eg.onehalf)]
        for dIdx in disks:
            cc = self.circumcenter(dIdx)
            cr = self.circumRadiiSqr[dIdx]
            if eg.segmentIntersectsCircle(cc,cr,querySeg):
                possiblePoints = eg.insideIntersectionsSegmentCircle(cc,cr,querySeg) + eg.outsideIntersectionsSegmentCircle(cc,cr,querySeg)
                for p in possiblePoints:
                    if p == querySeg.source():
                        continue
                    if p == querySeg.target():
                        continue
                    alreadyInQPs = False
                    for qp in queryPoints:
                        if qp == p:
                            alreadyInQPs = True
                            break
                    if alreadyInQPs:
                        continue
                    queryPoints.append(p)
        topoDisks = set()
        for q in queryPoints:
            topoDisk = []
            for i in disks:
                if eg.inCircle(self.circumcenter(i), self.circumRadiiSqr[i], q) != "outside":
                    intersectsEdge = False
                    for otherI in range(3):
                        for e, s in zip(limitingEdges, limitingSegments):
                            if self.triangles[i, otherI] in self.segments[self.triangleMap[e[0][0], e[0][1], 2]]:
                                continue
                            if eg.innerIntersect(q, self.point(self.triangles[i, otherI]), s.source(), s.target(),
                                                 False, False):
                                intersectsEdge = True
                                break
                        if intersectsEdge:
                            break
                    if intersectsEdge:
                        continue
                    topoDisk.append(i)
            topoDisk = tuple(list(sorted(topoDisk)))
            topoDisks.add(topoDisk)
        return topoDisks




    def getAllTriangleDisks(self,triIdx):
        #TODO: optmize
        #first figure out all triangles that intersect the circumcenter of triIdx
        myCC = self.circumcenter(triIdx)
        myCRsqr = self.circumRadiiSqr[triIdx]
        mIds = self.triangles[triIdx]
        limitingEdges,limitingSegments = self.getSegmentsIntersectingCircumCircle(triIdx)

        intersecting = self.getAllTruncatedCirclesIntersecting(triIdx)

        queryPoints = []
        for i in range(len(intersecting)):
            for j in range(i+1,len(intersecting)):
                myCCA = self.circumcenter(intersecting[i])
                myCRsqrA = self.circumRadiiSqr[intersecting[i]]
                myCCB = self.circumcenter(intersecting[j])
                myCRsqrB = self.circumRadiiSqr[intersecting[j]]
                for p in eg.getCircleIntersections(myCCA,myCRsqrA,myCCB,myCRsqrB):
                    queryPoints.append(p)
        topoDisks = set()
        for q in queryPoints:
            if eg.inCircle(myCC,myCRsqr,q) == "outside":
                continue
            topoDisk = []
            for i in intersecting:
                if eg.inCircle(self.circumcenter(i),self.circumRadiiSqr[i],q) != "outside":
                    intersectsEdge = False
                    for otherI in range(3):
                        for e, s in zip(limitingEdges, limitingSegments):
                            if self.triangles[i, otherI] in self.segments[self.triangleMap[e[0][0], e[0][1], 2]]:
                                continue
                            if eg.innerIntersect(q,self.point(self.triangles[i, otherI]), s.source(), s.target(),
                                                 True, True):
                                intersectsEdge = True
                                break
                        if intersectsEdge:
                            break
                    if intersectsEdge:
                        continue
                    topoDisk.append(i)
            if triIdx not in topoDisk:
                continue
            topoDisk = tuple(list(sorted(topoDisk)))
            topoDisks.add(topoDisk)
        return topoDisks

    def getGeometricSubproblemFromTopoDisk(self,topoDisk):
        #assumption here is, that there is no vertex on the inside. Maybe this assumption ought to be dropped lateron
        topoDisk = list(topoDisk)
        boundary = [[topoDisk[0],i] for i in range(3)]
        expanded = [topoDisk[0]]
        changed = True
        innerSegmentIds = []
        while changed:
            changed = False
            for i in range(len(boundary)):
                boundaryedge = boundary[i]
                nextedge = boundary[(i+1)%len(boundary)]
                nextedgeAsIdxs = [self.triangles[nextedge[0],(nextedge[1]+offset)%3] for offset in range(1,3)]
                if self.triangleMap[boundaryedge[0],boundaryedge[1],0] in topoDisk:
                    #need to replace this boundarypiece
                    nextF,nextI = self.triangleMap[boundaryedge[0],boundaryedge[1],0],self.triangleMap[boundaryedge[0],boundaryedge[1],1]
                    if nextF in expanded:
                        logging.error("circle expansion detected...")
                        return None
                    nexttwo = [[nextF,(nextI+1)%3],[nextF,(nextI+2)%3]]
                    if self.triangles[nexttwo[0][0],nexttwo[0][1]] not in nextedgeAsIdxs:
                        nexttwo = nexttwo[::-1]
                    if i == len(boundary)-1:
                        boundary = boundary[:i] + nexttwo
                    else:
                        boundary = boundary[:i] + nexttwo + boundary[i+1:]
                    changed = True
                    expanded.append(nextF)
                    if (segId := self.triangleMap[boundaryedge[0],boundaryedge[1],2]) != noneEdge:
                        innerSegmentIds.append(segId)
                    break
        #build link
        next = None
        link = []
        for i in range(len(boundary)):
            curF,curI = boundary[i]
            edgeAsIdxs = [self.triangles[curF,(curI+1)%3],self.triangles[curF,(curI+2)%3]]
            if next == None:
                nextF,nextI = boundary[i+1]
                nextEdgeAsIdxs = [self.triangles[nextF,(nextI+1)%3],self.triangles[nextF,(nextI+2)%3]]
                for id in edgeAsIdxs:
                    if id in nextEdgeAsIdxs:
                        next = id
                    else:
                        link.append(id)
            else:
                link.append(next)
                for id in edgeAsIdxs:
                    if id != next:
                        next = id
                        break

        segmentIds = []
        for face,internal in boundary:
            if self.triangleMap[face,internal,2] != noneEdge:
                segmentIds.append(self.triangleMap[face,internal,2])
        numBad = len(np.where(self.badTris[topoDisk] == True)[0])
        # clean up link and move their inbetween vertex to the inside. if the homotopy type is non-trivial, we return None instead

        numBoundaryDroppers = 0
        for face in topoDisk:
            if self.badTris[face]:
                i = eg.badAngle(self.point(self.triangles[face, 0]), self.point(self.triangles[face, 1]),
                                self.point(self.triangles[face, 2]))
                if self.triangleMap[face, i, 0] == outerFace:
                    numBoundaryDroppers += 1
                    numBad -= 1

        outsideIds = []
        for face in topoDisk:
            for internal in range(3):
                if (nId := self.triangleMap[face,internal,0]) != outerFace and (nId not in topoDisk) and (nId not in outsideIds):
                    outsideIds.append(self.triangleMap[face,internal,0])

        outside = []
        for face in outsideIds:
            outside.append([self.circumcenter(face),self.circumRadiiSqr[face]])

        return GeometricSubproblem([], topoDisk, link, self.exactVerts[link], self.numericVerts[link],  self.segments[innerSegmentIds],
                                   self.segments[segmentIds], self.segmentType[segmentIds], self.instanceSize,
                                   numBad,numBoundaryDroppers, outside,"topoDisk",0, self.gpaxs)


    def combinatorialDepth(self):
        dists = np.full((len(self.triangles)),-1)
        actionQueue = []
        for triIdx in self.validTriIdxs():
            if outerFace in self.triangleMap[triIdx,:,0]:
                dists[triIdx] = 0
                actionQueue.append(triIdx)
        while len(actionQueue) > 0:
            triIdx = actionQueue.pop(0)
            myDist = dists[triIdx]
            assert(myDist != -1)
            for i in range(3):
                nId = self.triangleMap[triIdx,i,0]
                if nId == outerFace or dists[nId] != -1:
                    continue
                dists[nId] = myDist + 1
                actionQueue.append(nId)
        return np.array(dists)

    def applyUnsafeActionAndReturnSafeAction(self,action):
        assert(not action.safe)
        removedPoints = []
        removedPointIds = []
        addedPoints=[]
        addedPointIds = []

        for id in action.removedPointIds:
            if  id in self.invalidVertIdxs() or id >= len(self.exactVerts):
                continue
            else:
                removedPoints.append(self.point(id))
                removedPointIds.append(id)
                self.removePoint(id)


        for id,p in zip(action.addedPointIds,action.addedPoints):
            added,actualId = False,None
            if id in self.invalidVertIdxs():
                added,actualId = self.addPoint(p,id)
            else:
                added,actualId = self.addPoint(p)
            if added:
                addedPoints.append(p)
                addedPointIds.append(actualId)
        return TriangulationAction(addedPoints,addedPointIds,removedPoints,removedPointIds)


    def applyAction(self,action):
        assert(action.safe)
        for id,p in zip(action.removedPointIds,action.removedPoints):
            assert( id not in self.invalidVertIdxs() )
            assert( p.x() == self.point(id).x() )
            assert( p.y() == self.point(id).y() )
            self.removePoint(id)

        for id,p in zip(action.addedPointIds,action.addedPoints):
            assert(id in self.invalidVertIdxs())
            added,newId = self.addPoint(p,id)
            assert(added)
            assert(id == newId)

    def undoAction(self,action):
        assert(action.safe)
        for id, p in zip(action.addedPointIds, action.addedPoints):
            assert (id not in self.invalidVertIdxs())
            assert (p.x() == self.point(id).x())
            assert (p.y() == self.point(id).y())
            self.removePoint(id)

        for id, p in zip(action.removedPointIds, action.removedPoints):
            assert (id in self.invalidVertIdxs())
            added, newId = self.addPoint(p, id)
            assert (added)
            assert (id == newId)

class TriangulationAction:
    def __init__(self,addedPoints,addedPointIds,removedPoints,removedPointIds,safe=True):
        self.addedPoints = addedPoints
        self.addedPointIds = addedPointIds
        self.removedPoints = removedPoints
        self.removedPointIds = removedPointIds
        self.safe = safe

class QualityImprover:
    def __init__(self, tri: Triangulation,seed=None):
        self.tri = tri
        self.solver = StarSolver(2,1,1,1.25,2,2,2)
        if seed != None:
            np.random.seed(seed)
        self.seed = seed
        self.convergenceDetectorDict = dict()


    def plotHistory(self,numSteinerHistory,numBadTriHistory,round,specialRounds,ax,twin_ax):

        if ax != None:
            ax.clear()
            twin_ax.clear()
            ax.plot(list(range(round)), numSteinerHistory, color="red")
            ax.plot(list(range(round)), numBadTriHistory, color="mediumorchid")
            ax.plot(list(range(round)), np.array(numSteinerHistory) + np.array(numBadTriHistory),
                                   color="black")
            ax.plot([0, max(numSteinerHistory[0] + numBadTriHistory[0], round - 1)],
                                   [numSteinerHistory[0] + numBadTriHistory[0],
                                    numSteinerHistory[0] + numBadTriHistory[0]], linestyle="dashed", color="black")
            ax.set_xlim((0, max(numSteinerHistory[0] + numBadTriHistory[0], round - 1)))
            ax.set_ylim((0, ax.get_ylim()[1]))
            ax.set_yticks([0, numSteinerHistory[0] + numBadTriHistory[0]])
            ax.plot([round - 1, max(numSteinerHistory[0] + numBadTriHistory[0], round - 1)],
                                   [numSteinerHistory[-1] + numBadTriHistory[-1],
                                    numSteinerHistory[-1] + numBadTriHistory[-1]], color="black", linestyle="dotted")
            twin_ax.set_yticks([numSteinerHistory[-1] + numBadTriHistory[-1]])
            twin_ax.set_ylim(ax.get_ylim())
            b,t = ax.get_ylim()
            for r in specialRounds:
                ax.plot([r,r],[b,t],color="blue")

    def buildUnsafeActionList(self,earlyStoppingAllowed:int,onlyChanged=False,fastAndGreedy=False):
        actionList = []
        hasGoodSolution = False

        nonSuperseeded = self.tri.getNonSuperseededBadTris()
        if onlyChanged:
            nonSuperseeded = np.array([id for id in nonSuperseeded if self.tri.triangleChanged[id]])

        self.tri.updateGeometricProblems(fastAndGreedy)

        #geometricsubproblem induced actions
        np.random.shuffle(self.tri.geometricLinkProblems)
        np.random.shuffle(self.tri.geometricCircleProblems)
        np.random.shuffle(self.tri.geometricSegmentProblems)
        for gp in self.tri.geometricSubproblemIterator():
            if onlyChanged and gp.wasSolved:
                continue
            #TODO: return all different types of solutions instead of the best one, but respect rounding?
            #for eval, sol in self.solver.solve(gp):
            eval,sol = self.solver.solve(gp)
            if sol != None:
                if eval < self.solver.cleanWeight:
                    hasGoodSolution = True
                actionList.append((eval,TriangulationAction(sol,[-1 for _ in sol],[],gp.getInsideSteiners(),False),gp))
        if hasGoodSolution and earlyStoppingAllowed != -1 and len(actionList) > earlyStoppingAllowed :
            return sorted(actionList,key=lambda x:x[0])

        #bad triangle induced actions
        #nonSuperseeded = self.tri.getNonSuperseededBadTris()  # list(np.where(self.tri.badTris == True)[0])

        if len(nonSuperseeded) != 0:
            locs = None

            if not onlyChanged:

                nonSuperseededMask = np.full(self.tri.badTris.shape, False)
                nonSuperseededMask[nonSuperseeded] = True
                # mask = nonSuperseededMask

                dists = self.tri.combinatorialDepth()
                centerFindIdx = None

                # tolerance for distance selector
                tolerance = 2
                # setting this to zero could result in situations that do not converge. careful!
                withOuterLayer = True#False

                assert (tolerance >= 0)
                mask = np.full(self.tri.badTris.shape, False)
                mode = "fromInside"  # "fromOutside"

                if mode == "fromInside":
                    val = np.max(dists[nonSuperseeded])
                    mask |= (((dists >= val - tolerance)) & (nonSuperseededMask))
                    if withOuterLayer:
                        mask |= ((dists == 0) & (self.tri.badTris))

                elif mode == "fromOutside":
                    val = np.min(dists[nonSuperseeded])
                    mask |= (((dists <= val + tolerance)) & (nonSuperseededMask))
                    if withOuterLayer:
                        mask |= ((dists == 0) & (self.tri.badTris))

                locs = np.where(mask)[0]
            else:
                locs = nonSuperseeded
            np.random.shuffle(locs)
            for id in locs:
                if earlyStoppingAllowed != -1 and len(actionList) > earlyStoppingAllowed :
                    return sorted(actionList,key=lambda x:x[0])
                center = None
                threat = self.convergenceDetectorDict.get(id, 0)
                if threat < 10:
                    bA = eg.badAngle(self.tri.point(self.tri.triangles[id, 0]),
                                     self.tri.point(self.tri.triangles[id, 1]),
                                     self.tri.point(self.tri.triangles[id, 2]))
                    if self.tri.triangleMap[id, bA, 2] != noneEdge:
                        # logging.error("alt")
                        center = eg.altitudePoint(Segment(self.tri.point(self.tri.triangles[id, (bA + 1) % 3]),
                                                          self.tri.point(self.tri.triangles[id, (bA + 2) % 3])),
                                                  self.tri.point(self.tri.triangles[id, bA]))
                    else:
                        # logging.error("circ")
                        center = self.tri.findComplicatedCenter(id)
                elif threat < 20:
                    center = Point(*self.tri.circumCenters[id])
                else:
                    center = Point(eg.zero, eg.zero)
                    for i in range(3):
                        center += self.tri.point(self.tri.triangles[id, i])
                    center = center.scale(FieldNumber(1) / FieldNumber(3))

                actionList.append((self.solver.cleanWeight-1/64,TriangulationAction([center],[-1],[],[],False),None))
                if earlyStoppingAllowed != -1 and len(actionList) > earlyStoppingAllowed:
                    break
        actionList = sorted(actionList,key=lambda x:x[0])
        return actionList

    def eval(self):
        #numSteinerHistory.append(len(self.tri.validVertIdxs()) - self.tri.instanceSize)
        #numBadTriHistory.append(len(np.where(self.tri.badTris == True)[0]))
        #return len(self.tri.validVertIdxs()) - self.tri.instanceSize + len(np.where(self.tri.badTris == True)[0])
        return len(self.tri.validVertIdxs()) - self.tri.instanceSize + 2* len(self.tri.getNonSuperseededBadTris()) + len(np.where(self.tri.badTris == True)[0])

    def solveEveryGP(self):
        for gp in self.tri.geometricSubproblemIterator():
            self.solver.solve(gp)

    def realEvalActionList(self,unsafeActions,depth:int):
        if depth == 0:
            result = []
            combinatorialState = self.tri.copyOfCombinatorialState()
            for action in unsafeActions:

                realAction = self.tri.applyUnsafeActionAndReturnSafeAction(action)
                result.append(self.eval())

                #afterwards vertex state should be the same...
                self.tri.undoAction(realAction)
                #and then combinatorial state is the same as well hopefully
                self.tri.applyCombinatorialState(combinatorialState)
            return result
        else:
            result = []
            for action in unsafeActions:
                result.append(self.realEvalAction(action,depth,True))
            return result




    def realEvalAction(self,action:TriangulationAction,depth:int,fastAndGreedy=False):
        assert(depth >= 0)
        actualAction = self.tri.applyUnsafeActionAndReturnSafeAction(action)
        myEval = -1
        if depth == 0:
            myEval = self.eval()
        elif depth == 1:

            unsafeSubactions = self.buildUnsafeActionList(-1, True,fastAndGreedy)
            combinatorialState = self.tri.copyOfCombinatorialState()
            #print(" "*(3-depth) + str(len(unsafeSubactions)))

            # if we have actions:
            for _, subaction, _ in unsafeSubactions:

                realSubAction = self.tri.applyUnsafeActionAndReturnSafeAction(subaction)
                subEval = self.eval()
                if myEval == -1 or subEval < myEval:
                    myEval = subEval

                #afterwards vertex state should be the same...
                self.tri.undoAction(realSubAction)
                #and then combinatorial state is the same as well hopefully
                self.tri.applyCombinatorialState(combinatorialState)

            # no action to be applied :(
            if myEval == -1:
                myEval = self.eval()

        else:
            #deep eval
            unsafeSubactions = self.buildUnsafeActionList(-1, True,fastAndGreedy)
            #print(" "*(3-depth) + str(len(unsafeSubactions)))

            #if we have actions:
            for _, subaction, _ in unsafeSubactions:
                subEval = self.realEvalAction(subaction, depth - 1)
                if myEval == -1 or subEval < myEval:
                    myEval = subEval

            #no action to be applied :(
            if myEval == -1:
                myEval = self.eval()

        self.tri.undoAction(actualAction)
        self.tri.updateGeometricProblems(fastAndGreedy)
        return myEval

    def addCenterOfTriangle(self,id,threat,depth):
        logging.info("convergence threat level: " + str(threat))
        if threat < 10:
            # if solved by altitudedrop, do so, else use complicated center
            center = None
            bA = eg.badAngle(self.tri.point(self.tri.triangles[id, 0]),
                             self.tri.point(self.tri.triangles[id, 1]),
                             self.tri.point(self.tri.triangles[id, 2]))
            if self.tri.triangleMap[id, bA, 2] != noneEdge:
                # logging.error("alt")
                center = eg.altitudePoint(Segment(self.tri.point(self.tri.triangles[id, (bA + 1) % 3]),
                                                  self.tri.point(self.tri.triangles[id, (bA + 2) % 3])),
                                          self.tri.point(self.tri.triangles[id, bA]))
            else:
                # logging.error("circ")
                center = self.tri.findComplicatedCenter(id)

            triang = list(self.tri.triangles[id])
            added,newPointId = self.tri.addPoint(center)
            if added:
                logging.info("successfully added complicated Center of triangle " + str(id) + " at depth " + str(depth))
                lastEdit = "circumcenter"
                added = True
                allIn = np.all([vIdx in self.tri.triangles[id] for vIdx in triang])
                if allIn:
                    logging.error("Addition did not flip the triangle even though it should...")
                return True,TriangulationAction([center],[newPointId],[],[])
            else:
                logging.error(str(self.seed) + ": failed to add complicated Center of triangle " + str(id) + " at depth " + str(depth))
                return False,None
        elif threat < 20:
            logging.info(str(self.seed) + ": reached convergence threat " + str(threat))
            center = Point(*self.tri.circumCenters[id])
            assert (center != None)
            added,newPointId = self.tri.addPoint(center)
            if added:
                logging.info("successfully added circumcenter of triangle " + str(id) + " at depth " + str(
                    depth))
                return True,TriangulationAction([center],[newPointId],[],[])
            else:
                logging.info(str(self.seed) + ": failed to add circumcenter of triangle " + str(
                    id) + " at depth " + str(depth))
                return False,None
        else:
            logging.error(
                str(self.seed) + ": reached convergence threat " + str(threat))
            center = Point(eg.zero, eg.zero)
            for i in range(3):
                center += self.tri.point(self.tri.triangles[id, i])
            center = center.scale(FieldNumber(1) / FieldNumber(3))
            assert (center != None)
            added,newPointId = self.tri.addPoint(center)
            if added:
                logging.info("successfully added centroid of triangle " + str(id) + " at depth " + str(
                    depth))
                return True,TriangulationAction([center],[newPointId],[],[])
            else:
                logging.error(
                    str(self.seed) + ": failed to add centroid of triangle " + str(id) + " at depth " + str(
                        depth))
                return False,None

    def improve(self):
        actionStack = []
        keepGoing = True
        lastEdit = "None"
        plotUpdater = 0
        round = 0
        specialRounds = []
        numSteinerHistory = []
        numBadTriHistory = []
        lastImprovement = round
        bestSofar = 1000
        self.convergenceDetectorDict = dict()
        convergenceEndCounter = 0
        while keepGoing:
            #logging.info("----- ACTIONSTACK -----")
            #for a in actionStack:
            #    logging.info(str([(float(p.x()),float(p.y())) for p in a.addedPoints]))
            #    logging.info(str(a.addedPointIds))
            #    logging.info(str([(float(p.x()),float(p.y())) for p in a.removedPoints]))
            #    logging.info(str(a.removedPointIds))

            #logging.info("-----------------------")

            for a in reversed(actionStack):
            #    self.tri.undoAction(a)
            #    self.tri.plotTriangulation()
                pass

            for a in actionStack:
            #    self.tri.applyAction(a)
            #    self.tri.plotTriangulation()
                pass

            logging.info(f"Round {round}: #Steiner = {len(self.tri.validVertIdxs()) - self.tri.instanceSize}, #>90° = {len( np.where(self.tri.badTris == True)[0])}, subproblems solved = {self.solver.succesfulSolves}, rep qual = {self.tri.getCoordinateQuality()}")
            numSteinerHistory.append(len(self.tri.validVertIdxs()) - self.tri.instanceSize)
            numBadTriHistory.append(len( np.where(self.tri.badTris == True)[0]))
            round += 1
            plotUpdater += 1
            # curEdit = "None"

            #convergence safeguards
            if len(np.where(self.tri.badTris)[0]) < bestSofar:
                bestSofar = len(np.where(self.tri.badTris)[0])
                lastImprovement = round

            if self.solver.patialTolerance > 0 and round - lastImprovement > 25:
                self.solver.patialTolerance = max(0,self.solver.patialTolerance - 1)
                logging.info(f"Tolerance set to {self.solver.patialTolerance}")
                self.solver.outsideTolerance = min(self.solver.patialTolerance,self.solver.outsideTolerance)
                lastImprovement = round
                specialRounds.append(round)

            #be less tolerant if three quarters of the triangles are gone
            if self.solver.patialTolerance > 1 and len(np.where(self.tri.badTris)[0]) <= max(1, numBadTriHistory[0] // 4):
                logging.info("Tolerance set to 1")
                self.solver.patialTolerance = 1
                self.solver.outsideTolerance = min(self.solver.patialTolerance,self.solver.outsideTolerance)
                specialRounds.append(round)

            #no longer be tolerant when only 5% of bad triangles are left
            if self.solver.patialTolerance > 0 and len(np.where(self.tri.badTris)[0]) <= max(1, numBadTriHistory[0] // 20):
                logging.info("Tolerance set to 0")
                self.solver.patialTolerance = 0
                self.solver.outsideTolerance = min(self.solver.patialTolerance,self.solver.outsideTolerance)
                specialRounds.append(round)

            #get best action
            #TODO: add early stopping up to k
            numMoves = 15
            actionList = self.buildUnsafeActionList(numMoves,False,False)
            logging.info("identified " + str(len(actionList)) +" actions.")
            actionAdded = False

            betterEvalActionPairs = []

            depth = 1
            actionList = actionList[:numMoves + 1]
            np.random.shuffle(actionList)

            actionValues = self.realEvalActionList([action for _,action,_ in actionList],depth)

            #i = 0
            #for _,action,_ in actionList[:numMoves+1]:
            #    logging.info(f"evaluating action {i}")
            #    i+=1
            #    betterEvalActionPairs.append((self.realEvalAction(action,depth),action))
            #self.tri.updateGeometricProblems()

            betterEvalActionPairs = sorted(list(zip(actionValues,[action for _,action,_ in actionList])),key=lambda x:x[0])
            #print([v for v,_ in betterEvalActionPairs])
            for _,action in betterEvalActionPairs:
                if len(action.addedPointIds) == 0 and len(action.removedPointIds) == 0:
                    continue
                if np.min([self.convergenceDetectorDict.get(id,0) for id in action.addedPointIds] + [self.convergenceDetectorDict.get(id,0) for id in action.addedPointIds]) > 30:
                    continue
                realAction = self.tri.applyUnsafeActionAndReturnSafeAction(action)
                if len(realAction.addedPointIds) == 0 and len(realAction.removedPointIds) == 0:
                    for id in action.addedPointIds:
                        self.convergenceDetectorDict[id] = self.convergenceDetectorDict.get(id,0) + 1
                    for id in action.removedPointIds:
                        self.convergenceDetectorDict[id] = self.convergenceDetectorDict.get(id,0) + 1
                    continue
                actionAdded = True
                actionStack.append(realAction)
                break

            #print(self.tri.watch)
            self.tri.watch=0
            self.plotHistory(numSteinerHistory,numBadTriHistory,round,specialRounds,self.tri.histoaxs,self.tri.histoaxtwin)
            self.tri.plotTriangulation()
            if plotUpdater == 1:
            #    self.tri.plotCoordinateQuality()
                self.tri.plotTriangulation()
                plotUpdater = 0
            if not actionAdded:
                if len(self.tri.getNonSuperseededBadTris()) == 0:
                    keepGoing = False
                else:
                    logging.error("only increased convergence threat this iteration...")
            else:
                if len(self.tri.getNonSuperseededBadTris()) == 0:
                    if convergenceEndCounter > 10:
                        keepGoing = False
                    else:
                        convergenceEndCounter += 1
        self.tri.plotTriangulation()
        self.plotHistory(numSteinerHistory,numBadTriHistory,round,specialRounds,self.tri.histoaxs,self.tri.histoaxtwin)
        plt.pause(0.5)
        return self.tri.solutionParse()

class SolutionMerger:
    def __init__(self, instance,triPool):
        self.triPool = triPool
        #self.solver = StarSolver(2,1,1,1.25,2,2,2)
        self.instance = instance
        self.instancesize = len(instance.points_x)
        self.kdPool = [KDTree(tri.numericVerts[self.instancesize:]) for tri in triPool]

    def attemptImprovement(self,tri:Triangulation):
        assert(len(tri.validVertIdxs()) == len(tri.exactVerts))#for now
        myKDTree = KDTree(tri.numericVerts[self.instancesize:])
        for x in tri.numericVerts:
            for y in tri.numericVerts:
                diff = y-x
                r = np.sqrt((diff[0]*diff[0]) + (diff[1]*diff[1]))
                myInsidePoints = myKDTree.query_ball_point(x,r)
                if len(myInsidePoints) <= 2:
                    continue
                if len(myInsidePoints)*5 > len(tri.numericVerts[self.instancesize:]):
                    continue
                for kdTree,triang in zip(self.kdPool,self.triPool):
                    inside = kdTree.query_ball_point(x,r)
                    if len(inside) == 0:
                        continue
                    if len(inside) < len(myInsidePoints):
                        print("attempting improvement...")
                        pass
                        #build new triangulation and improve it...
                        tr = Triangulation(self.instance)
                        insertSteinerpoints = []
                        for i in range(self.instancesize,len(tri.exactVerts)):
                            if i in myInsidePoints:
                                continue
                            insertSteinerpoints.append(tri.point(i))
                        for i in inside:
                            insertSteinerpoints.append(triang.point(i))
                        tr.addPoints(insertSteinerpoints)
                        qi = QualityImprover(tr)
                        tr = qi.improve()
                        if len(tri.validVertIdxs()) < len(tri.exactVerts):
                            print("found improvement!!!")








