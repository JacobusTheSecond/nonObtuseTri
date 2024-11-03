import numpy as np
import matplotlib.ticker
import matplotlib.pyplot as plt
from cgshop2025_pyutils import Cgshop2025Solution, Cgshop2025Instance
from cgshop2025_pyutils.geometry import FieldNumber, Point, Segment

import exact_geometry as eg
from constants import *
from GeometricSubproblem import GeometricSubproblem

import triangle as tr  # https://rufat.be/triangle/

import logging

from src.GeometricSubproblem import StarSolver

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
        self.plotWithIds = True#self.withValidate

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

        self.triangles = triangles
        self.segments = segments
        self.triangleMap = None
        self.exactVerts = np.array(exactVerts, dtype=Point)
        self.numericVerts = np.array(numericVerts)
        self.isValidVertex = np.array([True for _ in self.numericVerts], dtype=bool)

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

        self.geometricProblems = []
        self.updateGeometricProblems()

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
                t = plt.Polygon(self.numericVerts[tri], color='mediumorchid')
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
            self.vertexMap[self.triangles[triIdx, iVIdx]].append([triIdx, iVIdx])
        for vIdx in self.triangles[triIdx]:
            self.pointTopologyChanged[vIdx] = True
        for vIdx in self.triangles[triIdx]:
            for otherIdx, oppIVIdx in self.vertexMap[vIdx]:
                self.updateEdgeTopology(otherIdx)

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

    def createPoint(self, p: Point):
        invalids = self.invalidVertIdxs()
        if len(invalids) == 0:
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

    def updateGeometricProblems(self):

        # remove all geometric problems, whose face set has experienced a change
        for gpiIdx in reversed(range(len(self.geometricProblems))):
            hasToBeRemoved = False
            for triIdx in self.geometricProblems[gpiIdx].triIdxs:
                if self.triangleChanged[triIdx]:
                    hasToBeRemoved = True
                    break
            if hasToBeRemoved:
                self.geometricProblems.pop(gpiIdx)

        # add all changed bad triangles
        for triIdx in np.where(self.triangleChanged)[0]:
            if self.isValidTriangle[triIdx] and self.badTris[triIdx]:
                self.geometricProblems.append(self.getFaceAsEnclosement(triIdx))

        self.rebaseTriangleState()

        # add all faces around a single steinerpoint if it changed
        for vIdx in self.validVertIdxs():
            if self.pointTopologyChanged[vIdx] and vIdx >= self.instanceSize:
                self.geometricProblems.append(self.getEnclosementOfLink([vIdx]))
            self.pointTopologyChanged[vIdx] = False

        for triIdx in self.validTriIdxs():
            for i in range(3):
                if self.edgeTopologyChanged[triIdx, i]:
                    if self.triangles[triIdx, (i + 1) % 3] >= self.instanceSize and self.triangles[
                        triIdx, (i + 2) % 3] >= self.instanceSize:
                        # prevent doublecounting
                        if self.triangleMap[triIdx, i, 0] > triIdx:
                            self.geometricProblems.append(self.getEnclosementOfLink(
                                [self.triangles[triIdx, (i + 1) % 3], self.triangles[triIdx, (i + 2) % 3]]))
                self.edgeTopologyChanged[triIdx, i] = False
        #np.random.shuffle(self.geometricProblems)

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

    def addPoint(self, p: Point):

        # first figure out, in which triangle the point lies. If inside, split into three, if on, split adjacent
        # faces into two each
        hitTriIdxs = []
        grazedTriIdxs = []
        for triIdx in self.validTriIdxs():
            tri = self.triangles[triIdx]
            sides = np.array([eg.onWhichSide(Segment(self.point(self.triangles[triIdx, (i + 1) % 3]),
                                                     self.point(self.triangles[triIdx, (i + 2) % 3])), p) for i in
                              range(3)])
            if np.all((sides == "left")) or np.all((sides == "right")):
                hitTriIdxs.append([triIdx])
            elif np.all((sides == "left") | (sides == "colinear")) or np.all(
                    (sides == "right") | (sides == "colinear")):
                grazedTriIdxs.append([triIdx, np.argwhere(sides == "colinear")])
        if len(hitTriIdxs) == 1:
            # inside
            assert (len(grazedTriIdxs) == 0)
            hitTriIdx = hitTriIdxs[0][0]
            hitTri = self.triangles[hitTriIdx]

            newPointIdx = self.createPoint(p)
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

            for triIdx in self.validTriIdxs():
                for i in range(3):
                    if not (self.triangleMap[triIdx, i, 0] != noneFace):
                        print("och man")
            self.ensureDelauney([hitTriIdx, newLeftIdx, newRightIdx])
            # self.ensureDelauney(None)

            # self.plotTriangulation()

            return True
        elif len(grazedTriIdxs) == 0:
            # outside
            return False
        elif len(grazedTriIdxs) == 1:
            # boundary
            assert (len(grazedTriIdxs[0][1]) == 1)
            grazedIdx = grazedTriIdxs[0][0]
            grazed = self.triangles[grazedIdx]
            grazedIVIdx = grazedTriIdxs[0][1][0][0]

            newPointIdx = self.createPoint(p)
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

            self.ensureDelauney([grazedIdx, newTriIdx])
            # self.ensureDelauney(None)

            # self.plotTriangulation()

            return True
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

            newPointIdx = self.createPoint(p)
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

            self.ensureDelauney([grazedAIdx, grazedBIdx, newTriByBIdx, newTriByAIdx])
            # self.ensureDelauney(None)

            # self.plotTriangulation()

            return True
        else:
            # vertex
            return False

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
        return GeometricSubproblem(vIdxs, insideFaces, link, self.exactVerts[vIdxs + link],
                                   self.numericVerts[vIdxs + link], self.segments[insideConstraints],
                                   self.segments[boundaryConstraints], boundaryConstraintTypes, self.instanceSize,
                                   numBad, self.gpaxs)

    def getFaceAsEnclosement(self, triIdx):
        tri = self.triangles[triIdx]
        segmentIds = []
        for _, _, segmentId in self.triangleMap[triIdx]:
            if segmentId != noneEdge:
                segmentIds.append(segmentId)
        return GeometricSubproblem([], [triIdx], tri, self.exactVerts[tri], self.numericVerts[tri], [],
                                   self.segments[segmentIds], self.segmentType[segmentIds], self.instanceSize,
                                   1 if self.isBad(triIdx) else 0, self.gpaxs)

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
            triIdx, opp = self.vertexMap[vIdx][0]
            target = self.triangles[triIdx, (opp + 1) % 3]

        logging.debug("merging " + str(vIdx) + " into " + str(target))
        self.internalMergePoints(vIdx, target)
        self.ensureDelauney([tT for tT in touchedTriangles if self.isValidTriangle[tT]])



    def replaceEnclosement(self, gs: GeometricSubproblem, solution):
        if len(gs.insideSteiners) == 0 and len(solution) == 1:
            self.addPoint(solution[0])
        elif len(gs.insideSteiners) > 0:

            for vIdx in reversed(sorted(gs.getInsideSteiners())):
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
                self.addPoint(point)
            else:
                self.ensureDelauney(None)

            pass
        # 1. unlink all steinerpoints in the inside
        # 2. reuse unlinked steinerpoints and triangles to add the points from the solution
        pass

    def findComplicatedCenter(self, triIdx):
        withPlot = (self.internalaxs != None)
        if withPlot:
            self.internalaxs.clear()
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

        #we now traverse the voronoi diagram figuring out all local edges that intersect the circle
        handledFacePairs = []
        encounteredSegments = []
        closestPointToIntersectingSegment = []
        intersectingSegmentsAsFacePairs = []
        intersectingSegments = []
        candidateFaces = [vIdx for vIdx in self.triangles[triIdx]]
        alreadyHandled = []

        def hasBeenOrWillBeHandled(qFace):
            if qFace in alreadyHandled:
                return True
            if qFace in candidateFaces:
                return True
            return False

        def isInEdgeSet(edgeSet,qEdge):
            for edge in edgeSet:
                if (edge[0] == qEdge[0] and edge[1] == qEdge[1]) or (edge[0] == qEdge[1] and edge[1] == qEdge[0]):
                    return True
            return False

        while len(candidateFaces)>0:
            #construct face
            candidateFace = candidateFaces.pop(0)
            alreadyHandled.append(candidateFace)
            faceSegments = []
            originInternal = []
            opposingFace = []
            for actualTriIdx,internal in self.vertexMap[candidateFace]:
                for i in range(1,3):
                    #if not hasBeenOrWillBeHandled(self.triangles[actualTriIdx,(internal+i)%3]):
                    #    candidateFaces.append(self.triangles[actualTriIdx,(internal+i)%3])
                    candidateEdge = [actualTriIdx,self.triangleMap[actualTriIdx,(internal + i)%3,0]]
                    if not isInEdgeSet(faceSegments,candidateEdge):
                        opposingFace.append(self.triangles[actualTriIdx,(internal + (3-i))%3])
                        faceSegments.append(candidateEdge)
                        originInternal.append((internal + i)%3)

            for i in range(len(faceSegments)):
                origin,target = faceSegments[i]
                if target != outerFace and isInEdgeSet(handledFacePairs,[origin,target]):
                    continue
                handledFacePairs.append([origin,target])
                internal = originInternal[i]
                oppFace = opposingFace[i]#this is a vertex index
                if (segId := self.triangleMap[origin,internal,2]) != noneEdge:
                    if segId not in encounteredSegments:
                        if eg.segmentIntersectsCircle(myCC,myCRsq,Segment(self.point(self.segments[segId,0]),self.point(self.segments[segId,1]))):
                            encounteredSegments.append(segId)
                            p = self.numericVerts[self.segments[segId][0]]
                            q = self.numericVerts[self.segments[segId][1]]
                            if withPlot:
                                self.internalaxs.plot([p[0],q[0]],[p[1],q[1]],color="black")
                if target == outerFace:
                    diff = self.point(self.triangles[origin,(internal+1)%3]) - self.point(self.triangles[origin,(internal+2)%3])
                    voronoiOrth = Point(FieldNumber(0) - diff.y(), diff.x())
                    if eg.dot(voronoiOrth,self.point(self.triangles[origin,(internal+1)%3]) - Point(*self.circumCenters[origin])) < FieldNumber(0):
                        voronoiOrth = voronoiOrth.scale(FieldNumber(-1))
                    #voronoiOrth is now the direction pointing towards the defining segment
                    inter = eg.supportingRayIntersectSegment(Segment(Point(*self.circumCenters[origin]),Point(*self.circumCenters[origin])+voronoiOrth),Segment(self.point(self.triangles[origin,(internal+1)%3]),self.point(self.triangles[origin,(internal+2)%3])))
                    if inter == None:
                        seg1 = Segment(Point(*self.circumCenters[origin]),Point(*self.circumCenters[origin])+voronoiOrth)
                        seg2 = Segment(self.point(self.triangles[origin,(internal+1)%3]),self.point(self.triangles[origin,(internal+2)%3]))
                        self.internalaxs.plot([float(seg1.source().x()),float(seg1.target().x())],[float(seg1.source().y()),float(seg1.target().y())],color="red",zorder=1000000)
                        self.internalaxs.plot([float(seg2.source().x()),float(seg2.target().x())],[float(seg2.source().y()),float(seg2.target().y())],color="red",zorder=1000000)
                    inter = eg.roundExactOnSegment(Segment(self.point(self.triangles[origin,(internal+1)%3]),self.point(self.triangles[origin,(internal+2)%3])),inter)
                    assert(inter is not None)
                    actualSeg = Segment(Point(*self.circumCenters[origin]),inter)
                    if eg.segmentIntersectsCircle(myCC,myCRsq,actualSeg):
                        if not hasBeenOrWillBeHandled(oppFace):
                            candidateFaces.append(oppFace)
                        intersectingSegments.append(["segment",actualSeg])
                        intersectingSegmentsAsFacePairs.append([origin,target])
                        closestPointToIntersectingSegment.append(candidateFace)
                        numActualSeg = [[float(p.x()), float(p.y())] for p in [actualSeg.source(), actualSeg.target()]]
                        if withPlot:
                            self.internalaxs.scatter([numActualSeg[0][0], numActualSeg[1][0]],
                                             [numActualSeg[0][1], numActualSeg[1][1]], marker='*', color='lime',
                                             zorder=99)
                            self.internalaxs.plot([numActualSeg[0][0], numActualSeg[1][0]],
                                          [numActualSeg[0][1], numActualSeg[1][1]], color='lime', zorder=99)
                            pass

                else:
                    actualSeg = Segment(Point(*self.circumCenters[origin]),Point(*self.circumCenters[target]))
                    #print(origin,target)
                    if eg.segmentIntersectsCircle(myCC,myCRsq,actualSeg):
                        if not hasBeenOrWillBeHandled(oppFace):
                            candidateFaces.append(oppFace)
                        intersectingSegments.append(["ray",actualSeg])
                        intersectingSegmentsAsFacePairs.append([origin,target])
                        closestPointToIntersectingSegment.append(candidateFace)
                        numActualSeg = [[float(p.x()),float(p.y())] for p in [actualSeg.source(),actualSeg.target()]]
                        if withPlot:
                            self.internalaxs.scatter([numActualSeg[0][0],numActualSeg[1][0]], [numActualSeg[0][1],numActualSeg[1][1]],marker='*', color='green', zorder=100)
                            self.internalaxs.plot([numActualSeg[0][0],numActualSeg[1][0]], [numActualSeg[0][1],numActualSeg[1][1]], color='green', zorder=100)
                            pass


        candidateIntersections = []
        roundable=[]
        for i in range(len(intersectingSegments)):
            type,seg = intersectingSegments[i]
            closestPoint = self.point(closestPointToIntersectingSegment[i])

            #intersect segment with the three boundingbox segments, the circle and ever encountered segment.
            if type == "segment":
                #add the points themselves
                #TODO: remove double conting here, but not tooo important
                candidateIntersections.append([eg.distsq(closestPoint,seg.source()),seg.source()])
                roundable.append(True)
                candidateIntersections.append([eg.distsq(closestPoint,seg.target()),seg.target()])
                roundable.append(True)

            else:
                # add the points themselves
                # TODO: remove double conting here, but not tooo important
                candidateIntersections.append([eg.distsq(closestPoint, seg.source()), seg.source()])
                roundable.append(True)
                candidateIntersections.append([eg.distsq(closestPoint, seg.target()), seg.target()])
                roundable.append(False)


            #first intersect with boundingbox
            for bbseg in boundingSegments:
                inter = eg.innerIntersect(seg.source(),seg.target(),bbseg.source(),bbseg.target())
                if inter != None:
                    candidateIntersections.append([eg.distsq(closestPoint,inter),inter])
                    roundable.append(True)
                    if withPlot:
                        self.internalaxs.scatter([float(inter[0])], [float(inter[1])], marker="*", color='blue', zorder=100)

            #next intersect with encountered segments
            for encseg in encounteredSegments:
                inter = eg.innerIntersect(self.point(self.segments[encseg,0]),self.point(self.segments[encseg,1]),seg.source(),seg.target())
                if inter != None:
                    inter = eg.roundExactOnSegment(Segment(self.point(self.segments[encseg,0]),self.point(self.segments[encseg,1])),inter)
                    candidateIntersections.append([eg.distsq(closestPoint,inter),inter])
                    roundable.append(False)
                    if withPlot:
                        self.internalaxs.scatter([float(inter[0])], [float(inter[1])], marker="*", color='blue', zorder=100)

            #find intersections with circle. this is somehow really annoying
            p = seg.source()
            q = seg.target()
            inters = eg.insideIntersectionsSegmentCircle(myCC,myCRsq,seg)
            for inter in inters:
                candidateIntersections.append([eg.distsq(closestPoint,inter),inter])
                roundable.append(True)
                if withPlot:
                    self.internalaxs.scatter([float(inter[0])], [float(inter[1])], marker="*", color='blue', zorder=100)
                    pass

        candidatePoints = []
        roundablePoint = []
        for i in range(len(candidateIntersections)):
            d, p = candidateIntersections[i]
            pisRoundable = roundable[i]
            if withPlot:
                self.internalaxs.scatter([float(p[0])], [float(p[1])], marker="*", color='yellow', zorder=101)
            # outside
            if eg.distsq(myCC, p) > self.circumRadiiSqr[triIdx]:
                continue

            # outside slab
            if eg.dot(other - base, p - base) < FieldNumber(0):
                continue
            if eg.dot(base - other, p - other) < FieldNumber(0):
                continue
            if eg.dot(orth, p - base) < FieldNumber(0):
                continue

            candidatePoints.append([d, p])
            roundablePoint.append(pisRoundable)
            if withPlot:
                self.internalaxs.scatter([float(p[0])], [float(p[1])], marker="*", color='red', zorder=1000)

        # now all candidatePoints are guaranteed to lie inside the region. we need to check finally, if the three rays from base, other and far to the point intersect some segment
        actualInside = None
        actualDist = None
        actualRoundable = None
        for i in range(len(candidatePoints)):
            d,p = candidatePoints[i]
            pisRoundable = roundablePoint[i]
            intersects = False
            for i in range(3):
                triP = self.point(tri[i])
                for seg in self.segments:
                    segSource = self.point(seg[0])
                    segTarget = self.point(seg[1])
                    if (inter := eg.innerIntersect(segSource, segTarget, triP, p)) is not None:
                        intersects = True
            if not intersects:
                if actualDist is None or d > actualDist:
                    actualInside = Point(*p)
                    actualDist = d
                    actualRoundable = pisRoundable
        if withPlot:
            self.internalaxs.scatter([float(actualInside[0])], [float(actualInside[1])], marker="*", color='blue', zorder=1000)
            self.internalaxs.set_aspect('equal')
        if actualRoundable:
            acc = 10
            offset = FieldNumber(1)/FieldNumber(acc)
            for off in [Point(FieldNumber(0),FieldNumber(0)),Point(offset,FieldNumber(0)),Point(FieldNumber(0)-offset,FieldNumber(0)),Point(FieldNumber(0),offset),Point(FieldNumber(0),FieldNumber(0)-offset)]:
                newP = eg.roundExact(actualInside + off,acc)
                # outside
                if eg.distsq(myCC, newP) > self.circumRadiiSqr[triIdx]:
                    continue

                # outside slab
                if eg.dot(other - base, newP - base) < FieldNumber(0):
                    continue
                if eg.dot(base - other, newP - other) < FieldNumber(0):
                    continue
                if eg.dot(orth, newP - base) < FieldNumber(0):
                    continue

                logging.debug("rounded point!")
                actualInside = newP
                break

        return actualInside

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
        return dists


class QualityImprover:
    def __init__(self, tri: Triangulation,seed=None):
        self.tri = tri
        self.solver = StarSolver(2,1,1,1.25,2,1)
        if seed != None:
            np.random.seed(seed)


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

    def improve(self):
        keepGoing = True
        lastEdit = "None"
        plotUpdater = 0
        round = 0
        specialRounds = []
        numSteinerHistory = []
        numBadTriHistory = []
        lastImprovement = round
        bestSofar = 1000
        while keepGoing:
            logging.info(f"Round {round}: #Steiner = {len(self.tri.validVertIdxs()) - self.tri.instanceSize}, #>90° = {len( np.where(self.tri.badTris == True)[0])}, subproblems solved = {self.solver.succesfulSolves}, rep qual = {self.tri.getCoordinateQuality()}")
            numSteinerHistory.append(len(self.tri.validVertIdxs()) - self.tri.instanceSize)
            numBadTriHistory.append(len( np.where(self.tri.badTris == True)[0]))
            round += 1
            plotUpdater += 1
            # curEdit = "None"
            bestEval, bestSol, replacer = 0, None, None
            # self.tri.plotTriangulation()
            logging.debug("updating Geometric problems...")
            self.tri.updateGeometricProblems()
            np.random.shuffle(self.tri.geometricProblems)
            logging.debug("completed updating Geometric problems")
            logging.debug("drawing...")
            self.plotHistory(numSteinerHistory,numBadTriHistory,round,specialRounds,self.tri.histoaxs,self.tri.histoaxtwin)
            if plotUpdater == 1:
                #self.tri.plotCoordinateQuality()
                self.tri.plotTriangulation()
                plotUpdater = 0
            logging.debug("done drawing")
            self.tri.validateTriangleMap()
            logging.debug("scraping through "+str(len(self.tri.geometricProblems))+" many geometric subproblems")

            if len(np.where(self.tri.badTris)[0]) < bestSofar:
                bestSofar = len(np.where(self.tri.badTris)[0])
                lastImprovement = round

            #convergence safeguard
            if self.solver.patialTolerance > 0 and round - lastImprovement > 10:
                self.solver.patialTolerance = max(0,self.solver.patialTolerance - 1)
                logging.info(f"Tolerance set to {self.solver.patialTolerance}")
                lastImprovement = round
                specialRounds.append(round)

            #be less tolerant if half the triangles are gone
            if self.solver.patialTolerance > 1 and len(np.where(self.tri.badTris)[0]) <= max(1, numBadTriHistory[0] // 2):
                logging.info("Tolerance set to 1")
                self.solver.patialTolerance = 1
                specialRounds.append(round)

            #no longer be tolerant when only 10% of bad triangles are left
            if self.solver.patialTolerance > 0 and len(np.where(self.tri.badTris)[0]) <= max(1, numBadTriHistory[0] // 10):
                logging.info("Tolerance set to 0")
                self.solver.patialTolerance = 0
                specialRounds.append(round)

            for gp in self.tri.geometricProblems:
                # gp.plotMe()
                #start = time.time()
                eval, sol = self.solver.solve(gp)
                #print(f"{time.time()-start:.2f}")
                #print(")",end="")
                if eval != None and (bestEval == None or eval < bestEval):
                    logging.info("subproblem with eval " + str(eval) + " found")
                    bestEval = eval
                    bestSol = sol
                    replacer = gp
                    # curEdit = "altitude"

            logging.debug("done scraping")

            if bestEval == 0:
                badTris = list(np.where(self.tri.badTris == True)[0])

                if len(badTris) == 0:
                    keepGoing = False
                    continue
                added = False
                # we add complicated ungör erten center
                logging.debug("adding some complicated center...")
                #np.random.shuffle(badTris)
                dists = self.tri.combinatorialDepth()
                shallowest = np.min(np.array(dists)[badTris])
                locs = np.where((np.array(dists) <= shallowest+1) & (self.tri.badTris))[0]
                np.random.shuffle(locs)
                shallowIdx = locs[0]

                center = self.tri.findComplicatedCenter(shallowIdx)
                assert (center != None)
                if self.tri.addPoint(center):
                    logging.info("successfully added complicated Center of triangle " + str(shallowIdx) + " at depth " + str(dists[shallowIdx]))
                    lastEdit = "circumcenter"
                    added = True
                else:
                    logging.error("failed to add complicated Center of triangle " + str(shallowIdx) + " at depth " + str(dists[shallowIdx]))
            else:
                if self.tri.gpaxs != None:
                    replacer.plotMe()
                # lastEdit = curEdit
                logging.info("replacing identified subproblem")
                self.tri.replaceEnclosement(replacer, bestSol)
                logging.info("done replacing")
                self.tri.validateTriangleMap()
        self.tri.plotTriangulation()
        self.plotHistory(numSteinerHistory,numBadTriHistory,round,specialRounds,self.tri.histoaxs,self.tri.histoaxtwin)
        plt.pause(0.5)
        return self.tri.solutionParse()