import time

import matplotlib.pyplot as plt
import numpy as np
from cgshop2025_pyutils import Cgshop2025Solution, Cgshop2025Instance
from cgshop2025_pyutils.geometry import FieldNumber, Point, Segment

import exact_geometry as eg

import triangle as tr  # https://rufat.be/triangle/

def vprint(str, verbosity=0, vLevel=0, end="\n"):
    if verbosity > vLevel:
        print(str, end=end)

def convert(data):
    # convert to triangulation type
    points = np.column_stack((data.points_x, data.points_y))
    constraints = np.column_stack((data.region_boundary, np.roll(data.region_boundary, -1)))
    if (len(data.additional_constraints) != 0):
        constraints = np.concatenate((constraints, data.additional_constraints))
    A = dict(vertices=points, segments=constraints)
    return A

class Triangulation:
    def __init__(self, instance: Cgshop2025Instance,withValidate=False):
        self.outer = np.iinfo(int).max
        self.none = self.outer - 1
        self.withValidate = withValidate
        self.instanceSize = len(instance.points_x)
        self.exactVerts = []
        self.numericVerts = []
        for x, y in zip(instance.points_x, instance.points_y):
            self.exactVerts.append(Point(x, y))
            self.numericVerts.append([x, y])
        self.exactVerts = np.array(self.exactVerts, dtype=Point)
        self.numericVerts = np.array(self.numericVerts)
        Ain = tr.triangulate(convert(instance), 'p')
        self.segments = Ain['segments']
        self.triangles = Ain['triangles']
        self.instance_uid = instance.instance_uid

        self.vertexMap = [[] for v in self.exactVerts]
        for triIdx in range(len(self.triangles)):
            for vIdx in self.triangles[triIdx]:
                self.vertexMap[vIdx].append(triIdx)

        self.badTris = [idx for idx in range(len(self.triangles)) if self.isBad(idx)]

        self.localTopologyChanged = [True for v in self.exactVerts]

        self.voronoiEdges = [[] for tri in self.triangles]
        self.constrainedMask = [[] for tri in self.triangles]

        # temporary edgeSet for triangleNeighbourhoodMap
        fullMap = [[[] for j in self.exactVerts] for i in self.exactVerts]
        for i in range(len(self.triangles)):
            tri = self.triangles[i]
            for edge in [[tri[0], tri[1]], [tri[1], tri[2]], [tri[2], tri[0]]]:
                fullMap[edge[0]][edge[1]].append(i)
                fullMap[edge[1]][edge[0]].append(i)

        for i in range(len(self.triangles)):
            tri = self.triangles[i]
            for edge in [[tri[1], tri[2]], [tri[2], tri[0]], [tri[0], tri[1]]]:
                added = False
                for j in fullMap[edge[0]][edge[1]]:
                    if i != j:
                        self.voronoiEdges[i].append(j)
                        self.constrainedMask[i].append(self.getSegmentIdx(edge))
                        added = True
                if not added:
                    self.voronoiEdges[i].append(self.outer)
                    self.constrainedMask[i].append(self.getSegmentIdx(edge))
        self.constrainedMask = np.array(self.constrainedMask,dtype=int)
        self.voronoiEdges = np.array(self.voronoiEdges,dtype=int)

        self.circumCenters = [eg.circumcenter(*self.exactVerts[tri]) for tri in self.triangles]
        self.circumRadiiSqr = [eg.distsq(self.point(self.triangles[i][0]), self.circumCenters[i]) for i in range(len(self.triangles))]

        self.ensureDelauney()

        self.validate()

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

    def markTriangle(self, mark, axs, withNeighbor=True):
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
                axs.plot([float(self.circumCenters[mark].x()), float(self.circumCenters[i].x())],
                         [float(self.circumCenters[mark].y()), float(self.circumCenters[i].y())], zorder=1000,
                         color="yellow")

    def plotTriangulation(self, axs, mark=None):
        SC = len(self.exactVerts) - self.instanceSize
        name = ""
        badCount = 0
        if SC > 0:
            name += " [SC:" + str(SC) + "]"

        # axs.scatter([p[0] for p in self.numericVerts],[p[1] for p in self.numericVerts],marker=".")

        for tri in self.triangles:
            cords = self.numericVerts[tri]
            cords = np.concatenate((cords, [cords[0]]))
            axs.plot(*(cords.T), color='black', linewidth=1, zorder=98)
            if eg.isBadTriangle(*self.exactVerts[tri]):
                badCount += 1
                t = plt.Polygon(self.numericVerts[tri], color='b')
                axs.add_patch(t)
        name += " (>90°: " + str(badCount) + ")"

        if mark != None:
            for Idx in range(len(mark)):
                self.markTriangle(mark[Idx], axs, Idx == 0)

        for e in self.segments:
            axs.plot(*(self.numericVerts[e].T), color='red', linewidth=2, zorder=99)
        axs.scatter(*(self.numericVerts[:self.instanceSize].T), marker='.', color='black', zorder=100)
        axs.scatter(*(self.numericVerts[self.instanceSize:].T), marker='.', color='green', zorder=100)

        axs.set_aspect('equal')
        axs.title.set_text(name)

    def point(self,i:int):
        return Point(*self.exactVerts[i])

    def getSegmentIdx(self, querySeg):
        revseg = [querySeg[1], querySeg[0]]
        for i in range(len(self.segments)):
            seg = self.segments[i]
            if np.all(seg == querySeg) or np.all(seg == revseg):
                return i
        return self.none

    def setCircumCenter(self, idx):
        self.circumCenters[idx] = eg.circumcenter(*self.exactVerts[self.triangles[idx]])
        self.circumRadiiSqr[idx] = eg.distsq(self.point(self.triangles[idx][0]),self.circumCenters[idx])

    def unsetVertexMap(self, triIdx):
        for idx in self.triangles[triIdx]:
            self.vertexMap[idx].remove(triIdx)

    def setVertexMap(self, triIdx):
        for idx in self.triangles[triIdx]:
            self.vertexMap[idx].append(triIdx)
            self.localTopologyChanged[idx] = True

    def unsetBadness(self, triIdx):
        if self.isBad(triIdx):
            self.badTris.remove(triIdx)

    def setBadness(self, triIdx):
        if self.isBad(triIdx):
            self.badTris.append(triIdx)
        np.random.seed(133737)
        np.random.shuffle(self.badTris)

    def _unsafeRemapTriangle(self,idx,source,target):

        if idx != self.outer:
            self.unsetVertexMap(idx)
            self.unsetBadness(idx)
            self.triangles[idx] = np.where(self.triangles[idx] != source,self.triangles[idx],target)
            self.setVertexMap(idx)
            self.setCircumCenter(idx)
            self.setBadness(idx)

    def _unsafeRemapConstraint(self,idx,source,target):
        if idx != self.outer:
            self.constrainedMask[idx] = np.where(self.constrainedMask[idx] != source,self.constrainedMask[idx],target)

    def _unsafeRemapVoronoi(self,idx,source,target):
        if idx != self.outer:
            self.voronoiEdges[idx] = np.where(self.voronoiEdges[idx] != source,self.voronoiEdges[idx],target)

    def _unsafeRemapVoronoiAndConstraint(self, idx, sourceV, targetV, targetC):
        if idx != self.outer:
            self.constrainedMask[idx] = np.where(self.voronoiEdges[idx] != sourceV, self.constrainedMask[idx], targetC)
            self.voronoiEdges[idx] = np.where(self.voronoiEdges[idx] != sourceV, self.voronoiEdges[idx], targetV)

    def _unsafeCreateTriangle(self,tri,triVEdge,triMask):
        self.triangles = np.vstack((self.triangles, [tri]))
        self.voronoiEdges = np.vstack((self.voronoiEdges, [triVEdge]))
        self.constrainedMask = np.vstack((self.constrainedMask,[triMask]))

        myIdx = len(self.triangles)-1

        self.setVertexMap(myIdx)
        self.circumCenters.append(Point(FieldNumber(0), FieldNumber(0)))
        self.circumRadiiSqr.append(FieldNumber(0))
        self.setCircumCenter(myIdx)

        self.setBadness(myIdx)

    def _unsafeSetTriangle(self,triIdx,tri,triVedge,triMask):

        self.unsetVertexMap(triIdx)
        self.unsetBadness(triIdx)

        self.triangles[triIdx] = tri
        self.voronoiEdges[triIdx] = triVedge
        self.constrainedMask[triIdx] = triMask

        self.setVertexMap(triIdx)
        self.setCircumCenter(triIdx)
        self.setBadness(triIdx)

    def _unsafeCreateVertex(self,p:Point):
        self.exactVerts = np.vstack((self.exactVerts, [p]))
        self.numericVerts = np.vstack((self.numericVerts, [float(p.x()), float(p.y())]))
        self.vertexMap.append([])
        self.localTopologyChanged.append(True)

    def isBad(self, triIdx):
        return eg.isBadTriangle(*self.exactVerts[self.triangles[triIdx]])

    def validate(self):
        if self.withValidate:
            self.validateVoronoi()
            self.validateBadTris()
            self.validateConstraints()
            self.validateVertexMap()

    def validateVertexMap(self):
        for i in range(len(self.exactVerts)):
            for triIdx in self.vertexMap[i]:
                assert (i in self.triangles[triIdx])
        for triIdx in range(len(self.triangles)):
            for i in self.triangles[triIdx]:
                assert (triIdx in self.vertexMap[i])

    def validateConstraints(self):
        for idx in range(len(self.triangles)):
            tri = self.triangles[idx]
            for i in range(3):
                edge = [tri[i], tri[(i + 1) % 3]]
                segmentID = self.getSegmentIdx(edge)
                if segmentID == self.none:
                    assert (self.constrainedMask[idx][(i + 2) % 3] == self.none)
                else:
                    assert (segmentID == self.constrainedMask[idx][(i + 2) % 3])

    def validateBadTris(self):
        for idx in range(len(self.triangles)):
            if self.isBad(idx):
                assert (idx in self.badTris)
        for idx in self.badTris:
            assert (self.isBad(idx))
        seen = set()
        uniq = []
        for x in self.badTris:
            if x not in seen:
                seen.add(x)
            else:
                assert (False)

    def validateVoronoi(self):
        for triIdx in range(len(self.triangles)):
            ve = self.voronoiEdges[triIdx]
            for i in range(3):
                if ve[i] != self.outer:
                    assert (self.triangles[triIdx][(i + 1) % 3] in self.triangles[ve[i]])
                    assert (self.triangles[triIdx][(i + 2) % 3] in self.triangles[ve[i]])

    def _unsafeDeleteVertex(self, vIdx):
        self.exactVerts = np.delete(self.exactVerts, (vIdx), axis=0)
        self.numericVerts = np.delete(self.numericVerts, (vIdx), axis=0)
        self.vertexMap.pop(vIdx)

        # remap segments and triangles
        self.segments = np.where(self.segments>vIdx,self.segments-1,self.segments)
        self.triangles = np.where(self.triangles>vIdx,self.triangles-1,self.triangles)

    def _unsafeDeleteTriangle(self, triIdx):

        self.unsetVertexMap(triIdx)
        self.unsetBadness(triIdx)

        self.triangles = np.delete(self.triangles, (triIdx), axis=0)
        self.constrainedMask = np.delete(self.constrainedMask, (triIdx), axis=0)
        self.voronoiEdges = np.delete(self.voronoiEdges, (triIdx), axis=0)
        self.circumCenters.pop(triIdx)
        self.circumRadiiSqr.pop(triIdx)

        # remap vertexmap, voronoiEdges and badTris
        for idx in range(len(self.exactVerts)):
            newVMap = []
            for i in self.vertexMap[idx]:
                if i > triIdx:
                    newVMap.append(i - 1)
                elif i < triIdx:
                    newVMap.append(i)
                else:
                    assert (False)
            self.vertexMap[idx] = newVMap

        self.voronoiEdges = np.where((self.voronoiEdges != self.outer) & (self.voronoiEdges>triIdx),self.voronoiEdges-1,self.voronoiEdges)
        for idx in range(len(self.badTris)):
            if self.badTris[idx] > triIdx:
                self.badTris[idx] -= 1
            if self.badTris[idx] == triIdx:
                pass
                # print("uh oh")

    def _unsafeDeleteSegment(self, segIdx):
        self.segments = np.delete(self.segments, (segIdx), axis=0)
        self.constrainedMask = np.where((self.constrainedMask != self.none) & (self.constrainedMask > segIdx),self.constrainedMask-1,self.constrainedMask)

    def flipTrianglePair(self, triAIdx, triBIdx):

        # As perspective
        assert (triBIdx in self.voronoiEdges[triAIdx])
        neighbourAIdx = None
        for nAIdx in range(len(self.voronoiEdges[triAIdx])):
            if triBIdx == self.voronoiEdges[triAIdx][nAIdx]:
                neighbourAIdx = nAIdx
        assert (neighbourAIdx != None)
        assert (self.constrainedMask[triAIdx][neighbourAIdx] == self.none)

        # Bs perspective
        assert (triAIdx in self.voronoiEdges[triBIdx])
        neighbourBIdx = None
        for nBIdx in range(len(self.voronoiEdges[triBIdx])):
            if triAIdx == self.voronoiEdges[triBIdx][nBIdx]:
                neighbourBIdx = nBIdx
        assert (neighbourBIdx != None)
        assert (self.constrainedMask[triBIdx][neighbourBIdx] == self.none)

        triA = self.triangles[triAIdx]
        triB = self.triangles[triBIdx]

        if triA[(neighbourAIdx + 1) % 3] == triB[(neighbourBIdx + 1) % 3]:
            # same orientation of shared edge
            newA = [triA[neighbourAIdx], triB[neighbourBIdx], triA[(neighbourAIdx + 2) % 3]]
            newAVedge = [self.voronoiEdges[triBIdx][(neighbourBIdx + 1) % 3],
                         self.voronoiEdges[triAIdx][(neighbourAIdx + 1) % 3], triBIdx]
            newAVedgeMask = [self.constrainedMask[triBIdx][(neighbourBIdx + 1) % 3],
                             self.constrainedMask[triAIdx][(neighbourAIdx + 1) % 3], self.none]

            newB = [triA[neighbourAIdx], triB[neighbourBIdx], triA[(neighbourAIdx + 1) % 3]]
            newBVedge = [self.voronoiEdges[triBIdx][(neighbourBIdx + 2) % 3],
                         self.voronoiEdges[triAIdx][(neighbourAIdx + 2) % 3], triAIdx]
            newBVedgeMask = [self.constrainedMask[triBIdx][(neighbourBIdx + 2) % 3],
                             self.constrainedMask[triAIdx][(neighbourAIdx + 2) % 3], self.none]

            self._unsafeRemapVoronoi(self.voronoiEdges[triAIdx][(neighbourAIdx + 2) % 3], triAIdx, triBIdx)
            self._unsafeRemapVoronoi(self.voronoiEdges[triBIdx][(neighbourBIdx + 1) % 3], triBIdx, triAIdx)

            self._unsafeSetTriangle(triAIdx, newA, newAVedge, newAVedgeMask)
            self._unsafeSetTriangle(triBIdx, newB, newBVedge, newBVedgeMask)


        else:
            # different orientation of shared edge
            newA = [triA[neighbourAIdx], triB[neighbourBIdx], triA[(neighbourAIdx + 1) % 3]]
            newAVedge = [self.voronoiEdges[triBIdx][(neighbourBIdx + 1) % 3],
                         self.voronoiEdges[triAIdx][(neighbourAIdx + 2) % 3], triBIdx]
            newAVedgeMask = [self.constrainedMask[triBIdx][(neighbourBIdx + 1) % 3],
                             self.constrainedMask[triAIdx][(neighbourAIdx + 2) % 3], self.none]

            newB = [triA[neighbourAIdx], triB[neighbourBIdx], triA[(neighbourAIdx + 2) % 3]]
            newBVedge = [self.voronoiEdges[triBIdx][(neighbourBIdx + 2) % 3],
                         self.voronoiEdges[triAIdx][(neighbourAIdx + 1) % 3], triAIdx]
            newBVedgeMask = [self.constrainedMask[triBIdx][(neighbourBIdx + 2) % 3],
                             self.constrainedMask[triAIdx][(neighbourAIdx + 1) % 3], self.none]

            self._unsafeRemapVoronoi(self.voronoiEdges[triAIdx][(neighbourAIdx + 1) % 3], triAIdx, triBIdx)
            self._unsafeRemapVoronoi(self.voronoiEdges[triBIdx][(neighbourBIdx + 1) % 3], triBIdx, triAIdx)

            self._unsafeSetTriangle(triAIdx, newA, newAVedge, newAVedgeMask)
            self._unsafeSetTriangle(triBIdx, newB, newBVedge, newBVedgeMask)

    def ensureDelauney(self):

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
            j = self.voronoiEdges[i][jIdx]
            jMask = self.constrainedMask[i][jIdx]
            oppositeIndexInJ = None
            if jMask == self.none:
                onlyOn = True
                for v in range(3):
                    if self.triangles[j][v] not in self.triangles[i]:
                        oppositeIndexInJ = v
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

        self.validate()
        # they are stored as [triangleindex, inducing index]
        badEdgesInTriangleLand = []
        bannedEdges = []
        for i in range(len(self.triangles)):
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

            self.flipTrianglePair(i, j)

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
                    otherTriIdx = self.voronoiEdges[triIdx][badEdgesInTriangleLand[it][0][1]]
                    oppositeIt = None
                    for v in range(3):
                        if self.voronoiEdges[otherTriIdx][v] == triIdx:
                            oppositeIt = v
                    assert (oppositeIt != None)
                    badEdgesInTriangleLand[it].append([otherTriIdx, oppositeIt])
                else:
                    assert (False)

            for jIdx in range(3):
                _addEdgeToStack(i, jIdx)
            for iIdx in range(3):
                _addEdgeToStack(j, iIdx)
        self.validate()

    def dropAltitude(self, idx):
        tri = self.triangles[idx]
        if not eg.isBadTriangle(*self.exactVerts[tri]):
            # print("nowhere to drop from!")
            return False
            # assert(False)
        badIdx = eg.badAngle(*self.exactVerts[tri])
        if self.constrainedMask[idx][badIdx] == self.none:
            # print("nowhere to drop to!")
            return False
            # assert(False)
        otherIdx = self.voronoiEdges[idx][badIdx]

        # first things first, split the segment
        segment = [tri[(badIdx + 1) % 3], tri[(badIdx + 2) % 3]]
        segIdx = self.constrainedMask[idx][badIdx]

        # the point to be inserted on the segment
        ap = eg.altitudePoint(Segment(self.point(segment[0]), self.point(segment[1])),
                              self.exactVerts[tri[badIdx]])

        newPointIndex = len(self.exactVerts)

        self.segments[segIdx] = [segment[0], newPointIndex]
        self.segments = np.vstack((self.segments, [newPointIndex, segment[1]]))

        seg1Idx = segIdx
        seg2Idx = len(self.segments) - 1

        self._unsafeCreateVertex(ap)

        # now split both triangles attached to the split segment
        if otherIdx == self.outer:
            newTriIndex = len(self.triangles)
            # easy case!
            segAIdx = None
            segOtherIdx = None
            if tri[(badIdx + 1) % 3] in self.segments[seg1Idx]:
                segAIdx = seg1Idx
                segOtherIdx = seg2Idx
            else:
                segAIdx = seg2Idx
                segOtherIdx = seg1Idx

            newA = [tri[badIdx], tri[(badIdx + 1) % 3], newPointIndex]
            newAVedge = [self.outer, newTriIndex, self.voronoiEdges[idx][(badIdx + 2) % 3]]
            newAVedgeMask = [segAIdx, self.none, self.constrainedMask[idx][(badIdx + 2) % 3]]

            newTri = [tri[badIdx], newPointIndex, tri[(badIdx + 2) % 3]]
            newVedge = [self.outer, self.voronoiEdges[idx][(badIdx + 1) % 3], idx]
            newVedgeMask = [segOtherIdx, self.constrainedMask[idx][(badIdx + 1) % 3], self.none]

            self._unsafeRemapVoronoi(self.voronoiEdges[idx][(badIdx + 1) % 3], idx, newTriIndex)

            self._unsafeSetTriangle(idx, newA, newAVedge, newAVedgeMask)
            self._unsafeCreateTriangle(newTri, newVedge, newVedgeMask)

        else:
            # phew fuck me...
            newInsideIdx = len(self.triangles)
            newOutsideIdx = newInsideIdx + 1
            otherTri = self.triangles[otherIdx]
            opposingIdx = None
            for v in range(3):
                if not self.triangles[otherIdx][v] in tri:
                    opposingIdx = v
            assert (opposingIdx != None)

            segAIdx = None
            segOtherIdx = None
            if tri[(badIdx + 1) % 3] in self.segments[seg1Idx]:
                segAIdx = seg1Idx
                segOtherIdx = seg2Idx
            else:
                segAIdx = seg2Idx
                segOtherIdx = seg1Idx

            # if shared edge is oriented the same way
            if otherTri[(opposingIdx + 1) % 3] == tri[(badIdx + 1) % 3]:
                newA = [tri[badIdx], tri[(badIdx + 1) % 3], newPointIndex]
                newAVedge = [otherIdx, newInsideIdx, self.voronoiEdges[idx][(badIdx + 2) % 3]]
                newAMask = [segAIdx, self.none, self.constrainedMask[idx][(badIdx + 2) % 3]]

                newInside = [tri[badIdx], newPointIndex, tri[(badIdx + 2) % 3]]
                newInsideVedge = [newOutsideIdx, self.voronoiEdges[idx][(badIdx + 1) % 3], idx]
                newInsideMask = [segOtherIdx, self.constrainedMask[idx][(badIdx + 1) % 3], self.none]

                newB = [otherTri[opposingIdx], otherTri[(opposingIdx + 1) % 3], newPointIndex]
                newBVedge = [idx, newOutsideIdx, self.voronoiEdges[otherIdx][(opposingIdx + 2) % 3]]
                newBMask = [newAMask[0], self.none, self.constrainedMask[otherIdx][(opposingIdx + 2) % 3]]

                newOutside = [otherTri[opposingIdx], newPointIndex, otherTri[(opposingIdx + 2) % 3]]
                newOutsideVedge = [newInsideIdx, self.voronoiEdges[otherIdx][(opposingIdx + 1) % 3], otherIdx]
                newOutsideMask = [newInsideMask[0], self.constrainedMask[otherIdx][(opposingIdx + 1) % 3], self.none]

                self._unsafeRemapVoronoi(self.voronoiEdges[idx][(badIdx + 1) % 3], idx, newInsideIdx)
                self._unsafeRemapVoronoi(self.voronoiEdges[otherIdx][(opposingIdx + 1) % 3], otherIdx, newOutsideIdx)

                self._unsafeSetTriangle(idx, newA, newAVedge, newAMask)
                self._unsafeCreateTriangle(newInside, newInsideVedge, newInsideMask)

                self._unsafeSetTriangle(otherIdx, newB, newBVedge, newBMask)
                self._unsafeCreateTriangle(newOutside, newOutsideVedge, newOutsideMask)

            else:
                newA = [tri[badIdx], tri[(badIdx + 1) % 3], newPointIndex]
                newAVedge = [otherIdx, newInsideIdx, self.voronoiEdges[idx][(badIdx + 2) % 3]]
                newAMask = [segAIdx, self.none,
                            self.constrainedMask[idx][(badIdx + 2) % 3]]

                newInside = [tri[badIdx], newPointIndex, tri[(badIdx + 2) % 3]]
                newInsideVedge = [newOutsideIdx, self.voronoiEdges[idx][(badIdx + 1) % 3], idx]
                newInsideMask = [segOtherIdx,
                                 self.constrainedMask[idx][(badIdx + 1) % 3], self.none]

                newB = [otherTri[opposingIdx], otherTri[(opposingIdx + 2) % 3], newPointIndex]
                newBVedge = [idx, newOutsideIdx, self.voronoiEdges[otherIdx][(opposingIdx + 1) % 3]]
                newBMask = [newAMask[0], self.none, self.constrainedMask[otherIdx][(opposingIdx + 1) % 3]]

                newOutside = [otherTri[opposingIdx], newPointIndex, otherTri[(opposingIdx + 1) % 3]]
                newOutsideVedge = [newInsideIdx, self.voronoiEdges[otherIdx][(opposingIdx + 2) % 3], otherIdx]
                newOutsideMask = [newInsideMask[0], self.constrainedMask[otherIdx][(opposingIdx + 2) % 3], self.none]

                self._unsafeRemapVoronoi(self.voronoiEdges[idx][(badIdx + 1) % 3], idx, newInsideIdx)
                self._unsafeRemapVoronoi(self.voronoiEdges[otherIdx][(opposingIdx + 2) % 3], otherIdx, newOutsideIdx)

                self._unsafeSetTriangle(idx, newA, newAVedge, newAMask)
                self._unsafeCreateTriangle(newInside, newInsideVedge, newInsideMask)

                self._unsafeSetTriangle(otherIdx, newB, newBVedge, newBMask)
                self._unsafeCreateTriangle(newOutside, newOutsideVedge, newOutsideMask)

        self.ensureDelauney()
        return True

    def addPoint(self, p: Point):
        # first identify triangle hit by p
        hitTriIdx = None
        for triIdx in range(len(self.triangles)):
            sides = [eg.onWhichSide(Segment(self.point(self.triangles[triIdx][i]),
                                            self.point(self.triangles[triIdx][(i + 1) % 3])), p) for i in
                     range(3)]
            if np.all(sides == ["left", "left", "left"]) or np.all(sides == ["right", "right", "right"]):
                hitTriIdx = triIdx
                break
        if hitTriIdx == None:
            # print("not inside")
            return False

        hitTri = self.triangles[hitTriIdx]
        hitTriVedge = self.voronoiEdges[hitTriIdx]
        hitTriMask = self.constrainedMask[hitTriIdx]
        addedPointIdx = len(self.exactVerts)
        newLeftTriIdx = len(self.triangles)
        newRightTriIdx = newLeftTriIdx + 1

        newA = [addedPointIdx, hitTri[1], hitTri[2]]
        newAVedge = [hitTriVedge[0], newLeftTriIdx, newRightTriIdx]
        newAMask = [hitTriMask[0], self.none, self.none]

        newLeft = [hitTri[0], addedPointIdx, hitTri[2]]
        newLeftVedge = [hitTriIdx, hitTriVedge[1], newRightTriIdx]
        newLeftMask = [self.none, hitTriMask[1], self.none]

        newRight = [hitTri[0], hitTri[1], addedPointIdx]
        newRightVedge = [hitTriIdx, newLeftTriIdx, hitTriVedge[2]]
        newRightMask = [self.none, self.none, hitTriMask[2]]

        self._unsafeRemapVoronoi(hitTriVedge[1], hitTriIdx, newLeftTriIdx)
        self._unsafeRemapVoronoi(hitTriVedge[2], hitTriIdx, newRightTriIdx)

        # make sure, its a copy!
        self._unsafeCreateVertex(Point(FieldNumber(p.x().exact()), FieldNumber(p.y().exact())))

        self._unsafeSetTriangle(hitTriIdx, newA, newAVedge, newAMask)
        self._unsafeCreateTriangle(newLeft, newLeftVedge, newLeftMask)
        self._unsafeCreateTriangle(newRight, newRightVedge, newRightMask)

        self.ensureDelauney()

        return True

    def mergePoints(self, source, target):
        self.validate()
        # moves source index to target and deletes the triangles participating.
        # in reality we let some data dangle, to not fuck up all the maps
        participatingTris = [tri for tri in self.vertexMap[source] if
                             (tri != self.outer) and (target in self.triangles[tri])]
        sourceTris = [tri for tri in self.vertexMap[source] if (tri != self.outer) and (target not in self.triangles[tri])]

        # identify, if we are deleting something along a segment
        deleteSegIdx = None
        otherSegIdx = None
        targetSeg = None
        for segIdx in range(len(self.segments)):
            seg = self.segments[segIdx]
            if (source in seg) and (target in seg):
                deleteSegIdx = segIdx
            elif source in seg:
                otherSegIdx = segIdx
                if source == seg[0]:
                    targetSeg = [target, seg[1]]
                else:
                    targetSeg = [seg[0], target]

        # delete the segment
        if deleteSegIdx != None:
            if deleteSegIdx < otherSegIdx:
                # clean...
                temp = deleteSegIdx
                deleteSegIdx = otherSegIdx
                otherSegIdx = temp
            self.segments[otherSegIdx] = targetSeg
            self._unsafeDeleteSegment(deleteSegIdx)

            for triIdx in sourceTris:
                self._unsafeRemapConstraint(triIdx,deleteSegIdx,otherSegIdx)

            for triIdx in participatingTris:
                self._unsafeRemapConstraint(triIdx,deleteSegIdx,otherSegIdx)

        # change source to target for all source tris
        for triIdx in sourceTris:
            self._unsafeRemapTriangle(triIdx, source, target)

        # now for the ugly part: local and global remapping of the triangles to be deleted
        if len(participatingTris) == 2:

            leftTriIdx = participatingTris[0]
            rightTriIdx = participatingTris[1]

            leftTri = self.triangles[leftTriIdx]
            rightTri = self.triangles[rightTriIdx]

            leftSIdx = None
            leftTIdx = None

            rightSIdx = None
            rightTIdx = None

            for i in range(3):
                if leftTri[i] == source:
                    leftSIdx = i
                elif leftTri[i] == target:
                    leftTIdx = i

            for i in range(3):
                if rightTri[i] == source:
                    rightSIdx = i
                elif rightTri[i] == target:
                    rightTIdx = i

            # self.validateVoronoi()

            leftUpper = self.voronoiEdges[leftTriIdx][leftSIdx]
            leftLower = self.voronoiEdges[leftTriIdx][leftTIdx]

            self._unsafeRemapVoronoiAndConstraint(leftLower,leftTriIdx,leftUpper,self.constrainedMask[leftTriIdx][leftSIdx])
            self._unsafeRemapVoronoi(leftUpper,leftTriIdx,leftLower)

            rightUpper = self.voronoiEdges[rightTriIdx][rightSIdx]
            rightLower = self.voronoiEdges[rightTriIdx][rightTIdx]

            self._unsafeRemapVoronoiAndConstraint(rightLower,rightTriIdx,rightUpper,self.constrainedMask[rightTriIdx][rightSIdx])
            self._unsafeRemapVoronoi(rightUpper,rightTriIdx,rightLower)


            # remove triangles
            for deleteIdx in [max(leftTriIdx, rightTriIdx), min(leftTriIdx, rightTriIdx)]:
                self._unsafeDeleteTriangle(deleteIdx)

            # remove source-vertex
            self._unsafeDeleteVertex(source)

            self.validate()

        elif len(participatingTris) == 1:
            leftTriIdx = participatingTris[0]

            leftTri = self.triangles[leftTriIdx]

            leftSIdx = None
            leftTIdx = None

            for i in range(3):
                if leftTri[i] == source:
                    leftSIdx = i
                elif leftTri[i] == target:
                    leftTIdx = i

            leftUpper = self.voronoiEdges[leftTriIdx][leftSIdx]
            leftLower = self.voronoiEdges[leftTriIdx][leftTIdx]

            self._unsafeRemapVoronoiAndConstraint(leftLower,leftTriIdx,leftUpper,self.constrainedMask[leftTriIdx][leftSIdx])
            self._unsafeRemapVoronoi(leftUpper,leftTriIdx,leftLower)

            # remove triangles
            self._unsafeDeleteTriangle(leftTriIdx)

            # remove source-vertex
            self._unsafeDeleteVertex(source)

            self.validate()
        else:
            assert (False)

    def getLinkAroundVertex(self,v):
        # build Link
        link = []
        triIndices = []
        constraint = []

        startedAtDummy = False

        curTriIdx = self.vertexMap[v][0]
        curSelfIdx = None
        for i in range(3):
            if self.triangles[curTriIdx][i] == v:
                curSelfIdx = i
        curSelfIdx = (curSelfIdx + 1) % 3
        curIdx = self.triangles[curTriIdx][curSelfIdx]

        while len(link) == 0 or curIdx != link[0]:

            if (self.voronoiEdges[curTriIdx][curSelfIdx] == self.outer) and (startedAtDummy == False):
                # restart from this triangle
                startedAtDummy = True

                for i in range(3):
                    if self.triangles[curTriIdx][i] == curIdx or self.triangles[curTriIdx][i] == v:
                        continue
                    curSelfIdx = i
                curIdx = self.triangles[curTriIdx][curSelfIdx]
                triIndices = []
                link = [self.outer]
            elif (self.voronoiEdges[curTriIdx][curSelfIdx] == self.outer) and (startedAtDummy == True):
                # should be done
                link.append(self.triangles[curTriIdx][curSelfIdx])
                for i in range(3):
                    if self.triangles[curTriIdx][i] == curIdx or self.triangles[curTriIdx][i] == v:
                        continue
                    link.append(self.triangles[curTriIdx][i])
                triIndices.append(curTriIdx)

                curIdx = self.outer
            else:
                link.append(curIdx)
                triIndices.append(curTriIdx)

                steppedOverConstraint = False

                if self.constrainedMask[curTriIdx][curSelfIdx] != self.none:
                    # must have hit an edge
                    steppedOverConstraint = True

                oldIdx = curIdx
                curIdx = None
                for i in range(3):
                    if self.triangles[curTriIdx][i] == oldIdx or self.triangles[curTriIdx][i] == v:
                        continue
                    curIdx = self.triangles[curTriIdx][i]

                # step to next triangle
                curTriIdx = self.voronoiEdges[curTriIdx][curSelfIdx]

                for i in range(3):
                    if self.triangles[curTriIdx][i] == curIdx:
                        curSelfIdx = i

                if steppedOverConstraint:
                    constraint.append(curIdx)
        if link[0] == self.outer:
            link = link[1:]
            constraint = [link[0], link[len(link) - 1]]
        return link,constraint,triIndices

    def moveSteinerpoint(self, ignoreBadness=False, mainAx=None):
        globalMoved = False
        badMap = [[] for v in self.exactVerts]

        for triIdx in self.badTris:
            for vIdx in self.triangles[triIdx]:
                badMap[vIdx].append(triIdx)

        for idx in range(self.instanceSize, len(self.exactVerts)):
            if self.localTopologyChanged[idx] == False:
                continue

            self.localTopologyChanged[idx] = False
            # attempt to move the steinerpoint with index idx
            moved = False
            onlyVertex = True
            doWiggle = ignoreBadness
            oldPos = self.numericVerts[idx]
            for triIdx in badMap[idx]:
                if self.isBad(triIdx):
                    doWiggle = True
                    onlyVertex = False
                    break
            if doWiggle:



                exactPoints = []
                intrinsicConstraint = []

                link,constraint,enclosedTriangles = self.getLinkAroundVertex(idx)

                for i in range(len(link)):
                    linkIdx = link[i]
                    exactPoints.append(self.point(linkIdx))
                    if linkIdx in constraint:
                        intrinsicConstraint.append(i)
                solType = None
                cs = None
                if len(intrinsicConstraint) == 2:
                    if onlyVertex:
                        solType, cs = eg.findVertexCenterOfLinkConstrained(exactPoints,[intrinsicConstraint[0],
                                                              intrinsicConstraint[1]])
                    else:
                        solType, cs = eg.findCenterOfLinkConstrained(exactPoints, intrinsicConstraint[0],
                                                              intrinsicConstraint[1])
                elif len(intrinsicConstraint) == 0:
                    if onlyVertex:
                        solType, cs = eg.findVertexCenterOfLink(exactPoints)
                    else:
                        solType, cs = eg.findCenterOfLink(exactPoints)
                else:
                    pass
                    # print("uh oh!")
                if solType == "inside":
                    # move point!
                    if not ignoreBadness:
                        moved = True
                    else:
                        moved = len(badMap[idx]) > 0
                    if moved:
                        for i in link:
                            assert( i != self.outer)
                            self.localTopologyChanged[i] = True
                    # validate point
                    onlyLeft = True
                    onlyRight = True

                    a = Point(FieldNumber(cs[0].x().exact()), FieldNumber(cs[0].y().exact()))
                    for i in range(len(link)):
                        b = self.point(link[i])
                        c = self.point(link[(i + 1) % len(link)])
                        side = eg.onWhichSide(Segment(b, c), a)
                        if side == "left":
                            onlyRight = False
                        elif side == "right":
                            onlyLeft = False
                    if (onlyLeft == False) and (onlyRight == False):
                        print("oh no...")

                    self.exactVerts[idx] = Point(FieldNumber(cs[0].x().exact()), FieldNumber(cs[0].y().exact()))
                    self.numericVerts[idx] = [float(cs[0].x()), float(cs[0].y())]
                    for triIndex in enclosedTriangles:
                        self.setCircumCenter(triIndex)
                        self.unsetBadness(triIndex)
                    # self.validate()
                elif solType == "vertex":
                    # identiy the two triangles that can die
                    # if onBoundary:
                    # print("special case")
                    for i in link:
                        assert( i != "dummy")
                        self.localTopologyChanged[i] = True
                    self.mergePoints(idx, link[cs])
                    # self.validate()
                    moved = True
                    globalMoved = True
                    break
                elif solType == "None":
                    pass
                    # print("has no center")

            if moved:
                globalMoved = True
        if globalMoved:
            # self.validate()
            self.ensureDelauney()
        return globalMoved

    def improveQuality(self, axs=None, verbosity=0):
        print(verbosity)
        movingTime = 0
        droppingTime = 0
        addingTime = 0
        finalWiggle = False
        for round in range(5000):

            # plot solution
            if axs != None and (round %1) == 0:
                axs.clear()
                self.plotTriangulation(axs)
                plt.draw()
                plt.pause(0.001)

            # attempt moving every vertex to locally remove bad triangles
            start = time.time()
            vprint("moving...", end="", verbosity=verbosity)
            movedSomething = self.moveSteinerpoint(ignoreBadness=True, mainAx=axs)
            movingTime += time.time() - start
            vprint("success" if movedSomething else "failure", verbosity=verbosity)

            # if we locally improved the solution, try moving more stuff around
            if movedSomething:
                continue

            # attempt to drop an altitude onto a constraint
            start = time.time()
            added = False
            vprint("dropping...", end="", verbosity=verbosity)
            for triIdx in self.badTris:
                if self.dropAltitude(triIdx):
                    added = True
                    break
            vprint("success" if added else "failure", verbosity=verbosity)
            droppingTime += time.time() - start

            # if we added something, start next round, trying to locally improve the point locations again
            if added:
                continue

            # figure out, if we solved the instance
            if len(self.badTris) == 0:
                if not finalWiggle:

                    # if this is the first time we have found a solution, attempt to move every point again, potentially
                    # removing steinerpoints (via starting the next round)
                    finalWiggle = True
                    self.localTopologyChanged = [True for v in self.exactVerts]
                    continue
                else:

                    # otherwise we are done!
                    if axs != None:
                        axs.clear()
                        self.plotTriangulation(axs)
                        plt.draw()
                        plt.pause(0.001)
                    print(movingTime, droppingTime, addingTime, end="")
                    return self.solutionParse()
            start = time.time()

            withRounding = True
            if withRounding:
                # if we end up here, we attempt fixing a bad triangle by adding its circumcenter as a steiner point
                for triIdx in self.badTris:
                    vprint("adding rounded circumcenter of " + str(triIdx) + "...", end="",verbosity=verbosity)
                    added = self.addPoint(eg.roundExact(self.circumCenters[triIdx]))
                    if added:
                        if verbosity > 0:
                            print("success")
                        break
                    if not added and verbosity > 0:
                        print("failure")
            if added:
                addingTime += time.time() - start
                continue
            # if rounding didnt work, try to add an exact version
            for triIdx in self.badTris:
                vprint("adding exact circumcenter of " + str(triIdx) + " with representation length "+str(len(str(self.circumCenters[triIdx].x().exact())) + len(str(self.circumCenters[triIdx].y().exact()))) + "...", end="",verbosity=verbosity)
                added = self.addPoint(self.circumCenters[triIdx])
                if added:
                    if verbosity > 0:
                        print("success")
                    break
                if not added and verbosity > 0:
                    print("failure")

            # if we end up here, all circumcenter of bad triangles lie outside the bounding polygon, or on
            # an edge of the triangulation. In this case we simply add a point in the middle of the triangle. this will
            # later be moved to an advantageous location anyways, so we dont care
            if not added:
                # hacky solution for now
                for i in self.badTris:
                    vprint("adding centroid of " + str(i)+ "...", end="",verbosity=verbosity)
                    avg = Point(FieldNumber(0), FieldNumber(0))
                    for v in range(3):
                        avg += self.point(self.triangles[i][v])
                    avg = avg.scale(FieldNumber(1) / FieldNumber(3))
                    added = self.addPoint(avg)
                    if added:
                        vprint("success",verbosity=verbosity)
                        break
                    vprint("failure",verbosity=verbosity)

            if not added:
                #if we end up here, we are fucked
                print("huh")
            addingTime += time.time() - start
        return None
