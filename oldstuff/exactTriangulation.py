import time

import matplotlib.pyplot as plt
import numpy as np
from cgshop2025_pyutils import Cgshop2025Solution, Cgshop2025Instance
from cgshop2025_pyutils.geometry import FieldNumber, Point, Segment

import exact_geometry as eg

import Triangulation as newTri

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
    #####
    # in and out
    #####
    def __init__(self, instance: Cgshop2025Instance, withValidate=False, seed=None,axs=None):

        fig, triAxs = plt.subplots(1,1)
        self.tri = newTri.Triangulation(instance,axs=triAxs)

        self.seed = seed

        self.plotTime = 0.005

        self.outer = np.iinfo(int).max
        self.noneFace = self.outer - 1
        self.noneEdge = self.outer - 2
        self.noneVertex = self.outer - 3
        self.noneIntervalVertex = self.outer - 4

        self.withValidate = withValidate
        self.instanceSize = len(instance.points_x)
        self.exactVerts = []
        self.numericVerts = []
        for x, y in zip(instance.points_x, instance.points_y):
        #for x, y in zip(xs, ys):
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

        self.pointTopologyChanged = [True for v in self.exactVerts]

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
                        self.constrainedMask[i].append(self.getSegmentIdx(np.array(edge)))
                        added = True
                if not added:
                    self.voronoiEdges[i].append(self.outer)
                    self.constrainedMask[i].append(self.getSegmentIdx(np.array(edge)))
        self.constrainedMask = np.array(self.constrainedMask, dtype=int)
        self.voronoiEdges = np.array(self.voronoiEdges, dtype=int)

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

        self.segmentType = [False for seg in self.segments]
        for triIdx in range(len(self.triangles)):
            for i in range(3):
                if (edgeId := self.constrainedMask[triIdx, i]) != self.noneEdge:
                    if self.voronoiEdges[triIdx, i] == self.outer:
                        self.segmentType[edgeId] = True
        self.segmentType = np.array(self.segmentType)

        if axs is not None:
            axs.clear()
            axs.set_facecolor('lightgray')
            self.plotTriangulation(axs)
            plt.draw()
            plt.pause(self.plotTime)

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

    #####
    # self validation
    #####
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
                segmentID = self.getSegmentIdx(np.array(edge))
                if segmentID == self.noneEdge:
                    assert (self.constrainedMask[idx][(i + 2) % 3] == self.noneEdge)
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

    #####
    # visualization
    #####
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

        for triIdx in range(len(self.triangles)):
            tri = self.triangles[triIdx]
            cords = self.numericVerts[tri]
            cords = np.concatenate((cords, [cords[0]]))
            for i in range(3):
                if (self.triangles[triIdx][(i + 1) % 3] >= self.instanceSize) and (
                            self.triangles[triIdx][(i + 2) % 3] >= self.instanceSize) and  (self.edgeTopologyChanged[triIdx,i] or (self.voronoiEdges[triIdx,i] != self.outer and self.edgeTopologyChanged[self.voronoiEdges[triIdx,i],self.oppositeInternalIndex(triIdx,i)])):
                    axs.plot(*(cords[[(i+1)%3,(i+2)%3]].T), color='black', linewidth=1, zorder=98, linestyle="dotted")
                else:
                    axs.plot(*(cords[[(i+1)%3,(i+2)%3]].T), color='black', linewidth=1, zorder=98)
            if eg.isBadTriangle(*self.exactVerts[tri]):
                badCount += 1
                t = plt.Polygon(self.numericVerts[tri], color='mediumorchid')
                axs.add_patch(t)
            else:
                t = plt.Polygon(self.numericVerts[tri], color='palegoldenrod')
                axs.add_patch(t)
        name += " (>90°: " + str(badCount) + ")"

        if mark != None:
            for Idx in range(len(mark)):
                self.markTriangle(mark[Idx], axs, Idx == 0)

        for edgeId in range(len(self.segments)):
            e = self.segments[edgeId]
            t = self.segmentType[edgeId]
            color = 'indigo' if t else 'forestgreen'
            axs.plot(*(self.numericVerts[e].T), color=color, linewidth=2, zorder=99)
        min = 12
        max = 30
        sizes = np.array(self.pointTopologyChanged, dtype=int) * (max - min) + min
        axs.scatter(*(self.numericVerts[:self.instanceSize].T), s=min, color='black', zorder=100)
        axs.scatter(*(self.numericVerts[self.instanceSize:].T), s=sizes[self.instanceSize:], color='red', zorder=100)

        axs.set_aspect('equal')
        axs.title.set_text(name)

    #####
    # getters
    #####
    def point(self, i: int):
        return Point(*self.exactVerts[i])

    def isBad(self, triIdx):
        return eg.isBadTriangle(*self.exactVerts[self.triangles[triIdx]])

    def oppositeInternalIndex(self, triIdx, iVIdx):
        if (self.voronoiEdges[triIdx][iVIdx] == self.outer) or (self.voronoiEdges[triIdx][iVIdx] == self.noneFace):
            return self.noneIntervalVertex
        for i in range(3):
            if self.voronoiEdges[self.voronoiEdges[triIdx][iVIdx]][i] == triIdx:
                return i
        return self.noneIntervalVertex

    def oppositeInternalIndexOfPoint(self, triIdx, vIdx):
        for i in range(3):
            if self.triangles[triIdx] == vIdx:
                return self.oppositeInternalIndex(triIdx, i)
        return self.noneIntervalVertex

    def oppositeInternalIndexOfEdge(self, triIdx, pIdx, qIdx):
        for i in range(3):
            if (self.triangles[triIdx][i] != pIdx) and (self.triangles[triIdx][i] != qIdx):
                return i
        return self.noneIntervalVertex

    def getSegmentIdx(self, querySeg):
        locs = np.concatenate(
            (np.argwhere(np.all((self.segments == querySeg), -1)),
             np.argwhere(np.all((self.segments == querySeg[::-1]), -1))))
        if len(locs) == 1:
            return locs[0, 0]
        else:
            return self.noneEdge

    def trianglesOnEdge(self, pIdx, qIdx):
        return np.array([triIdx for triIdx in self.vertexMap[pIdx] if qIdx in self.triangles[triIdx]])

    def internalIndex(self, triIdx, vIdx):
        for i in range(3):
            if self.triangles[triIdx][i] == vIdx:
                return i
        return self.noneIntervalVertex

    #####
    # naive setters
    #####
    def setCircumCenter(self, idx):
        self.circumCenters[idx] = eg.circumcenter(*self.exactVerts[self.triangles[idx]])
        self.circumRadiiSqr[idx] = eg.distsq(self.point(self.triangles[idx][0]), self.circumCenters[idx])

        closest = None
        closestdist = None
        for vIdx in range(len(self.exactVerts)):
            dist = eg.distsq(self.circumCenters[idx], self.point(vIdx))
            if (closest is None) or (dist < closestdist):
                closest = vIdx
                closestdist = dist
        self.closestToCC[idx] = closest
        self.closestDist[idx] = closestdist

    def unsetVertexMap(self, triIdx):
        for idx in self.triangles[triIdx]:
            self.vertexMap[idx].remove(triIdx)

    def updateEdgeTopology(self,triIdx):
        for otherIdx in self.voronoiEdges[triIdx]:
            if otherIdx != self.outer and otherIdx != self.noneFace:
                for i in range(3):
                    self.edgeTopologyChanged[otherIdx,i] = True

    def setVertexMap(self, triIdx):
        for idx in self.triangles[triIdx]:
            self.vertexMap[idx].append(triIdx)
            self.pointTopologyChanged[idx] = True
        for vIdx in self.triangles[triIdx]:
            for otherIdx in self.vertexMap[vIdx]:
                self.updateEdgeTopology(otherIdx)

    def unsetBadness(self, triIdx):
        if triIdx in self.badTris:
            self.badTris.remove(triIdx)

    def setBadness(self, triIdx):
        if self.isBad(triIdx):
            self.badTris = self.badTris + [triIdx]
            dists = self.closestDist[self.badTris]
            args = dists.argsort()
            # self.badTris = list(np.array(self.badTris)[args[::-1]])
            if self.seed != None:
                np.random.seed(self.seed)
                np.random.shuffle(self.badTris)

    #####
    # unsafe modifiers
    #####
    # vertex
    def _unsafeCreateVertex(self, p: Point):
        self.exactVerts = np.vstack((self.exactVerts, [p]))
        self.numericVerts = np.vstack((self.numericVerts, [float(p.x()), float(p.y())]))
        self.vertexMap.append([])
        self.pointTopologyChanged = np.hstack((self.pointTopologyChanged, True))

        for triIdx in range(len(self.triangles)):
            dist = eg.distsq(self.circumCenters[triIdx], p)
            if (dist < self.closestDist[triIdx]):
                self.closestToCC[triIdx] = len(self.exactVerts) - 1
                self.closestDist[triIdx] = dist

    def _unsafeDeleteVertex(self, vIdx):
        self.exactVerts = np.delete(self.exactVerts, (vIdx), axis=0)
        self.numericVerts = np.delete(self.numericVerts, (vIdx), axis=0)
        self.pointTopologyChanged = np.delete(self.pointTopologyChanged, (vIdx), axis=0)
        self.vertexMap.pop(vIdx)

        # remap segments and triangles
        self.segments = np.where(self.segments > vIdx, self.segments - 1, self.segments)
        self.triangles = np.where(self.triangles > vIdx, self.triangles - 1, self.triangles)

        updateDists = []
        for triIdx in range(len(self.triangles)):
            if self.closestToCC[triIdx] == vIdx:
                updateDists.append(triIdx)

        for triIdx in updateDists:
            closest = None
            closestdist = None
            for vIdx in range(len(self.exactVerts)):
                dist = eg.distsq(self.circumCenters[triIdx], self.point(vIdx))
                if (closest is None) or (dist < closestdist):
                    closest = vIdx
                    closestdist = dist
            self.closestToCC[triIdx] = closest
            self.closestDist[triIdx] = closestdist

    def _unsafeUnlinkAndDeleteVertex(self, vIdx):
        assert (len(self.vertexMap[vIdx]) == 0)
        oldSeg = []
        segIds = []
        for segIdx in range(len(self.segments)):
            seg = self.segments[segIdx]
            if vIdx == seg[0]:
                oldSeg.append(seg[1])
                segIds.append(segIdx)
            elif vIdx == seg[1]:
                oldSeg.append(seg[0])
                segIds.append(segIdx)
        if len(segIds) == 2:
            self.segments[segIds[0]] = oldSeg
            self._unsafeDeleteSegment(segIds[1])
        elif len(segIds) != 0:
            assert (False)
        self._unsafeDeleteVertex(vIdx)

    def _superUnsafeUnshackleVertex(self, vIdx):
        oldSeg = []
        segIds = []
        for segIdx in range(len(self.segments)):
            seg = self.segments[segIdx]
            if vIdx == seg[0]:
                oldSeg.append(seg[1])
                segIds.append(segIdx)
            elif vIdx == seg[1]:
                oldSeg.append(seg[0])
                segIds.append(segIdx)
        if len(segIds) == 2:
            segIds = np.sort(segIds)
            self.segments[segIds[0]] = oldSeg
            self._unsafeDeleteSegment(segIds[1])
            for triIdx in self.vertexMap[vIdx]:
                for i in range(3):
                    if self.triangles[triIdx, i] != vIdx and self.constrainedMask[triIdx, i] != self.noneEdge:
                        self.voronoiEdges[triIdx, i] = self.noneFace
                        self.constrainedMask[triIdx, i] = self.noneEdge
            return segIds[0]
        elif len(segIds) != 0:
            assert (False)
        return self.noneEdge

    def _unsafeUnlinkVertex(self, vIdx):
        assert (len(self.vertexMap[vIdx]) == 0)
        return self._superUnsafeUnshackleVertex(vIdx)

    def _unsafeMoveVertex(self, vIdx, p: Point):
        for triIdx in self.vertexMap[vIdx]:
            self.unsetBadness(triIdx)
        self.exactVerts[vIdx] = Point(FieldNumber(p.x().exact()), FieldNumber(p.y().exact()))
        self.numericVerts[vIdx] = [float(p.x()), float(p.y())]
        for triIdx in self.vertexMap[vIdx]:
            self.setBadness(triIdx)
            self.setCircumCenter(triIdx)

        # update all dists, that originally pointed to vIdx
        updateDists = []
        for triIdx in range(len(self.triangles)):
            if self.closestToCC[triIdx] == vIdx:
                updateDists.append(triIdx)

        for triIdx in updateDists:
            closest = None
            closestdist = None
            for vIdx in range(len(self.exactVerts)):
                dist = eg.distsq(self.circumCenters[triIdx], self.point(vIdx))
                if (closest is None) or (dist < closestdist):
                    closest = vIdx
                    closestdist = dist
            self.closestToCC[triIdx] = closest
            self.closestDist[triIdx] = closestdist

        # update all other
        for triIdx in range(len(self.triangles)):
            dist = eg.distsq(self.circumCenters[triIdx], p)
            if (dist < self.closestDist[triIdx]):
                self.closestToCC[triIdx] = len(self.exactVerts) - 1
                self.closestDist[triIdx] = dist


        for otherIdx in self.vertexMap[vIdx]:
            self.updateEdgeTopology(otherIdx)

    # segment
    def _unsafeDeleteSegment(self, segIdx):
        self.segments = np.delete(self.segments, (segIdx), axis=0)
        self.segmentType = np.delete(self.segmentType, (segIdx), axis=0)
        self.constrainedMask = np.where((self.constrainedMask != self.noneEdge) & (self.constrainedMask > segIdx),
                                        self.constrainedMask - 1, self.constrainedMask)

    def _unsafeSplitSegment(self, segIdx, pIdx):
        seg = [self.segments[segIdx][0], self.segments[segIdx][1]]
        self.segments[segIdx] = [seg[0], pIdx]
        self.segments = np.vstack((self.segments, [pIdx, seg[1]]))
        self.segmentType = np.hstack((self.segmentType, self.segmentType[segIdx]))

    # triangle
    # these should be streamlined...
    def _unsafeRemapTriangle(self, idx, source, target):
        if idx != self.outer:
            self.unsetVertexMap(idx)
            self.unsetBadness(idx)
            self.triangles[idx] = np.where(self.triangles[idx] != source, self.triangles[idx], target)
            self.setVertexMap(idx)
            self.setCircumCenter(idx)
            self.setBadness(idx)

    def _unsafeRemapConstraint(self, idx, source, target):
        if idx != self.outer:
            self.constrainedMask[idx] = np.where(self.constrainedMask[idx] != source, self.constrainedMask[idx], target)

    def _unsafeRemapVoronoi(self, idx, source, target):
        if idx != self.outer:
            self.voronoiEdges[idx] = np.where(self.voronoiEdges[idx] != source, self.voronoiEdges[idx], target)

    def _unsafeRemapVoronoiAndConstraint(self, idx, sourceV, targetV, targetC):
        if idx != self.outer:
            self.constrainedMask[idx] = np.where(self.voronoiEdges[idx] != sourceV, self.constrainedMask[idx], targetC)
            self.voronoiEdges[idx] = np.where(self.voronoiEdges[idx] != sourceV, self.voronoiEdges[idx], targetV)

    def _unsafeCreateTriangles(self,tris,triVEdges,triMasks):
        myIdxs = []
        for i in range(len(tris)):
            tri = tris[i]
            triVEdge = triVEdges[i]
            triMask = triMasks[i]

            self.triangles = np.vstack((self.triangles, [tri]))
            self.voronoiEdges = np.vstack((self.voronoiEdges, [triVEdge]))
            self.constrainedMask = np.vstack((self.constrainedMask, [triMask]))
            self.edgeTopologyChanged = np.vstack((self.edgeTopologyChanged, [True, True, True]))
            self.closestToCC = np.hstack((self.closestToCC, [self.noneVertex]))
            self.closestDist = np.hstack((self.closestDist, [FieldNumber(0)]))
            self.circumCenters.append(Point(FieldNumber(0), FieldNumber(0)))
            self.circumRadiiSqr.append(FieldNumber(0))

            myIdxs.append(len(self.triangles) - 1)

        for myIdx in myIdxs:

            self.setVertexMap(myIdx)
            self.setCircumCenter(myIdx)

            self.setBadness(myIdx)

    def _unsafeCreateTriangle(self, tri, triVEdge, triMask):
        self.triangles = np.vstack((self.triangles, [tri]))
        self.voronoiEdges = np.vstack((self.voronoiEdges, [triVEdge]))
        self.constrainedMask = np.vstack((self.constrainedMask, [triMask]))
        self.edgeTopologyChanged = np.vstack((self.edgeTopologyChanged, [True,True,True]))
        self.closestToCC = np.hstack((self.closestToCC, [self.noneVertex]))
        self.closestDist = np.hstack((self.closestDist, [FieldNumber(0)]))
        self.circumCenters.append(Point(FieldNumber(0), FieldNumber(0)))
        self.circumRadiiSqr.append(FieldNumber(0))

        myIdx = len(self.triangles) - 1

        self.setVertexMap(myIdx)
        self.setCircumCenter(myIdx)

        self.setBadness(myIdx)

    def _superUnsafeSetTriangle(self, triIdx, tri, triVedge, triMask):
        # should only be used with unlinked triangles
        self.triangles[triIdx] = tri
        self.voronoiEdges[triIdx] = triVedge
        self.constrainedMask[triIdx] = triMask

        self.setVertexMap(triIdx)
        self.setCircumCenter(triIdx)
        self.setBadness(triIdx)

    def _unsafeSetTriangle(self, triIdx, tri, triVedge, triMask):

        self.unsetVertexMap(triIdx)
        self.unsetBadness(triIdx)

        self._superUnsafeSetTriangle(triIdx, tri, triVedge, triMask)

    def _superUnsafeDeleteTriangle(self, triIdx):

        # this is only acceptable if the triangle has been unlinked before

        self.triangles = np.delete(self.triangles, (triIdx), axis=0)
        self.constrainedMask = np.delete(self.constrainedMask, (triIdx), axis=0)
        self.voronoiEdges = np.delete(self.voronoiEdges, (triIdx), axis=0)
        self.closestToCC = np.delete(self.closestToCC, (triIdx), axis=0)
        self.closestDist = np.delete(self.closestDist, (triIdx), axis=0)
        self.edgeTopologyChanged = np.delete(self.edgeTopologyChanged, (triIdx), axis=0)
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

        self.voronoiEdges = np.where((self.voronoiEdges != self.outer) & (self.voronoiEdges > triIdx),
                                     self.voronoiEdges - 1, self.voronoiEdges)
        for idx in range(len(self.badTris)):
            if self.badTris[idx] > triIdx:
                self.badTris[idx] -= 1
            if self.badTris[idx] == triIdx:
                pass
                # print("uh oh")

    def _unsafeDeleteTriangle(self, triIdx):

        self.unsetVertexMap(triIdx)
        self.unsetBadness(triIdx)

        self._superUnsafeDeleteTriangle(triIdx)

    def _unsafeUnlinkTriangle(self, triIdx):
        self.unsetVertexMap(triIdx)
        self.unsetBadness(triIdx)

        for i in range(3):
            oppTri = self.voronoiEdges[triIdx][i]
            if (oppTri != self.outer) and (oppTri != self.noneFace):
                oppInd = self.oppositeInternalIndex(triIdx, i)
                self.voronoiEdges[oppTri, oppInd] = self.noneFace

        # for visualization
        self.triangles[triIdx] = [0, 0, 0]

    def _unsafeLinkAndSetTriangle(self, triIdx, tri):
        opps = []
        newVEdge = []
        newMask = []
        for i in range(3):
            edge = [tri[(i + 1) % 3], tri[(i + 2) % 3]]
            opp = self.trianglesOnEdge(edge[0], edge[1])
            if len(opp) == 1:
                oppInd = self.oppositeInternalIndexOfEdge(opp[0], *edge)
                assert (self.voronoiEdges[opp, oppInd] == self.noneFace)
                opps.append([opp, oppInd])
                newVEdge.append(opp[0])
                newMask.append(self.constrainedMask[opp[0]][oppInd])
            elif len(opp) == 0:
                if (edgeId := self.getSegmentIdx(np.array(edge))) == self.noneEdge:
                    newVEdge.append(self.noneFace)
                    newMask.append(self.noneEdge)
                else:
                    if self.segmentType[edgeId] == True:
                        newVEdge.append(self.outer)
                        newMask.append(edgeId)
                    else:
                        newVEdge.append(self.noneFace)
                        newMask.append(edgeId)
            else:
                assert (False)
        self._superUnsafeSetTriangle(triIdx, tri, newVEdge, newMask)
        for opp, oppInd in opps:
            self.voronoiEdges[opp, oppInd] = triIdx

    def _unsafeUnlinkAndDeleteTriangle(self, triIdx):
        for i in range(3):
            oppTri = self.voronoiEdges[triIdx][i]
            oppInd = self.oppositeInternalIndex(triIdx, i)
            self.voronoiEdges[oppTri, oppInd] = self.noneFace
        self._unsafeDeleteTriangle(triIdx)

    def _unsafeCreateAndLink(self, tri):
        opps = []
        newVEdge = []
        newMask = []
        for i in range(3):
            edge = [tri[(i + 1) % 3], tri[(i + 2) % 3]]
            opp = self.trianglesOnEdge(*edge)
            if len(opp) == 1:
                oppInd = self.oppositeInternalIndexOfEdge(opp[0], *edge)
                assert (self.voronoiEdges[opp, oppInd] == self.noneFace)
                opps.append([opp, oppInd])
                newVEdge.append(opp[0])
                newMask.append(self.constrainedMask[opp[0]][oppInd])
            elif len(opp) == 0:
                if (edgeId := self.getSegmentIdx(np.array(edge))) == self.noneEdge:
                    newVEdge.append(self.noneFace)
                    newMask.append(self.noneEdge)
                else:
                    if self.segmentType[edgeId] == True:
                        newVEdge.append(self.outer)
                        newMask.append(edgeId)
                    else:
                        newVEdge.append(self.noneFace)
                        newMask.append(edgeId)
            else:
                assert (False)
        self._unsafeCreateTriangle(tri, newVEdge, newMask)
        for opp, oppInd in opps:
            self.voronoiEdges[opp, oppInd] = len(self.triangles) - 1

    #####
    # safe modifiers
    #####
    def flipTrianglePair(self, triAIdx, triBIdx):

        # As perspective
        assert (triBIdx in self.voronoiEdges[triAIdx])
        neighbourAIdx = None
        for nAIdx in range(len(self.voronoiEdges[triAIdx])):
            if triBIdx == self.voronoiEdges[triAIdx][nAIdx]:
                neighbourAIdx = nAIdx
        assert (neighbourAIdx != None)
        assert (self.constrainedMask[triAIdx][neighbourAIdx] == self.noneEdge)

        # Bs perspective
        assert (triAIdx in self.voronoiEdges[triBIdx])
        neighbourBIdx = None
        for nBIdx in range(len(self.voronoiEdges[triBIdx])):
            if triAIdx == self.voronoiEdges[triBIdx][nBIdx]:
                neighbourBIdx = nBIdx
        assert (neighbourBIdx != None)
        assert (self.constrainedMask[triBIdx][neighbourBIdx] == self.noneEdge)

        triA = self.triangles[triAIdx]
        triB = self.triangles[triBIdx]

        if triA[(neighbourAIdx + 1) % 3] == triB[(neighbourBIdx + 1) % 3]:
            # same orientation of shared edge
            newA = [triA[neighbourAIdx], triB[neighbourBIdx], triA[(neighbourAIdx + 2) % 3]]
            newAVedge = [self.voronoiEdges[triBIdx][(neighbourBIdx + 1) % 3],
                         self.voronoiEdges[triAIdx][(neighbourAIdx + 1) % 3], triBIdx]
            newAVedgeMask = [self.constrainedMask[triBIdx][(neighbourBIdx + 1) % 3],
                             self.constrainedMask[triAIdx][(neighbourAIdx + 1) % 3], self.noneEdge]

            newB = [triA[neighbourAIdx], triB[neighbourBIdx], triA[(neighbourAIdx + 1) % 3]]
            newBVedge = [self.voronoiEdges[triBIdx][(neighbourBIdx + 2) % 3],
                         self.voronoiEdges[triAIdx][(neighbourAIdx + 2) % 3], triAIdx]
            newBVedgeMask = [self.constrainedMask[triBIdx][(neighbourBIdx + 2) % 3],
                             self.constrainedMask[triAIdx][(neighbourAIdx + 2) % 3], self.noneEdge]

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
                             self.constrainedMask[triAIdx][(neighbourAIdx + 2) % 3], self.noneEdge]

            newB = [triA[neighbourAIdx], triB[neighbourBIdx], triA[(neighbourAIdx + 2) % 3]]
            newBVedge = [self.voronoiEdges[triBIdx][(neighbourBIdx + 2) % 3],
                         self.voronoiEdges[triAIdx][(neighbourAIdx + 1) % 3], triAIdx]
            newBVedgeMask = [self.constrainedMask[triBIdx][(neighbourBIdx + 2) % 3],
                             self.constrainedMask[triAIdx][(neighbourAIdx + 1) % 3], self.noneEdge]

            self._unsafeRemapVoronoi(self.voronoiEdges[triAIdx][(neighbourAIdx + 1) % 3], triAIdx, triBIdx)
            self._unsafeRemapVoronoi(self.voronoiEdges[triBIdx][(neighbourBIdx + 1) % 3], triBIdx, triAIdx)

            self._unsafeSetTriangle(triAIdx, newA, newAVedge, newAVedgeMask)
            self._unsafeSetTriangle(triBIdx, newB, newBVedge, newBVedgeMask)

    def ensureDelauney(self,seedTris=None):

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
            if jMask == self.noneEdge:
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
        if seedTris is None:
            for i in range(len(self.triangles)):
                for jIdx in range(3):
                    _addEdgeToStack(i, jIdx)
        else:
            for i in seedTris:
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

    def dropAltitude(self, idx, onlyInner=False):
        tri = self.triangles[idx]
        badIdx = eg.badAngle(*self.exactVerts[tri])
        assert (badIdx != -1)
        if self.constrainedMask[idx][badIdx] == self.noneEdge:
            # print("nowhere to drop to!")
            return False
            # assert(False)
        if onlyInner and self.voronoiEdges[idx][badIdx] == self.outer:
            return False
        otherIdx = self.voronoiEdges[idx][badIdx]

        # first things first, split the segment
        segment = [tri[(badIdx + 1) % 3], tri[(badIdx + 2) % 3]]
        segIdx = self.constrainedMask[idx][badIdx]

        # the point to be inserted on the segment
        ap = None
        #if self.voronoiEdges[idx][badIdx] == self.outer:
        #    shorterIdx = (badIdx + 1) % 3
        #    longerIdx = (badIdx + 2) % 3
        #    if eg.distsq(self.point(tri[longerIdx]), self.point(tri[badIdx])) < eg.distsq(self.point(tri[shorterIdx]),
        #                                                                                 self.point(tri[badIdx])):
        #        shorterIdx = (badIdx + 2) % 3
        #        longerIdx = (badIdx + 1) % 3
        #    p = self.point(tri[badIdx])
        #    orth = p - self.point(tri[shorterIdx])
        #    orth = Point(FieldNumber(0)-orth.y(),orth.x())
        #    if eg.dot(orth,self.point(tri[longerIdx]) - p) < FieldNumber(0):
        #        orth = orth.scale(FieldNumber(-1))
        #
        #    inter = eg.supportingRayIntersectSegment(Segment(p,p+orth),Segment(self.point(tri[shorterIdx]),self.point(tri[longerIdx])))#
        #
        #    #dont drop past the middle point
        #    if eg.distsq(self.point(tri[shorterIdx]),inter) < eg.distsq(self.point(tri[longerIdx]),inter):
        #        ap = inter
        #    else:
        #        ap = eg.altitudePoint(Segment(self.point(segment[0]), self.point(segment[1])),
        #                              self.exactVerts[tri[badIdx]])
        #else:
        ap = eg.altitudePoint(Segment(self.point(segment[0]), self.point(segment[1])),
                                  self.exactVerts[tri[badIdx]])

        ap2 = eg.altitudePoint(Segment(self.point(segment[0]), self.point(segment[1])),
                                  self.exactVerts[tri[badIdx]])

        self.tri.addPoint(ap2)
        #for triIdx in range(len(self.tri.triangles)):
        #    tri = self.tri.triangles[triIdx]
        #    for i in range(3):
        #        for j in range(1,3):
        #            neighbourInternal = (i+j)%3
        #            neighbourId,opp,_ = self.tri.triangleMap[triIdx,neighbourInternal]
        #            if neighbourId == np.iinfo(int).max:
        #                continue
        #            if j == 1:
        #                self.tri.getEnclosementOfLink([tri[(i+1)%3],tri[(i+2)%3],self.tri.triangles[neighbourId,opp]])
        #            if j == 2:
        #                self.tri.getEnclosementOfLink([self.tri.triangles[neighbourId,opp],tri[(i+1)%3],tri[(i+2)%3]])


        newPointIndex = len(self.exactVerts)

        self._unsafeSplitSegment(segIdx, newPointIndex)

        # self.segments[segIdx] = [segment[0], newPointIndex]
        # self.segments = np.vstack((self.segments, [newPointIndex, segment[1]]))

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
            newAVedgeMask = [segAIdx, self.noneEdge, self.constrainedMask[idx][(badIdx + 2) % 3]]

            newTri = [tri[badIdx], newPointIndex, tri[(badIdx + 2) % 3]]
            newVedge = [self.outer, self.voronoiEdges[idx][(badIdx + 1) % 3], idx]
            newVedgeMask = [segOtherIdx, self.constrainedMask[idx][(badIdx + 1) % 3], self.noneEdge]

            self._unsafeRemapVoronoi(self.voronoiEdges[idx][(badIdx + 1) % 3], idx, newTriIndex)

            self._unsafeCreateTriangle(newTri, newVedge, newVedgeMask)
            self._unsafeSetTriangle(idx, newA, newAVedge, newAVedgeMask)

            self.ensureDelauney([idx,newTriIndex])

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
                newAMask = [segAIdx, self.noneEdge, self.constrainedMask[idx][(badIdx + 2) % 3]]

                newInside = [tri[badIdx], newPointIndex, tri[(badIdx + 2) % 3]]
                newInsideVedge = [newOutsideIdx, self.voronoiEdges[idx][(badIdx + 1) % 3], idx]
                newInsideMask = [segOtherIdx, self.constrainedMask[idx][(badIdx + 1) % 3], self.noneEdge]

                newB = [otherTri[opposingIdx], otherTri[(opposingIdx + 1) % 3], newPointIndex]
                newBVedge = [idx, newOutsideIdx, self.voronoiEdges[otherIdx][(opposingIdx + 2) % 3]]
                newBMask = [newAMask[0], self.noneEdge, self.constrainedMask[otherIdx][(opposingIdx + 2) % 3]]

                newOutside = [otherTri[opposingIdx], newPointIndex, otherTri[(opposingIdx + 2) % 3]]
                newOutsideVedge = [newInsideIdx, self.voronoiEdges[otherIdx][(opposingIdx + 1) % 3], otherIdx]
                newOutsideMask = [newInsideMask[0], self.constrainedMask[otherIdx][(opposingIdx + 1) % 3],
                                  self.noneEdge]

                self._unsafeRemapVoronoi(self.voronoiEdges[idx][(badIdx + 1) % 3], idx, newInsideIdx)
                self._unsafeRemapVoronoi(self.voronoiEdges[otherIdx][(opposingIdx + 1) % 3], otherIdx, newOutsideIdx)

                self._unsafeCreateTriangles([newInside, newOutside], [newInsideVedge, newOutsideVedge],
                                            [newInsideMask, newOutsideMask])

                self._unsafeSetTriangle(idx, newA, newAVedge, newAMask)
                self._unsafeSetTriangle(otherIdx, newB, newBVedge, newBMask)

            else:
                newA = [tri[badIdx], tri[(badIdx + 1) % 3], newPointIndex]
                newAVedge = [otherIdx, newInsideIdx, self.voronoiEdges[idx][(badIdx + 2) % 3]]
                newAMask = [segAIdx, self.noneEdge,
                            self.constrainedMask[idx][(badIdx + 2) % 3]]

                newInside = [tri[badIdx], newPointIndex, tri[(badIdx + 2) % 3]]
                newInsideVedge = [newOutsideIdx, self.voronoiEdges[idx][(badIdx + 1) % 3], idx]
                newInsideMask = [segOtherIdx,
                                 self.constrainedMask[idx][(badIdx + 1) % 3], self.noneEdge]

                newB = [otherTri[opposingIdx], otherTri[(opposingIdx + 2) % 3], newPointIndex]
                newBVedge = [idx, newOutsideIdx, self.voronoiEdges[otherIdx][(opposingIdx + 1) % 3]]
                newBMask = [newAMask[0], self.noneEdge, self.constrainedMask[otherIdx][(opposingIdx + 1) % 3]]

                newOutside = [otherTri[opposingIdx], newPointIndex, otherTri[(opposingIdx + 1) % 3]]
                newOutsideVedge = [newInsideIdx, self.voronoiEdges[otherIdx][(opposingIdx + 2) % 3], otherIdx]
                newOutsideMask = [newInsideMask[0], self.constrainedMask[otherIdx][(opposingIdx + 2) % 3],
                                  self.noneEdge]

                self._unsafeRemapVoronoi(self.voronoiEdges[idx][(badIdx + 1) % 3], idx, newInsideIdx)
                self._unsafeRemapVoronoi(self.voronoiEdges[otherIdx][(opposingIdx + 2) % 3], otherIdx, newOutsideIdx)


                self._unsafeCreateTriangles([newInside, newOutside], [newInsideVedge, newOutsideVedge],
                                            [newInsideMask, newOutsideMask])
                #self._unsafeCreateTriangle(newOutside, newOutsideVedge, newOutsideMask)
                #self._unsafeCreateTriangle(newInside, newInsideVedge, newInsideMask)

                self._unsafeSetTriangle(idx, newA, newAVedge, newAMask)
                self._unsafeSetTriangle(otherIdx, newB, newBVedge, newBMask)
            self.ensureDelauney([idx,otherIdx,newInsideIdx,newOutsideIdx])
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
        newAMask = [hitTriMask[0], self.noneEdge, self.noneEdge]

        newLeft = [hitTri[0], addedPointIdx, hitTri[2]]
        newLeftVedge = [hitTriIdx, hitTriVedge[1], newRightTriIdx]
        newLeftMask = [self.noneEdge, hitTriMask[1], self.noneEdge]

        newRight = [hitTri[0], hitTri[1], addedPointIdx]
        newRightVedge = [hitTriIdx, newLeftTriIdx, hitTriVedge[2]]
        newRightMask = [self.noneEdge, self.noneEdge, hitTriMask[2]]

        self._unsafeRemapVoronoi(hitTriVedge[1], hitTriIdx, newLeftTriIdx)
        self._unsafeRemapVoronoi(hitTriVedge[2], hitTriIdx, newRightTriIdx)

        # make sure, its a copy!
        self._unsafeCreateVertex(Point(FieldNumber(p.x().exact()), FieldNumber(p.y().exact())))

        self._unsafeCreateTriangles([newLeft,newRight],[newLeftVedge,newRightVedge],[newLeftMask,newRightMask])
        #self._unsafeCreateTriangle(newLeft, newLeftVedge, newLeftMask)
        #self._unsafeCreateTriangle(newRight, newRightVedge, newRightMask)
        self._unsafeSetTriangle(hitTriIdx, newA, newAVedge, newAMask)

        self.ensureDelauney([hitTriIdx,newLeftTriIdx,newRightTriIdx])

        return True

    def mergePoints(self, source, target):
        self.validate()
        # moves source index to target and deletes the triangles participating.
        # in reality we let some data dangle, to not fuck up all the maps
        participatingTris = [tri for tri in self.vertexMap[source] if
                             (tri != self.outer) and (target in self.triangles[tri])]
        sourceTris = [tri for tri in self.vertexMap[source] if
                      (tri != self.outer) and (target not in self.triangles[tri])]

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
                self._unsafeRemapConstraint(triIdx, deleteSegIdx, otherSegIdx)

            for triIdx in participatingTris:
                self._unsafeRemapConstraint(triIdx, deleteSegIdx, otherSegIdx)

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

            self._unsafeRemapVoronoiAndConstraint(leftLower, leftTriIdx, leftUpper,
                                                  self.constrainedMask[leftTriIdx][leftSIdx])
            self._unsafeRemapVoronoi(leftUpper, leftTriIdx, leftLower)

            rightUpper = self.voronoiEdges[rightTriIdx][rightSIdx]
            rightLower = self.voronoiEdges[rightTriIdx][rightTIdx]

            self._unsafeRemapVoronoiAndConstraint(rightLower, rightTriIdx, rightUpper,
                                                  self.constrainedMask[rightTriIdx][rightSIdx])
            self._unsafeRemapVoronoi(rightUpper, rightTriIdx, rightLower)

            # remove triangles
            for deleteIdx in [max(leftTriIdx, rightTriIdx), min(leftTriIdx, rightTriIdx)]:
                self._unsafeDeleteTriangle(deleteIdx)

            self.ensureDelauney(self.vertexMap[target])

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

            self._unsafeRemapVoronoiAndConstraint(leftLower, leftTriIdx, leftUpper,
                                                  self.constrainedMask[leftTriIdx][leftSIdx])
            self._unsafeRemapVoronoi(leftUpper, leftTriIdx, leftLower)

            # remove triangles
            self._unsafeDeleteTriangle(leftTriIdx)

            self.ensureDelauney(self.vertexMap[target])

            # remove source-vertex
            self._unsafeDeleteVertex(source)

            self.validate()
        else:
            assert (False)

    def getLinkAroundVertex(self, v):
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

                if self.constrainedMask[curTriIdx][curSelfIdx] != self.noneEdge:
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
        return link, constraint, triIndices

    def getLinkAroundEdge(self, triIdx, vIdx):
        def walkAround(startF, startV):
            link = []
            faces = [startF]
            indices = [startV]
            curF = startF
            curV = startV
            constrainedVertex = []
            while curF != self.outer:
                nextF = self.voronoiEdges[curF][curV]
                faces.append(nextF)
                if nextF == self.outer:
                    indices.append(self.noneEdge)
                    for i in range(1, 3):
                        v = self.triangles[curF][(curV + i) % 3]
                        if (v not in edge):
                            link.append(v)
                            constrainedVertex.append(False)
                    curF = nextF

                else:
                    nextV = None
                    for i in range(3):
                        v = self.triangles[nextF][i]
                        if (v in self.triangles[curF]) and (v not in edge):
                            nextV = i
                    if nextV == None:
                        indices.append(self.noneEdge)
                        break
                    else:
                        indices.append(nextV)
                        link.append(self.triangles[nextF][nextV])
                        if self.constrainedMask[curF][curV] != self.noneEdge:
                            constrainedVertex.append(True)
                        else:
                            constrainedVertex.append(False)
                        curF = nextF
                        curV = nextV
            return link, faces, indices, constrainedVertex

        # the points to be deleted are self.triangles[triIdx][(vIdx+1)%3] and self.triangles[triIdx][(vIdx+2)%3]
        edge = [self.triangles[triIdx][(vIdx + 1) % 3], self.triangles[triIdx][(vIdx + 2) % 3]]
        otherIdx = self.voronoiEdges[triIdx][vIdx]
        if otherIdx == self.outer:
            linkL, facesL, indicesL, conL = walkAround(triIdx, (vIdx + 1) % 3)
            linkR, facesR, indicesR, conR = walkAround(triIdx, (vIdx + 2) % 3)
            assert (linkL[0] == linkR[0])
            assert (facesR[-1] == self.outer)
            assert (facesL[-1] == self.outer)
            link = list(reversed(linkL)) + linkR[1:]
            con = [0, len(link) - 1]
            return link, con

        else:
            oppVIdx = None
            for i in range(3):
                if self.triangles[otherIdx][i] not in edge:
                    oppVIdx = i
            linkUL, facesUL, indicesUL, conUL = walkAround(triIdx, (vIdx + 1) % 3)
            linkUR, facesUR, indicesUR, conUR = walkAround(triIdx, (vIdx + 2) % 3)

            link = []
            constraint = []

            if facesUR[0] == facesUR[-1]:
                # the one vertex does not lie on a boundary. hype!!
                if facesUL[0] == facesUL[-1]:
                    # one contiguous region!! hype
                    for i in range(len(linkUL)):
                        link.append(linkUL[i])
                        if conUL[i] == True:
                            constraint.append(i)
                    for i in reversed(range(len(linkUR))):
                        if i == 0:
                            if conUR[i] == True:
                                constraint.append(0)
                        elif i == len(linkUR) - 1:
                            if conUR[i] == True:
                                constraint.append(len(link) - 1)
                        else:
                            link.append(linkUR[i])
                            if conUR[i] == True:
                                constraint.append(len(link) - 1)
                else:
                    # UL hit outerface
                    otherLink, otherFaces, otherIndices, otherCon = walkAround(otherIdx, (oppVIdx + 1) % 3)
                    if otherFaces[-1] != self.outer:
                        otherLink, otherFaces, otherIndices, otherCon = walkAround(otherIdx, (oppVIdx + 2) % 3)
                    for i in reversed(range(len(linkUL))):
                        link.append(linkUL[i])
                        if conUL[i] == True:
                            constraint.append(len(link) - 1)
                    for i in range(len(linkUR)):
                        if i == 0:
                            if conUR[i] == True:
                                constraint.append(len(link) - 1)
                        else:
                            link.append(linkUR[i])
                            if conUR[i] == True:
                                constraint.append(len(link) - 1)
                    for i in range(len(otherLink)):
                        if i == 0:
                            if otherCon[i] == True:
                                constraint.append(len(link) - 1)
                        else:
                            link.append(otherLink[i])
                            if otherCon[i] == True:
                                constraint.append(len(link) - 1)
                return link, constraint
            else:
                otherLink, otherFaces, otherIndices, otherCon = linkUR, facesUR, indicesUR, conUR
                goodLink, goodFaces, goodIndices, goodCon = walkAround(otherIdx, (oppVIdx + 1) % 3)
                shitLink = []
                shitFaces = []
                shitIndices = []
                shitCon = []

                if goodFaces[0] == goodFaces[-1]:
                    shitLink, shitFaces, shitIndices, shitCon = walkAround(otherIdx, (oppVIdx + 2) % 3)
                else:
                    shitLink, shitFaces, shitIndices, shitCon = goodLink, goodFaces, goodIndices, goodCon
                    goodLink, goodFaces, goodIndices, goodCon = walkAround(otherIdx, (oppVIdx + 2) % 3)

                # shitlink is guaranteed to end at the outer face
                if goodFaces[0] == goodFaces[-1]:
                    # first the reversed shitfaces
                    for i in reversed(range(len(shitLink))):
                        link.append(shitLink[i])
                        if shitCon[i] == True:
                            constraint.append(len(link) - 1)

                    # then the good faces
                    for i in range(len(goodLink)):
                        if i == 0:
                            if goodCon[i] == True:
                                constraint.append(len(link) - 1)
                        else:
                            link.append(goodLink[i])
                            if goodCon[i] == True:
                                constraint.append(len(link) - 1)

                    # then the other faces
                    for i in range(len(otherLink)):
                        if i == 0:
                            if otherCon[i] == True:
                                constraint.append(len(link) - 1)
                        else:
                            link.append(otherLink[i])
                            if otherCon[i] == True:
                                constraint.append(len(link) - 1)
                else:
                    # here the edge spans from one outer vertex to another, so there can not be a constraint entering the link!
                    for i in reversed(linkUR):
                        link.append(i)
                    for i in linkUL[1:]:
                        link.append(i)
                    nextShitLink = False
                    lastShitEdge = self.triangles[shitFaces[-2]]
                    lastULEdge = self.triangles[facesUL[-2]]
                    for i in lastULEdge:
                        if i in lastShitEdge:
                            nextShitLink = True
                    if nextShitLink:
                        for i in reversed(shitLink):
                            link.append(i)
                        for i in goodLink[1:]:
                            link.append(i)
                    else:
                        for i in reversed(goodLink):
                            link.append(i)
                        for i in shitLink[1:]:
                            link.append(i)
                return link, constraint

    def _internalAttempt(self, link, constraint, edge, toFlip=[], axs=None):
        for i in link:
            if len([v for v in link if v == i])>1:
                return False
        points = [self.point(i) for i in link]

        soltype = "None"
        cs = None
        if len(constraint) == 2:
            soltype, cs = eg.findCenterOfLinkConstrained(points, constraint[0], constraint[1])
        elif len(constraint) == 0:
            soltype, cs = eg.findCenterOfLink(points)

        if soltype == "None":
            return False

        for flip in toFlip:
            self.flipTrianglePair(flip[0], flip[1])

            if axs is not None:
                axs.clear()
                self.plotTriangulation(axs)
                plt.draw()
                plt.pause(self.plotTime)

        if soltype == "inside":
            idx = 0

            if len(constraint) == 2:
                both = 0
                for seg in self.segments:
                    if np.all([edge[1], link[constraint[0]]] == seg) or np.all([link[constraint[0]], edge[1]] == seg):
                        both += 1
                    if np.all([edge[1], link[constraint[1]]] == seg) or np.all([link[constraint[1]], edge[1]] == seg):
                        both += 1
                if both == 2:
                    idx = 1
            elif len(constraint) == 0:
                # there ought be at least one, that is not constraint. idx should point to that. If not then we handle it later
                for seg in self.segments:
                    if edge[0] in seg:
                        idx = 1
                        break

            otherIdx = 1 - idx

            faceIdPool = []
            facePool = []

            # first unlink all faces that lie near otherIdx
            for i in reversed(range(len(self.vertexMap[edge[otherIdx]]))):
                triId = self.vertexMap[edge[otherIdx]][i]
                faceIdPool.append(triId)
                facePool.append([v if v != edge[otherIdx] else edge[idx] for v in self.triangles[triId]])
                self._unsafeUnlinkTriangle(triId)
                if axs is not None:
                    axs.clear()
                    self.plotTriangulation(axs)
                    plt.draw()
                    plt.pause(self.plotTime)

            addorFacePool = []

            # if the two points share a constraint, then we do not need to introduce any faces. if the two
            # points dont share a constraint, then weneed to add a face for every constraint point (at most 2) in any
            # case, we delete exactly two faces, so we can reuse these

            if len(constraint) != 2:

                # first unshackle point edge[idx] from any potential boundaries
                edgeId = self._superUnsafeUnshackleVertex(edge[idx])
                if edgeId != self.noneEdge:
                    addorFacePool.append([edge[idx], self.segments[edgeId][0], self.segments[edgeId][1]])

                # next unlink the vertex to be deleted, to ensure consistency on constrained edges
                edgeId = self._unsafeUnlinkVertex(edge[otherIdx])
                if edgeId != self.noneEdge:
                    addorFacePool.append([edge[idx], self.segments[edgeId][0], self.segments[edgeId][1]])
                # addorFacePool now contains the two faces that must be added in order to maintain trhangles to
                # all boundaries

            else:
                # Unlink the vertex to be deleted, to ensure consistency on constrained edges, this might add one face
                edgeId = self._unsafeUnlinkVertex(edge[otherIdx])
                if edgeId != self.noneEdge:
                    if edge[idx] not in self.segments[edgeId]:
                        # this happens exactly if both edge[idx] and edge[otherIdx] are constraint, but edge[otherIdx]
                        # is NOT constraint to edge[idx] but rather on the boundary of the link
                        addorFacePool.append([edge[idx], self.segments[edgeId][0], self.segments[edgeId][1]])

            # next move the unshackled point. this should now lie in the center of the star
            self._unsafeMoveVertex(edge[idx], cs[0])

            if axs is not None:
                axs.clear()
                self.plotTriangulation(axs)
                plt.draw()
                plt.pause(self.plotTime)

            # now reintroduce the faces. careful tho, there might be two, that have edge[idx] twice. collect these two, and merge them
            # len(collector) should always dominate
            collector = []
            for i in range(len(faceIdPool)):
                id = faceIdPool[i]
                face = facePool[i]
                both = 0
                for j in range(3):
                    if face[j] == edge[idx]:
                        both += 1
                if both == 2:
                    collector.append(i)
                elif both == 1:
                    self._unsafeLinkAndSetTriangle(id, face)

                    if axs is not None:
                        axs.clear()
                        self.plotTriangulation(axs)
                        plt.draw()
                        plt.pause(self.plotTime)

            assert (len(collector) >= len(addorFacePool))
            for it in range(len(addorFacePool)):
                self._unsafeLinkAndSetTriangle(faceIdPool[collector[it]], addorFacePool[it])

            self.ensureDelauney(self.vertexMap[edge[idx]])

            for id in reversed(sorted(np.array(faceIdPool)[collector[len(addorFacePool):]])):
                self._superUnsafeDeleteTriangle(id)
            self._unsafeDeleteVertex(edge[otherIdx])

            if axs is not None:
                axs.clear()
                self.plotTriangulation(axs)
                plt.draw()
                plt.pause(self.plotTime)
            return True
        else:
            # print("=",end="")
            # this is now easy.
            # unlink all neighbouring triangles, and relink them to the solving vertex. delete all others

            facePool = []
            faceIdPool = []
            idx = 0
            otherIdx = 1
            # first unlink all faces that lie near BOTH vertices
            for i in reversed(range(len(self.vertexMap[edge[otherIdx]]))):
                triId = self.vertexMap[edge[otherIdx]][i]
                faceIdPool.append(triId)
                facePool.append([v if v != edge[otherIdx] else edge[idx] for v in self.triangles[triId]])
                self._unsafeUnlinkTriangle(triId)
                if axs is not None:
                    axs.clear()
                    self.plotTriangulation(axs)
                    plt.draw()
                    plt.pause(self.plotTime)

            idx = 1
            otherIdx = 0

            for i in reversed(range(len(self.vertexMap[edge[otherIdx]]))):
                triId = self.vertexMap[edge[otherIdx]][i]
                if triId in faceIdPool:
                    continue
                faceIdPool.append(triId)
                facePool.append([v if v != edge[otherIdx] else edge[idx] for v in self.triangles[triId]])
                self._unsafeUnlinkTriangle(triId)
                if axs is not None:
                    axs.clear()
                    self.plotTriangulation(axs)
                    plt.draw()
                    plt.pause(self.plotTime)

            self._unsafeUnlinkVertex(edge[0])
            self._unsafeUnlinkVertex(edge[1])

            digest = 0
            movedIndexList = []
            for it in range(len(link)):
                triPoints = [self.point(link[cs]), self.point(link[it]), self.point(link[(it + 1) % len(link)])]
                if not eg.colinear(Segment(triPoints[0], triPoints[1]), triPoints[2]):
                    self._unsafeLinkAndSetTriangle(faceIdPool[digest], [link[cs], link[it], link[(it + 1) % len(link)]])
                    movedIndexList.append(faceIdPool[digest])
                    digest += 1
                    if axs is not None:
                        axs.clear()
                        self.plotTriangulation(axs)
                        plt.draw()
                        plt.pause(self.plotTime)

            self.ensureDelauney(movedIndexList)

            for id in reversed(sorted(np.array(faceIdPool)[digest:])):
                self._superUnsafeDeleteTriangle(id)
            self._unsafeDeleteVertex(max(edge[0], edge[1]))
            self._unsafeDeleteVertex(min(edge[0], edge[1]))

            if axs is not None:
                axs.clear()
                self.plotTriangulation(axs)
                plt.draw()
                plt.pause(self.plotTime)

            return True

    def attemptComplicatedEdgeContraction(self, triIdx, vIdx, axs=None, withFaceExpansion=False):
        neither = True
        if self.edgeTopologyChanged[triIdx,vIdx]:
            neither = False
            self.edgeTopologyChanged[triIdx, vIdx] = False
        oppVIdx = self.oppositeInternalIndex(triIdx,vIdx)
        oppTriIdx = self.voronoiEdges[triIdx,vIdx]
        if oppTriIdx != self.outer and self.edgeTopologyChanged[oppTriIdx,oppVIdx]:
            neither = False
            self.edgeTopologyChanged[oppTriIdx, oppVIdx] = False

        if neither:
            return False
        link, constraint = self.getLinkAroundEdge(triIdx, vIdx)
        if len(constraint) > 2:
            return False
        edge = [self.triangles[triIdx][(vIdx + 1) % 3], self.triangles[triIdx][(vIdx + 2) % 3]]
        expanders = [None]
        if withFaceExpansion:
            for s in edge:
                for triIdx in self.vertexMap[s]:
                    iSId = self.internalIndex(triIdx, s)
                    if self.voronoiEdges[triIdx][iSId] != self.outer and self.constrainedMask[triIdx][
                        iSId] == self.noneEdge:
                        oppV = self.triangles[self.voronoiEdges[triIdx][iSId]][self.oppositeInternalIndex(triIdx, iSId)]
                        if oppV not in link:
                            expanders.append(
                                [self.triangles[triIdx][(iSId + 1) % 3], self.triangles[triIdx][(iSId + 2) % 3], oppV,
                                 triIdx, self.voronoiEdges[triIdx][iSId]])
        for expander in expanders:
            if expander == None:
                if self._internalAttempt(link, constraint, edge, axs=axs):
                    return True
            else:
                l = expander[0]
                r = expander[1]
                ex = expander[2]
                toFlip = [expander[3:]]

                tempLink = link.copy()
                tempConstraint = []

                for i in range(len(link)):
                    if (link[i] == l and link[(i + 1) % len(link)] == r) or (
                            link[i] == r and link[(i + 1) % len(link)] == l):
                        tempLink = tempLink[:(i + 1)] + [ex] + tempLink[(i + 1):]
                        for c in constraint:
                            if c > i:
                                tempConstraint.append(c + 1)
                            else:
                                tempConstraint.append(c)
                        break
                if self._internalAttempt(tempLink, tempConstraint, edge, toFlip, axs):
                    return True
        return False

    def moveSteinerpoint(self, ignoreBadness=False, mainAx=None):
        globalMoved = False
        badMap = [[] for v in self.exactVerts]

        movedIndexList = []

        for triIdx in self.badTris:
            for vIdx in self.triangles[triIdx]:
                badMap[vIdx].append(triIdx)

        for idx in range(self.instanceSize, len(self.exactVerts)):
            if self.pointTopologyChanged[idx] == False:
                continue

            self.pointTopologyChanged[idx] = False
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

                link, constraint, enclosedTriangles = self.getLinkAroundVertex(idx)

                for i in range(len(link)):
                    linkIdx = link[i]
                    exactPoints.append(self.point(linkIdx))
                    if linkIdx in constraint:
                        intrinsicConstraint.append(i)
                solType = None
                cs = None
                if len(intrinsicConstraint) == 2:
                    if onlyVertex:
                        solType, cs = eg.findVertexCenterOfLinkConstrained(exactPoints, [intrinsicConstraint[0],
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
                            assert (i != self.outer)
                            self.pointTopologyChanged[i] = True
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
                        pass
                        # print("oh no...")

                    self._unsafeMoveVertex(idx,cs[0])
                    #self.exactVerts[idx] = Point(FieldNumber(cs[0].x().exact()), FieldNumber(cs[0].y().exact()))
                    #self.numericVerts[idx] = [float(cs[0].x()), float(cs[0].y())]
                    #for triIndex in enclosedTriangles:
                    #    self.setCircumCenter(triIndex)
                    #    self.unsetBadness(triIndex)
                    self.pointTopologyChanged[idx] = False
                    for triIndex in enclosedTriangles:
                        movedIndexList.append(triIndex)
                    self.ensureDelauney(movedIndexList)
                    return True
                elif solType == "vertex":
                    # identiy the two triangles that can die
                    # if onBoundary:
                    # print("special case")
                    for i in link:
                        self.pointTopologyChanged[i] = True
                    self.mergePoints(idx, link[cs])
                    return True
                elif solType == "None":
                    pass
                    # print("has no center")
        return False

    def improveQuality(self, axs=None, verbosity=0):
        drawTime = 0
        removalTime = 0
        movingTime = 0
        droppingTime = 0
        addingTime = 0
        finalWiggle = False

        expectedSolutionSize = len(self.badTris)

        for round in range(5000):

            start = time.time()
            # plot solution
            if axs is not None and (finalWiggle or (round % 5) == 0):
                axs.clear()
                axs.set_facecolor('lightgray')
                self.plotTriangulation(axs)
                plt.draw()
                plt.pause(self.plotTime)
            drawTime += time.time() - start

            start = time.time()
            removed = False
            vprint("removing...", end="", verbosity=verbosity)
            # new rule: try to contract edge
            for triIdx,vIdx in np.argwhere(self.edgeTopologyChanged == True):
                if (self.triangles[triIdx][(vIdx + 1) % 3] >= self.instanceSize) and (
                        self.triangles[triIdx][(vIdx + 2) % 3] >= self.instanceSize):
                    if self.attemptComplicatedEdgeContraction(triIdx, vIdx, None, finalWiggle):
                        # print("-", end="")
                        removed = True
                        break
                if removed:
                    break
            if removed:
                removalTime += time.time() - start
                vprint("success", verbosity=verbosity)
                continue

            # try greedily more faces, but still only edges

            # if (len(self.exactVerts)-self.instanceSize) * 3 > expectedSolutionSize:
            #    for triIdx in range(len(self.triangles)):
            #        for vIdx in range(3):
            #            if (self.triangles[triIdx][(vIdx + 1) % 3] >= self.instanceSize) and (
            #                    self.triangles[triIdx][(vIdx + 2) % 3] >= self.instanceSize) and (
            #                    triIdx < self.voronoiEdges[triIdx][vIdx]) and (
            #                    (self.pointTopologyChanged[self.triangles[triIdx][(vIdx + 1) % 3]]) or (
            #                    self.pointTopologyChanged[self.triangles[triIdx][(vIdx + 2) % 3]])):
            #                if self.attemptComplicatedEdgeContraction(triIdx, vIdx,None,True):
            #                    #print("-", end="")
            #                    self.ensureDelauney()
            #                    removed = True
            #                    break
            #        if removed:
            #            break
            #    removalTime += time.time() - start
            #    if removed:
            #        vprint("success", verbosity=verbosity)
            #        continue
            #    vprint("failure", verbosity=verbosity)

            # attempt moving every vertex to locally remove bad triangles
            start = time.time()
            vprint("moving...", end="", verbosity=verbosity)
            movedSomething = self.moveSteinerpoint(ignoreBadness=finalWiggle, mainAx=axs)
            movingTime += time.time() - start
            vprint("success" if movedSomething else "failure", verbosity=verbosity)

            # if we locally improved the solution, try moving more stuff around
            if movedSomething:
                continue

            # attempt to drop an altitude onto a constraint
            start = time.time()
            added = False
            # vprint("dropping...", end="", verbosity=verbosity)
            # for triIdx in self.badTris:
            #    if self.dropAltitude(triIdx,onlyInner = True):
            #        added = True
            #        break
            # attempt to drop onto outer face
            if not added:
                for triIdx in self.badTris:
                    if self.dropAltitude(triIdx, onlyInner=False):
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
                    self.pointTopologyChanged = [True for v in self.exactVerts]
                    self.edgeTopologyChanged = np.full(self.triangles.shape,True)
                    continue
                else:

                    # otherwise we are done!
                    if axs != None:
                        axs.clear()
                        self.plotTriangulation(axs)
                        plt.draw()
                        plt.pause(1)
                    # print(f"\ndraw: {drawTime:0,.2f}; remove: {removalTime:0,.2f}; move: {movingTime:0,.2f}; drop: {droppingTime:0,.2f}; add: {addingTime:0,.2f};       ",end="")
                    return self.solutionParse()
            start = time.time()

            start = time.time()
            added = False
            withRounding = True
            if withRounding:
                # if we end up here, we attempt fixing a bad triangle by adding its circumcenter as a steiner point
                for triIdx in self.badTris:
                    # this shouldnt be here, but whatever
                    # badIdx = eg.badAngle(self.point(self.triangles[triIdx][0]),self.point(self.triangles[triIdx][1]),self.point(self.triangles[triIdx][2]))
                    # badPoint = self.point(self.triangles[triIdx][badIdx])
                    # dropsThrough = False
                    # for segIdx in range(len(self.segments)):
                    #    seg = self.segments[segIdx]
                    #    if eg.innerIntersect(self.point(seg[0]), self.point(seg[1]), badPoint, self.circumCenters[triIdx]) is not None:
                    #        dropsThrough = True
                    # if dropsThrough:
                    #    continue
                    vprint("adding rounded circumcenter of " + str(triIdx) + "...", end="", verbosity=verbosity)
                    added = self.addPoint(eg.roundExact(self.circumCenters[triIdx]))
                    if added:
                        if verbosity > 0:
                            vprint("success", verbosity=verbosity)
                        break
                    if not added and verbosity > 0:
                        vprint("failure", verbosity=verbosity)
            if added:
                addingTime += time.time() - start
                continue
            # if rounding didnt work, try to add an exact version
            for triIdx in self.badTris:
                # this shouldnt be here, but whatever
                # badIdx = eg.badAngle(self.point(self.triangles[triIdx][0]), self.point(self.triangles[triIdx][1]),
                #                     self.point(self.triangles[triIdx][2]))
                # badPoint = self.point(self.triangles[triIdx][badIdx])
                # dropsThrough = False
                # for segIdx in range(len(self.segments)):
                #    seg = self.segments[segIdx]
                #    if eg.innerIntersect(self.point(seg[0]), self.point(seg[1]), badPoint,
                #                         self.circumCenters[triIdx]) is not None:
                #        dropsThrough = True
                # if dropsThrough:
                #    continue
                vprint("adding exact circumcenter of " + str(triIdx) + " with representation length " + str(
                    len(str(self.circumCenters[triIdx].x().exact())) + len(
                        str(self.circumCenters[triIdx].y().exact()))) + "...", end="", verbosity=verbosity)
                added = self.addPoint(self.circumCenters[triIdx])
                if added:
                    if verbosity > 0:
                        vprint("success", verbosity=verbosity)
                    break
                if not added and verbosity > 0:
                    vprint("failure", verbosity=verbosity)

            addingTime += time.time() - start
            if added:
                continue

            # if we end up here, all circumcenter of bad triangles lie outside the bounding polygon, or on
            # an edge of the triangulation. In this case we simply add a point in the middle of the triangle. this will
            # later be moved to an advantageous location anyways, so we dont care
            if not added:
                # hacky solution for now
                for i in self.badTris:
                    vprint("adding centroid of " + str(i) + "...", end="", verbosity=verbosity)
                    avg = Point(FieldNumber(0), FieldNumber(0))
                    for v in range(3):
                        avg += self.point(self.triangles[i][v])
                    avg = avg.scale(FieldNumber(1) / FieldNumber(3))
                    added = self.addPoint(avg)
                    if added:
                        vprint("success", verbosity=verbosity)
                        break
                    vprint("failure", verbosity=verbosity)

            if not added:
                pass
                # if we end up here, we are fucked
                # print("huh")
            addingTime += time.time() - start
        return None
