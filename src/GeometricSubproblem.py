import logging

import numpy as np
from constants import *
import matplotlib.pyplot as plt
import exact_geometry as eg
from cgshop2025_pyutils.geometry import Point, Segment, FieldNumber, intersection_point


class GeometricSubproblem:
    def __init__(self, vIdxs, triIdxs, boundary, exactPoints, numericPoints, innerCons, boundaryCons, boundaryType,
                 steinercutoff, numBadTris, axs=None):
        # some housekeeping to have some handle on what to delete later

        # lets make this better
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
        for s, t in innerCons:
            insideCons.append([np.where(self.localMap == s)[0][0], np.where(self.localMap == t)[0][0]])

        outsideCons = []
        for s, t in boundaryCons:
            outsideCons.append([np.where(self.localMap == s)[0][0], np.where(self.localMap == t)[0][0]])

        insideCons = np.array(insideCons)
        outsideCons = np.array(outsideCons)
        outsideType = boundaryType

        # move points from boundary to inside
        deleteList = None
        moveInside = []
        while deleteList is None or len(deleteList) > 0:
            deleteList = []
            for i in range(len(boundaryIdxs)):
                if boundaryIdxs[i] == boundaryIdxs[(i + 2) % len(boundaryIdxs)]:
                    deleteList.append(i)
                    deleteList.append((i + 1) % len(boundaryIdxs))
                    moveInside.append(boundaryIdxs[(i + 1) % len(boundaryIdxs)])
            boundaryIdxs = np.delete(boundaryIdxs, deleteList)
        insideIdxs = np.hstack((insideIdxs, np.array(moveInside, dtype=int)))

        # now boundary should be free of repitions. assert so (otherwise we have a loop somehwere which sucks)
        assert (len(boundaryIdxs) == len(list(set(boundaryIdxs))))

        # we still have steinerpoints on the boundary, and duplicates in insidevertices and boundary vertices. remove
        # these
        deleteList = []
        moveInside = []
        for i in range(len(boundaryIdxs)):
            bIdx = boundaryIdxs[i]
            if self.isSteiner[bIdx]:
                if len(outsideCons) > 0:
                    edgeIds = np.where((outsideCons[:, 0] == bIdx) | (outsideCons[:, 1] == bIdx))[0]
                    edges = outsideCons[edgeIds]
                    assert (len(edges) <= 2)
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
        insideIdxs = np.hstack((insideIdxs, np.array([v for v in moveInside if v not in insideIdxs], dtype=int)))

        # boundary should now be free of steinerpoints on proper boundaries and as such should also be intersectionfree with insideIdxs. assert so
        for inVIdx in insideIdxs:
            assert (inVIdx not in boundaryIdxs)

        # we are left with removing unnecessary steinerpoints in the middle
        deleteList = []
        insideSteiners = []
        for i in range(len(insideIdxs)):
            iIdx = insideIdxs[i]
            if self.isSteiner[iIdx]:
                insideSteiners.append(iIdx)
                deleteList.append(i)
                if len(insideCons) > 0:
                    edgeIds = np.where((insideCons[:, 0] == iIdx) | (insideCons[:, 1] == iIdx))[0]
                    edges = insideCons[edgeIds]
                    assert (len(edges) == 2 or len(edges) == 0)
                    if len(edges) == 2:
                        newSeg = []
                        for e in edges:
                            newSeg.append(e[np.where(e != iIdx)][0])
                        insideCons[edgeIds[0]] = newSeg
                        insideCons = np.delete(insideCons, [edgeIds[1]], axis=0)

        assert (len(outsideCons) == len(outsideType))

        self.insideVertices = np.delete(insideIdxs, deleteList)
        self.insideSteiners = np.array(insideSteiners)
        self.boundaryVertices = boundaryIdxs
        self.boundarySegs = np.vstack((boundaryIdxs, np.roll(boundaryIdxs, -1))).T
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

        self.wasSolved = False
        self.eval = None
        self.sol = None

        # self.plotMe()

    def plotMe(self):
        self.axs.clear()

        for edgeId in range(len(self.boundarySegs)):
            e = self.boundarySegs[edgeId]
            color = "black"
            if self.boundaryType[edgeId] == "halfin":
                color = "forestgreen"
            elif self.boundaryType[edgeId] == "boundary":
                color = "blue"
            self.axs.plot(*self.numericVerts[e].T, color=color, linewidth=2, zorder=98)

        for edgeId in range(len(self.insideSegs)):
            e = self.insideSegs[edgeId]
            color = 'lime'
            self.axs.plot(*self.numericVerts[e].T, color=color, linewidth=2, zorder=99)

        for vIdx in list(self.boundaryVertices):
            color = 'red' if self.isSteiner[vIdx] else "black"
            self.axs.scatter(*self.numericVerts[[vIdx]].T, s=20, color=color, zorder=100)

        for vIdx in list(self.insideSteiners) + list(self.insideVertices):
            color = 'red' if self.isSteiner[vIdx] else "black"
            self.axs.scatter(*self.numericVerts[[vIdx]].T, s=30, marker="*", color=color, zorder=100)

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
        if len(self.insideSteiners) == 0:
            return []
        return self.localMap[self.insideSteiners]

class StarSolver:
    def __init__(self,faceWeight,insideSteinerWeight,boundarySteinerWeight,halfInSteinerWeight,unconstrainedSteinerWeight,tolerance,axs=None):
        self.faceWeight = faceWeight
        self.insideSteinerWeight = insideSteinerWeight
        self.boundarySteinerWeight = boundarySteinerWeight
        self.halfInSteinerWeight = halfInSteinerWeight
        self.unconstrainedSteinerWeight = unconstrainedSteinerWeight
        self.cleanWeight = - 1/64 #some eps
        self.patialTolerance = tolerance
        self.axs = axs
        self.points = []
        #self.boundary = []
        #self.boundaryType = []
        #self.constraints = []
        #self.circles = []
        #self.rays = []
        self.succesfulSolves = 0

    def getOrientationOfPoints(self,points):
        lowestRightIdx = 0
        lowestRight = points[0]
        for i in range(1,len(points)):
            if points[i].y() < lowestRight.y():
                lowestRight = points[i]
                lowestRightIdx = i
            elif points[i].y() == lowestRight.y() and points[i].x() > lowestRight.x():
                lowestRight = points[i]
                lowestRightIdx = i
        prevI = (lowestRightIdx + (len(points) - 1))%len(points)
        nextI = (lowestRightIdx + 1)%len(points)
        side = eg.onWhichSide(Segment(points[prevI],lowestRight),points[nextI])
        return side

    def internalK0Solve(self,gp, boundary, constraints):
        #check if it is solved by some single vertex. this does not even require convex position
        solutions = []
        hasClean = False
        if len(constraints) > 1:
            return hasClean,solutions

        baseEval = - (self.faceWeight*gp.numBadTris) - (self.insideSteinerWeight*len(gp.insideSteiners))
        eligibleIdxs = range(len(self.points)) if len(constraints) == 0 else constraints[0]
        for pIdx in eligibleIdxs:
            p = self.points[pIdx]
            isGood = True
            for i,j in boundary:
                if i == pIdx or j == pIdx:
                    continue
                if eg.onWhichSide(Segment(self.points[i],self.points[j]),p) != "left":
                    isGood = False
                    break
                if eg.isBadTriangle(self.points[i],self.points[j],p):
                    isGood = False
                    break
            if isGood:
                solutions.append([baseEval+self.cleanWeight,[]])
                hasClean = True
        return hasClean,solutions

    def cutOutOfIntervalSet(self,intervalSet,interval):
        newIntervalSet = []
        for oldInterval in intervalSet:
            if interval[0] < oldInterval[0] and oldInterval[1] < interval[1]:
                continue
            elif oldInterval[0] <= interval[0] and oldInterval[1] < interval[1]:
                newIntervalSet.append([oldInterval[0], min(interval[0], oldInterval[1])])
            elif interval[0] < oldInterval[0] and interval[1] <= oldInterval[1]:
                newIntervalSet.append([max(interval[1], oldInterval[0]), oldInterval[1]])
            elif oldInterval[0] <= interval[0] and interval[1] <= oldInterval[1]:
                newIntervalSet.append([oldInterval[0], interval[0]])
                newIntervalSet.append([interval[1], oldInterval[1]])
            else:
                newIntervalSet.append(oldInterval)
        return newIntervalSet

    def internalConstrainedK1Intervals(self,gp,seg,boundary,rays,circles,allowBoundary = False):

        edgeArrangement = [FieldNumber(0), FieldNumber(1)]
        for c, rsq in circles:
            if (inters := eg.outsideIntersectionsSegmentCircle(c, rsq, seg)) is not None:
                for inter in inters:
                    edgeArrangement.append(eg.getParamOfPointOnSegment(seg, inter))

        for ray in rays:
            if (inter := intersection_point(ray, seg)) is not None:
                edgeArrangement.append(eg.getParamOfPointOnSegment(seg, inter))

        edgeArrangement = list(sorted(edgeArrangement))

        goodIntervals = []
        for i in range(len(edgeArrangement) - 1):
            nextI = i + 1
            s = edgeArrangement[i]
            t = edgeArrangement[i + 1]
            if not allowBoundary:
                if t <= FieldNumber(0):
                    continue
                if s >= FieldNumber(1):
                    continue

            mid = eg.onehalf * (s + t)
            q = seg.source() + (seg.target() - seg.source()).scale(mid)

            if not allowBoundary:
                if mid <= FieldNumber(0):
                    continue
                if mid >= FieldNumber(1):
                    continue
            isGood = True
            # test if q is good
            for i, j in boundary:
                bi, bj = self.points[i], self.points[j]
                if allowBoundary:
                    if q == bi or q == bj:
                        continue
                if eg.colinear(Segment(bi, bj), q):
                    isGood = False
                    break
                if eg.isBadTriangle(bi, bj, q):
                    isGood = False
                    break
            if isGood:
                if len(goodIntervals) == 0:
                    goodIntervals.append([s, t])
                else:
                    if goodIntervals[-1][1] == s:
                        goodIntervals[-1] = [goodIntervals[-1][0], t]
                    else:
                        goodIntervals.append([s, t])
        return goodIntervals

    def internalConstrainedK1Solve(self,gp,seg,boundary,rays,circles):
        baseEval = - (self.faceWeight*gp.numBadTris) - (self.insideSteinerWeight*len(gp.insideSteiners))

        goodIntervals = self.internalConstrainedK1Intervals(gp,seg,boundary,rays,circles)

        for s, t in goodIntervals:
            if s != t:
                mid = eg.onehalf * (s + t)
                q = seg.source() + (seg.target() - seg.source()).scale(mid)
                newQ = eg.roundExactOnSegment(seg, q)
                if s < eg.getParamOfPointOnSegment(seg, newQ) < t:
                    return True, [[baseEval + 1 + self.cleanWeight, [newQ]]]
                else:
                    return False, [[baseEval + 1, [q]]]

        if len(goodIntervals) > 0:
            mid = goodIntervals[0][0]
            q = seg.source() + (seg.target() - seg.source()).scale(mid)
            return False, [[baseEval + 1, [q]]]

        return False, []

    def internalK1Solve(self,gp,boundary,boundaryType,constraints,baseRays,circles,additionalRays=None,filterFunction=None,permitBoundary=True):

        #generate set of candidates
        if len(constraints) > 1:
            return False,[]

        if len(constraints) == 1 and additionalRays is not None:
            assert(False)

        #check for convex position maybe this needs to be rethought if we allow dropout of segments
        for i in range(len(self.points)):
            nextI = (i+1)%len(self.points)
            nextnextI = (i+2)%len(self.points)
            if eg.onWhichSide(Segment(self.points[i],self.points[nextI]),self.points[nextnextI]) == "right":
                return False,[]

        sameRays = False
        if additionalRays is None:
            additionalRays = baseRays
            sameRays = True
        if filterFunction is None:
            filterFunction = lambda x : True

        baseEval = - (self.faceWeight*gp.numBadTris) - (self.insideSteinerWeight*len(gp.insideSteiners))

        if len(constraints) == 1:

            #intersect everything with the constrained segment
            seg = Segment(self.points[constraints[0][0]],self.points[constraints[0][1]])
            return self.internalConstrainedK1Solve(gp,seg,boundary,baseRays,circles)

        else:
            #unconstrained :)

            #first attempt so solve inside
            candidateIntersections = []
            for rayAIdx in range(len(additionalRays)):
                rayA = additionalRays[rayAIdx]

                #intersect ray with other rays
                for rayBIdx in range(rayAIdx+1 if sameRays else 0,len(baseRays)):
                    rayB = baseRays[rayBIdx]
                    if (inter := eg.innerIntersect(rayA.source(),rayA.target(),rayB.source(),rayB.target())) is not None:
                        candidateIntersections.append(inter)

                #intersect ray with circles
                for c,rsq in circles:
                    if (inters := eg.outsideIntersectionsSegmentCircle(c,rsq,rayA)) is not None:
                        for inter in inters:
                            if inter == rayA.source() or inter == rayA.target():
                                continue
                            candidateIntersections.append(inter)

            #circle circle intersections, but only for neighbouring segments, and if result is inside
            for i in range(len(boundary)):
                nextI = (i+1)%len(boundary)
                nextnextI = (i+2)%len(boundary)
                if boundary[i][1] != boundary[nextI][0]:
                    continue
                bi,bni,bnni,bnnni = boundary[i][0],boundary[nextI][0],boundary[nextnextI][0],boundary[nextnextI][1]


                bA = eg.badAngle(self.points[bi],self.points[bni],self.points[bnni])
                if bA == 1 or bA == -1:
                    if not eg.colinear(Segment(self.points[bi],self.points[bnni]),self.points[bnnni]):
                        candidateIntersections.append(eg.altitudePoint(Segment(self.points[bi],self.points[bnni]),self.points[bni]))

            candidateIntersections = [ci for ci in candidateIntersections if filterFunction(ci)]

            #TODO: better evaluate for partially solved faces, that respects on which boundary is being dropped. this requires more complicated hashes i think...
            preemptive = dict()
            hasClean = False
            for ci in candidateIntersections:
                bads = []
                badsHash = 0
                isVeryBad = False
                for bIdx in range(len(boundary)):
                    if len(bads) > self.patialTolerance:
                        break
                    i,j = boundary[bIdx]
                    bi,bj = self.points[i],self.points[j]
                    if eg.colinear(Segment(bi,bj),ci):
                        isVeryBad = True
                        break
                    #only allow bad triangles, if it wants to drop to the outside
                    bA = eg.badAngle(bi,bj,ci)
                    #if (bA :=eg.badAngle(bi,bj,ci)) == 2 or bA == -1:
                    if bA == 2:
                        bads.append(bIdx)
                        badsHash += (2**i)
                    elif bA == 1 or bA == 0:
                        isVeryVad = True
                        break
                if len(bads) > self.patialTolerance:
                    continue
                if isVeryBad:
                    continue
                preemptive[badsHash] = preemptive.get(badsHash,[])+[[bads,ci]]

            solutions = []

            for badsHash in preemptive:
                centroid = Point(FieldNumber(0),FieldNumber(0))
                count = 0
                pointList = preemptive[badsHash]
                myEval = baseEval
                logging.debug("bads: " + str(pointList[0][0]) + " boundary type: " + str(boundaryType))
                for bIdx in pointList[0][0]:
                    #add weight corresponding to the expected cost of solving, which is one on the boundary, and faceweight everywhere else
                    #myEval += self.faceWeight
                    if boundaryType[bIdx] == "boundary":
                        myEval += self.boundarySteinerWeight + (1 / 32) #slightly increase weight to make it not be better than just solving the face, improving stability
                    else:
                        myEval += self.faceWeight
                for _,ci in pointList:
                    centroid += ci
                    count += 1
                centroid = centroid.scale(FieldNumber(1)/FieldNumber(count))
                if (not sameRays) and (len(additionalRays) == 2) and (additionalRays[0].source() == additionalRays[1].source() and additionalRays[0].target() == additionalRays[1].target()) and additionalRays[0].squared_length() > eg.zero:
                    centroid = eg.roundExactOnSegment(additionalRays[0],centroid)
                else:
                    centroid = eg.roundExact(centroid)
                if filterFunction(centroid):
                    centroidHash = 0
                    isVeryBad = False
                    for bIdx in range(len(boundary)):
                        i,j = boundary[bIdx]
                        bi,bj = self.points[i],self.points[j]
                        if eg.onWhichSide(Segment(bi,bj),centroid) == "right":
                            isVeryBad = True
                            break
                        if eg.isBadTriangle(bi,bj,centroid):
                            centroidHash += (2**i)
                    if not isVeryBad and centroidHash == badsHash:
                        if badsHash == 0:
                            myEval += self.cleanWeight
                            hasClean = True
                        pointList = [[[],centroid]] + pointList
                for _,p in pointList:
                    solutions.append([myEval+1,[p]])

            #now append all solutions that live on segments
            if permitBoundary:
                for bIdx in range(len(boundary)):
                    i,j = boundary[bIdx]
                    seg = Segment(self.points[i], self.points[j])

                    #this is only considered clean, if the internal call is clean and we drop onto boundary
                    hasClean,sols = self.internalConstrainedK1Solve(gp,seg,[[idx,jdx] for idx,jdx in boundary if not (idx == i and jdx == j)],additionalRays,circles)
                    for _,sol in sols:
                        if len(sol) == 1:
                            if not filterFunction(sol[0]):
                                continue
                        myEval = baseEval
                        if boundaryType[bIdx] == "boundary":
                            myEval += self.boundarySteinerWeight
                            if hasClean:
                                myEval += self.cleanWeight
                        elif boundaryType[bIdx] == "halfin":
                            myEval += self.halfInSteinerWeight
                        else:
                            myEval += self.unconstrainedSteinerWeight
                        solutions.append([myEval,sol])

            return hasClean, solutions


    def constructRays(self,boundary,rootPoints=None):
        if rootPoints == None:
            rootPoints = [[FieldNumber(0),FieldNumber(1)] for _ in boundary]



        rays = []
        for bIdx in range(len(boundary)):
            i,j = boundary[bIdx]
            rps = rootPoints[bIdx]
            diff = self.points[j]-self.points[i]
            orth = Point(eg.zero-diff.y(),diff.x())
            if eg.onWhichSide(Segment(self.points[i],self.points[j]), self.points[i] + orth) != "left":
                orth = orth.scale(FieldNumber(-1))

            for root in rps:
                ray = Segment(self.points[i] + diff.scale(root),self.points[i] + diff.scale(root)+orth)
                intersected = False
                for offset in range(1,len(boundary)):
                    qId,nextQId = boundary[(bIdx+offset)%len(boundary)]
                    if eg.onWhichSide(ray,self.points[qId]) != "left" and eg.onWhichSide(ray,self.points[nextQId]) != "right":
                        inter = eg.supportingRayIntersectSegment(ray,Segment(self.points[qId],self.points[nextQId]))
                        if inter == None:
                            #this means we are walking behind i think
                            continue
                        rays.append(Segment(ray.source(),inter))
                        #if gp.axs != None:
                        #    gp.axs.plot([eg.numericPoint(self.rays[-1].source())[0], eg.numericPoint(self.rays[-1].target())[0]],
                        #     [eg.numericPoint(self.rays[-1].source())[1], eg.numericPoint(self.rays[-1].target())[1]], color="blue")
                        intersected = True
                        break
                if not intersected:
                    assert(False)

        return rays

    def constructCircles(self, boundary):
        circles = []
        for i,j in boundary:
            c = (self.points[i]+self.points[j]).scale(eg.onehalf)
            rsq = eg.distsq(c,self.points[j])
            circles.append([c,rsq])
        return circles

    def constructRaysAndCircles(self,boundary,rootPoints=None):
        circles = self.constructCircles(boundary)
        rays = self.constructRays(boundary,rootPoints)
        return rays,circles


    def internalK2Solve(self,gp,boundary,boundaryType,constraints,rays,circles):
        #this will be fun.
        if len(constraints) > 1:
            return False,[]
        if len(constraints) == 1:
            #identify two subfaces, determine constraining interval, and solve other face orthogonal to constraining interval
            con = constraints[0]
            subface1 = [con[0]]
            sf1boundaryType = []
            it = con[0]
            while it != con[1]:
                sf1boundaryType.append(boundaryType[it])
                it = (it + 1) % len(self.points)
                subface1.append(it)
            sf1boundaryType.append("None")

            segSF2Orientation = Segment(self.points[con[0]],self.points[con[1]])
            subface1Boundary = np.vstack((subface1,np.roll(subface1,-1))).T
            sf1rays,sf1circles = self.constructRaysAndCircles(subface1Boundary)
            forSubface2Intervals = self.internalConstrainedK1Intervals(gp, segSF2Orientation, subface1Boundary[:-1], sf1rays, sf1circles)
            sf1Filter = lambda x: len([interval for interval in forSubface1Intervals if interval[0] <= eg.getParamOfPointOnSegment(segSF1Orientation,eg.altitudePoint(segSF1Orientation,x)) <= interval[1]]) > 0

            subface2 = [con[1]]
            sf2boundaryType = []
            it = con[1]
            while it != con[0]:
                sf2boundaryType.append(boundaryType[it])
                it = (it + 1) % len(self.points)
                subface2.append(it)
            sf2boundaryType.append("None")

            segSF1Orientation = Segment(self.points[con[1]],self.points[con[0]])
            subface2Boundary = np.vstack((subface2,np.roll(subface2,-1))).T
            sf2rays,sf2circles = self.constructRaysAndCircles(subface2Boundary)
            forSubface1Intervals = self.internalConstrainedK1Intervals(gp, segSF1Orientation, subface2Boundary[:-1], sf2rays, sf2circles)
            sf2Filter = lambda x: len([interval for interval in forSubface2Intervals if interval[0] <= eg.getParamOfPointOnSegment(segSF2Orientation,eg.altitudePoint(segSF2Orientation,x)) <= interval[1]]) > 0



            orthRaysInSF1 = []
            for interval in forSubface1Intervals:
                orthRaysInSF1.append(self.constructRays(subface1Boundary,[[] for _ in subface1Boundary[:-1]]+[interval]))

            orthRaysInSF2 = []
            for interval in forSubface2Intervals:
                orthRaysInSF2.append(self.constructRays(subface2Boundary,[[] for _ in subface2Boundary[:-1]]+[interval]))

            hasClean = False
            sols = []

            #first attempt to solve either subface by a vertex. if that is possible, check if there is a solution for the other face that does not create a bad face
            restoreV = self.patialTolerance
            self.patialTolerance = 0
            sf1HasClean,sf1Sols = self.internalK0Solve(gp,subface1Boundary,[])
            if len(sf1Sols) > 0:
                #subface 1 can be solved by a vertex. attempt to solve other subface with one vertex
                sf2HasClean,sf2Sols = self.internalSolve(gp,1,subface2Boundary,sf2boundaryType,[],sf2rays,sf2circles,permitBoundary=False)
                for myEval,sol in sf2Sols:
                    hasClean = hasClean or sf2HasClean
                    sols.append([myEval,sol])

            #other way around
            sf2HasClean,sf2Sols = self.internalK0Solve(gp,subface2Boundary,[])
            if len(sf2Sols) > 0:
                #subface 1 can be solved by a vertex. attempt to solve other subface with one vertex
                sf1HasClean,sf1Sols = self.internalSolve(gp,1,subface1Boundary,sf1boundaryType,[],sf1rays,sf1circles,permitBoundary=False)
                for myEval,sol in sf1Sols:
                    hasClean = hasClean or sf1HasClean
                    sols.append([myEval,sol])


            self.patialTolerance = restoreV




            for orthRays in orthRaysInSF1:
                sf1HasClean,sf1Sols = self.internalK1Solve(gp,subface1Boundary[:-1],sf1boundaryType[:-1],[],sf1rays,sf1circles,orthRays,sf1Filter)
                hasClean = hasClean or sf1HasClean
                for myEval,sol in sf1Sols:
                    if len(sol) == 1:
                        other = eg.altitudePoint(segSF1Orientation,sol[0])
                        if other == segSF1Orientation.source() or other == segSF1Orientation.target():
                            sols.append([myEval,sol])
                        else:
                            sols.append([myEval+1,sol+[other]])

            pass

            for orthRays in orthRaysInSF2:
                sf2HasClean,sf2Sols = self.internalK1Solve(gp,subface2Boundary[:-1],sf2boundaryType[:-1],[],sf2rays,sf2circles,orthRays,sf2Filter)
                hasClean = hasClean or sf2HasClean
                for myEval,sol in sf2Sols:
                    if len(sol) == 1:
                        other = eg.altitudePoint(segSF2Orientation,sol[0])
                        if other == segSF2Orientation.source() or other == segSF2Orientation.target():
                            sols.append([myEval,sol])
                        else:
                            sols.append([myEval+1,sol+[other]])

            #now construct new ray and circle objects
            return hasClean,sols
        else:
            return False,[]


    def internalSolve(self,gp:GeometricSubproblem,maxK,boundary,boundaryType,constraints,rays,circles,permitBoundary=True):
        if maxK == 0:
            return self.internalK0Solve(gp,boundary,constraints)
        else:
            lowerHasClean,lowerLevelSols = self.internalSolve(gp,maxK-1,boundary,boundaryType,constraints,rays,circles)
            if lowerHasClean:
                return lowerHasClean,lowerLevelSols
            if maxK == 1:
                hasClean, sols =  self.internalK1Solve(gp,boundary,boundaryType,constraints,rays,circles,permitBoundary=permitBoundary)
                return (hasClean or lowerHasClean), sols + lowerLevelSols
            if maxK >= 2:
                #TODO
                hasClean, sols =  self.internalK2Solve(gp,boundary,boundaryType,constraints,rays,circles)
                return (hasClean or lowerHasClean), sols + lowerLevelSols
        assert(False)

    def solve(self, gp : GeometricSubproblem):
        if gp.wasSolved:
            return gp.eval, gp.sol

        if len(gp.insideVertices)>0:
            return 0,None

        #gp.plotMe()

        #prepare internal data

        self.points = [Point(*v) for v in gp.exactVerts[gp.boundaryVertices]]
        boundary = []
        constraints = []
        boundaryType = gp.boundaryType

        for bon in gp.boundarySegs:
            internalBon = [np.where(gp.boundaryVertices == bon[0])[0][0], np.where(gp.boundaryVertices == bon[1])[0][0]]
            boundary.append(internalBon)

        for con in gp.insideSegs:
            if len(np.where(gp.boundaryVertices == con[1])[0]) == 0:
                pass
            internalCon = [np.where(gp.boundaryVertices == con[0])[0][0], np.where(gp.boundaryVertices == con[1])[0][0]]
            constraints.append(internalCon)

        boundary = np.array(boundary)
        constraints = np.array(constraints)

        if self.getOrientationOfPoints(self.points) == "right":
            self.points = self.points[::-1]
            boundaryType = np.roll(boundaryType[::-1],-1)
            #self.boundary = np.full(self.boundary.shape,len(gp.boundaryVertices)-1) - self.boundary
            constraints = np.full(constraints.shape,len(gp.boundaryVertices)-1) - constraints

        assert(self.getOrientationOfPoints(self.points) == "left")

        rays,circles = self.constructRaysAndCircles(boundary)

        #call internal solvers
        k = max(1,len(gp.insideSteiners))
        if gp.numBadTris == 0:
            k -= 1
        hasClean, solutions = self.internalSolve(gp,k,boundary,boundaryType,constraints,rays,circles)
        bestEval = 0
        bestSol = None
        for eval,sol in solutions:
            if eval < bestEval:
                bestEval = eval
                bestSol = sol
        gp.wasSolved = True
        gp.eval = bestEval
        gp.sol = bestSol
        self.succesfulSolves += 1
        return bestEval,bestSol