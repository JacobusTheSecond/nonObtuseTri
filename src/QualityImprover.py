import copy

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from cgshop2025_pyutils import InstanceDatabase,ZipSolutionIterator,ZipWriter,Cgshop2025Solution,verify
from cgshop2025_pyutils.geometry import FieldNumber, Point, Segment

from exact_geometry import isBadTriangle, badAngle, badVertex, badness,innerIntersect,circumcenter,altitudePoint
from hacky_internal_visualization_stuff import plotExact, plot, plot_solution #internal ugly functions I dont want you to see


import triangle as tr #https://rufat.be/triangle/

import math

rules = dict(dropLimit=1000,tryDropBeforeSplit=False,cornerRule=False,aggressiveDropping=False,withRandomRefine=True)

def numericalangles(A:Point,B:Point,C:Point):
    # Square of lengths be a2, b2, c2
    a2 = float(Segment(B,C).squared_length())
    b2 = float(Segment(A,C).squared_length())
    c2 = float(Segment(A,B).squared_length())

    # length of sides be a, b, c
    a = math.sqrt(a2)
    b = math.sqrt(b2)
    c = math.sqrt(c2)

    assert(a != 0)
    assert(b != 0)
    assert(c != 0)

    # From Cosine law
    alpha = math.acos((b2 + c2 - a2) / (2 * b * c))
    betta = math.acos((a2 + c2 - b2) / (2 * a * c))
    gamma = math.acos((a2 + b2 - c2) / (2 * a * b))

    # Converting to degree
    alpha = alpha * 180 / math.pi
    betta = betta * 180 / math.pi
    gamma = gamma * 180 / math.pi

    return np.array((alpha,betta,gamma))

def tallyBad(Aplot, Aexact):
    badTri = []
    bad = []
    C = tr.triangulate(Aplot, 'p')
    assert(len(C['vertices']) == len(Aplot['vertices']))
    for tri in C['triangles']:
        if isBadTriangle(Aexact['vertices'][tri[0]], Aexact['vertices'][tri[1]], Aexact['vertices'][tri[2]]):
            bad.append(float(badness(Aexact['vertices'][tri[0]], Aexact['vertices'][tri[1]], Aexact['vertices'][tri[2]])))
            badTri.append(tri)
    return (bad,badTri)

def isSegment(Aexact,querySeg):
    revseg = [querySeg[1],querySeg[0]]
    for i in range(len(Aexact['segments'])):
        seg = Aexact['segments'][i]
        if np.all(seg == querySeg) or np.all(seg == revseg):
            return i
    return None


def dropAltitudesOnSegments(Ain,Aexact,Aplot,axs,dontDrop = False, aggressive = False):
    global rules
    num = rules['dropLimit']
    aggressive = rules['aggressiveDropping']
    cornerrule = rules['cornerRule']

    curnum = 0
    inserted = True
    bad = []
    badTri = []
    restartIdx = 0
    completeIt = False
    foundBad = 0
    altitudeMap = dict()
    while not (completeIt and inserted == False):
        # print(completeIt,inserted)
        if inserted == False:
            restartIdx = 0
        completeIt = restartIdx == 0
        if completeIt:
            foundBad = 0
        #print(Aexact['segments'])
        B = tr.triangulate(Aplot, "p")
        #quietShow(Ain,Aplot,Aexact,"internal",axs,False)
        inserted = False
        bad = []
        for i in range(restartIdx, len(B['triangles'])):
            triIdx = B['triangles'][i]
            triangle = B['vertices'][triIdx]
            a, b, c = (Aexact['vertices'][triIdx[0]], Aexact['vertices'][triIdx[1]], Aexact['vertices'][triIdx[2]])
            if (isBadTriangle(a, b, c)):
                foundBad += 1
                # bad triangle
                bad.append(float(badness(a, b, c)))
                badTri.append(triIdx)
                # identify circumcenter (should be exact)
                cc = circumcenter(a, b, c)

                idxs = triIdx
                idxs = np.roll(idxs,-badAngle(a,b,c))

                p = badVertex(a,b,c)

                closestSegmentIndex = None
                closestSegment = None

                closestSegmentIndex = isSegment(B,[idxs[1], idxs[2]])
                if closestSegmentIndex != None:
                    closestSegment = (Aexact['vertices'][idxs[1]], Aexact['vertices'][idxs[2]])

                if closestSegment == None and aggressive:

                    # check with every constraint edge if the line from p to the circumcenter intersects it
                    # store the closest intersecting segment in closestSegment
                    closestIntersection = None
                    baseline = Segment(p, cc)
                    for segmentIndex in range(len(B['segments'])):
                        segment = B['segments'][segmentIndex]
                        intersection = innerIntersect(p, cc, Aexact['vertices'][segment[0]], Aexact['vertices'][segment[1]])
                        if intersection != None:
                            if closestIntersection == None or \
                                    np.sqrt(float(Segment(p, intersection).squared_length())) < \
                                    np.sqrt(float(Segment(p, closestIntersection).squared_length())):
                                closestSegment = (Aexact['vertices'][segment[0]], Aexact['vertices'][segment[1]])
                                closestIntersection = intersection
                                closestSegmentIndex = segmentIndex

                if closestSegment != None and not dontDrop:
                    # we can insert circumcenter and retriangulate
                    # need to split the edge
                    ap = altitudePoint(Segment(closestSegment[0], closestSegment[1]), p)

                    if idxs[0] in altitudeMap:
                        altitudeMap[idxs[0]].append(len(Aexact['vertices']))
                    else:
                        altitudeMap[idxs[0]] = [len(Aexact['vertices'])]


                    # add altitude point
                    Aexact['vertices'].append(ap)
                    ap1 = float(ap.x())
                    ap2 = float(ap.y())
                    Aplot['vertices'] = np.vstack((Aplot['vertices'], np.array([[ap1, ap2]])))

                    # remove segment
                    deletor = Aexact['segments'][closestSegmentIndex]
                    Aexact['segments'] = np.vstack((np.delete(Aexact['segments'], (closestSegmentIndex), axis=0),
                                                    np.array([[deletor[0], len(Aexact['vertices']) - 1],
                                                              [len(Aexact['vertices']) - 1, deletor[1]]])))

                    #ax = quietShowHorrible(Ain,Aplot,Aexact,"internal",axs)
                    #ax.scatter([Aplot['vertices'][idxs[0]][0]],[Aplot['vertices'][idxs[0]][1]], color='yellow', zorder=100)
                    #ax.scatter([ap1],[ap2], color='yellow', zorder=100)
                    #plt.show()

                    if cornerrule:
                        #print(altitudeMap[idxs[0]])
                        toadd = None
                        if len(altitudeMap[idxs[0]])>1:
                            #check if the segments, the altitude dropped into has a common point
                            for seg in Aexact['segments']:
                                if np.all([deletor[0],altitudeMap[idxs[0]][-2]] == seg) or np.all([altitudeMap[idxs[0]][-2],deletor[0]] == seg):
                                    assert(toadd == None)
                                    toadd = [idxs[0],deletor[0]]
                                elif np.all([deletor[1], altitudeMap[idxs[0]][-2]] == seg) or np.all([altitudeMap[idxs[0]][-2], deletor[1]] == seg):
                                    assert (toadd == None)
                                    toadd = [idxs[0], deletor[1]]
                        if toadd != None:
                            if isSegment(Aexact,toadd) == None:
                                Aexact['segments'] = np.vstack((Aexact['segments'],[toadd]))
                                #ax = quietShowHorrible(Ain, Aplot, Aexact, "internal", axs)
                                #ax.plot([Aplot['vertices'][toadd[0]][0],Aplot['vertices'][toadd[1]][0]],[Aplot['vertices'][toadd[0]][1],Aplot['vertices'][toadd[1]][1]], color='yellow',zorder=100)
                                #ax.scatter([ap1], [ap2], color='yellow', zorder=100)
                                #plt.show()

                    Aplot['segments'] = Aexact['segments']

                    inserted = True
                    restartIdx = i + 1
                    completeIt = restartIdx == 0
                    curnum += 1
                    if curnum >= num:
                        dontDrop=True
                    break
    return (inserted,Aexact,Aplot)

#returns opposing index if true, else none
def edgeOnTri(edge,tri):
    revedge = [edge[1],edge[0]]
    for i in range(3):
        if np.all(edge == [tri[i],tri[(i+1)%3]]) or np.all(revedge == [tri[i],tri[(i+1)%3]]):
            return (i+2)%3
    return None

def createSteinerPoints(Ain,Aplot, Aexact, maxNum, verbosity=0):
    global rules
    tryDropping = rules['tryDropBeforeSplit']
    wiggleTally = 0
    wiggleMap = []
    C = None
    upper = max(5, maxNum)
    lower = 1
    C = tr.triangulate(Aplot, "pq0.000001U90.00001S1")
    if (len(Aplot['segments']) == len(C['segments'])):
        while lower + 1 < upper:
            mid = (lower + upper) // 2
            C = tr.triangulate(Aplot, "pq0.000001U90.00001S" + str(mid))
            if (len(Aplot['segments']) == len(C['segments'])):
                lower = mid
            else:
                upper = mid
    C = tr.triangulate(Aplot, "pq0.000001U90.00001S" + str(lower))

    if len(Aplot['vertices']) < len(C['vertices']):
        inserted = True

    if (len(Aplot['segments']) != len(C['segments'])):
        # if we are here, we need to identify the split segment, and akin to the triangle code we need to split that segment exactly in the middle
        assert (len(Aplot['segments']) + 1 == len(C['segments']))
        # first identify split segment
        mirroringIndex = len(C['vertices']) - 1
        indicesUsingIndex = []
        for seg in C['segments']:
            if seg[0] == mirroringIndex:
                indicesUsingIndex.append(seg[1])
            elif seg[1] == mirroringIndex:
                indicesUsingIndex.append(seg[0])
        assert (len(indicesUsingIndex) == 2)

        #first attempt to drop some altitude onto it
        couldDrop = False
        if tryDropping:
            temp = tr.triangulate(Aplot,'p')
            #ax = quietShowHorrible(Ain,Aplot,Aexact,"internal")
            #plt.show()
            for tri in temp['triangles']:
                opposing = edgeOnTri(indicesUsingIndex,tri)
                if opposing == None:
                    continue
                sortedTri = np.roll(tri,-opposing)

                badn = badness(Aexact['vertices'][sortedTri[0]],Aexact['vertices'][sortedTri[1]],Aexact['vertices'][sortedTri[2]])
                badA = badAngle(Aexact['vertices'][sortedTri[0]],Aexact['vertices'][sortedTri[1]],Aexact['vertices'][sortedTri[2]])

                if badA == 0 or badA == -1 and badn != FieldNumber(0):
                    ap = altitudePoint(Segment(Aexact['vertices'][sortedTri[1]], Aexact['vertices'][sortedTri[2]]), Aexact['vertices'][sortedTri[0]])

                    # add altitude point
                    Aexact['vertices'].append(ap)
                    ap1 = float(ap.x())
                    ap2 = float(ap.y())
                    Aplot['vertices'] = np.vstack((Aplot['vertices'], np.array([[ap1, ap2]])))

                    # remove segment
                    Aplot['segments'] = C['segments']
                    Aexact['segments'] = C['segments']

                    #ax = quietShowHorrible(Ain, Aplot, Aexact, "internal")
                    #ax.scatter([Aplot['vertices'][-1][0]], [Aplot['vertices'][-1][1]], color='yellow', zorder=100)
                    # ax.scatter([ap1],[ap2], color='yellow', zorder=100)
                    #plt.show()

                    couldDrop = True
                    break
        #print(couldDrop)
        if not couldDrop:

            sum = Aexact['vertices'][indicesUsingIndex[0]] + Aexact['vertices'][indicesUsingIndex[1]]
            midpoint = Point(sum.x() * FieldNumber(0.5), sum.y() * FieldNumber(0.5))

            Aexact['vertices'].append(midpoint)
            mp1 = float(midpoint.x())
            mp2 = float(midpoint.y())
            Aplot['vertices'] = np.vstack((Aplot['vertices'], np.array([[mp1, mp2]])))

            # segment should now be able to copy from C
            Aplot['segments'] = C['segments']
            Aexact['segments'] = C['segments']

    else:
        wiggleTally = 0
        oldLen = len(Aexact['vertices'])
        for v in np.array(C['vertices'][len(Aplot['vertices']) - len(C['vertices']):]):
            Aexact['vertices'].append(Point(v[0], v[1]))
        for v in Aexact['vertices']:
            wiggleMap.append([])

        for tri in C['triangles']:
            if isBadTriangle(Aexact['vertices'][tri[0]], Aexact['vertices'][tri[1]], Aexact['vertices'][tri[2]]):
                for idx in tri:
                    wiggleMap[idx].append(tri)

        for i in range(oldLen, len(Aexact['vertices'])):
            badButShouldnt = []
            # check if i participates in a triangle in C that SHOULD be good, but is bad by small error
            for tri in wiggleMap[i]:
                if tri[0] == i or tri[1] == i or tri[2] == i:
                    # i participates in tri

                    # if tri is bad, but only slightly:
                    if max(numericalangles(Aexact['vertices'][tri[0]], Aexact['vertices'][tri[1]],
                                           Aexact['vertices'][tri[2]])) < 90.01:
                        badButShouldnt.append(tri)
            # next we should sanity-check that i is not the obtuse angle
            if len(badButShouldnt) > 0:
                tri = badButShouldnt[0]
                while tri[0] != i:
                    tri = np.roll(tri, -1)
                badA = badAngle(Aexact['vertices'][tri[0]], Aexact['vertices'][tri[1]], Aexact['vertices'][tri[2]])
                if badA <= 0:  # i.e. either none or i itself, which it should not be
                    if verbosity > 0:
                        print("too scared to wiggle...")
                else:
                    otherA = 3 - badA
                    # I should do math here but oh well:
                    origin = Point(0, 0)
                    other = Aexact['vertices'][tri[otherA]] - Aexact['vertices'][tri[badA]]
                    orth = Segment(origin, Point(FieldNumber(0) - other.y(), other.x()))
                    # project it
                    Aexact['vertices'][i] = Aexact['vertices'][tri[badA]] + altitudePoint(orth, Aexact['vertices'][i] -
                                                                                          Aexact['vertices'][tri[badA]])
                    wiggleTally += 1
        addedpoints = [[float(Aexact['vertices'][i].x()), float(Aexact['vertices'][i].y())] for i in
                       range(oldLen, len(Aexact['vertices']))]
        Aplot['vertices'] = np.vstack((Aplot['vertices'], np.array(addedpoints)))
    if verbosity > 0:
        print("wiggled", wiggleTally, "new vertices")
    return (Aplot,Aexact)

def removeNumericallyBadTriangles(Ain,Aplot,Aexact,badTri):

    #deepcopy
    Aplotcopy = dict(vertices=copy.deepcopy(Aplot['vertices']), segments=copy.deepcopy(Aplot['segments']))
    Aexactcopy = dict(vertices=[], segments=copy.deepcopy(Aexact['segments']))
    for p in Aexact['vertices']:
        Aexactcopy['vertices'].append(Point(FieldNumber(p.x().exact()),FieldNumber(p.y().exact())))

    vs = []
    for tri in badTri:
        for v in tri:
            if v >= len(Ain['vertices']) and not v in vs:
                vs.append(v)
    vs = list(reversed(sorted(vs)))
    for v in vs:
        # merge participating segmets
        deletorIdx = []
        newEdge = []
        for i in range(len(Aexactcopy['segments'])):
            if Aexactcopy['segments'][i][0] == v:
                deletorIdx.append(i)
                newEdge.append(Aexactcopy['segments'][i][1])
            elif Aexactcopy['segments'][i][1] == v:
                deletorIdx.append(i)
                newEdge.append(Aexactcopy['segments'][i][0])
        assert(len(deletorIdx) == 0 or len(deletorIdx) == 2)
        if len(deletorIdx)==2:
            Aexactcopy['segments'][deletorIdx[0]] = newEdge
            Aexactcopy['segments'] = np.delete(Aexactcopy['segments'], (deletorIdx[1]), axis=0)

        # remap all segments
        for i in range(len(Aexactcopy['segments'])):
            if Aexactcopy['segments'][i][0] >= v:
                Aexactcopy['segments'][i][0] -= 1
            if Aexactcopy['segments'][i][1] >= v:
                Aexactcopy['segments'][i][1] -= 1
        Aexactcopy['vertices'].pop(v)
        Aplotcopy['vertices'] = np.delete(Aplotcopy['vertices'], (v), axis=0)
    Aplotcopy['segments'] = Aexactcopy['segments']
    return (Aplotcopy,Aexactcopy)

def quietShow(Ain, Aplot, Aexact, name, axs,withDelay=True):
    pausetime = 0.01
    #fig, axs = plt.subplots(1, 1)
    axs.clear()
    C = tr.triangulate(Aplot, 'p')
    plotExact(Ain, Aexact, C, name, axs)
    #plt.show()
    plt.draw()
    if withDelay:
        plt.pause(pausetime)
    else:
        plt.pause(0.001)

def quietShowHorrible(Ain, Aplot, Aexact, name):
    pausetime = 0.03
    fig, axs = plt.subplots(1, 1)
    axs.clear()
    C = tr.triangulate(Aplot, 'p')
    plotExact(Ain, Aexact, C, name, axs)
    return axs
    #plt.draw()
    #plt.pause(pausetime)

def numericalWiggle(Ain,Aplot, Aexact,verbosity=0):
    wiggleTally = 0
    #lastly we try to wiggle everything, if needed

    C = tr.triangulate(Aplot, 'p')

    #multiple passes just to be sure?
    #fuck it, wiggle order is random!
    for round in range(3):

        fullMap = []
        for v in Aexact['vertices']:
            fullMap.append([])
        for tri in C['triangles']:
            if max(numericalangles(Aexact['vertices'][tri[0]], Aexact['vertices'][tri[1]],
                                           Aexact['vertices'][tri[2]])) < 90.01 and isBadTriangle(
                            Aexact['vertices'][tri[0]], Aexact['vertices'][tri[1]], Aexact['vertices'][tri[2]]):
                for idx in tri:
                    fullMap[idx].append(tri)

        toWiggleCount = 0
        for tris in fullMap:
            if len(tris) > 0:
                toWiggleCount += 1

        vertexlist = list(range(len(Ain['vertices']),len(Aexact['vertices'])))
        np.random.shuffle(vertexlist)
        for i in vertexlist:
            #check if i lives on an constraint edge, which would suck!
            restrained = False
            for seg in Aexact['segments']:
                if seg[0]==i or seg[1]==i:
                    restrained = True
            if restrained:
                #there is probably something we can do here, but whatever for now
                continue
            badButShouldnt = []
            # check if i participates in a triangle in C that SHOULD be good, but is bad by small error
            for tri in fullMap[i]:
                if tri[0] == i or tri[1] == i or tri[2] == i:
                    # i participates in tri

                    # if tri is bad, but only slightly:
                    if max(numericalangles(Aexact['vertices'][tri[0]], Aexact['vertices'][tri[1]],
                                           Aexact['vertices'][tri[2]])) < 90.01 and isBadTriangle(
                            Aexact['vertices'][tri[0]], Aexact['vertices'][tri[1]], Aexact['vertices'][tri[2]]):
                        badButShouldnt.append(tri)
            # next we should sanity-check that i is not the obtuse angle
            if len(badButShouldnt) > 0:
                tri = badButShouldnt[0]
                while tri[0] != i:
                    tri = np.roll(tri, -1)
                badA = badAngle(Aexact['vertices'][tri[0]], Aexact['vertices'][tri[1]], Aexact['vertices'][tri[2]])
                if badA == 0 and toWiggleCount>6:
                    if verbosity > 0:
                        print("too scared to wiggle...")
                elif badA == 0 and toWiggleCount<=6:
                    dropper = Segment(Aexact['vertices'][tri[1]], Aexact['vertices'][tri[2]])
                    ap = altitudePoint(dropper, Aexact['vertices'][tri[0]])

                    # add altitude point
                    Aexact['vertices'].append(ap)
                    ap1 = float(ap.x())
                    ap2 = float(ap.y())
                    Aplot['vertices'] = np.vstack((Aplot['vertices'], np.array([[ap1, ap2]])))

                    segmentIdx = isSegment(Aexact,[tri[1],tri[2]])
                    if segmentIdx != None:
                        deletor = Aexact['segments'][segmentIdx]
                        Aexact['segments'] = np.vstack((np.delete(Aexact['segments'], (segmentIdx), axis=0),
                                                        np.array([[deletor[0], len(Aexact['vertices']) - 1],
                                                                  [len(Aexact['vertices']) - 1, deletor[1]]])))
                        Aplot['segments'] = Aexact['segments']

                    return (Aplot, Aexact,True)
                else:
                    otherA = 3 - badA
                    # I should do math here but oh well:
                    origin = Point(0, 0)
                    other = Aexact['vertices'][tri[otherA]] - Aexact['vertices'][tri[badA]]
                    orth = Segment(origin, Point(FieldNumber(0) - other.y(), other.x()))
                    # project it
                    Aexact['vertices'][i] = Aexact['vertices'][tri[badA]] + altitudePoint(orth,
                                                                                          Aexact['vertices'][i] -
                                                                                          Aexact['vertices'][
                                                                                              tri[badA]])
                    wiggleTally+=1
    if verbosity > 0:
        print("wiggled",wiggleTally,"old vertices")
    return (Aplot,Aexact,False)

def parseAsSolution(Ain,Aplot,Aexact,name):
    C = tr.triangulate(Aplot, 'p')
    inneredges = []
    for tri in C['triangles']:
        edges = ([tri[0],tri[1]],[tri[1],tri[2]],[tri[2],tri[0]])
        #check if edge is already added or are in segments
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
        for i in range(len(Ain['vertices']),len(Aexact['vertices'])):
            sx.append(Aexact['vertices'][i].x().exact())
            sy.append(Aexact['vertices'][i].y().exact())
    return Cgshop2025Solution(instance_uid = name, steiner_points_x=sx,steiner_points_y=sy,edges=inneredges)

def randomStableSingleVertexDelete(Ain,Aplot,Aexact):
    bad,badTri = tallyBad(Aplot,Aexact)
    assert(len(bad) == 0)

    #free vertices:
    freeVertices = list(range(len(Ain['vertices']),len(Aexact['vertices'])))
    for seg in Aexact['segments']:
        for idx in seg:
            if idx in freeVertices:
                freeVertices.remove(idx)

    deleteList = []
    currentSegments = copy.deepcopy(Aexact['segments'])

    #np.random.shuffle(freeVertices)
    for idx in reversed(freeVertices):
        tempDeleteList = deleteList + [idx]
        Aplotcopy = dict(vertices=copy.deepcopy(Aplot['vertices']))
        Aexactcopy = dict(vertices=[])
        for p in Aexact['vertices']:
            Aexactcopy['vertices'].append(Point(FieldNumber(p.x().exact()),FieldNumber(p.y().exact())))

        for v in tempDeleteList:
            Aplotcopy['vertices'] = np.delete(Aplotcopy['vertices'], (v), axis=0)
            Aexactcopy['vertices'].pop(v)

        tempSegs = copy.deepcopy(currentSegments)

        for i in range(len(tempSegs)):
            if tempSegs[i][0] == idx:
                print("huh...")
            if tempSegs[i][1] == idx:
                print("huh...")
            if tempSegs[i][0] > idx:
                tempSegs[i][0] -= 1
            if tempSegs[i][1] > idx:
                tempSegs[i][1] -= 1
        Aplotcopy['segments'] = tempSegs
        Aexactcopy['segments'] = Aplotcopy['segments']

        bad, badTri = tallyBad(Aplotcopy, Aexactcopy)

        if len(badTri) == 0:
            currentSegments = tempSegs
            deleteList.append(idx)

    Aplotcopy = dict(vertices=copy.deepcopy(Aplot['vertices']))
    Aexactcopy = dict(vertices=[])
    for p in Aexact['vertices']:
        Aexactcopy['vertices'].append(Point(FieldNumber(p.x().exact()), FieldNumber(p.y().exact())))

    for v in deleteList:
        Aplotcopy['vertices'] = np.delete(Aplotcopy['vertices'], (v), axis=0)
        Aexactcopy['vertices'].pop(v)

    Aplotcopy['segments'] = currentSegments
    Aexactcopy['segments'] = Aplotcopy['segments']

    return (len(deleteList) != 0,Aplotcopy,Aexactcopy)



def improveQuality(instance,withShow=True,axs=None,verbosity=0):
    if withShow:
        assert axs != None
        plt.ion()

    if verbosity > 0:
        print("Phase 0")

    #########
    #
    # Phase 0: init
    #          we work with the triangulation data type required for triangle library. To make it numerically stable
    #          we mirror everything in an exact copy working with exact coordinates.
    #
    #          Ain:    copy of input triangulated once to properly init boundary (if non-boundary points lie on it)
    #          Aplot:  numerically unstable data-structure used as input to triangle
    #          Aexact: exact copy with exact coordinates
    #
    #########
    Ain = tr.triangulate(convert(instance),'p')
    Ain = dict(vertices=Ain['vertices'],segments=Ain['segments'])
    while True:#to modify later for improvement stages i guess
        Aexact = dict(segments=Ain['segments'])
        Aexact['vertices'] = []
        for x, y in zip(instance.points_x, instance.points_y):
            Aexact['vertices'].append(Point(x, y))

        Aplot = dict(vertices=Ain['vertices'],segments=Ain['segments'])


        bad,badTri = tallyBad(Aplot,Aexact)
        minSofar = len(bad)
        if rules['dropLimit'] == "dynamic":
            rules['dropLimit'] = max(7,minSofar//10)
        roundsSinceImprov = 0

        #1000 rounds should always suffice, but who knows...
        for n in range(1000):

            startLen = len(Aexact['vertices'])

            if withShow:
                quietShow(Ain,Aplot,Aexact,instance.instance_uid,axs)

            if verbosity > 0:
                print("Phase 1")
            #########
            #
            # Phase 1: AltitudeDropper
            #          if there are altitudes that should be dropped onto constraint segments, do it!
            #
            #########
            inserted,Aexact,Aplot = dropAltitudesOnSegments(Ain,Aexact,Aplot,axs)

            if withShow:
                quietShow(Ain,Aplot,Aexact,instance.instance_uid,axs)
            #if inserted:
            #    continue
            if verbosity > 0:
                print("badnesses:",list(np.sort(bad)))

            if verbosity > 0:
                print("Phase 2")
            #########
            #
            # Phase 2: Horrible Heuristic
            #          If we are essentially done, but there are few numerically unstable triangles, try to wiggle, and if
            #          not possible, remove numerically bad triangles
            #
            #########
            bad,badTri = tallyBad(Aplot,Aexact)
            if len(bad) > 0 and min(bad) > -1:
                Aplot,Aexact,horribleInsert = numericalWiggle(Ain,Aplot,Aexact,verbosity)
                Aplot,Aexact = removeNumericallyBadTriangles(Ain,Aplot,Aexact,badTri)
                if withShow:
                    quietShow(Ain, Aplot, Aexact, instance.instance_uid, axs)
                continue

            if verbosity > 0:
                print("Phase 3")
            #########
            #
            # Phase 3: Steiner Points
            #          If there are properly bad triangles, call triangle-library to introduce as many steiner-points as
            #          possible. However triangle is not allowed to split constrained segments due to numerical stability
            #          Afterwards, all newly introduced points are wiggled (projected onto lines) around if necessary to
            #          have exact coordinates
            #
            #########
            if len(badTri)>0:
                oldLen = len(Aplot['vertices'])
                Aplot,Aexact = createSteinerPoints(Ain,Aplot,Aexact,2*len(badTri),verbosity)

            if withShow:
                quietShow(Ain,Aplot,Aexact,instance.instance_uid,axs)

            if verbosity > 0:
                print("Phase 4")
            #########
            #
            # Phase 4: wiggle everything
            #          To counteract numerical instability, every free point is checked, if it needs to be wiggled
            #
            #########
            Aplot,Aexact,horribleInsert = numericalWiggle(Ain,Aplot,Aexact,verbosity)

            bad,badTri = tallyBad(Aplot,Aexact)
            curLen = len(Aexact['vertices'])

            if withShow:
                quietShow(Ain,Aplot,Aexact,instance.instance_uid,axs)
            if len(badTri) > 0 and oldLen >= curLen:
                if verbosity > 0:
                    print("Huh...")

            #########
            #
            # Phase 5: Verify and delete unnecessary vertices
            #          Check if every triangle is indeed not bad, otherwise start next round
            #          If all triangles are good, remove random vertices reducing solution size if possible
            #
            #########
            if len(badTri) == 0:

                if verbosity > 0:
                    print("Phase 5")

                if rules['withRandomRefine']:
                    removed = True
                    while removed:
                        removed,Aplot,Aexact = randomStableSingleVertexDelete(Ain,Aplot,Aexact)

                return parseAsSolution(Ain,Aplot,Aexact,instance.instance_uid)
    return None

def convert(data):

    #convert to triangulation type
    points = np.column_stack((data.points_x, data.points_y))
    constraints = np.column_stack((data.region_boundary, np.roll(data.region_boundary, -1)))
    if (len(data.additional_constraints) != 0):
        constraints = np.concatenate((constraints, data.additional_constraints))
    A = dict(vertices=points, segments=constraints)
    return A


