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

def numericalangles(A:Point,B:Point,C:Point):
    # Square of lengths be a2, b2, c2
    a2 = float(Segment(B,C).squared_length())
    b2 = float(Segment(A,C).squared_length())
    c2 = float(Segment(A,B).squared_length())

    # length of sides be a, b, c
    a = math.sqrt(a2)
    b = math.sqrt(b2)
    c = math.sqrt(c2)

    # From Cosine law
    alpha = math.acos((b2 + c2 - a2) / (2 * b * c))
    betta = math.acos((a2 + c2 - b2) / (2 * a * c))
    gamma = math.acos((a2 + b2 - c2) / (2 * a * b))

    # Converting to degree
    alpha = alpha * 180 / math.pi
    betta = betta * 180 / math.pi
    gamma = gamma * 180 / math.pi

    return np.array((alpha,betta,gamma))

def dropAltitudesOnSegments(Aexact,Aplot,aggressive = False):
    inserted = True
    bad = []
    badTri = []
    restartIdx = 0
    completeIt = False
    foundBad = 0
    while not (completeIt and inserted == False):
        # print(completeIt,inserted)
        if inserted == False:
            restartIdx = 0
        completeIt = restartIdx == 0
        if completeIt:
            foundBad = 0
        B = tr.triangulate(Aplot, "p")
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

                for i in range(len(B['segments'])):
                    if np.all([idxs[1], idxs[2]] == B['segments'][i]) or np.all([idxs[2], idxs[1]] == B['segments'][i]):
                        closestSegmentIndex = i
                        closestSegment = (Aexact['vertices'][idxs[1]], Aexact['vertices'][idxs[2]])

                if closestSegment != None and aggressive:

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

                if closestSegment != None:
                    # we can insert circumcenter and retriangulate
                    # need to split the edge
                    ap = altitudePoint(Segment(closestSegment[0], closestSegment[1]), p)

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
                    Aplot['segments'] = Aexact['segments']
                    inserted = True
                    restartIdx = i + 1
                    completeIt = restartIdx == 0
                    break
    return (inserted,foundBad,Aexact,Aplot,bad,badTri)

def createSteinerPoints(Aplot, Aexact, foundBad, verbosity=0):
    wiggleTally = 0
    wiggleMap = []
    C = None
    upper = max(5, foundBad)
    lower = 1
    C = tr.triangulate(Aplot, "pq0.000001U90.001S1")
    if (len(Aplot['segments']) == len(C['segments'])):
        while lower + 1 < upper:
            mid = (lower + upper) // 2
            C = tr.triangulate(Aplot, "pq0.000001U90.001S" + str(mid))
            if (len(Aplot['segments']) == len(C['segments'])):
                lower = mid
            else:
                upper = mid
    C = tr.triangulate(Aplot, "pq0.000001U90.001S" + str(lower))

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
    vs = []
    for tri in badTri:
        for v in tri:
            if v >= len(Ain['vertices']) and not v in vs:
                vs.append(v)
    vs = list(reversed(sorted(vs)))
    for v in vs:
        # merge participating segmets
        left = None
        right = None
        for i in range(len(Aexact['segments'])):
            if Aexact['segments'][i][0] == v:
                left = i
            elif Aexact['segments'][i][1] == v:
                right = i
        if (left != None and right != None):
            Aexact['segments'][left][0] = Aexact['segments'][right][0]
            Aexact['segments'] = np.delete(Aexact['segments'], (right), axis=0)

        # remap all segments
        for i in range(len(Aexact['segments'])):
            if Aexact['segments'][i][0] >= v:
                Aexact['segments'][i][0] -= 1
            if Aexact['segments'][i][1] >= v:
                Aexact['segments'][i][1] -= 1
        Aexact['vertices'].pop(v)
        Aplot['vertices'] = np.delete(Aplot['vertices'], (v), axis=0)
    Aplot['segments'] = Aexact['segments']
    return (Aplot,Aexact)

def quietShow(Ain, Aplot, Aexact, name, axs):
    pausetime = 0.03
    axs.clear()
    C = tr.triangulate(Aplot, 'p')
    plotExact(Ain, Aexact, C, name, axs)
    plt.draw()
    plt.pause(pausetime)

def numericalWiggle(Ain,Aplot, Aexact,verbosity=0):
    wiggleTally = 0
    #lastly we try to wiggle everything, if needed

    fullMap = []
    C = tr.triangulate(Aplot, 'p')
    for v in Aexact['vertices']:
        fullMap.append([])
    for tri in C['triangles']:
        if isBadTriangle(Aexact['vertices'][tri[0]],Aexact['vertices'][tri[1]],Aexact['vertices'][tri[2]]):
            for idx in tri:
                fullMap[idx].append(tri)

    toWiggleCount = 0
    for tris in fullMap:
        if len(tris)>0:
            toWiggleCount+=1

    #multiple passes just to be sure?
    #fuck it, wiggle order is random!
    for round in range(3):
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
                    #TODO: unsafe assumption that dropper is not a constraint segment
                    dropper = Segment(Aexact['vertices'][tri[1]], Aexact['vertices'][tri[2]])
                    ap = altitudePoint(dropper, Aexact['vertices'][tri[0]])

                    # add altitude point
                    Aexact['vertices'].append(ap)
                    ap1 = float(ap.x())
                    ap2 = float(ap.y())
                    Aplot['vertices'] = np.vstack((Aplot['vertices'], np.array([[ap1, ap2]])))

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

def improveQuality(instance,withShow=True,axs=None,verbosity=0):
    if withShow:
        assert axs != None
        plt.ion()

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

    Aexact = dict(segments=Ain['segments'])
    Aexact['vertices'] = []
    for x, y in zip(instance.points_x, instance.points_y):
        Aexact['vertices'].append(Point(x, y))

    Aplot = dict(vertices=Ain['vertices'],segments=Ain['segments'])

    #1000 rounds should always suffice, but who knows...
    for n in range(1000):

        #########
        #
        # Phase 1: AltitudeDropper
        #          if there are altitudes that should be dropped onto constraint segments, do it!
        #
        #########
        inserted,foundBad,Aexact,Aplot,bad,badTri = dropAltitudesOnSegments(Aexact,Aplot)

        if withShow:
            quietShow(Ain,Aplot,Aexact,instance.instance_uid,axs)
        if inserted:
            continue
        if verbosity > 0:
            print("badnesses:",list(np.sort(bad)))
        #########
        #
        # Phase 2: Horrible Heuristic
        #          If we are essentially done, but there are few numerically unstable triangles, try to wiggle, and if
        #          not possible, remove numerically bad triangles
        #
        #########
        if len(bad) > 0 and min(bad) > -1:
            Aplot,Aexact,horribleInsert = numericalWiggle(Ain,Aplot,Aexact,verbosity)
            Aplot,Aexact = removeNumericallyBadTriangles(Ain,Aplot,Aexact,badTri)
            if withShow:
                quietShow(Ain, Aplot, Aexact, instance.instance_uid, axs)
            continue

        #########
        #
        # Phase 3: Steiner Points
        #          If there are properly bad triangles, call triangle-library to introduce as many steiner-points as
        #          possible. However triangle is not allowed to split constrained segments due to numerical stability
        #          Afterwards, all newly introduced points are wiggled (projected onto lines) around if necessary to
        #          have exact coordinates
        #
        #########
        if foundBad>0:
            Aplot,Aexact = createSteinerPoints(Aplot,Aexact,foundBad,verbosity)

        if withShow:
            quietShow(Ain,Aplot,Aexact,instance.instance_uid,axs)

        #########
        #
        # Phase 4: wiggle everything
        #          To counteract numerical instability, every free point is checked, if it needs to be wiggled
        #
        #########
        Aplot,Aexact,horribleInsert = numericalWiggle(Ain,Aplot,Aexact,verbosity)

        if withShow:
            quietShow(Ain,Aplot,Aexact,instance.instance_uid,axs)
        if foundBad > 0 and inserted == False:
            if verbosity > 0:
                print("Huh...")

        #########
        #
        # Phase 5: Verify
        #          Check if every triangle is indeed not bad, otherwise start next round
        #
        #########
        foundBad = 0
        C = tr.triangulate(Aplot, 'p')
        for tri in C['triangles']:
            if isBadTriangle(Aexact['vertices'][tri[0]],Aexact['vertices'][tri[1]],Aexact['vertices'][tri[2]]):
                foundBad += 1
        if foundBad == False:
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


