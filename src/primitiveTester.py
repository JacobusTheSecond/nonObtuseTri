import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from cgshop2025_pyutils import InstanceDatabase,ZipSolutionIterator,verify
from cgshop2025_pyutils.geometry import FieldNumber, Point, Segment

from exact_geometry import circumcenter

plot_counter = 0
exactXs = []#['3875', '3100', '3100', '62601667508888178997054489444025704335300117512125/19630270826788759296118908858769753999444803584', '1899917829515038938394782879407253755847532425620638735885/514980698071987321961523398039070475475390338814705664']
exactYs = []#['6525', '5650', '5550', '24628274106659577357595059784296789698159498569599/4907567706697189824029727214692438499861200896', '749097676914591684180344343685625341929643366843720232827/171660232690662440653841132679690158491796779604901888']
exactPoints = [Point(FieldNumber(exactXs[i]),FieldNumber(exactYs[i])) for i in range(len(exactYs))]
xs = [float(p.x()) for p in exactPoints]#[3,7,8.5,7,3,1.5]
ys = [float(p.y()) for p in exactPoints]#[1,1,3.5,6,6,3.5]
constrainted = False
coA = 1
coB = 3
#exactPoints = [Point(FieldNumber(xs[i]),FieldNumber(ys[i])) for i in range(len(xs))]

from cgshop2025_pyutils.geometry import FieldNumber, Point, Segment

#dotproduct
def dot(X:Point, Y:Point) -> FieldNumber:
    return (X[0]*Y[0]) + (X[1]*Y[1])

#badness of triangle. If the badness is < 0, then the triangle is obtuse
def badness(A:Point,B:Point,C:Point)->FieldNumber:
    def badnessPrimitive(A: Point, B: Point, C: Point) -> FieldNumber:
        return dot(C - B, A - B)
    return min(badnessPrimitive(A,B,C),badnessPrimitive(B,C,A),badnessPrimitive(C,A,B))

#some angle primitives
def isBadAngle(A:Point,B:Point,C:Point):
    return dot(C-B,A-B)<FieldNumber(0)

def isBadTriangle(A:Point,B:Point,C:Point):
    return isBadAngle(A,B,C) or isBadAngle(B,C,A) or isBadAngle(C,A,B)

#returns the index of the point at the obtuse angle (if it exists)
def badAngle(A:Point,B:Point,C:Point):
    if(isBadAngle(A,B,C)):
        return 1
    elif(isBadAngle(B,C,A)):
        return 2
    elif(isBadAngle(C,A,B)):
        return 0
    else:
        return -1

#returns the point at the obtuse angle if it exists
def badVertex(A:Point,B:Point,C:Point):
    if(isBadAngle(A,B,C)):
        return B
    elif(isBadAngle(B,C,A)):
        return C
    elif(isBadAngle(C,A,B)):
        return A
    else:
        return None

#exactly computes the point resulting from dropping the altitue from C onto AB
def altitudePoint(AB:Segment,C:Point) -> Point:
    def altitudePrimitive(B: Point, C: Point):
        scale = dot(B, C) / dot(B, B)
        return Point(scale * B.x(), scale * B.y())
    A = AB.source()
    B = AB.target()
    return A+altitudePrimitive(B-A,C-A)

def circumcenter(a:Point,b:Point,c:Point) -> Point:
    ax = a[0]
    ay = a[1]
    bx = b[0]
    by = b[1]
    cx = c[0]
    cy = c[1]
    d = FieldNumber(2) * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
    return Point(ux, uy)

def innerIntersect(p1:Point, p2:Point, p3:Point, p4:Point):
    x1,y1 = p1
    x2,y2 = p2
    x3,y3 = p3
    x4,y4 = p4
    denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
    if denom == FieldNumber(0): # parallel
        return None
    ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
    if ua < FieldNumber(0) or ua > FieldNumber(1): # out of range
        return None
    ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
    if ub <= FieldNumber(0) or ub >= FieldNumber(1): # out of range
        return None
    x = x1 + ua * (x2-x1)
    y = y1 + ua * (y2-y1)
    return Point(x,y)

def exactSign(f : FieldNumber):
    if f < FieldNumber(0):
        return FieldNumber(-1)
    if f == FieldNumber(0):
        return FieldNumber(0)
    else:
        return FieldNumber(1)

def sign(f:FieldNumber):
    if f < FieldNumber(0):
        return -1
    if f == FieldNumber(0):
        return 0
    else:
        return 1

def isOnLeftSide(ab:Segment,p:Point):
    diff = ab.target() - ab.source()
    sourceOrth = Point(diff.y(),FieldNumber(0)-diff.x())
    if(dot(p-ab.source(),sourceOrth)<FieldNumber(0)):
        return True
    return False

def colinear(ab:Segment,p:Point):
    diff = ab.target() - ab.source()
    sourceOrth = Point(diff.y(),FieldNumber(0)-diff.x())
    if(dot(p-ab.source(),sourceOrth)==FieldNumber(0)):
        return True
    return False

def isOnRightSide(ab:Segment,p:Point):
    diff = ab.target() - ab.source()
    sourceOrth = Point(diff.y(),FieldNumber(0)-diff.x())
    if(dot(p-ab.source(),sourceOrth)>FieldNumber(0)):
        return True
    return False

def onWhichSide(ab:Segment,p:Point):
    diff = ab.target() - ab.source()
    sourceOrth = Point(diff.y(),FieldNumber(0)-diff.x())
    dotp = dot(p-ab.source(),sourceOrth)
    if dotp>FieldNumber(0):
        return "right"
    elif dotp< FieldNumber(0):
        return "left"
    return "colinear"

def inCircle(m:Point,rsqr:FieldNumber,p:Point):
    dotmp = dot(p-m,p-m)
    if dotmp > rsqr:
        return "outside"
    elif dotmp < rsqr:
        return "inside"
    else:
        return "on"

def binaryIntersection(m:Point,rsqr:FieldNumber,pq:Segment):
    p = pq.source()
    q = pq.target()
    pQuery = inCircle(m,rsqr,p)
    qQuery = inCircle(m,rsqr,q)
    inside = None
    outside = None
    if pQuery == "on":
        return "exact", p
    elif pQuery == "outside":
        outside = p
    else:
        inside = p
    if qQuery == "on":
        return "exact", q
    elif qQuery == "outside":
        outside = q
    else:
        inside = q
    if inside == None or outside == None:
        return "exact",None
    else:
        while Segment(inside,outside).squared_length() > FieldNumber(0.00000000001):
            mid = inside.scale(FieldNumber(0.5)) + outside.scale(FieldNumber(0.5))
            midQ = inCircle(m,rsqr,mid)
            if midQ == "on":
                return "exact", mid
            elif midQ == "outside":
                outside = mid
            else:
                inside = mid
        return "nonexact",outside

def supportingRayIntersect(seg1:Segment, seg2:Segment):
    x1,y1 = seg1.source()
    x2,y2 = seg1.target()
    x3,y3 = seg2.source()
    x4,y4 = seg2.target()
    denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
    if denom == FieldNumber(0): # parallel
        return None
    ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
    if ua <= FieldNumber(0): # out of range
        return None
    ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
    if ub <= FieldNumber(0): # out of range
        return None
    x = x1 + ua * (x2-x1)
    y = y1 + ua * (y2-y1)
    return Point(FieldNumber(x.exact()),FieldNumber(y.exact()))

def supportingRayIntersectSegment(ray:Segment, seg:Segment):
    x1,y1 = ray.source()
    x2,y2 = ray.target()
    x3,y3 = seg.source()
    x4,y4 = seg.target()
    denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
    if denom == FieldNumber(0): # parallel
        return None
    ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
    if ua < FieldNumber(0): # out of range
        return None
    ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
    if ub < FieldNumber(0) or ub > FieldNumber(1): # out of range
        return None
    x = x1 + ua * (x2-x1)
    y = y1 + ua * (y2-y1)
    return Point(FieldNumber(x.exact()),FieldNumber(y.exact()))


def _linkCanBeSolvedByVertex(points,constraint=None):

    ran = range(len(points)) if constraint == None else constraint

    for i in ran:
        c = points[i]
        solves = True

        hasZero = False

        for j in range(len(points)):
            a = points[j]
            b = points[(j+1)%len(points)]

            if isBadTriangle(a,b,c):
                solves = False
                break

            if colinear(Segment(a,b), c):
                hasZero = True
        if constraint == None and hasZero:
            solves = False

        if solves:
            return i
    return None

def findCenterOfLink(points,num=0,axs=None):
    #get orientation
    onlyLeft = True
    onlyRight = True

    for i in range(len(points)):
        a = points[i]
        b = points[(i+1)%len(points)]
        c = points[(i+2)%len(points)]
        side = onWhichSide(Segment(a,b),c)
        if  side == "left":
            onlyRight = False
        elif side == "right":
            onlyLeft = False

    if (onlyLeft == False) and (onlyRight == False):
        return "None",None
    if onlyRight:
        return findCenterOfLink(list(reversed(points)), num, axs)

    vertexSol = _linkCanBeSolvedByVertex(points)
    if vertexSol != None:
        return "vertex", vertexSol

    ran = list(range(len(points)))

    orth = []

    for i in range(len(ran)):
        idx = ran[i]
        nextIdx = ran[(i+1)%len(ran)]
        diff = points[nextIdx] - points[idx]
        orth.append(Point(FieldNumber(0) - diff.y(),diff.x()))

    endofrays = [[] for i in range(len(ran))]
    for i in range(len(ran)):
        for ioff in range(2):
            p = points[ran[(i+ioff)%len(ran)]]
            o = orth[i]

            #the points lie in convex position. we can check if the ray p,o intersects the interior by checking the neighbouring segments

            rayisInside = True
            if ioff == 0:
                if dot(points[ran[(i+1)%len(ran)]]-p,points[ran[i-1]]-p)>= FieldNumber(0):
                    rayisInside = False
            else:
                if dot(p-points[i],points[(i+2)%len(ran)]-p) <= FieldNumber(0):
                    rayisInside = False

            if rayisInside:

                closestIntersect = None
                closestDistsq = None
                for j in range(len(ran)):
                    if (j == (i+ioff)%len(ran)) or ((j+1)%len(ran) == (i+ioff)%len(ran)):
                        continue
                    a = points[ran[j]]
                    b = points[ran[(j+1)%len(ran)]]

                    inter = supportingRayIntersectSegment(Segment(p,p+o),Segment(a,b))
                    if inter != None:
                        distsq = Segment(p,inter).squared_length()
                        if closestDistsq == None or distsq < closestDistsq:
                            closestDistsq = distsq
                            closestIntersect = inter
                endofrays[i].append(closestIntersect)

                if closestIntersect != None:
                    px = float(p.x())
                    py = float(p.y())
                    qx = float(endofrays[i][-1].x())
                    qy = float(endofrays[i][-1].y())
                    axs.plot([px,qx],[py,qy])
            else:
                endofrays[i].append(None)



    intersections = []
    for i in range(len(ran)):
        a = points[ran[i]]
        b = points[ran[(i+1)%len(ran)]]
        c = points[ran[(i+2)%len(ran)]]
        if colinear(Segment(a,b),c):
            continue
        bA = badAngle(a,b,c)
        if (bA == -1) or (bA == 1):
            inter = altitudePoint(Segment(a,c),b)
            dontAdd = False
            for it in range(len(ran)):
                if isBadTriangle(inter, points[ran[it]], points[ran[(it + 1) % len(ran)]]):
                    dontAdd = True
            if not dontAdd:
                if inter not in intersections:
                    intersections.append(inter)

    global circleNum
    circleNum = len(intersections)

    for i in range(len(ran)):
        isFirst = (i == 0)
        for ioff in range(2):
            for j in range(i+ioff,len(ran)):
                for joff in range(2):

                    if (endofrays[i][ioff] == None) or (endofrays[j][joff] == None):
                        continue

                    oIdx = (i + ioff)%len(ran)
                    oJdx = (j + joff)%len(ran)
                    if oIdx == oJdx:
                        continue
                    idx = ran[oIdx]
                    jdx = ran[oJdx]

                    if idx == jdx:
                        continue

                    #colinearity checks
                    if onWhichSide(Segment(points[idx], points[idx] + orth[i]),points[(idx+1)%len(points)]) == "colinear" or onWhichSide(Segment(points[idx], points[idx] + orth[i]),points[idx-1]) == "colinear":
                        continue


                    #colinearity checks
                    if onWhichSide(Segment(points[jdx], points[jdx] + orth[j]),points[(jdx+1)%len(points)]) == "colinear" or onWhichSide(Segment(points[jdx], points[jdx] + orth[j]),points[jdx-1]) == "colinear":
                        continue

                    inter = innerIntersect(points[idx],endofrays[i][ioff],points[jdx],endofrays[j][joff])

                    if inter != None:
                        dontAdd = False
                        for it in range(len(ran)):
                            if isBadTriangle(inter, points[ran[it]], points[ran[(it + 1) % len(ran)]]):
                                dontAdd = True
                        if not dontAdd:
                            if inter not in intersections:
                                intersections.append(inter)
                        #axs.plot([float(points[idx].x()),float(inter.x())],[float(points[idx].y()),float(inter.y())],color="black",zorder=100000000)
                        #axs.plot([float(points[jdx].x()),float(inter.x())],[float(points[jdx].y()),float(inter.y())],color="black",zorder=100000000)
                        #if len(intersections) == 2:
                        #    print(i,j,ioff,joff,idx,jdx)
                        #    axs.scatter([float(points[idx].x()),float(points[jdx].x())],[float(points[idx].y()),float(points[jdx].y())],color="red")

    global altNum
    altNum = len(intersections)

    #line-circle numerical
    for i in range(len(ran)):
        isFirst = (i == 0)
        for ioff in range(2):

            if endofrays[i][ioff] == None:
                continue

            idx = ran[(i+ioff)%len(ran)]

            p = points[idx]
            q = endofrays[i][ioff]
            o = orth[i]

            # colinearity checks
            if onWhichSide(Segment(p,p+o),
                           points[(idx + 1) % len(points)]) == "colinear" or onWhichSide(
                    Segment(points[idx], points[idx] + orth[i]), points[idx - 1]) == "colinear":
                continue

            for j in range(len(ran)):
                addors = []
                m = points[ran[j]].scale(FieldNumber(0.5)) + points[ran[(j+1)%len(ran)]].scale(FieldNumber(0.5))
                rsqr = Segment(m,points[ran[j]]).squared_length()
                inCirc = inCircle(m,rsqr,p)
                if inCirc == "on":
                    #dont add vertices!
                    #addors.append(p)
                    #if m is in direction of o, there should be a second one
                    dotomp = dot(o,m-p)
                    if dotomp > FieldNumber(0):
                        #there is a second one
                        mid = altitudePoint(Segment(p,p+o),m)

                        #if this is 0, then its colinear and addor will be the next point along the boundary, which is NOT inside
                        if Segment(mid,m).squared_length() != 0:
                            diff = p-mid
                            addor = mid - diff
                            if Segment(p,addor).squared_length() < Segment(p,q).squared_length():
                                addors.append(addor)
                elif inCirc == "outside":
                    mid = altitudePoint(Segment(p,p+o),m)
                    secondInCirc = inCircle(m,rsqr,mid)
                    if secondInCirc == "on":
                        #if we are tangent, we have to check that we are not inserting a vertex
                        if not ( (mid == points[ran[j]]) or (mid == points[ran[(j+1)%len(ran)]]) ):
                            if Segment(p,mid).squared_length() < Segment(p,q).squared_length():
                                addors.append(mid)
                    elif secondInCirc == "inside":
                        ex,addor1 = binaryIntersection(m,rsqr,Segment(p,mid))
                        diff = addor1 - mid
                        addor2 = mid - diff
                        if onWhichSide(Segment(points[ran[j]],points[ran[(j+1)%len(ran)]]),addor1) == "left":
                            if Segment(p,addor1).squared_length() < Segment(p,q).squared_length():
                                addors.append(addor1)
                        if onWhichSide(Segment(points[ran[j]],points[ran[(j+1)%len(ran)]]),addor2) == "left":
                            if Segment(p,addor2).squared_length() < Segment(p,q).squared_length():
                                addors.append(addor2)
                else:
                    ex,addor = binaryIntersection(m,rsqr,Segment(p,q))
                    if addor != None:
                        addors.append(addor)
                for addor in addors:
                    dontAdd = False
                    for it in range(len(ran)):
                        if isBadTriangle(addor, points[ran[it]], points[ran[(it + 1) % len(ran)]]):
                            dontAdd = True
                    if not dontAdd:
                        if addor not in intersections:
                            intersections.append(addor)

    if len(intersections) == 0:
        return "None",None
    else:
        #form sum of all and add to intersections as best candidate
        result = []
        centroid = Point(FieldNumber(0),FieldNumber(0))
        for inter in intersections:
            zeroVol = False
            for it in range(len(ran)):
                if colinear(Segment(points[ran[it]], points[ran[(it + 1) % len(ran)]]),inter):
                    zeroVol = True
            if not zeroVol:
                result.append(Point(FieldNumber(inter.x().exact()),FieldNumber(inter.y().exact())))
                centroid = centroid + inter
        centroid = centroid.scale(FieldNumber(1)/FieldNumber(len(intersections)))
        #keep representation simple
        centroid = Point(FieldNumber(int(float(centroid.x())*10)/10),FieldNumber(int(float(centroid.y())*10)/10))

        if len(result) == 0:
            return "inside",None

        #print(centroid)

        dontAdd = False
        for it in range(len(ran)):
            if isBadTriangle(centroid, points[ran[it]], points[ran[(it + 1) % len(ran)]]):
                dontAdd = True
        if not dontAdd:
            if centroid not in intersections:
                altNum += 1
                circleNum += 1
                return "inside",[centroid] + result

        return "inside",result

def findCenterOfLinkConstrained(points,constraintA,constraintB,num=0,axs=None):
    #get orientation
    onlyLeft = True
    onlyRight = True

    for i in range(len(points)):
        a = points[i]
        b = points[(i+1)%len(points)]
        c = points[(i+2)%len(points)]
        side = onWhichSide(Segment(a,b),c)
        if  side == "left":
            onlyRight = False
        elif side == "right":
            onlyLeft = False

    if (onlyLeft == False) and (onlyRight == False):
        return "None",None

    if onlyRight:
        soltype,sol = findCenterOfLinkConstrained(list(reversed(points)),len(points)-1-constraintA,len(points)-1-constraintB,num,axs)
        if soltype == "vertex":
            return "vertex",len(points)-1-sol
        else:
            return soltype,sol

    vertexSol = _linkCanBeSolvedByVertex(points,[constraintA,constraintB])
    if vertexSol != None:
        return "vertex", vertexSol

    ran = list(range(len(points)))

    onBoundary = False
    boundaryIdx = None

    if ((constraintA + 1)%len(ran) == constraintB):
        onBoundary = True
        boundaryIdx = constraintA
    if ((constraintB + 1)%len(ran) == constraintA):
        onBoundary = True
        boundaryIdx = constraintB

    querySegment = Segment(points[constraintA], points[constraintB])

    orth = []

    for i in range(len(ran)):
        idx = ran[i]
        nextIdx = ran[(i+1)%len(ran)]
        diff = points[nextIdx] - points[idx]
        orth.append(Point(FieldNumber(0) - diff.y(),diff.x()))

    intersections = []

    #circle circle intersection is not needed
    global circleNum
    circleNum = len(intersections)

    #intersect segment with rays
    for i in range(len(ran)):
        for ioff in range(2):

            oIdx = (i + ioff) % len(ran)
            idx = ran[oIdx]
            p = points[idx]
            o = orth[i]

            if (idx == constraintB) or (idx == constraintA):
                continue

            inter = supportingRayIntersectSegment(Segment(p,p+o),querySegment)

            if inter != None:
                dontAdd = False
                for it in range(len(ran)):
                    if onBoundary and it == boundaryIdx:
                        continue
                    if isBadTriangle(inter, points[ran[it]], points[ran[(it + 1) % len(ran)]]):
                        dontAdd = True
                if not dontAdd:
                    if inter not in intersections:
                        intersections.append(inter)

    global altNum
    altNum = len(intersections)

    #line-circle numerical
    p = querySegment.source()
    o = querySegment.target() - querySegment.source()

    for j in range(len(ran)):
        addors = []
        m = points[ran[j]].scale(FieldNumber(0.5)) + points[ran[(j+1)%len(ran)]].scale(FieldNumber(0.5))
        rsqr = Segment(m,points[ran[j]]).squared_length()
        inCirc = inCircle(m,rsqr,p)
        if inCirc == "on":
            #dont add vertices!
            #addors.append(p)
            if inCircle(m,rsqr,p+o)!="on":
                #if m is in direction of o, there should be a second one
                dotomp = dot(o,m-p)
                if dotomp > FieldNumber(0):
                    #there is a second one
                    mid = altitudePoint(Segment(p,p+o),m)

                    #if this is 0, then its colinear and addor will be the next point along the boundary, which is NOT inside
                    if Segment(mid,m).squared_length() != 0:
                        diff = p-mid
                        addors.append(mid - diff)
        elif inCirc == "outside":
            mid = altitudePoint(Segment(p,p+o),m)
            secondInCirc = inCircle(m,rsqr,mid)
            if secondInCirc == "on":
                #if we are tangent, we have to check that we are not inserting a vertex
                if not ( (mid == points[ran[j]]) or (mid == points[ran[(j+1)%len(ran)]]) ):
                    addors.append(mid)
            elif secondInCirc == "inside":
                ex,addor1 = binaryIntersection(m,rsqr,Segment(p,mid))
                diff = addor1 - mid
                addor2 = mid - diff
                if onWhichSide(Segment(points[ran[j]],points[ran[(j+1)%len(ran)]]),addor1) == "left":
                    addors.append(addor1)
                if onWhichSide(Segment(points[ran[j]],points[ran[(j+1)%len(ran)]]),addor2) == "left":
                    addors.append(addor2)
        else:
            #p is in the circle already. then there is only one!
            #s is upperbound on 2*r
            s = None
            if rsqr < FieldNumber(1):
                s = FieldNumber(2)
            elif rsqr < FieldNumber(2):
                s = rsqr * FieldNumber(2)
            else:
                s = rsqr
            olsqr = Segment(p,p+o).squared_length()
            if olsqr < FieldNumber(1):
                s = s / olsqr
            q = p + o.scale(s)
            #q should be guaranteed to lie outside!
            assert(inCircle(m,rsqr,q) == "outside")
            ex,addor = binaryIntersection(m,rsqr,Segment(p,q))
            addors.append(addor)
        for addor in addors:
            dontAdd = False
            for it in range(len(ran)):
                if onBoundary and it == boundaryIdx:
                    continue
                if isBadTriangle(addor, points[ran[it]], points[ran[(it + 1) % len(ran)]]):
                    dontAdd = True
            if not dontAdd:
                if addor not in intersections:
                    intersections.append(addor)

    if len(intersections) == 0:
        return "None",None
    else:
        #form sum of all and add to intersections as best candidate
        result = []
        centroid = Point(FieldNumber(0), FieldNumber(0))
        for inter in intersections:
            zeroVol = False
            for it in range(len(ran)):
                if colinear(Segment(points[ran[it]], points[ran[(it + 1) % len(ran)]]), inter):
                    zeroVol = True
            if not zeroVol:
                result.append(Point(FieldNumber(inter.x().exact()), FieldNumber(inter.y().exact())))
                centroid = centroid + inter
        centroid = centroid.scale(FieldNumber(1) / FieldNumber(len(intersections)))

        if len(result) == 0:
            return "None", None

        # print(centroid)

        dontAdd = False
        for it in range(len(ran)):
            if isBadTriangle(centroid, points[ran[it]], points[ran[(it + 1) % len(ran)]]):
                dontAdd = True
        if not dontAdd:
            if centroid not in intersections:
                altNum += 1
                circleNum += 1
                return "inside", [centroid] + result

        return "inside", result

num = 0
totalNum = 0
cs = []


def primitiveTester():
    fig, axs = plt.subplots(1, 1)
    def on_click(event):
        global xs
        global ys
        global num
        global totalNum
        global cs
        (x,y) = (event.xdata,event.ydata)
        exactPoints.append(Point(FieldNumber(x),FieldNumber(y)))
        xs.append(x)
        ys.append(y)
        axs.clear()
        axs.scatter(xs,ys,color="blue")

        if len(xs)>=4:
            for i in range(len(xs)):
                axs.plot([xs[i],xs[(i+1)%len(xs)]],[ys[i],ys[(i+1)%len(xs)]])


            for i in range(len(xs)):
                ax,bx = xs[i],xs[(i+1)%len(xs)]
                ay,by =  ys[i],ys[(i+1)%len(xs)]
                mx = (ax + bx) / 2
                my = (ay + by) / 2
                dx = ax - mx
                dy = ay - my
                r = np.sqrt((dx*dx) + (dy*dy))
                circle = plt.Circle((mx,my), r, color="yellow", fill=False, zorder=1000)
                axs.add_patch(circle)

            cs = None
            global constrainted
            global coA
            global coB

            cs = None
            soltype = None
            if constrainted:
                soltype, cs = findCenterOfLinkConstrained(exactPoints, coA, coB,axs=axs)
            else:
                soltype, cs = findCenterOfLink(exactPoints,axs=axs)
            if soltype == "None":
                print("no center exists")
            else:
                num = min(max(0, num), len(cs) - 1)
                totalNum = len(cs)

                c = None
                if soltype == "vertex":
                    c = exactPoints[cs]
                else:
                    c = cs[0]
                cx = float(c.x())
                cy = float(c.y())
                for i in range(len(xs)):
                    axs.plot([xs[i], cx], [ys[i], cy])

                    if not isBadTriangle(c,exactPoints[i],exactPoints[(i+1)%len(xs)]):

                        t = plt.Polygon([[xs[i],ys[i]],[xs[(i+1)%len(xs)],ys[(i+1)%len(xs)]],[cx,cy]], color='g')
                        axs.add_patch(t)

                global circleNum
                global altNum

                for i in range(circleNum):
                    cx = float(cs[i].x())
                    cy = float(cs[i].y())
                    if i == 0:
                        axs.scatter([cx],[cy],color="orange",zorder=100000000)
                    else:
                        axs.scatter([cx],[cy],color="red",marker=".",zorder=1000000)

                for i in range(circleNum,len(cs)):
                    cx = float(cs[i].x())
                    cy = float(cs[i].y())
                    axs.scatter([cx],[cy],color="blue" if i < altNum else "black",marker=".",zorder=100000)
        axs.set_xlim([0, 10])
        axs.set_ylim([0, 10])
        axs.set_aspect('equal')
        plt.draw()

    def on_press(event):
        global num
        global totalNum
        global xs
        global ys
        global cs

        axs.clear()
        axs.scatter(xs,ys,color="blue")

        if event.key == '+':
            num = min(num +1,totalNum)
        if event.key == '-':
            num = max(num-1,0)

        if len(xs) >= 4:
            for i in range(len(xs)):
                axs.plot([xs[i], xs[(i + 1) % len(xs)]], [ys[i], ys[(i + 1) % len(xs)]])

            for i in range(len(xs)):
                ax, bx = xs[i], xs[(i + 1) % len(xs)]
                ay, by = ys[i], ys[(i + 1) % len(xs)]
                mx = (ax + bx) / 2
                my = (ay + by) / 2
                dx = ax - mx
                dy = ay - my
                r = np.sqrt((dx * dx) + (dy * dy))
                circle = plt.Circle((mx, my), r, color="yellow", fill=False, zorder=1000)
                axs.add_patch(circle)

            #cs = findCenterOfLinkConstrained(exactPoints, 0, 1)
            if cs == None:
                print("no center exists")
            else:
                num = min(max(0, num), len(cs) - 1)
                totalNum = len(cs)
                c = cs[num]
                cx = float(c.x())
                cy = float(c.y())
                for i in range(len(xs)):
                    axs.plot([xs[i], cx], [ys[i], cy])

                    if not isBadTriangle(c, exactPoints[i], exactPoints[(i + 1) % len(xs)]):
                        t = plt.Polygon([[xs[i], ys[i]], [xs[(i + 1) % len(xs)], ys[(i + 1) % len(xs)]], [cx, cy]],
                                        color='g')
                        axs.add_patch(t)

                global circleNum
                global altNum

                for i in range(circleNum):
                    cx = float(cs[i].x())
                    cy = float(cs[i].y())
                    if i == 0:
                        axs.scatter([cx], [cy], color="orange", zorder=100000000)
                    else:
                        axs.scatter([cx], [cy], color="red", marker=".", zorder=1000000)

                for i in range(circleNum, len(cs)):
                    cx = float(cs[i].x())
                    cy = float(cs[i].y())
                    axs.scatter([cx], [cy], color="blue" if i < altNum else "black", marker=".", zorder=100000)
        axs.set_xlim([0, 10])
        axs.set_ylim([0, 10])
        axs.set_aspect('equal')
        plt.draw()

    global num
    global totalNum
    global xs
    global ys
    global cs
    axs.clear()
    axs.scatter(xs,ys,color="blue")

    if len(xs) >= 4:
        for i in range(len(xs)):
            axs.plot([xs[i], xs[(i + 1) % len(xs)]], [ys[i], ys[(i + 1) % len(xs)]])


        for i in range(len(xs)):
            ax, bx = xs[i], xs[(i + 1) % len(xs)]
            ay, by = ys[i], ys[(i + 1) % len(xs)]
            mx = (ax + bx) / 2
            my = (ay + by) / 2
            dx = ax - mx
            dy = ay - my
            r = np.sqrt((dx * dx) + (dy * dy))
            circle = plt.Circle((mx, my), r, color="yellow", fill=False, zorder=1000)
            axs.add_patch(circle)
        global constrainted
        global coA
        global coB
        cs = None
        soltype = None
        if constrainted:
            soltype, cs = findCenterOfLinkConstrained(exactPoints, coA, coB)
        else:
            soltype, cs = findCenterOfLink(exactPoints)
        if soltype == "None":
            print("no center exists")
        else:
            print(soltype)
            num = min(max(0, num), len(cs) - 1)
            totalNum = len(cs)

            c = None
            if soltype == "vertex":
                c = exactPoints[cs]
            else:
                c = cs[0]
            cx = float(c.x())
            cy = float(c.y())
            for i in range(len(xs)):
                axs.plot([xs[i], cx], [ys[i], cy])

                if not isBadTriangle(c, exactPoints[i], exactPoints[(i + 1) % len(xs)]):
                    t = plt.Polygon([[xs[i], ys[i]], [xs[(i + 1) % len(xs)], ys[(i + 1) % len(xs)]], [cx, cy]],
                                    color='g')
                    axs.add_patch(t)

            global circleNum
            global altNum

            for i in range(circleNum):
                cx = float(cs[i].x())
                cy = float(cs[i].y())
                if i == 0:
                    axs.scatter([cx], [cy], color="orange", zorder=100000000)
                else:
                    axs.scatter([cx], [cy], color="red", marker=".", zorder=1000000)

            for i in range(circleNum, len(cs)):
                cx = float(cs[i].x())
                cy = float(cs[i].y())
                axs.scatter([cx], [cy], color="blue" if i < altNum else "black", marker=".", zorder=100000)
    if len(exactPoints)==0:
        axs.set_xlim([0, 10])
        axs.set_ylim([0, 10])
    axs.set_aspect('equal')
    plt.draw()

    fig.canvas.mpl_connect('key_press_event',on_press)
    fig.canvas.mpl_connect('button_press_event', on_click)

    plt.show()

if __name__=="__main__":
    primitiveTester()