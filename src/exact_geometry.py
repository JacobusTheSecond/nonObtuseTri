from cgshop2025_pyutils.geometry import FieldNumber, Point, Segment
import numpy as np
from matplotlib import pyplot as plt

zero = FieldNumber(0)
onehalf = FieldNumber(0.5)

#dotproduct
def dot(X:Point, Y:Point) -> FieldNumber:
    return (X[0]*Y[0]) + (X[1]*Y[1])

def distsq(x:Point,y:Point) -> FieldNumber:
    diff = y-x
    return dot(diff,diff)
    #return Segment(x,y).squared_length()

#badness of triangle. If the badness is < 0, then the triangle is obtuse
def badness(A:Point,B:Point,C:Point)->FieldNumber:
    def badnessPrimitive(A: Point, B: Point, C: Point) -> FieldNumber:
        return dot(C - B, A - B)
    return min(badnessPrimitive(A,B,C),badnessPrimitive(B,C,A),badnessPrimitive(C,A,B))

#some angle primitives
def isBadAngle(A:Point,B:Point,C:Point):
    return dot(C-B,A-B)<zero

def isBadTriangle(A:Point,B:Point,C:Point):
    ab = B-A
    ac = C-A
    bc = C-B
    return (dot(ab,ac) < zero) or (dot(ab,bc)>zero) or (dot(ac,bc)<zero)
    #return isBadAngle(A,B,C) or isBadAngle(B,C,A) or isBadAngle(C,A,B)

#returns the index of the point at the obtuse angle (if it exists)
def badAngle(A:Point,B:Point,C:Point):
    ab = B-A
    ac = C-A
    bc = C-B
    if dot(ab, ac) < zero:
        return 0
    elif dot(ab, bc) > zero:
        return 1
    elif dot(ac, bc) < zero:
        return 2
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
    daa = dot(a,a)
    dbb = dot(b,b)
    dcc = dot(c,c)
    d = onehalf / (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = (daa * (by - cy) + dbb * (cy - ay) + dcc * (ay - by))
    uy = (daa * (cx - bx) + dbb * (ax - cx) + dcc * (bx - ax))
    cc = (Point(ux, uy).scale(d))
    return cc

def innerIntersect(p1:Point, p2:Point, p3:Point, p4:Point,insideFirst=False,insideSecond=True):
    x1,y1 = p1
    x2,y2 = p2
    x3,y3 = p3
    x4,y4 = p4
    denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
    if denom == zero: # parallel
        return None
    ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
    if insideFirst:
        if ua <= zero or ua >= FieldNumber(1): # out of range
            return None
    else:
        if ua < zero or ua > FieldNumber(1): # out of range
            return None
    ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
    if insideSecond:
        if ub <= zero or ub >= FieldNumber(1): # out of range
            return None
    else:
        if ub < zero or ub > FieldNumber(1): # out of range
            return None
    x = x1 + ua * (x2-x1)
    y = y1 + ua * (y2-y1)
    return Point(x,y)

def exactSign(f : FieldNumber):
    if f < zero:
        return FieldNumber(-1)
    if f == zero:
        return zero
    else:
        return FieldNumber(1)

def sign(f:FieldNumber):
    if f < zero:
        return -1
    if f == zero:
        return 0
    else:
        return 1

def isOnLeftSide(ab:Segment,p:Point):
    diff = ab.target() - ab.source()
    sourceOrth = Point(diff.y(),zero-diff.x())
    if(dot(p-ab.source(),sourceOrth)<zero):
        return True
    return False

def colinear(ab:Segment,p:Point):
    diff = ab.target() - ab.source()
    sourceOrth = Point(diff.y(),zero-diff.x())
    if(dot(p-ab.source(),sourceOrth)==zero):
        return True
    return False

def isOnRightSide(ab:Segment,p:Point):
    diff = ab.target() - ab.source()
    sourceOrth = Point(diff.y(),zero-diff.x())
    if(dot(p-ab.source(),sourceOrth)>zero):
        return True
    return False

def onWhichSide(ab:Segment,p:Point):
    diff = ab.target() - ab.source()
    sourceOrth = Point(diff.y(),zero-diff.x())
    dotp = dot(p-ab.source(),sourceOrth)
    if dotp>zero:
        return "right"
    elif dotp< zero:
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

def segmentIntersectsCircle(m:Point,rsqr:FieldNumber,pq:Segment):
    if inCircle(m,rsqr,pq.source()) != "outside" or inCircle(m,rsqr,pq.target()) != "outside":
        return True
    if pq.squared_length() == FieldNumber(0):
        return inCircle(m,rsqr,pq.source()) != "outside"
    mid = altitudePoint(pq,m)
    if FieldNumber(0) <= dot(mid-pq.source(),pq.target() - pq.source()) <= pq.squared_length():
        if inCircle(m,rsqr,mid) != "outside":
            return True
    return False

def insideIntersectionsSegmentCircle(m:Point,rsqr:FieldNumber,pq:Segment):
    if not segmentIntersectsCircle(m,rsqr,pq):
        return []
    p = pq.source()
    q = pq.target()
    inCircP = inCircle(m,rsqr,p)
    inCircQ = inCircle(m,rsqr,q)
    if pq.squared_length() == FieldNumber(0):
        if inCircP == "on":
            return [p]
        else:
            return []
    if inCircP == "inside" and inCircQ == "inside":
        return []
    if inCircP == "outside" and inCircQ == "outside":
        #mid must be inside or on, otherwise we would have aborted earlier
        mid = altitudePoint(pq,m)
        if inCircle(m,rsqr,mid) == "on":
            return [mid]
        else:
            return [binaryIntersectionInside(m,rsqr,Segment(mid,p))[1],binaryIntersectionInside(m,rsqr,Segment(mid,q))[1]]
    if inCircP == "inside" or inCircQ == "inside":
        return [binaryIntersectionInside(m,rsqr,pq)[1]]
    if inCircP == "on" and inCircQ == "on":
        return [p,q]
    onPoint = p
    outPoint = q
    if inCircP == "outside":
        onPoint = q
        outPoint = p
    mid = altitudePoint(Segment(onPoint,outPoint),m)
    if FieldNumber(0) >= dot(mid-onPoint,outPoint - onPoint):
        return [onPoint]
    return [onPoint,mid+mid-onPoint]

def outsideIntersectionsSegmentCircle(m:Point,rsqr:FieldNumber,pq:Segment):
    if not segmentIntersectsCircle(m,rsqr,pq):
        return []
    p = pq.source()
    q = pq.target()
    inCircP = inCircle(m,rsqr,p)
    inCircQ = inCircle(m,rsqr,q)
    if pq.squared_length() == FieldNumber(0):
        if inCircP == "on":
            return [p]
        else:
            return []
    if inCircP == "inside" and inCircQ == "inside":
        return []
    if inCircP == "outside" and inCircQ == "outside":
        #mid must be inside or on, otherwise we would have aborted earlier
        mid = altitudePoint(pq,m)
        if inCircle(m,rsqr,mid) == "on":
            return [mid]
        else:
            return [binaryIntersection(m,rsqr,Segment(mid,p))[1],binaryIntersection(m,rsqr,Segment(mid,q))[1]]
    if inCircP == "inside" or inCircQ == "inside":
        return [binaryIntersection(m,rsqr,pq)[1]]
    if inCircP == "on" and inCircQ == "on":
        return [p,q]
    onPoint = p
    outPoint = q
    if inCircP == "outside":
        onPoint = q
        outPoint = p
    mid = altitudePoint(Segment(onPoint,outPoint),m)
    if FieldNumber(0) >= dot(mid-onPoint,outPoint - onPoint):
        return [onPoint]
    return [onPoint,mid+mid-onPoint]

def outsideClipSegmentToCircle(m:Point,rsqr:FieldNumber,pq:Segment):
    if not segmentIntersectsCircle(m,rsqr,pq):
        return []
    p = pq.source()
    q = pq.target()
    inCircP = inCircle(m,rsqr,p)
    inCircQ = inCircle(m,rsqr,q)
    if pq.squared_length() == FieldNumber(0):
        if inCircP == "on":
            return [p]
        elif inCircP == "inside":
            return [p,p]
    if inCircP == "inside" and inCircQ == "inside":
        return [p,q]
    if inCircP == "outside" and inCircQ == "outside":
        #mid must be inside or on, otherwise we would have aborted earlier
        mid = altitudePoint(pq,m)
        if inCircle(m,rsqr,mid) == "on":
            return [mid]
        else:
            return [binaryIntersection(m,rsqr,Segment(mid,p))[1],binaryIntersection(m,rsqr,Segment(mid,q))[1]]
    if inCircP == "inside" and inCircQ != "inside":
        return [p,binaryIntersection(m,rsqr,pq)[1]]
    if inCircQ == "inside" and inCircP != "inside":
        return [binaryIntersection(m,rsqr,pq)[1],q]
    if inCircP == "on" and inCircQ == "on":
        return [p,q]
    onPoint = p
    outPoint = q
    if inCircP == "outside":
        onPoint = q
        outPoint = p
    mid = altitudePoint(Segment(onPoint,outPoint),m)
    if FieldNumber(0) >= dot(mid-onPoint,outPoint - onPoint):
        return [onPoint]
    return [onPoint,mid+mid-onPoint]

def completeOutsideIntersectionSegmentCircle(m:Point,rsqr:FieldNumber,pq:Segment):
    if not segmentIntersectsCircle(m,rsqr,pq):
        return []
    p = pq.source()
    q = pq.target()
    inCircP = inCircle(m,rsqr,p)
    inCircQ = inCircle(m,rsqr,q)
    if pq.squared_length() == FieldNumber(0):
        if inCircP == "on":
            return [p,p]
        elif inCircP == "inside":
            return [p,p]
        else:
            return []
    if inCircP == "inside" and inCircQ == "inside":
        return [p,q]
    if inCircP == "outside" and inCircQ == "outside":
        #mid must be inside or on, otherwise we would have aborted earlier
        mid = altitudePoint(pq,m)
        if inCircle(m,rsqr,mid) == "on":
            return [mid]
        else:
            return [binaryIntersection(m,rsqr,Segment(mid,p))[1],binaryIntersection(m,rsqr,Segment(mid,q))[1]]
    if inCircP == "inside" or inCircQ == "inside":
        return [binaryIntersection(m,rsqr,pq)[1]]
    if inCircP == "on" and inCircQ == "on":
        return [p,q]
    onPoint = p
    outPoint = q
    if inCircP == "outside":
        onPoint = q
        outPoint = p
    mid = altitudePoint(Segment(onPoint,outPoint),m)
    if FieldNumber(0) >= dot(mid-onPoint,outPoint - onPoint):
        return [onPoint]
    return [onPoint,mid+mid-onPoint]

def insideIntersectionsRayCircle(m:Point,rsqr:FieldNumber,pq:Segment):
    if not rayIntersectsCircle(m,rsqr,pq):
        return []
    assert(pq.squared_length() != FieldNumber(0))
    p = pq.source()
    q = pq.target()
    inCircP = inCircle(m,rsqr,p)
    inCircQ = inCircle(m,rsqr,q)

    if inCircP == "outside":
        #mid must be inside or on, otherwise we would have aborted earlier
        mid = altitudePoint(pq,m)
        if inCircle(m,rsqr,mid) == "on":
            return [mid]
        else:
            return [binaryIntersectionInside(m,rsqr,Segment(mid,p))[1],binaryIntersectionInside(m,rsqr,Segment(mid,mid+mid-p))[1]]
    if inCircP == "inside":
        #copy Q to be safe
        if inCircQ == "inside":
            diff = q-p
            if dot(diff,diff) < FieldNumber(1):
                diff = diff.scale(FieldNumber(1)/dot(diff,diff))
            #diff now has length at least 1
            if rsqr > FieldNumber(1):
                diff = diff.scale(FieldNumber(2)*rsqr)
            #diff now has length at least 2r
            return [binaryIntersectionInside(m,rsqr,Segment(p,p+diff))[1]]
        else:
            return [binaryIntersectionInside(m,rsqr,Segment(p,q))[1]]

    if inCircP == "on":
        mid = altitudePoint(pq,m)
        if FieldNumber(0) >= dot(mid-p,q - p):
            return [p]
        return [p,mid+mid-p]

def rayIntersectsCircle(m:Point,rsqr:FieldNumber,pq:Segment):
    if inCircle(m,rsqr,pq.source()) != "outside":
        return True
    mid = altitudePoint(pq,m)
    if FieldNumber(0) <= dot(mid-pq.source(),pq.target() - pq.source()):
        if inCircle(m,rsqr,mid) != "outside":
            return True
    return False

def binaryIntersection(m:Point,rsqr:FieldNumber,pq:Segment,acc=1000):
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
        while Segment(inside,outside).squared_length() > (FieldNumber(1)/FieldNumber(acc)):
            mid = inside.scale(FieldNumber(0.5)) + outside.scale(FieldNumber(0.5))
            midQ = inCircle(m,rsqr,mid)
            if midQ == "on":
                return "exact", mid
            elif midQ == "outside":
                outside = mid
            else:
                inside = mid
        return "nonexact",outside

def binaryIntersectionInside(m:Point,rsqr:FieldNumber,pq:Segment,acc=1000):
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
        while Segment(inside,outside).squared_length()  > (FieldNumber(1)/FieldNumber(acc)):
            mid = inside.scale(FieldNumber(0.5)) + outside.scale(FieldNumber(0.5))
            midQ = inCircle(m,rsqr,mid)
            if midQ == "on":
                return "exact", mid
            elif midQ == "outside":
                outside = mid
            else:
                inside = mid
        return "nonexact",inside

def supportingRayIntersect(seg1:Segment, seg2:Segment):
    x1,y1 = seg1.source()
    x2,y2 = seg1.target()
    x3,y3 = seg2.source()
    x4,y4 = seg2.target()
    denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
    if denom == zero: # parallel
        return None
    ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
    if ua <= zero: # out of range
        return None
    ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
    if ub <= zero: # out of range
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
    if denom == zero: # parallel
        return None
    ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
    if ua < zero: # out of range
        return None
    ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
    if ub < zero or ub > FieldNumber(1): # out of range
        return None
    x = x1 + ua * (x2-x1)
    y = y1 + ua * (y2-y1)
    return Point(FieldNumber(x.exact()),FieldNumber(y.exact()))

def _linkCanBeSolvedByVertex(points,constraint=None):

    ran = range(len(points)) if constraint == None else constraint

    for i in ran:
        c = points[i]
        solves = True

        for j in range(len(points)):
            a = points[j]
            b = points[(j+1)%len(points)]

            if isBadTriangle(a,b,c):
                solves = False
                break

            if (colinear(Segment(a,b), c)):
                if ((j+1)%len(points) == i or i == j):
                    continue
                else:
                    solves = False

        if solves:
            return i
    return None

def findVertexCenterOfLink(points):
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
        vertexSol = _linkCanBeSolvedByVertex(list(reversed(points)))
        #soltype,sol =  findVertexCenterOfLink(list(reversed(points)))
        if vertexSol != None:
            return "vertex", len(points) - 1 - vertexSol
        return "None",None
    if onlyLeft:
        vertexSol = _linkCanBeSolvedByVertex(points)
        if vertexSol != None:
            return "vertex", vertexSol
        return "None",None


def findVertexCenterOfLinkConstrained(points,constraint=None):
    onlyLeft = True
    onlyRight = True

    for i in range(len(points)):
        a = points[i]
        b = points[(i + 1) % len(points)]
        c = points[(i + 2) % len(points)]
        side = onWhichSide(Segment(a, b), c)
        if side == "left":
            onlyRight = False
        elif side == "right":
            onlyLeft = False

    if (onlyLeft == False) and (onlyRight == False):
        return "None", None
    #if onlyRight:
    #    soltype, sol = findVertexCenterOfLinkConstrained(list(reversed(points)),[len(points)-1-constraint[0],len(points)-1-constraint[1]])
    #    if soltype == "vertex":
    #        return "vertex", len(points) - 1 - sol
    #    else:
    #        return soltype, sol
    if onlyRight:
        vertexSol = _linkCanBeSolvedByVertex(list(reversed(points)),[len(points)-1-constraint[0],len(points)-1-constraint[1]])
        if vertexSol != None:
            return "vertex", len(points) - 1 - vertexSol
        return "None",None
    if onlyLeft:
        vertexSol = _linkCanBeSolvedByVertex(points,constraint)
        if vertexSol != None:
            return "vertex", vertexSol
        return "None",None


def roundExactFieldNumber(f:FieldNumber,acc=100):
    return FieldNumber(int(float(f) * acc))/FieldNumber(acc)

def roundExact(p:Point,acc=10):
    return Point(FieldNumber(int(float(p.x()) * acc)), FieldNumber(int(float(p.y()) * acc))).scale(FieldNumber(1)/FieldNumber(acc))

def roundExactOnSegment(seg:Segment,p:Point,acc=100):
    #debugIdx = np.random.randint(1000)
    #logging.debug("debugIdx:"+str(debugIdx))
    assert(seg.squared_length() > zero)
    tx = (p-seg.source()).x()/(seg.target()-seg.source()).x()
    ty = (p-seg.source()).y()/(seg.target()-seg.source()).y()
    if (seg.target()-seg.source()).x() == zero:
        tx = ty
    elif (seg.target()-seg.source()).y() == zero:
        ty = tx
    if (tx != ty):
        assert(False)
    assert(zero <= tx <= FieldNumber(1))
    offset = FieldNumber(1)/FieldNumber(acc)
    if zero <= (roundor:=roundExactFieldNumber(tx,acc)) <= FieldNumber(1):
        return seg.source() + (seg.target()-seg.source()).scale(roundor)
    elif zero <= (roundor+offset) <= FieldNumber(1):
        return seg.source() + (seg.target()-seg.source()).scale(roundor+offset)
    elif zero <= (roundor-offset) <= FieldNumber(1):
        return seg.source() + (seg.target()-seg.source()).scale(roundor-offset)
    else:
        assert(False)


def roundExactBoundor(p:Point,acc=10):
    result = []
    result.append(Point(FieldNumber(int(float(p.x()) * acc)), FieldNumber(int(float(p.y()) * acc))).scale(FieldNumber(1)/FieldNumber(acc)))
    result.append(Point(FieldNumber(int(float(p.x()) * acc)+1), FieldNumber(int(float(p.y()) * acc))).scale(FieldNumber(1)/FieldNumber(acc)))
    result.append(Point(FieldNumber(int(float(p.x()) * acc)), FieldNumber(int(float(p.y()) * acc)+1)).scale(FieldNumber(1)/FieldNumber(acc)))
    result.append(Point(FieldNumber(int(float(p.x()) * acc)+1), FieldNumber(int(float(p.y()) * acc)+1)).scale(FieldNumber(1)/FieldNumber(acc)))
    return result

def roundExactOnSegmentBounder(seg:Segment,p:Point,acc=100):
    #debugIdx = np.random.randint(1000)
    #logging.debug("debugIdx:"+str(debugIdx))
    result = []
    if seg.squared_length() == zero:
        return result
    for p in roundExactBoundor(p,acc):
        projP = altitudePoint(seg,p)
        if zero <= getParamOfPointOnSegment(seg,projP) <= FieldNumber(1):
            result.append(projP)
    return result
    #assert(seg.squared_length() > zero)
    #tx = (p-seg.source()).x()/(seg.target()-seg.source()).x()
    #ty = (p-seg.source()).y()/(seg.target()-seg.source()).y()
    #if (seg.target()-seg.source()).x() == zero:
    #    tx = ty
    #elif (seg.target()-seg.source()).y() == zero:
    #    ty = tx
    #if (tx != ty):
    #    assert(False)
    #assert(zero <= tx <= FieldNumber(1))
    #offset = FieldNumber(1)/FieldNumber(acc)
    #roundor = roundExactFieldNumber(tx,acc)
    #if zero <= roundor <= FieldNumber(1):
    #    result.append( seg.source() + (seg.target()-seg.source()).scale(roundor) )
    #if zero <= (roundor+offset) <= FieldNumber(1):
    #    result.append( seg.source() + (seg.target()-seg.source()).scale(roundor+offset))
    #return result

def _unsafeOrientedFindCenterOfLink(points,num=0,axs=None):
    numpoints = len(points)
    #ran = list(range(len(points)))

    orth = []

    for i in range(numpoints):
        idx = i
        nextIdx = (i+1)%numpoints
        diff = points[nextIdx] - points[idx]
        orth.append(Point(zero - diff.y(),diff.x()))

    endofrays = [[] for i in range(numpoints)]
    for i in range(numpoints):
        for ioff in range(2):
            p = points[(i+ioff)%numpoints]
            o = orth[i]

            #the points lie in convex position. we can check if the ray p,o intersects the interior by checking the neighbouring segments

            rayisInside = True
            if ioff == 0:
                if dot(points[(i+1)%numpoints]-p,points[i-1]-p)>= zero:
                    rayisInside = False
            else:
                if dot(p-points[i],points[(i+2)%numpoints]-p) <= zero:
                    rayisInside = False

            if rayisInside:

                closestIntersect = None
                closestDistsq = None
                for j in range(numpoints):
                    if (j == (i+ioff)%numpoints) or ((j+1)%numpoints == (i+ioff)%numpoints):
                        continue
                    a = points[j]
                    b = points[(j+1)%numpoints]

                    inter = supportingRayIntersectSegment(Segment(p,p+o),Segment(a,b))
                    if inter != None:
                        distsq = Segment(p,inter).squared_length()
                        if closestDistsq == None or distsq < closestDistsq:
                            closestDistsq = distsq
                            closestIntersect = inter
                endofrays[i].append(closestIntersect)
            else:
                endofrays[i].append(None)



    intersections = []
    for i in range(numpoints):
        a = points[i]
        b = points[(i+1)%numpoints]
        c = points[(i+2)%numpoints]
        if colinear(Segment(a,b),c):
            continue
        bA = badAngle(a,b,c)
        if (bA == -1) or (bA == 1):
            inter = altitudePoint(Segment(a,c),b)
            dontAdd = False
            for it in range(numpoints):
                if isBadTriangle(inter, points[it], points[(it + 1) % numpoints]):
                    dontAdd = True
            if not dontAdd:
                if inter not in intersections:
                    intersections.append(inter)

    global circleNum
    circleNum = len(intersections)

    for i in range(numpoints):
        isFirst = (i == 0)
        for ioff in range(2):
            for j in range(i+ioff,numpoints):
                for joff in range(2):

                    if (endofrays[i][ioff] == None) or (endofrays[j][joff] == None):
                        continue

                    oIdx = (i + ioff)%numpoints
                    oJdx = (j + joff)%numpoints
                    if oIdx == oJdx:
                        continue
                    idx = oIdx
                    jdx = oJdx

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
                        for it in range(numpoints):
                            if isBadTriangle(inter, points[it], points[(it + 1) % numpoints]):
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
    for i in range(numpoints):
        isFirst = (i == 0)
        for ioff in range(2):

            if endofrays[i][ioff] == None:
                continue

            idx = (i+ioff)%numpoints

            p = points[idx]
            q = endofrays[i][ioff]
            o = orth[i]

            # colinearity checks
            if onWhichSide(Segment(p,p+o),
                           points[(idx + 1) % len(points)]) == "colinear" or onWhichSide(
                    Segment(points[idx], points[idx] + orth[i]), points[idx - 1]) == "colinear":
                continue

            for j in range(numpoints):
                addors = []
                m = points[j].scale(FieldNumber(0.5)) + points[(j+1)%numpoints].scale(FieldNumber(0.5))
                rsqr = Segment(m,points[j]).squared_length()
                inCirc = inCircle(m,rsqr,p)
                if inCirc == "on":
                    #dont add vertices!
                    #addors.append(p)
                    #if m is in direction of o, there should be a second one
                    dotomp = dot(o,m-p)
                    if dotomp > zero:
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
                        if not ( (mid == points[j]) or (mid == points[(j+1)%numpoints]) ):
                            if Segment(p,mid).squared_length() < Segment(p,q).squared_length():
                                addors.append(mid)
                    elif secondInCirc == "inside":
                        ex,addor1 = binaryIntersection(m,rsqr,Segment(p,mid))
                        diff = addor1 - mid
                        addor2 = mid - diff
                        if onWhichSide(Segment(points[j],points[(j+1)%numpoints]),addor1) == "left":
                            if Segment(p,addor1).squared_length() < Segment(p,q).squared_length():
                                addors.append(addor1)
                        if onWhichSide(Segment(points[j],points[(j+1)%numpoints]),addor2) == "left":
                            if Segment(p,addor2).squared_length() < Segment(p,q).squared_length():
                                addors.append(addor2)
                else:
                    ex,addor = binaryIntersection(m,rsqr,Segment(p,q))
                    if addor != None:
                        addors.append(addor)
                for addor in addors:
                    dontAdd = False
                    for it in range(numpoints):
                        if isBadTriangle(addor, points[it], points[(it + 1) % numpoints]):
                            dontAdd = True
                    if not dontAdd:
                        if addor not in intersections:
                            intersections.append(addor)

    if len(intersections) == 0:
        return "None",None
    else:
        #form sum of all and add to intersections as best candidate
        result = []
        centroid = Point(zero,zero)
        count = 0
        for inter in intersections:
            zeroVol = False
            for it in range(numpoints):
                if colinear(Segment(points[it], points[(it + 1) % numpoints]),inter):
                    zeroVol = True
            if not zeroVol:
                result.append(Point(FieldNumber(inter.x().exact()),FieldNumber(inter.y().exact())))
                centroid = centroid + inter
                count += 1

        if len(result) == 0:
            return "inside",None

        centroid = centroid.scale(FieldNumber(1)/FieldNumber(count))
        #keep representation simple
        centroid = roundExact(centroid)

        #print(centroid)

        dontAdd = False
        for it in range(numpoints):
            if isBadTriangle(centroid, points[it], points[(it + 1) % numpoints]):
                dontAdd = True
        if not dontAdd:
            if centroid not in intersections:
                altNum += 1
                circleNum += 1
                return "inside",[centroid] + result

        return "inside",result

def unsafeOrientedFindCenterOfLink(points,num=0,axs=None):

    trianglebases = []
    for i in range(len(points)):
        nextI = (i + 1) % len(points)
        trianglebases.append([i, nextI])
        if axs != None:
            axs.plot([numericPoint(points[j])[0] for j in [i, nextI]],
                     [numericPoint(points[j])[1] for j in [i, nextI]], color="black")

    circles = []
    for i, j in trianglebases:
        c = (points[i] + points[j]).scale(onehalf)
        rsq = distsq(c, points[j])
        circles.append([c, rsq])
        if axs != None:
            circle = plt.Circle((numericPoint(c)[0], numericPoint(c)[1]), np.sqrt(float(rsq)), color="yellow",
                                fill=False, zorder=1000)
            axs.add_patch(circle)

    rays = []
    for i, j in trianglebases:
        diff = points[j] - points[i]
        orth = Point(zero - diff.y(), diff.x())
        if onWhichSide(Segment(points[i], points[j]), points[i] + orth) != "left":
            orth = orth.scale(FieldNumber(-1))
        startRay = Segment(points[i], points[i] + orth)
        endRay = Segment(points[j], points[j] + orth)
        for offset in range(len(points)):
            qId = (j + offset) % len(points)
            nextQId = (qId + 1) % len(points)
            if onWhichSide(endRay, points[qId]) != "left" and onWhichSide(endRay, points[nextQId]) == "left":
                inter = supportingRayIntersectSegment(endRay, Segment(points[qId], points[nextQId]))
                assert (inter != None)
                rays.append(Segment(endRay.source(), inter))
                if axs != None:
                    axs.plot([numericPoint(rays[-1].source())[0], numericPoint(rays[-1].target())[0]],
                             [numericPoint(rays[-1].source())[1], numericPoint(rays[-1].target())[1]], color="blue")
                break

        for offset in reversed(range(len(points))):
            qId = (i + offset) % len(points)
            nextQId = (qId + (len(points) - 1)) % len(points)
            if onWhichSide(startRay, points[qId]) != "right" and onWhichSide(startRay, points[nextQId]) == "right":
                inter = supportingRayIntersectSegment(startRay, Segment(points[qId], points[nextQId]))
                assert (inter != None)
                rays.append(Segment(startRay.source(), inter))
                if axs != None:
                    axs.plot([numericPoint(rays[-1].source())[0], numericPoint(rays[-1].target())[0]],
                             [numericPoint(rays[-1].source())[1], numericPoint(rays[-1].target())[1]], color="blue")
                break
        # axs.plot([numericPoint(points[j])[0],numericPoint(points[j]+orth)[0]],[numericPoint(points[j])[1],numericPoint(points[j]+orth)[1]],color="blue")

    candidatePoints = []
    for i in range(len(rays)):
        for j in range(i + 1, len(rays)):
            if (inter := innerIntersect(rays[i].source(), rays[i].target(), rays[j].source(),
                                        rays[j].target())) is not None:
                candidatePoints.append(inter)

    for ray in rays:
        for c, rsq in circles:
            if (inters := outsideIntersectionsSegmentCircle(c, rsq, ray)) is not None:
                for inter in inters:
                    candidatePoints.append(inter)

    insidePoints = []
    for cp in candidatePoints:
        isGood = True
        for a, b in trianglebases:
            if isBadTriangle(cp, points[a], points[b]):
                isGood = False
                break
        if isGood:
            insidePoints.append(cp)

    # construct centroids
    solutions = []
    centroid = Point(zero, zero)
    count = 0
    for ip in insidePoints:
        centroid += ip
        count += 1
    if count != 0:
        centroid = centroid.scale(FieldNumber(1) / FieldNumber(count))
        centroid = roundExact(centroid)
        isGood = True
        for a, b in trianglebases:
            if isBadTriangle(centroid, points[a], points[b]):
                isGood = False
                break
        if isGood:
            solutions = [centroid]
    results = solutions + insidePoints
    return "inside" if len(results) > 0 else "None", results if len(results) > 0 else None

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
        vertexSol = _linkCanBeSolvedByVertex(list(reversed(points)))
        if vertexSol != None:
            return "vertex", len(points) - 1 - vertexSol

        soltype, sol =  unsafeOrientedFindCenterOfLink(list(reversed(points)),num,axs)
        if soltype == "vertex":
            return "vertex",len(points)-1-sol
        else:
            return soltype,sol

    if onlyLeft:
        vertexSol = _linkCanBeSolvedByVertex(points)
        if vertexSol != None:
            return "vertex", vertexSol
        return unsafeOrientedFindCenterOfLink(points,num,axs)

def unsafeOrientedFindCenterOfLinkConstrained(points,constraintA,constraintB,num=0,axs=None):
    numpoints = len(points)

    onBoundary = False
    boundaryIdx = None

    if ((constraintA + 1) % numpoints == constraintB):
        onBoundary = True
        boundaryIdx = constraintA
    if ((constraintB + 1) % numpoints == constraintA):
        onBoundary = True
        boundaryIdx = constraintB

    querySegment = Segment(points[constraintA], points[constraintB])

    orth = []

    for i in range(numpoints):
        idx = i
        nextIdx = (i + 1) % numpoints
        diff = points[nextIdx] - points[idx]
        orth.append(Point(zero - diff.y(), diff.x()))

    intersections = []

    # circle circle intersection is not needed
    global circleNum
    circleNum = len(intersections)

    # intersect segment with rays
    for i in range(numpoints):
        for ioff in range(2):

            oIdx = (i + ioff) % numpoints
            idx = oIdx
            p = points[idx]
            o = orth[i]

            if (idx == constraintB) or (idx == constraintA):
                continue

            inter = supportingRayIntersectSegment(Segment(p, p + o), querySegment)

            if inter != None:
                dontAdd = False
                for it in range(numpoints):
                    if onBoundary and it == boundaryIdx:
                        continue
                    if isBadTriangle(inter, points[it], points[(it + 1) % numpoints]):
                        dontAdd = True
                if not dontAdd:
                    if inter not in intersections:
                        intersections.append(inter)

    global altNum
    altNum = len(intersections)

    # line-circle numerical
    p = querySegment.source()
    o = querySegment.target() - querySegment.source()

    for j in range(numpoints):
        addors = []
        m = points[j].scale(FieldNumber(0.5)) + points[(j + 1) % numpoints].scale(FieldNumber(0.5))
        rsqr = Segment(m, points[j]).squared_length()
        inCirc = inCircle(m, rsqr, p)
        if inCirc == "on":
            # dont add vertices!
            # addors.append(p)
            if inCircle(m, rsqr, p + o) != "on":
                # if m is in direction of o, there should be a second one
                dotomp = dot(o, m - p)
                if dotomp > FieldNumber(0):
                    # there is a second one
                    mid = altitudePoint(Segment(p, p + o), m)

                    # if this is 0, then its colinear and addor will be the next point along the boundary, which is NOT inside
                    if Segment(mid, m).squared_length() != 0:
                        diff = p - mid
                        if distsq(p,mid - diff) <= distsq(p,p+o):
                            addors.append(mid - diff)
        elif inCirc == "outside":
            mid = altitudePoint(Segment(p, p + o), m)
            secondInCirc = inCircle(m, rsqr, mid)
            if secondInCirc == "on":
                # if we are tangent, we have to check that we are not inserting a vertex
                if not ((mid == points[j]) or (mid == points[(j + 1) % numpoints])):
                    if distsq(p,mid) <= distsq(p,p+o):
                        addors.append(mid)
            elif secondInCirc == "inside":
                ex, addor1 = binaryIntersection(m, rsqr, Segment(p, mid))
                diff = addor1 - mid
                addor2 = mid - diff
                if onWhichSide(Segment(points[j], points[(j + 1) % numpoints]), addor1) == "left":
                    addors.append(addor1)
                if onWhichSide(Segment(points[j], points[(j + 1) % numpoints]), addor2) == "left":
                    addors.append(addor2)
        else:
            # p is in the circle already. then there is only one!
            # s is upperbound on 2*r
            s = None
            if rsqr < FieldNumber(1):
                s = FieldNumber(2)
            elif rsqr < FieldNumber(2):
                s = rsqr * FieldNumber(2)
            else:
                s = rsqr
            olsqr = Segment(p, p + o).squared_length()
            if olsqr < FieldNumber(1):
                s = s / olsqr
            q = p + o.scale(s)
            # q should be guaranteed to lie outside!
            assert (inCircle(m, rsqr, q) == "outside")
            ex, addor = binaryIntersection(m, rsqr, Segment(p, q))
            addors.append(addor)
        for addor in addors:
            dontAdd = False
            for it in range(numpoints):
                if onBoundary and it == boundaryIdx:
                    continue
                if isBadTriangle(addor, points[it], points[(it + 1) % numpoints]):
                    dontAdd = True
            if not dontAdd:
                if addor not in intersections:
                    intersections.append(addor)

    if len(intersections) == 0:
        return "None", None
    else:
        # form sum of all and add to intersections as best candidate
        result = []
        centroid = Point(zero, zero)
        count = 0
        for inter in intersections:
            zeroVol = False
            for it in range(numpoints):
                if onBoundary and it == boundaryIdx:
                    continue
                if colinear(Segment(points[it], points[(it + 1) % numpoints]), inter):
                    zeroVol = True
            if not zeroVol:
                result.append(Point(FieldNumber(inter.x().exact()), FieldNumber(inter.y().exact())))
                centroid = centroid + inter
                count += 1

        if len(result) == 0:
            return "None", None

        centroid = centroid.scale(FieldNumber(1) / FieldNumber(count))
        centroid = roundExactOnSegment(querySegment,centroid,100)

        # print(centroid)

        dontAdd = False
        for it in range(numpoints):
            if onBoundary and it == boundaryIdx:
                continue
            if isBadTriangle(centroid, points[it], points[(it + 1) % numpoints]):
                dontAdd = True
        if not dontAdd:
            if centroid not in intersections:
                altNum += 1
                circleNum += 1
                return "inside", [centroid] + result

        return "inside", result

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
        soltype,sol = unsafeOrientedFindCenterOfLinkConstrained(list(reversed(points)),len(points)-1-constraintA,len(points)-1-constraintB,num,axs)
        if soltype == "vertex":
            return "vertex",len(points)-1-sol
        else:
            return soltype,sol

    if onlyLeft:
        vertexSol = _linkCanBeSolvedByVertex(points,[constraintA,constraintB])
        if vertexSol != None:
            return "vertex", vertexSol
        return unsafeOrientedFindCenterOfLinkConstrained(points,constraintA,constraintB,num,axs)

def getParamOfPointOnSegment(seg:Segment,p:Point):
    assert(seg.squared_length() > zero)
    tx = (p-seg.source()).x()/(seg.target()-seg.source()).x()
    ty = (p-seg.source()).y()/(seg.target()-seg.source()).y()
    if (seg.target()-seg.source()).x() == zero:
        tx = ty
    elif (seg.target()-seg.source()).y() == zero:
        ty = tx
    if tx != ty:
        return None
    return tx

def numericPoint(p:Point):
    return [float(p.x()),float(p.y())]

def circleIntersectsCircle(p:Point,rpsq:FieldNumber,q:Point,rqsq:FieldNumber,acc=10):

    if p == q:
        return True

    bigP,bigRsqr,smallP,smallRsqr = p,rpsq,q,rqsq
    if rpsq < rqsq:
        bigP,bigRsqr,smallP,smallRsqr = q,rqsq,p,rpsq

    diff = smallP - bigP
    if inCircle(bigP,bigRsqr,bigP+diff)=="inside":
        if dot(diff,diff) < FieldNumber(1):
            diff = diff.scale(FieldNumber(1)/dot(diff,diff))
        if bigRsqr > FieldNumber(1):
            diff = diff.scale(bigRsqr)
    assert(inCircle(bigP,bigRsqr,bigP+diff)!="inside")

    baseSeg = outsideClipSegmentToCircle(bigP,bigRsqr,Segment(bigP,bigP+diff))
    assert(len(baseSeg) == 2)
    seg = outsideClipSegmentToCircle(smallP,smallRsqr,Segment(baseSeg[0],baseSeg[1]))
    if len(seg) == 0:
        return False
    #assert(len(seg) != 1)
    return True

def getCircleIntersections(p:Point,rpsq:FieldNumber,q:Point,rqsq:FieldNumber,acc=10):

    if p == q:
        return []

    bigP,bigRsqr,smallP,smallRsqr = p,rpsq,q,rqsq
    if rpsq < rqsq:
        bigP,bigRsqr,smallP,smallRsqr = q,rqsq,p,rpsq

    diff = smallP - bigP
    if inCircle(bigP,bigRsqr,bigP+diff)=="inside":
        if dot(diff,diff) < FieldNumber(1):
            diff = diff.scale(FieldNumber(1)/dot(diff,diff))
        if bigRsqr > FieldNumber(1):
            diff = diff.scale(bigRsqr)
    assert(inCircle(bigP,bigRsqr,bigP+diff)!="inside")

    baseSeg = outsideClipSegmentToCircle(bigP,bigRsqr,Segment(bigP,bigP+diff))
    assert(len(baseSeg) == 2)
    seg = outsideClipSegmentToCircle(smallP,smallRsqr,Segment(baseSeg[0],baseSeg[1]))
    sols = []
    if len(seg) == 0:
        return []
    left = None
    right = None
    if len(seg) == 1:
        bounder = roundExactOnSegmentBounder(Segment(bigP,smallP),seg[0])
        left = bounder[0]
        right = bounder[0]
        for p in bounder:
            if dot(bigP-p,bigP-p) < dot(bigP-left,bigP-left):
                left = p
            if dot(smallP-p,smallP-p) < dot(smallP-right,smallP-right):
                right = p
        #sols.append(seg[0])
    elif len(seg) == 2:


        left = seg[0]
        right = seg[1]

    if inCircle(bigP, bigRsqr, left) == "inside" and inCircle(bigP, bigRsqr, right) == "inside":
        return []

    while(distsq(left,right) > (FieldNumber(1)/FieldNumber(acc))):
        mid = (left+right).scale(onehalf)
        if bigRsqr - dot(bigP-mid,bigP-mid) > smallRsqr - dot(smallP-mid,smallP-mid):
            left = mid
        else:
            right = mid

    if distsq(bigP,left) > bigRsqr or distsq(smallP,right) > smallRsqr:
        return []

    orth1 = Point(diff.y(),zero - diff.x())
    orth2 = Point(zero - diff.y(),diff.x())
    seg1 = Segment(left+orth1,left+orth2)
    seg2 = Segment(right+orth1,right+orth2)
    for p in outsideIntersectionsSegmentCircle(bigP,bigRsqr,seg1):
        sols.append(p)
    for p in outsideIntersectionsSegmentCircle(smallP,smallRsqr,seg1):
        sols.append(p)
    for p in insideIntersectionsSegmentCircle(smallP,smallRsqr,seg1):
        sols.append(p)
    for p in outsideIntersectionsSegmentCircle(smallP,smallRsqr,seg2):
        sols.append(p)
    for p in insideIntersectionsSegmentCircle(smallP,smallRsqr,seg2):
        sols.append(p)

    return sols

def getCircleIntersectionsOutside(p:Point,rpsq:FieldNumber,q:Point,rqsq:FieldNumber,acc=10):

    if p == q:
        return []

    bigP,bigRsqr,smallP,smallRsqr = p,rpsq,q,rqsq
    if rpsq < rqsq:
        bigP,bigRsqr,smallP,smallRsqr = q,rqsq,p,rpsq

    diff = smallP - bigP
    if inCircle(bigP,bigRsqr,bigP+diff)=="inside":
        if dot(diff,diff) < FieldNumber(1):
            diff = diff.scale(FieldNumber(1)/dot(diff,diff))
        if bigRsqr > FieldNumber(1):
            diff = diff.scale(bigRsqr)
    assert(inCircle(bigP,bigRsqr,bigP+diff)!="inside")

    baseSeg = outsideClipSegmentToCircle(bigP,bigRsqr,Segment(bigP,bigP+diff))
    assert(len(baseSeg) == 2)
    seg = outsideClipSegmentToCircle(smallP,smallRsqr,Segment(baseSeg[0],baseSeg[1]))
    sols = []
    if len(seg) == 0:
        return []
    left = None
    right = None
    if len(seg) == 1:
        bounder = roundExactOnSegmentBounder(Segment(bigP,smallP),seg[0])
        left = bounder[0]
        right = bounder[0]
        for p in bounder:
            if dot(bigP-p,bigP-p) < dot(bigP-left,bigP-left):
                left = p
            if dot(smallP-p,smallP-p) < dot(smallP-right,smallP-right):
                right = p
        #sols.append(seg[0])
    elif len(seg) == 2:


        left = seg[0]
        right = seg[1]

    if inCircle(bigP, bigRsqr, left) == "inside" and inCircle(bigP, bigRsqr, right) == "inside":
        return []

    while(distsq(left,right) > (FieldNumber(1)/FieldNumber(acc))):
        mid = (left+right).scale(onehalf)
        if bigRsqr - dot(bigP-mid,bigP-mid) > smallRsqr - dot(smallP-mid,smallP-mid):
            left = mid
        else:
            right = mid

    if distsq(bigP,left) > bigRsqr or distsq(smallP,right) > smallRsqr:
        return []

    orth1 = Point(diff.y(),zero - diff.x())
    orth2 = Point(zero - diff.y(),diff.x())
    seg1 = Segment(left+orth1,left+orth2)
    seg2 = Segment(right+orth1,right+orth2)
    for p in outsideIntersectionsSegmentCircle(bigP,bigRsqr,seg1):
        if inCircle(smallP,smallRsqr,p) == "outside":
            sols.append(p)
    for p in outsideIntersectionsSegmentCircle(smallP,smallRsqr,seg1):
        if inCircle(bigP,bigRsqr,p) == "outside":
            sols.append(p)
    for p in outsideIntersectionsSegmentCircle(bigP,bigRsqr,seg2):
        if inCircle(smallP,smallRsqr,p) == "outside":
            sols.append(p)
    for p in outsideIntersectionsSegmentCircle(smallP,smallRsqr,seg2):
        if inCircle(bigP,bigRsqr,p) == "outside":
            sols.append(p)

    return sols







