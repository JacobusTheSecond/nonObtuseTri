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