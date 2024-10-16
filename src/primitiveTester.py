import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from cgshop2025_pyutils import InstanceDatabase,ZipSolutionIterator,verify
from cgshop2025_pyutils.geometry import FieldNumber, Point, Segment

plot_counter = 0
xs = []
ys = []
exactPoints = []

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

def dot(a : Point, b:Point):
    return (a.x() * b.x()) + (a.y() * b.y())

def sourceProjection(ab:Segment,pq:Segment):
    a = ab.source()
    b = ab.target()
    p = pq.source()
    q = pq.target()
    if (sign(dot(b-a,q-a)) == sign(dot(b-a,p-a))):
        return None
    else:
        s = dot(b-a,q-a)/(dot(b-a,q-a) - dot(b-a,p-a))
        print(float(s))
        #assert(FieldNumber(0) < s and s < 1)
        return p.scale(s) + q.scale(FieldNumber(1)-s)

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


def getOrthogonalRight(s : Segment):
    diff = s.target() - s.source()
    sourceOrth = s.source() + Point(diff.y(),FieldNumber(0)-diff.x())
    targetOrth = s.target() + Point(diff.y(),FieldNumber(0)-diff.x())
    return (sourceOrth,targetOrth)

def primitiveTester():
    fig, axs = plt.subplots(1, 1)
    def on_click(event):
        global xs
        global ys
        (x,y) = (event.xdata,event.ydata)
        exactPoints.append(Point(FieldNumber(x),FieldNumber(y)))
        xs.append(x)
        ys.append(y)
        axs.clear()
        axs.scatter(xs,ys,color="blue")

        if len(xs)>=4:
            axs.plot([xs[0],xs[1]],[ys[0],ys[1]])
            axs.plot([xs[2],xs[3]],[ys[2],ys[3]])
            i = sourceProjection(Segment(exactPoints[0],exactPoints[1]),Segment(exactPoints[2],exactPoints[3]))
            if i != None:
                floatI = (float(i.x()),float(i.y()))
                axs.plot([xs[0],floatI[0]],[ys[0],floatI[1]],color="red" if isOnRightSide(Segment(exactPoints[0],exactPoints[1]),i) else "green")

        if len(xs)>=4:
            midpoint = exactPoints[0].scale(FieldNumber(0.5)) + exactPoints[1].scale(FieldNumber(0.5))
            axs.scatter([float(midpoint.x())],[float(midpoint.y())])

            circle = plt.Circle((float(midpoint.x()), float(midpoint.y())), float(np.sqrt(float(Segment(midpoint,exactPoints[0]).squared_length()))), color="yellow", fill=False, zorder=1000)
            axs.add_patch(circle)

            mode,i = binaryIntersection(midpoint,Segment(midpoint,exactPoints[0]).squared_length(),Segment(exactPoints[2],exactPoints[3]))

            print(mode)
            if i != None:
                floatI = (float(i.x()),float(i.y()))
                axs.plot([xs[0],floatI[0]],[ys[0],floatI[1]],color="red" if isOnRightSide(Segment(exactPoints[0],exactPoints[1]),i) else "green")

        if(len(xs)==2):
            exactSO,exactTO = getOrthogonalRight(Segment(exactPoints[0],exactPoints[1]))
            sO = (float(exactSO.x()),float(exactSO.y()))


            axs.plot([xs[0],sO[0]],[ys[0],sO[1]])
        axs.set_xlim([0, 10])
        axs.set_ylim([0, 10])
        axs.set_aspect('equal')
        plt.draw()


    #fig.canvas.mpl_connect('key_press_event',on_press)
    fig.canvas.mpl_connect('button_press_event', on_click)

    global xs,ys
    axs.clear()
    axs.scatter(xs, ys)
    axs.set_xlim([0, 10])
    axs.set_ylim([0, 10])
    axs.set_aspect('equal')
    plt.draw()
    plt.show()

if __name__=="__main__":
    primitiveTester()