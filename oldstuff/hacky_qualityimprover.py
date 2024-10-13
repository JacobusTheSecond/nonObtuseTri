import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from cgshop2025_pyutils import InstanceDatabase,ZipSolutionIterator,ZipWriter,Cgshop2025Instance

import triangle as tr #https://rufat.be/triangle/

from pydantic import BaseModel, Field, model_validator

import math
import requests


class MinNotObtTriangInstance(BaseModel):
    """
    This schema defines the data format for the CG_SHOP_2025_Instance, which involves finding a
    Non-Obtuse Triangulation with the minimum number of Steiner points. Point coordinates are
    integers, while Steiner point coordinates can be rational numbers.

    The instance includes a region boundary, represented as a list of counter-clockwise oriented
    point indices, describing the area to be triangulated. This boundary can be a convex hull or
    any simple polygon.

    The triangulation must include all points and adhere to the specified constraints, meaning
    that there must be a direct line connection between every two points in the constraints.
    However, the triangulation can split segments of both the boundary and the constraints.

    Special cases of instances:
    - Only points with the region boundary as the convex hull: This represents an unconstrained
      triangulation. Although you may need to add Steiner points, there are no constraints, and
      the region boundary is provided for convenience.
    - No constraints with the region boundary containing all points: This scenario is the
      classical problem of triangulating a simple polygon, for which a solution with a linear
      number of Steiner points is known to exist.

    """

    instance_uid: str = Field(..., description="Unique identifier of the instance.")
    num_points: int = Field(
        ...,
        description="Number of points in the instance. All points must be part of the final triangulation.",
    )
    points_x: list[int] = Field(..., description="List of x-coordinates of the points.")
    points_y: list[int] = Field(..., description="List of y-coordinates of the points.")
    region_boundary: list[int] = Field(
        ...,
        description=(
            "Boundary of the region to be triangulated, given as a list of counter-clockwise oriented "
            "point indices. The triangulation may split boundary segments. The first point is not "
            "repeated at the end of the list."
        ),
    )
    num_constraints: int = Field(
        ..., description="Number of constraints in the instance."
    )
    additional_constraints: list[list[int]] = Field(
        default_factory=list,
        description=(
            "List of constraints additional to the region_boundary, each given as a list of two point indices. The triangulation may split "
            "constraint segments, but must include a straight line between the two points."
        ),
    )

    @model_validator(mode="after")
    def validate_points(self):
        if (
            len(self.points_x) != self.num_points
            or len(self.points_y) != self.num_points
        ):
            msg = "Number of points does not match the length of the x/y lists."
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_region_boundary(self):
        if len(self.region_boundary) < 3:
            msg = "The region boundary must have at least 3 points."
            raise ValueError(msg)
        for idx in self.region_boundary:
            if idx < 0 or idx >= self.num_points:
                msg = "Invalid point index in region boundary."
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_constraints(self):
        for constraint in self.additional_constraints:
            if len(constraint) != 2:
                msg = "Constraints must have exactly two points."
                raise ValueError(msg)
            for idx in constraint:
                if idx < 0 or idx >= self.num_points:
                    msg = "Invalid point index in constraint."
                    raise ValueError(msg)
        return self

def getInstance(uid):
    url = "https://cgshop.ibr.cs.tu-bs.de/bip-api/min_nonobt_triang/instances/" + uid + "/raw"
    # sending get request and saving the response as response object
    r = requests.get(url=url)

    # extracting data in json format
    instance = r.json()
    return MinNotObtTriangInstance(**instance)

def getInstanceUIDs(limit):
    url = "https://cgshop.ibr.cs.tu-bs.de/bip-api/min_nonobt_triang/instances?limit="+str(limit)+"&offset=0&add_total_count=false"
    # sending get request and saving the response as response object
    r = requests.get(url=url)

    # extracting data in json format
    instances = r.json()

    return [instance['uid'] for instance in instances]



# returns square of distance b/w two points
def lengthSquare(X, Y):
    xDiff = X[0] - Y[0]
    yDiff = X[1] - Y[1]
    return xDiff * xDiff + yDiff * yDiff

def angles(A,B,C):
    # Square of lengths be a2, b2, c2
    a2 = lengthSquare(B, C)
    b2 = lengthSquare(A, C)
    c2 = lengthSquare(A, B)

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

def altitudepoint(A,B,C):
    x = A
    u = B
    v = C
    n = v-u
    n /= np.linalg.norm(n,2)
    return u + n*np.dot(x-u,n)



def maxAngle(A, B, C):
    # Square of lengths be a2, b2, c2
    a2 = lengthSquare(B, C)
    b2 = lengthSquare(A, C)
    c2 = lengthSquare(A, B)

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

    return max(alpha,betta,gamma)

def plot(A,B,name,ax,mark=-1):
    SC = len(B['vertices']) - len(A['vertices'])
    if SC > 0:
        name += " [SC:" + str(SC) + "]"
    cc = ""
    cr = ""
    marker = ""
    if 'triangles' in B:
        badCount = 0
        for i in range(len(B['triangles'])):
            tri = B['triangles'][i]
            cords = B['vertices'][tri]
            cords = np.concatenate((cords, [cords[0]]))
            ax.plot(*(cords.T), color='black', linewidth=1)
            if i == mark:
                badCount += 1
                t = plt.Polygon(B['vertices'][tri], color='g')
                ax.add_patch(t)
                cc = circumcenter(*B['vertices'][tri])
                cr = np.linalg.norm(B['vertices'][tri][0] - cc)
                marker = B['vertices'][tri]
            #print(*B['vertices'][tri])
            if i != mark and maxAngle(*B['vertices'][tri]) > 90.0001:
                badCount += 1
                t = plt.Polygon(B['vertices'][tri], color='b')
                ax.add_patch(t)
        name += " (>90°: " + str(badCount) + ")"
    for e in A['segments']:
        ax.plot(*(B['vertices'][e].T), color='red', linewidth=2)
    ax.scatter(*(B['vertices'][:len(A['vertices'])].T), marker='.', color='black', zorder=100)
    ax.scatter(*(B['vertices'][len(A['vertices']):].T), marker='.', color='green', zorder=100)
    if cc!="" :
        ax.scatter([cc[0]],[cc[1]], marker='.', color='yellow', zorder=1000)
        circle = plt.Circle(cc,cr,color="yellow",fill=False,zorder=1000)
        ax.add_patch(circle)
        #find intersection with all segments
        p = marker[np.argmax(angles(*marker))]
        closestSegment = None
        closestIntersection = None
        baseline = np.dstack((p,np.array(cc)))[0]
        ax.plot(*baseline, color='yellow', linewidth=2,zorder=1000)
        for segment in B['segments']:
            l1 = B['vertices'][segment]
            intersection = intersect(p,np.array(cc),l1[0],l1[1])
            if intersection!=None:
                if closestIntersection == None or np.linalg.norm(p-intersection) < np.linalg.norm(p-closestIntersection):
                    closestSegment = B['vertices'][segment]
                    closestIntersection = intersection
        if closestIntersection != None:
            ax.scatter(*closestIntersection,color="red",zorder=10000)
            ax.scatter(*altitudepoint(p,closestSegment[0],closestSegment[1]),color="yellow",zorder=100000)

    ax.set_aspect('equal')
    ax.title.set_text(name)

def circumcenter(a,b,c):
    ax = a[0]
    ay = a[1]
    bx = b[0]
    by = b[1]
    cx = c[0]
    cy = c[1]
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
    return (ux, uy)

def intersect(p1, p2, p3, p4):
    x1,y1 = p1
    x2,y2 = p2
    x3,y3 = p3
    x4,y4 = p4
    denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
    if denom == 0: # parallel
        return None
    ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
    if ua < 0 or ua > 1: # out of range
        return None
    ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
    if ub <= 0 or ub >= 1: # out of range
        return None
    x = x1 + ua * (x2-x1)
    y = y1 + ua * (y2-y1)
    return (x,y)

class Triangle:
    def __init__(self,triangulation,idx):
        self.indices = triangulation['triangles'][idx]
        self.points = triangulation['vertices'][self.indices]
        self.mask = []
        for i in range(3):
            v = (np.sort(np.dstack((self.indices,np.roll(self.indices,-1))),axis=2)[0])[i]
            self.mask.append(np.any(np.all(np.dstack((np.sort(triangulation['segments'],axis=1)[:,0] == v[0],np.sort(triangulation['segments'],axis=1)[:,1] == v[1])),axis=2)))
        #print(self.mask)
import time

def improveQuality(Ain):
    pausetime = 0.03
    #A is input
    #Triangle(B,0)
    Ain = tr.triangulate(Ain,'p')
    Ain = dict(vertices=Ain['vertices'],segments=Ain['segments'])
    A = dict(vertices=Ain['vertices'],segments=Ain['segments'])
    #plt.ion()
    #fig, axs = plt.subplots(1, 1)
    for n in range(1000):
        inserted = True
        while inserted == True:
            B = tr.triangulate(A,"p")
            inserted = False
            foundBad = 0
            for i in range(len(B['triangles'])):
                triangle = B['vertices'][B['triangles'][i]]
                if(maxAngle(*triangle)>90.00001):
                    foundBad += 1
                    #bad triangle
                    #plot(Ain, B, "constrainedDelauney", axs)
                    #plt.draw()
                    #plt.pause(10)
                    #axs.clear()

                    #identify circumcenter
                    cc = circumcenter(*triangle)

                    ## check if it can be inserted without problem

                    #p is point of offending angle
                    p = triangle[np.argmax(angles(*triangle))]

                    #check with every constraint edge if the line from p to the circumcenter intersects it
                    #store the closest intersecting segment in closestSegment
                    closestSegment = None
                    closestIntersection = None
                    closestSegmentIndex = None
                    baseline = np.dstack((p, np.array(cc)))[0]
                    for segmentIndex in range(len(B['segments'])):
                        segment = B['segments'][segmentIndex]
                        l1 = B['vertices'][segment]
                        intersection = intersect(p, np.array(cc), l1[0], l1[1])
                        if intersection != None:
                            if closestIntersection == None or np.linalg.norm(p - intersection) < np.linalg.norm(
                                    p - closestIntersection):
                                closestSegment = B['vertices'][segment]
                                closestIntersection = intersection
                                closestSegmentIndex = segmentIndex

                    if closestIntersection != None:
                        #we can insert circumcenter and retriangulate
                        #need to split the edge
                        ap = altitudepoint(p,closestSegment[0],closestSegment[1])

                        #add altitude point
                        A['vertices'] = np.vstack((A['vertices'],np.array([ap])))

                        #remove segment
                        deletor = A['segments'][closestSegmentIndex]
                        A['segments'] = np.vstack((np.delete(A['segments'],(closestSegmentIndex),axis=0),np.array([[deletor[0],len(A['vertices'])-1],[len(A['vertices'])-1,deletor[1]]])))
                        inserted = True
                        break
        if inserted:
            continue
        if foundBad>0:
            C = None
            for c in reversed(range(1,max(5,foundBad//2))):
                C = tr.triangulate(A, "pq0.000001U90.000001S"+str(c))
                if(len(A['segments']) == len(C['segments'])):
                    break
            if (len(A['segments']) == len(C['segments'])):
                print("FUCK!")
            if len(A['vertices']) < len(C['vertices']):
                inserted = True
            A['vertices'] = np.vstack((A['vertices'], np.array(C['vertices'][len(A['vertices'])-len(C['vertices']):])))
            A['segments'] = C['segments']
            B = tr.triangulate(A,"p")
            #plot(Ain, B, "constrainedDelauney", axs)
            #plt.draw()
            #plt.pause(pausetime)
            #axs.clear()

        #if inserted == False:
        #    for i in range(len(B['triangles'])):
        #        triangle = B['vertices'][B['triangles'][i]]
        #        if(maxAngle(*triangle)>90.00001):
        #            foundBad = True
        #            #bad triangle
        #            plot(Ain, B, "constrainedDelauney", axs, mark=i)
        #            plt.xlim(-500,10500)
        #            plt.ylim(-500,10500)
        #            plt.draw()
        #            plt.pause(pausetime)
        #            axs.clear()

                    #identify circumcenter
        #            cc = circumcenter(*triangle)

                    ## check if it can be inserted without problem

                    #p is point of offending angle
        #            p = triangle[np.argmax(angles(*triangle))]

                    #check with every constraint edge if the line from p to the circumcenter intersects it
                    #store the closest intersecting segment in closestSegment
        #            closestSegment = None
        #            closestIntersection = None
        #            baseline = np.dstack((p, np.array(cc)))[0]
        #            for segment in B['segments']:
        #                l1 = B['vertices'][segment]
        #                intersection = intersect(p, np.array(cc), l1[0], l1[1])
        #                if intersection != None:
        #                    if closestIntersection == None or np.linalg.norm(p - intersection) < np.linalg.norm(
        #                            p - closestIntersection):
        #                        closestSegment = B['vertices'][segment]
        #                        closestIntersection = intersection
        #            if closestIntersection == None:
                        #we can insert circumcenter and retriangulate
        #                A['vertices'] = np.vstack((A['vertices'],np.array([cc])))
        #                inserted = True
        #                break
        if foundBad == True and inserted == False:
            print("Huh...")
        if foundBad == False:
            #plt.ioff()
            #plot(Ain, B, "constrainedDelauney", axs)
            #plt.show()
            return B
    return None



def thinTriange(n):
    #flat triangle
    points = []
    for i in range(n):
        points.append([i,n*i])
    points.append([0,(n-1)*n])
    constraints = []
    for i in range(len(points)):
        constraints.append([i,(i+1)%(len(points))])
    A=dict(vertices=np.array(points),segments=np.array(constraints))
    return A

def convert(data):

    #convert to triangulation type
    points = np.column_stack((data.points_x, data.points_y))
    constraints = np.column_stack((data.region_boundary, np.roll(data.region_boundary, -1)))
    if (len(data.additional_constraints) != 0):
        constraints = np.concatenate((constraints, data.additional_constraints))
    A = dict(vertices=points, segments=constraints)
    return A

def main():

    idb = InstanceDatabase("/Users/styx/Downloads/challenge_instances_cgshop25_rev1.zip")

    if Path("/Users/styx/Desktop/solutions.zip").exists():
        Path("/Users/styx/Desktop/solutions.zip").unlink()

    solutions = []
    solutionsizes = []
    i = 0
    for instance in idb:
        i+=1
        ins = convert(instance)
        solution = improveQuality(ins)
        if solution != None:
            print("found solution for instance",i,"of size",len(solution['vertices']) - len(ins['vertices']))
            solutions.append(solution)
            solutionsizes.append(len(solution['vertices']) - len(ins['vertices']))
        else:
            print("Fuck instance",i)

    # Write the solutions to a new zip file
    #with ZipWriter("example_solutions.zip") as zw:
    #    for solution in solutions:
    #        zw.add_solution(solution)
    exit()

    #summaryPlot(thinTriange(10))
    #get instance
    instanceUIDs = getInstanceUIDs(1000)
    for i in range(len(instanceUIDs)):
        #if i != 4:
        #    continue
        #print("triangulating and plotting",instanceUIDs[i])
        print(i)
        data = getInstance(instanceUIDs[i])
        cdata = convert(data)
        #if(i==4):
        #    segments = np.concatenate((np.array([[21,39],[39,35]]),cdata['segments'][1:]))
        #    for idx in range(len(segments)):
        #        for jdx in range(len(segments[idx])):
        #            if(segments[idx,jdx] > 29):
        #                segments[idx,jdx] -= 1
        #    cdata['segments'] = segments
        #    deletor = cdata['vertices'][29]
        #    cdata['vertices'] = np.delete(cdata['vertices'],29,0)
        improveQuality(cdata)


if __name__=="__main__":
    main()



