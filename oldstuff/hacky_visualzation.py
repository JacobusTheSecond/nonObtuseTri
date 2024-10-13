import matplotlib.pyplot as plt
import numpy as np

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

def triangulateAndPlot(A,opt,axs,name):
    B = A
    #print(B)
    if opt != "":
        B = tr.triangulate(A,opt)
        #B['vertices'] = np.concatenate((B['vertices'],np.array([deletor])))
        #C =  dict(vertices=np.concatenate((B['vertices'],np.array([deletor]))), segments=A['segments'])
        #B = tr.triangulate(C,opt)
    SC = len(B['vertices']) - len(A['vertices'])
    if SC > 0:
        name += " [SC:" + str(SC)+"]"
    if 'triangles' in B:
        badCount = 0
        for tri in B['triangles']:
            cords = B['vertices'][tri]
            cords = np.concatenate((cords, [cords[0]]))
            axs.plot(*(cords.T), color='black', linewidth=1)
            if maxAngle(*B['vertices'][tri]) > 90.0001:
                badCount+=1
                t = plt.Polygon(B['vertices'][tri], color='b')
                axs.add_patch(t)
        name += " (>90°: " +str(badCount)+")"
    for e in A['segments']:
        axs.plot(*(B['vertices'][e].T), color='red',linewidth=2)
    axs.scatter(*(B['vertices'][:len(A['vertices'])].T), marker='.', color='black', zorder=100)
    axs.scatter(*(B['vertices'][len(A['vertices']):].T), marker='.', color='green', zorder=100)
    axs.set_aspect('equal')
    axs.title.set_text(name)

def summaryPlot(A):
    #plot
    num = 1000
    numstr = "S"+str(num)
    fig, axs = plt.subplots(2, 3)
    triangulateAndPlot(A,"", axs[0, 0],  "Input")
    triangulateAndPlot(A,"pq0.000001U90"+numstr, axs[0, 1], "OBT 1")
    triangulateAndPlot(A,"pq15U170"+numstr, axs[0, 2],  "CCDT 15")
    triangulateAndPlot(A,"p", axs[1, 0], "D")
    triangulateAndPlot(A,"pq30U90"+numstr, axs[1, 1], "OBT 30")
    triangulateAndPlot(A,"pq30U170"+numstr, axs[1, 2], "CCDT 30")
    plt.show()

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
    #summaryPlot(thinTriange(10))
    #get instance
    instanceUIDs = getInstanceUIDs(1000)
    for i in range(12,len(instanceUIDs)):
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
        summaryPlot(cdata)


if __name__=="__main__":
    main()



