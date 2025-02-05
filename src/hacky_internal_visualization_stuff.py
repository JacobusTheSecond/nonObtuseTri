#from exact_geometry import circumcenter, badVertex, innerIntersect, isBadTriangle, altitudePoint
import math

import numpy as np
import matplotlib.pyplot as plt
from cgshop2025_pyutils import Cgshop2025Instance,Cgshop2025Solution, VerificationResult
from cgshop2025_pyutils.geometry import FieldNumber, Point, Segment
import matplotlib
import exact_geometry as eg

def plotExact(Ain,Aexact,B,name,ax,mark=-1):
    SC = len(B['vertices']) - len(Ain['vertices'])
    if SC > 0:
        name += " [SC:" + str(SC) + "]"
    cc = None
    cr = None
    marker = None

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
                cc = eg.circumcenter(Aexact['vertices'][tri[0]],Aexact['vertices'][tri[1]],Aexact['vertices'][tri[2]])
                cr = np.sqrt(float(Segment(eg.badVertex(Aexact['vertices'][tri[0]],Aexact['vertices'][tri[1]],Aexact['vertices'][tri[2]]),cc).squared_length()))
                marker = (Aexact['vertices'][tri[0]],Aexact['vertices'][tri[1]],Aexact['vertices'][tri[2]])
            #print(*B['vertices'][tri])
            if i != mark and eg.isBadTriangle(Aexact['vertices'][tri[0]],Aexact['vertices'][tri[1]],Aexact['vertices'][tri[2]]):
                badCount += 1
                t = plt.Polygon(B['vertices'][tri], color='b')
                ax.add_patch(t)
        name += " (>90°: " + str(badCount) + ")"
    for e in Aexact['segments']:
        ax.plot(*(B['vertices'][e].T), color='red', linewidth=2)
    ax.scatter(*(B['vertices'][:len(Ain['vertices'])].T), marker='.', color='black', zorder=100)
    ax.scatter(*(B['vertices'][len(Ain['vertices']):].T), marker='.', color='green', zorder=100)

    if cc!=None:
        #ax.scatter([float(cc.x())],[float(cc.y())], marker='.', color='yellow', zorder=1000)
        #circle = plt.Circle((float(cc.x()),float(cc.y())),cr,color="yellow",fill=False,zorder=1000)
        #ax.add_patch(circle)
        a,b,c = marker
        # p is point of offending angle
        p = eg.badVertex(a, b, c)

        # check with every constraint edge if the line from p to the circumcenter intersects it
        # store the closest intersecting segment in closestSegment
        closestSegment = None
        closestIntersection = None
        closestSegmentIndex = None
        baseline = Segment(p, cc)
        for segmentIndex in range(len(B['segments'])):
            segment = B['segments'][segmentIndex]
            intersection = eg.innerIntersect(p, cc, Aexact['vertices'][segment[0]], Aexact['vertices'][segment[1]])
            if intersection != None:
                if closestIntersection == None or np.sqrt(float(Segment(p,intersection).squared_length())) < np.sqrt(float(Segment(p,closestIntersection).squared_length())):
                    closestSegment = (Aexact['vertices'][segment[0]], Aexact['vertices'][segment[1]])
                    closestIntersection = intersection
                    closestSegmentIndex = segmentIndex

        if closestIntersection != None:
            # we can insert circumcenter and retriangulate
            # need to split the edge
            ap = eg.altitudePoint(Segment(closestSegment[0], closestSegment[1]),p)
            ax.scatter([float(closestIntersection.x())],[closestIntersection.y()],color="red",zorder=10000)
            ax.scatter([float(ap.x())],[ap.y()],color="yellow",zorder=100000)


    ax.set_aspect('equal')
    ax.title.set_text(name)

def plot(A,B,name,ax,mark=-1):
    SC = len(B['vertices']) - len(A['vertices'])
    if SC > 0:
        name += " [SC:" + str(SC) + "]"
    cc = None
    cr = None
    marker = None
    Bvs = []
    for v in B['vertices']:
        Bvs.append(Point(v[0],v[1]))

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
                cc = eg.circumcenter(Bvs[tri[0]],Bvs[tri[1]],Bvs[tri[2]])
                cr = np.sqrt(float(Segment(eg.badVertex(Bvs[tri[0]],Bvs[tri[1]],Bvs[tri[2]]),cc).squared_length()))
                marker = (Bvs[tri[0]],Bvs[tri[1]],Bvs[tri[2]])
            #print(*B['vertices'][tri])
            if i != mark and eg.isBadTriangle(Bvs[tri[0]],Bvs[tri[1]],Bvs[tri[2]]):
                badCount += 1
                t = plt.Polygon(B['vertices'][tri], color='b')
                ax.add_patch(t)
        name += " (>90°: " + str(badCount) + ")"
    for e in A['segments']:
        ax.plot(*(B['vertices'][e].T), color='red', linewidth=2)
    ax.scatter(*(B['vertices'][:len(A['vertices'])].T), marker='.', color='black', zorder=100)
    ax.scatter(*(B['vertices'][len(A['vertices']):].T), marker='.', color='green', zorder=100)

    if cc!=None:
        ax.scatter([float(cc.x())],[float(cc.y())], marker='.', color='yellow', zorder=1000)
        circle = plt.Circle((float(cc.x()),float(cc.y())),cr,color="yellow",fill=False,zorder=1000)
        ax.add_patch(circle)
        a,b,c = marker
        # p is point of offending angle
        p = eg.badVertex(a, b, c)

        # check with every constraint edge if the line from p to the circumcenter intersects it
        # store the closest intersecting segment in closestSegment
        closestSegment = None
        closestIntersection = None
        closestSegmentIndex = None
        baseline = Segment(p, cc)
        for segmentIndex in range(len(B['segments'])):
            segment = B['segments'][segmentIndex]
            intersection = eg.innerIntersect(p, cc, Bvs[segment[0]], Bvs[segment[1]])
            if intersection != None:
                if closestIntersection == None or np.linalg.norm(p - intersection) < np.linalg.norm(
                        p - closestIntersection):
                    closestSegment = (Bvs[segment[0]], Bvs[segment[1]])
                    closestIntersection = intersection
                    closestSegmentIndex = segmentIndex

        if closestIntersection != None:
            # we can insert circumcenter and retriangulate
            # need to split the edge
            ap = eg.altitudePoint(Segment(closestSegment[0], closestSegment[1]),p)
            ax.scatter([float(closestIntersection.x())],[closestIntersection.y()],color="red",zorder=10000)
            ax.scatter([float(ap.x())],[ap.y()],color="yellow",zorder=100000)


    ax.set_aspect('equal')
    ax.title.set_text(name)

def get_angle_plot(line1, line2, offset = 1, color = None, origin = [0,0], len_x_axis = 1, len_y_axis = 1):

    l1xy = line1.get_xydata()

    # Angle between line1 and x-axis
    slope1 = (l1xy[1][1] - l1xy[0][1]) / float(l1xy[1][0] - l1xy[0][0])
    angle1 = abs(math.degrees(math.atan(slope1))) # Taking only the positive angle

    l2xy = line2.get_xydata()

    # Angle between line2 and x-axis
    slope2 = (l2xy[1][1] - l2xy[0][1]) / float(l2xy[1][0] - l2xy[0][0])
    angle2 = abs(math.degrees(math.atan(slope2)))

    theta1 = min(angle1, angle2)
    theta2 = max(angle1, angle2)

    angle = theta2 - theta1

    if color is None:
        color = line1.get_color() # Uses the color of line 1 if color parameter is not passed.

    return matplotlib.patches.Arc(origin, len_x_axis*offset, len_y_axis*offset, angle=0, theta1=theta1, theta2=theta2, color=color)

def plot_instance(ax:matplotlib.axes.Axes, instance : Cgshop2025Instance):
    ax.scatter(instance.points_x,instance.points_y,color="black",s=10,zorder=100)

    for i in range(len(instance.region_boundary)):
        x1, y1 = (
            instance.points_x[instance.region_boundary[i]],
            instance.points_y[instance.region_boundary[i]],
        )
        x2, y2 = (
            instance.points_x[
                instance.region_boundary[(i + 1) % len(instance.region_boundary)]
            ],
            instance.points_y[
                instance.region_boundary[(i + 1) % len(instance.region_boundary)]
            ],
        )
        ax.plot([x1, x2], [y1, y2], color="indigo",linewidth=3)
        # Plot constraints
    for constraint in instance.additional_constraints:
        x1, y1 = instance.points_x[constraint[0]], instance.points_y[constraint[0]]
        x2, y2 = instance.points_x[constraint[1]], instance.points_y[constraint[1]]
        ax.plot([x1, x2], [y1, y2], color="mediumspringgreen",linewidth=3)

    ax.set_aspect("equal")

def plot_solution(ax:matplotlib.axes.Axes,instance : Cgshop2025Instance, solution : Cgshop2025Solution,result : VerificationResult,prefix="",withNames=True):
    ax.scatter(instance.points_x,instance.points_y,color="black",s=10,zorder=100)

    exactPoints = [Point(instance.points_x[i],instance.points_y[i]) for i in range(len(instance.points_x))] + [Point(FieldNumber(solution.steiner_points_x[i]),FieldNumber(solution.steiner_points_y[i])) for i in range(len(solution.steiner_points_x))]

    steinerx = [float(FieldNumber(v)) for v in solution.steiner_points_x]
    steinery = [float(FieldNumber(v)) for v in solution.steiner_points_y]

    totalx = instance.points_x + steinerx
    totaly = instance.points_y + steinery

    ax.scatter(steinerx,steinery,color="red",s=25,zorder=100)
    for i in range(len(instance.region_boundary)):
        x1, y1 = (
            instance.points_x[instance.region_boundary[i]],
            instance.points_y[instance.region_boundary[i]],
        )
        x2, y2 = (
            instance.points_x[
                instance.region_boundary[(i + 1) % len(instance.region_boundary)]
            ],
            instance.points_y[
                instance.region_boundary[(i + 1) % len(instance.region_boundary)]
            ],
        )
        ax.plot([x1, x2], [y1, y2], color="indigo",linewidth=3)
        # Plot constraints
    for constraint in instance.additional_constraints:
        x1, y1 = instance.points_x[constraint[0]], instance.points_y[constraint[0]]
        x2, y2 = instance.points_x[constraint[1]], instance.points_y[constraint[1]]
        ax.plot([x1, x2], [y1, y2], color="mediumspringgreen",linewidth=3)

    for edge in solution.edges:
        x1, y1 = totalx[edge[0]], totaly[edge[0]]
        x2, y2 = totalx[edge[1]], totaly[edge[1]]
        ax.plot([x1, x2], [y1, y2], color="black",zorder=-1)

    ax.set_aspect("equal")
    if withNames:
        ax.set_title(prefix+" #Steiner:"+str(len(steinery))+" #non-obtuse:"+str(result.num_obtuse_triangles))
