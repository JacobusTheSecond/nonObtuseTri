from exact_geometry import circumcenter, badVertex, innerIntersect, isBadTriangle, altitudePoint
import numpy as np
import matplotlib.pyplot as plt
from cgshop2025_pyutils import Cgshop2025Instance,Cgshop2025Solution, VerificationResult
from cgshop2025_pyutils.geometry import FieldNumber, Point, Segment
import matplotlib

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
                cc = circumcenter(Aexact['vertices'][tri[0]],Aexact['vertices'][tri[1]],Aexact['vertices'][tri[2]])
                cr = np.sqrt(float(Segment(badVertex(Aexact['vertices'][tri[0]],Aexact['vertices'][tri[1]],Aexact['vertices'][tri[2]]),cc).squared_length()))
                marker = (Aexact['vertices'][tri[0]],Aexact['vertices'][tri[1]],Aexact['vertices'][tri[2]])
            #print(*B['vertices'][tri])
            if i != mark and isBadTriangle(Aexact['vertices'][tri[0]],Aexact['vertices'][tri[1]],Aexact['vertices'][tri[2]]):
                badCount += 1
                t = plt.Polygon(B['vertices'][tri], color='b')
                ax.add_patch(t)
        name += " (>90°: " + str(badCount) + ")"
    for e in Ain['segments']:
        ax.plot(*(B['vertices'][e].T), color='red', linewidth=2)
    ax.scatter(*(B['vertices'][:len(Ain['vertices'])].T), marker='.', color='black', zorder=100)
    ax.scatter(*(B['vertices'][len(Ain['vertices']):].T), marker='.', color='green', zorder=100)

    if cc!=None:
        #ax.scatter([float(cc.x())],[float(cc.y())], marker='.', color='yellow', zorder=1000)
        #circle = plt.Circle((float(cc.x()),float(cc.y())),cr,color="yellow",fill=False,zorder=1000)
        #ax.add_patch(circle)
        a,b,c = marker
        # p is point of offending angle
        p = badVertex(a, b, c)

        # check with every constraint edge if the line from p to the circumcenter intersects it
        # store the closest intersecting segment in closestSegment
        closestSegment = None
        closestIntersection = None
        closestSegmentIndex = None
        baseline = Segment(p, cc)
        for segmentIndex in range(len(B['segments'])):
            segment = B['segments'][segmentIndex]
            intersection = innerIntersect(p, cc, Aexact['vertices'][segment[0]], Aexact['vertices'][segment[1]])
            if intersection != None:
                if closestIntersection == None or np.sqrt(float(Segment(p,intersection).squared_length())) < np.sqrt(float(Segment(p,closestIntersection).squared_length())):
                    closestSegment = (Aexact['vertices'][segment[0]], Aexact['vertices'][segment[1]])
                    closestIntersection = intersection
                    closestSegmentIndex = segmentIndex

        if closestIntersection != None:
            # we can insert circumcenter and retriangulate
            # need to split the edge
            ap = altitudePoint(Segment(closestSegment[0], closestSegment[1]),p)
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
                cc = circumcenter(Bvs[tri[0]],Bvs[tri[1]],Bvs[tri[2]])
                cr = np.sqrt(float(Segment(badVertex(Bvs[tri[0]],Bvs[tri[1]],Bvs[tri[2]]),cc).squared_length()))
                marker = (Bvs[tri[0]],Bvs[tri[1]],Bvs[tri[2]])
            #print(*B['vertices'][tri])
            if i != mark and isBadTriangle(Bvs[tri[0]],Bvs[tri[1]],Bvs[tri[2]]):
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
        p = badVertex(a, b, c)

        # check with every constraint edge if the line from p to the circumcenter intersects it
        # store the closest intersecting segment in closestSegment
        closestSegment = None
        closestIntersection = None
        closestSegmentIndex = None
        baseline = Segment(p, cc)
        for segmentIndex in range(len(B['segments'])):
            segment = B['segments'][segmentIndex]
            intersection = innerIntersect(p, cc, Bvs[segment[0]], Bvs[segment[1]])
            if intersection != None:
                if closestIntersection == None or np.linalg.norm(p - intersection) < np.linalg.norm(
                        p - closestIntersection):
                    closestSegment = (Bvs[segment[0]], Bvs[segment[1]])
                    closestIntersection = intersection
                    closestSegmentIndex = segmentIndex

        if closestIntersection != None:
            # we can insert circumcenter and retriangulate
            # need to split the edge
            ap = altitudePoint(Segment(closestSegment[0], closestSegment[1]),p)
            ax.scatter([float(closestIntersection.x())],[closestIntersection.y()],color="red",zorder=10000)
            ax.scatter([float(ap.x())],[ap.y()],color="yellow",zorder=100000)


    ax.set_aspect('equal')
    ax.title.set_text(name)

def plot_solution(ax:matplotlib.axes.Axes,instance : Cgshop2025Instance, solution : Cgshop2025Solution,result : VerificationResult):
    ax.scatter(instance.points_x,instance.points_y,color="black",marker='.',zorder=100)
    steinerx = [float(FieldNumber(v)) for v in solution.steiner_points_x]
    steinery = [float(FieldNumber(v)) for v in solution.steiner_points_y]

    totalx = instance.points_x + steinerx
    totaly = instance.points_y + steinery

    ax.scatter(steinerx,steinery,color="green",marker='.',zorder=100)
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
        ax.plot([x1, x2], [y1, y2], color="blue", linestyle="-",linewidth=3)
        # Plot constraints
    for constraint in instance.additional_constraints:
        x1, y1 = instance.points_x[constraint[0]], instance.points_y[constraint[0]]
        x2, y2 = instance.points_x[constraint[1]], instance.points_y[constraint[1]]
        ax.plot([x1, x2], [y1, y2], color="red", linestyle="-",linewidth=3)


    for edge in solution.edges:
        x1, y1 = totalx[edge[0]], totaly[edge[0]]
        x2, y2 = totalx[edge[1]], totaly[edge[1]]
        ax.plot([x1, x2], [y1, y2], color="black", linestyle="-",zorder=-1)
    ax.set_aspect("equal")
    ax.set_title(instance.instance_uid+" #Steiner: "+str(len(steinery))+" #non-obtuse: "+str(result.num_obtuse_triangles))
