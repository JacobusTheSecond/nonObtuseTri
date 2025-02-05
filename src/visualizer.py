import logging

import matplotlib.pyplot as plt
import matplotlib
from sqlalchemy.testing import against

from src.solutionManagement import updateSummaries

matplotlib.use("TkAgg")
import numpy as np
from pathlib import Path
from cgshop2025_pyutils import InstanceDatabase, ZipSolutionIterator, verify, ZipWriter

from hacky_internal_visualization_stuff import plot_solution, plot_instance
from solutionManagement import loadSolutions, triangulationFromSolution
from exact_geometry import dot
from constants import noneEdge


def verifyAll(solname="solutions.zip"):

    filepath = Path(__file__)
    zips = filepath.parent.parent/"challenge_instances_cgshop25" / "zips"

    idb = InstanceDatabase(zips/"challenge_instances_cgshop25_rev1.zip")
    i=0

    for solution in ZipSolutionIterator(zips/solname):
        i+=1
        instance = idb[solution.instance_uid]
        result = verify(instance,solution)
        print(f"{i} {solution.instance_uid}: {result}")
        assert not result.errors, "Expect no errors."
        if result.num_obtuse_triangles > 0:
            fig, axs = plt.subplots(1, 1)
            plot_solution(axs, instance, solution)
            plt.show()

def showInstance():
    pass

def showSolutions(solname="solutions.zip"):
    filepath = Path(__file__)
    zips = filepath.parent.parent/"challenge_instances_cgshop25" / "zips"

    idb = InstanceDatabase(zips/"challenge_instances_cgshop25_rev1.zip")

    for solution in ZipSolutionIterator(zips/solname):
        fig, axs = plt.subplots(1, 1)
        instance = idb[solution.instance_uid]
        result = verify(instance,solution)
        print(f"{solution.instance_uid}: {result}")
        plot_solution(axs,instance,solution, result)
        plt.show()

plot_counter = 0

def updatePlot(ax1,ax2,ax3,diff,diffHeat,zippedList,idb,name,baseName):
    ax1.clear()
    ax2.clear()
    ax3.clear()
    global plot_counter
    extremum = max(abs(np.max(diffHeat)),abs(np.min(diffHeat)))
    gain = 0.85
    ax1.imshow(diffHeat, cmap='PiYG', interpolation='nearest', norm=matplotlib.colors.SymLogNorm(linthresh=0.5,linscale=1,vmin=-gain*extremum,vmax=gain*extremum,base=2))
    for i in range(len(diff)):
        for j in range(len(diff[i])):
            if i * 30 + j == plot_counter:
                ax1.text(j, i, diff[i, j], ha="center", va="center", color="blue", fontweight='bold')
            else:
                if diff[i, j] != "0" and diff[i, j] != 0:
                    ax1.text(j, i, diff[i, j], ha="center", va="center", color="black")

    sol1 = zippedList[plot_counter][0]
    sol2 = zippedList[plot_counter][1]
    instance = idb[sol1.instance_uid]
    print(f"Instance with {len(instance.points_x)} points and {len(instance.region_boundary) + len(instance.additional_constraints)} constrained edges.")

    #sol1.plotTriangulation()
    #sol2.plotTriangulation()

    #ax1.set_title(instance.instance_uid)
    #return

    result1 = verify(instance, sol1)
    result2 = verify(instance, sol2)
    # print(f"{solution.instance_uid}: {result}")
    plot_solution(ax2, instance, sol1, result1,prefix=baseName[plot_counter])
    plot_solution(ax3, instance, sol2, result2,prefix=name[plot_counter])
    ax1.set_title(instance.instance_uid)

def plotByType(solutions):
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S", level=logging.INFO)

    filepath = Path(__file__)
    idb = InstanceDatabase(filepath.parent.parent/"challenge_instances_cgshop25"/"zips"/"challenge_instances_cgshop25_rev1.zip")

    bests = []
    for name in solutions:
        best = loadSolutions(name)
        i = 0
        for sol,solname in best:
            if len(bests) == i:
                bests.append([sol,solname])
            elif len(sol.steiner_points_x) < len(bests[i][0].steiner_points_x):
                assert sol.instance_uid == bests[i][0].instance_uid
                bests[i] = [sol,solname]
                #logging.info(str(solname)+" is better at "+str(sol.instance_uid) +" with solution size "+str(len(sol.steiner_points_x)))
            i += 1

    triangulations = []
    anglelist = []
    constraintList = []
    for best,_ in bests:
        triangulations.append(triangulationFromSolution(idb[best.instance_uid],best,[None,None,None,None,None]))

    def primitiveAngle(a,b,c):
        dotprod = float(dot(a-b,c-b))
        anorm = np.sqrt(float(dot(a-b,a-b)))
        cnorm = np.sqrt(float(dot(c-b,c-b)))
        return 360 * np.arccos(dotprod / anorm / cnorm) / 2 / np.pi
        # a · b = | a | | b | cos θ

    for trig in triangulations:
        myAngleTrios = []
        myOnConstraint = []
        triIdx = 0
        for tri in trig.triangles:



            a = trig.point(tri[0])
            b = trig.point(tri[1])
            c = trig.point(tri[2])
            myAngleTrios.append([primitiveAngle(a,b,c),primitiveAngle(b,c,a),primitiveAngle(c,a,b)])
            myOnConstraint.append([trig.triangleMap[triIdx][0][2] != noneEdge or trig.triangleMap[triIdx][2][2] != noneEdge,trig.triangleMap[triIdx][0][2] != noneEdge or trig.triangleMap[triIdx][1][2] != noneEdge,trig.triangleMap[triIdx][1][2] != noneEdge or trig.triangleMap[triIdx][2][2] != noneEdge])


            triIdx += 1

        anglelist.append(myAngleTrios)
        constraintList.append(myOnConstraint)

    ids = list(range(10))

    fig,ax = plt.subplots()

    showCalled = False
    for id in ids:
        sol,_ = bests[id]
        angles = anglelist[id]
        constraints = constraintList[id]


        ax.clear()
        fig.canvas.manager.set_window_title(f"{sol.instance_uid}_histogram")

        constraintMaxAngle = []
        unconstraitMaxAngle = []
        for angleTrio,constraintTrio in zip(angles,constraints):
            argmax = np.argsort(angleTrio)[-1]
            if constraintTrio[argmax]:
                constraintMaxAngle.append(angleTrio[argmax])
            else:
                unconstraitMaxAngle.append(angleTrio[argmax])

        ax.hist([constraintMaxAngle,unconstraitMaxAngle], 90, range=(0, 90), stacked=True,color=["darkblue","blue"],edgecolor='k',rwidth=0.8)

        #ax.scatter(x=[item[0] for item in buckets[k]],y=[item[1] for item in buckets[k]])
        if not showCalled:
            plt.show()
            showCalled = True

        plt.draw()
        fig.tight_layout()
        ax.clear()


        constraintMaxAngle = []
        unconstraitMaxAngle = []
        for angleTrio,constraintTrio in zip(angles,constraints):
            argmax = np.argsort(angleTrio)[-2]
            if constraintTrio[argmax]:
                constraintMaxAngle.append(angleTrio[argmax])
            else:
                unconstraitMaxAngle.append(angleTrio[argmax])

        ax.hist([constraintMaxAngle,unconstraitMaxAngle], 90, range=(0, 90), stacked=True,color=["darkgreen","green"],edgecolor='k',rwidth=0.8)


        plt.draw()
        fig.tight_layout()
        ax.clear()


        constraintMaxAngle = []
        unconstraitMaxAngle = []
        for angleTrio,constraintTrio in zip(angles,constraints):
            argmax = np.argsort(angleTrio)[-3]
            if constraintTrio[argmax]:
                constraintMaxAngle.append(angleTrio[argmax])
            else:
                unconstraitMaxAngle.append(angleTrio[argmax])

        ax.hist([constraintMaxAngle,unconstraitMaxAngle], 90, range=(0, 90), stacked=True,color=["darkred","red"],edgecolor='k',rwidth=0.8)


        plt.draw()

        #ax.hist([countsA,countsB], binsA,stacked=True)
        #ax.hist(binsA[:-1], binsA,weights=countsA+countsB,zorder=0)
        #bins[:-1], bins, weights = counts

        fig.tight_layout()

        plt.pause(0.01)







    buckets = dict()
    keys = ["simple-polygon-exterior","point-set","ortho","simple-polygon"]
    for k in keys:
        buckets[k] = []

    for sol,_ in bests:
        uid = sol.instance_uid
        x = len(idb[uid].points_x)
        y = len(sol.steiner_points_x)
        handled = False
        for k in keys:
            if k in uid:
                handled = True
                buckets[k].append((x,y))
            if handled:
                break
        if not handled:
            print("uh oh...")

    fig,ax = plt.subplots()
    plt.show()
    for k in keys:

        ax.clear()
        fig.canvas.manager.set_window_title(f"{k}_plot")
        xdict = dict()
        for x,y in buckets[k]:
            xdict[x] = xdict.get(x,[]) + [y]
        print(xdict)
        for x in xdict.keys():
            ax.boxplot([xdict[x]],positions=[x],widths=[8],showfliers=False)
            newxs = np.random.normal(x,0.5,size=len(xdict[x]))
            ax.plot(newxs,xdict[x],"r.",alpha=0.2)

        xs = [item[0] for item in buckets[k]]
        ys = [item[1] for item in buckets[k]]
        ax.plot(np.unique(xs),np.poly1d(np.polyfit(xs,ys,1))(np.unique(xs)),label=str(np.poly1d(np.polyfit(xs,ys,1))))
        ax.legend(loc="upper left")
        ax.set_ylim(-5,230)
        ax.set_aspect("equal")

        fig.tight_layout()

        #ax.scatter(x=[item[0] for item in buckets[k]],y=[item[1] for item in buckets[k]])
        plt.draw()
        plt.pause(0.01)



def compareSolutions(base,others):
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S", level=logging.INFO)

    filepath = Path(__file__)
    idb = InstanceDatabase(filepath.parent.parent/"challenge_instances_cgshop25"/"zips"/"challenge_instances_cgshop25_rev1.zip")

    bestBases = []
    for name in base:
        best = loadSolutions(name)
        i = 0
        for sol,solname in best:
            if len(bestBases) == i:
                bestBases.append([sol,solname])
            elif len(sol.steiner_points_x) < len(bestBases[i][0].steiner_points_x):
                assert sol.instance_uid == bestBases[i][0].instance_uid
                bestBases[i] = [sol,solname]
                #logging.info(str(solname)+" is better at "+str(sol.instance_uid) +" with solution size "+str(len(sol.steiner_points_x)))
            i += 1

    bestOthers = []
    for name in others:
        best = loadSolutions(name)
        i = 0
        for sol,solname in best:
            if len(bestOthers) == i:
                bestOthers.append([sol,solname])
            elif len(sol.steiner_points_x) < len(bestOthers[i][0].steiner_points_x):
                assert sol.instance_uid == bestOthers[i][0].instance_uid
                bestOthers[i] = [sol,solname]
                #logging.info(str(solname)+" is better at "+str(sol.instance_uid) +" with solution size "+str(len(sol.steiner_points_x)))
            i += 1

    global plot_counter
    plot_counter = 0
    zippedList = []
    diff = []
    baseName = []
    name = []
    fullList = []

    fig = plt.figure()
    gs = fig.add_gridspec(nrows=2,ncols=2,height_ratios=[1,2])
    ax1 = fig.add_subplot(gs[0,:])
    ax2 = fig.add_subplot(gs[1,0])
    ax3 = fig.add_subplot(gs[1,1])

    fig2,otherAx = plt.subplots()
    fig3,otherotherAx = plt.subplots()

    avgImprovePercent = 0
    for base,other in zip(bestBases,bestOthers):
        #logging.info("identifying best for " + str(base[0].instance_uid))
        a = base[0]
        b = other[0]
        #idx = np.argmin([len(o.steiner_points_x) for o in other])
        #other = b if len(b.steiner_points_x) < len(c.steiner_points_x) else c
        assert a.instance_uid == b.instance_uid
        #zippedList.append([a,b])
        #diff.append(len(a.steiner_points_x) - len(b.steiner_points_x))
        #baseName.append(bases[baseIdx])
        #name.append(others[idx])
        #atrig = triangulationFromSolution(idb[a.instance_uid],a,[ax2,None,None,None,None])
        #btrig = triangulationFromSolution(idb[b.instance_uid],b,[ax3,None,None,None,None])
        asize = len(a.steiner_points_x)
        bsize = len(b.steiner_points_x)
        diff = len(a.steiner_points_x) - len(b.steiner_points_x)
        diffText = None
        if asize == bsize:
            diffText = "0"
        elif asize < bsize:
            avgImprovePercent -= 100*((bsize/asize)-1)
            diffText = "-"+str(int(100*((bsize/asize)-1)))+"%"
        else:
            avgImprovePercent += 100*(1-(bsize/asize))
            diffText = str(int(100*(1 - (bsize/asize))))+"%"
        fullList.append([[a,b],len(a.steiner_points_x),(diff,diffText),str(other[1].parent.name)+"/"+str(other[1].name),str(base[1].parent.name)+"/"+str(base[1].name),idb[a.instance_uid].num_points])
    avgImprovePercent/=len(bestBases)
    logging.info(f"average improvement in percent: {avgImprovePercent:3.1f}%")
    #fullList = sorted(fullList,key = lambda entry : str(entry[0][0].instance_uid))
    fullList = sorted(fullList,key = lambda entry : entry[5])
    zippedList = [e[0] for e in fullList]
    base = [e[1] for e in fullList]
    diff = [e[2][0] for e in fullList]
    diffTexts = [e[2][1] for e in fullList]
    name = [e[3] for e in fullList]
    baseName = [e[4] for e  in fullList]


    minimum = min(diff)
    maximum = max(diff)
    extremum = max(abs(minimum),abs(maximum))

    diffheat = (((np.array(base) + np.array(diff))/np.array(base))-1)
    diffheat = np.reshape(diffheat,(5,30))

    diff = np.reshape(diff,(5,30))
    percentage = True
    if percentage:
        diffTexts = np.reshape(diffTexts,(5,30))
    else:
        diffTexts = diff

    limit = len(zippedList)

    def on_click(event):
        if event.inaxes==ax1:
            global plot_counter
            plot_counter = 30 * (min(max(0,int(event.ydata+0.5)),4)) + min(max(0,int(event.xdata+0.5)),29)
            plot_counter = min(max(0,plot_counter),149)
            updatePlot(ax1, ax2, ax3, diffTexts, diffheat, zippedList, idb, name, baseName)
            sol1 = zippedList[plot_counter][0]
            sol2 = zippedList[plot_counter][1]
            instance = idb[sol1.instance_uid]
            otherAx.clear()
            plot_instance(otherAx, instance)
            otherotherAx.clear()
            result2 = verify(instance, sol2)
            # print(f"{solution.instance_uid}: {result}")
            plot_solution(otherotherAx, instance, sol2, result2,withNames=False)

            otherAx.axis("off")
            fig2.tight_layout()
            otherotherAx.axis("off")
            fig3.tight_layout()
            fig2.canvas.manager.set_window_title(f"{sol1.instance_uid}_instance")
            fig3.canvas.manager.set_window_title(f"{sol1.instance_uid}_solution")

            plt.draw()

    def on_press(event):
        global plot_counter
        if event.key == '+':
            plot_counter = min(plot_counter+1,limit-1)
        if event.key == '-':
            plot_counter = max(plot_counter-1,0)
        if event.key == 'down':
            plot_counter = min(plot_counter+30,limit-1)
        if event.key == 'right':
            plot_counter = min(plot_counter+1,limit-1)
        if event.key == 'left':
            plot_counter = max(plot_counter-1,0)
        if event.key == 'up':
            plot_counter = max(plot_counter-30,0)

        updatePlot(ax1, ax2, ax3, diffTexts, diffheat, zippedList, idb, name, baseName)
        sol1 = zippedList[plot_counter][0]
        sol2 = zippedList[plot_counter][1]
        instance = idb[sol1.instance_uid]
        otherAx.clear()
        plot_instance(otherAx, instance)
        otherotherAx.clear()
        result2 = verify(instance, sol2)
        # print(f"{solution.instance_uid}: {result}")
        plot_solution(otherotherAx, instance, sol2, result2,withNames=False)
        otherAx.axis("off")
        fig2.tight_layout()
        otherotherAx.axis("off")
        fig3.tight_layout()

        fig2.canvas.manager.set_window_title(f"{sol1.instance_uid}_instance")
        fig3.canvas.manager.set_window_title(f"{sol1.instance_uid}_solution")

        plt.draw()



    fig.canvas.mpl_connect('key_press_event',on_press)
    fig.canvas.mpl_connect('button_press_event', on_click)

    updatePlot(ax1,ax2,ax3,diffTexts,diffheat,zippedList,idb,name,baseName)
    sol1 = zippedList[plot_counter][0]
    sol2 = zippedList[plot_counter][1]
    instance = idb[sol1.instance_uid]
    otherAx.clear()
    plot_instance(otherAx, instance)
    otherotherAx.clear()
    result2 = verify(instance, sol2)
    # print(f"{solution.instance_uid}: {result}")
    plot_solution(otherotherAx, instance, sol2, result2,withNames=False)
    otherAx.axis("off")
    fig2.tight_layout()
    otherotherAx.axis("off")
    fig3.tight_layout()

    fig2.canvas.manager.set_window_title(f"{sol1.instance_uid}_instance")
    fig3.canvas.manager.set_window_title(f"{sol1.instance_uid}_solution")


    plt.show()

def plotHistory():
    filepath = Path(__file__)
    idb = InstanceDatabase(filepath.parent.parent/"challenge_instances_cgshop25"/"zips"/"challenge_instances_cgshop25_rev1.zip")
    fig,ax = plt.subplots()
    plt.show()
    history = filepath.parent.parent/"history"
    for i in range(len(list(history.iterdir()))):
        for solution in ZipSolutionIterator(history/f"{i}.zip"):
            tr = triangulationFromSolution(idb[solution.instance_uid], solution, [ax, None, None, None, None])
            tr.plotTriangulation()
            ax.axis("off")
            ax.title.set_text("")
            fig.canvas.manager.set_window_title(f"{i}")
            fig.tight_layout()
            plt.draw()
            plt.pause(0.1)


if __name__=="__main__":

    #showSolutions()
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S", level=logging.INFO)

    updateSummaries()

    plotHistory()

    filepath = Path(__file__)
    numeric_solutions = filepath.parent.parent/"instance_solutions" / "numeric_solutions"
    exact_solutions = filepath.parent.parent/"instance_solutions" / "exact_solutions"
    new = filepath.parent.parent/"instance_solutions" / "newestVersion"
    seeded = filepath.parent.parent/"instance_solutions" / "seededRuns"
    seededEndFace = filepath.parent.parent/"instance_solutions"/"seededWithFECleaning"
    seededFace = filepath.parent.parent/"instance_solutions"/"seededWithFaceExpansion"
    withFace = filepath.parent.parent/"instance_solutions"/"withFaceExpansion"
    out = filepath.parent.parent/"instance_solutions"/"out"
    gigaSeeded = filepath.parent.parent/"instance_solutions"/"gigaSeeded"
    output = filepath.parent.parent/"instance_solutions"/"output"
    withComplicatedCenter = filepath.parent.parent/"instance_solutions"/"withComplicatedCenter"
    withConstrainedVoronoi = filepath.parent.parent/"instance_solutions"/"withConstrainedVoronoi"
    new1 = filepath.parent.parent/"instance_solutions"/"constrainedVoronoiFromInside"
    new2 = filepath.parent.parent/"instance_solutions"/"constrainedVoronoiFromOutside"
    new3Old = filepath.parent.parent/"instance_solutions"/"288CircleArrNoOutRNoSegsOld"
    new3 = filepath.parent.parent/"instance_solutions"/"288CircleArrNoOutRNoSegs"
    new4 = filepath.parent.parent/"instance_solutions"/"288CircleArr2OutRNoSegs"
    new5 = filepath.parent.parent/"instance_solutions"/"withDepth1Greedy"
    merged = filepath.parent.parent/"instance_solutions"/"merged_summaries"
    merged_3 = filepath.parent.parent/"instance_solutions"/"merged_summaries_3"
    merged_5 = filepath.parent.parent/"instance_solutions"/"merged_summaries_5"
    merged_bak = filepath.parent.parent/"instance_solutions"/"merged_summaries_bak"
    mergemerge = filepath.parent.parent/"out_merged_summaries"

    allexceptnumeric = [exact_solutions,new,seeded,seededEndFace,seededFace,withFace,withComplicatedCenter,output,gigaSeeded,withConstrainedVoronoi]
        #allexceptnumeric = allexceptnumeric + [v for v in list.iterdir()]

    #compareSolutions(base=[v for v in seeded.iterdir() if len([w for w in out.iterdir() if v.name == w.name])>0],others=[v for v in out.iterdir()])
    againstMergeBak = False
    #plotByType([merged,merged_3,merged_5,merged_bak,mergemerge])
    if againstMergeBak:
        compareSolutions(others=[mergemerge],base=[merged,merged_3,merged_5])#
    else:
        compareSolutions(others=[merged,merged_3,merged_5,merged_bak,mergemerge],base=[new1,new2,new3Old,new3,new4,new5])  #
        #compareSolutions(others=[new5, new4, new3, new3Old],base= allexceptnumeric)  #

    #compareSolutions(base=[v for v in seeded.iterdir()],others=[v for v in out.iterdir()])
