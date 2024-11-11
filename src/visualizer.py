import logging
import os
import pickle

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import numpy as np
from pathlib import Path
from cgshop2025_pyutils import InstanceDatabase, ZipSolutionIterator, verify, ZipWriter

from hacky_internal_visualization_stuff import plot_solution


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

    ax1.imshow(diffHeat, cmap='PiYG', interpolation='nearest', vmin=-extremum, vmax=extremum)
    for i in range(len(diff)):
        for j in range(len(diff[i])):
            if i * 30 + j == plot_counter:
                ax1.text(j, i, diff[i, j], ha="center", va="center", color="blue", fontweight='bold')
            else:
                if diff[i, j] != 0:
                    ax1.text(j, i, diff[i, j], ha="center", va="center", color="black")

    sol1 = zippedList[plot_counter][0]
    sol2 = zippedList[plot_counter][1]

    instance = idb[sol1.instance_uid]
    result1 = verify(instance, sol1)
    result2 = verify(instance, sol2)
    # print(f"{solution.instance_uid}: {result}")
    plot_solution(ax2, instance, sol1, result1,prefix=baseName[plot_counter])
    plot_solution(ax3, instance, sol2, result2,prefix=name[plot_counter])
    ax1.set_title(instance.instance_uid)

def loadSolutions(foldername):

    best = []
    #first unpickle
    fn = foldername.name
    summaryName = foldername.parent.parent/"solution_summaries"/(str(fn)+".zip")
    pickelName = foldername.parent.parent/"solution_summaries"/(str(fn)+".pkl")
    if summaryName.exists():
        i = 0
        names = pickle.load(open(pickelName, "rb"))

        for sol in ZipSolutionIterator(summaryName):
            if len(best) == i:
                best.append([sol,names[i]])
            elif len(sol.steiner_points_x) < len(best[i][0].steiner_points_x):
                assert sol.instance_uid == best[i][0].instance_uid
                best[i] = [sol,names[i]]
                logging.info(str(names[i])+" is better at "+str(sol.instance_uid) +" with solution size "+str(len(sol.steiner_points_x)))
            i += 1

    #first build the list
    if foldername.exists():
        for solname in foldername.iterdir():
            i = 0
            logging.info("reading "+str(solname))
            for sol in ZipSolutionIterator(solname):
                if len(best) == i:
                    best.append([sol,solname])
                elif len(sol.steiner_points_x) < len(best[i][0].steiner_points_x):
                    assert sol.instance_uid == best[i][0].instance_uid
                    best[i] = [sol,solname]
                    logging.info(str(solname)+" is better at "+str(sol.instance_uid) +" with solution size "+str(len(sol.steiner_points_x)))
                i += 1

    if len(best) == 0:
        logging.error("No matching data found for folder "+str(foldername))
        return []

    #now rebuild summary
    if summaryName.exists():
        summaryName.unlink()

    if pickelName.exists():
        pickelName.unlink()
    #Write the solutions to a new zip file
    toWriteNames = []
    with ZipWriter(summaryName) as zw:
        logging.info("writting sol summary at "+str(summaryName))
        for solution,name in best:
            zw.add_solution(solution)
            toWriteNames.append(name)
    with open(pickelName, "wb") as f:
        logging.info("writting name summary at "+str(pickelName))
        pickle.dump(toWriteNames, f)

    return best


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
                logging.info(str(solname)+" is better at "+str(sol.instance_uid) +" with solution size "+str(len(sol.steiner_points_x)))
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
                logging.info(str(solname)+" is better at "+str(sol.instance_uid) +" with solution size "+str(len(sol.steiner_points_x)))
            i += 1

    global plot_counter
    plot_counter = 0
    zippedList = []
    diff = []
    baseName = []
    name = []
    fullList = []

    for base,other in zip(bestBases,bestOthers):
        #logging.info("identifying best for " + str(base[0].instance_uid))
        a = base[0]
        b = other[0]
        #idx = np.argmin([len(o.steiner_points_x) for o in other])
        #other = b if len(b.steiner_points_x) < len(c.steiner_points_x) else c
        assert a.instance_uid == b.instance_uid
        zippedList.append([a,b])
        diff.append(len(a.steiner_points_x) - len(b.steiner_points_x))
        #baseName.append(bases[baseIdx])
        #name.append(others[idx])
        fullList.append([[a,b],len(a.steiner_points_x),len(a.steiner_points_x) - len(b.steiner_points_x),str(other[1].parent.name)+"/"+str(other[1].name),str(base[1].parent.name)+"/"+str(base[1].name),idb[a.instance_uid].num_points])

    #fullList = sorted(fullList,key = lambda entry : str(entry[0][0].instance_uid))
    #fullList = sorted(fullList,key = lambda entry : entry[5])
    zippedList = [e[0] for e in fullList]
    base = [e[1] for e in fullList]
    diff = [e[2] for e in fullList]
    name = [e[3] for e in fullList]
    baseName = [e[4] for e  in fullList]


    minimum = min(diff)
    maximum = max(diff)
    extremum = max(abs(minimum),abs(maximum))

    diffheat = (((np.array(base) + np.array(diff))/np.array(base))-1)
    diffheat = np.reshape(diffheat,(5,30))

    diff = np.reshape(diff,(5,30))

    limit = len(zippedList)

    fig = plt.figure()
    gs = fig.add_gridspec(nrows=2,ncols=2,height_ratios=[1,2])
    ax1 = fig.add_subplot(gs[0,:])
    ax2 = fig.add_subplot(gs[1,0])
    ax3 = fig.add_subplot(gs[1,1])

    def on_click(event):
        if event.inaxes==ax1:
            global plot_counter
            plot_counter = 30 * (min(max(0,int(event.ydata+0.5)),4)) + min(max(0,int(event.xdata+0.5)),29)
            plot_counter = min(max(0,plot_counter),149)
            updatePlot(ax1, ax2, ax3, diff, diffheat, zippedList, idb, name, baseName)
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

        updatePlot(ax1, ax2, ax3, diff, diffheat, zippedList, idb, name, baseName)
        plt.draw()



    fig.canvas.mpl_connect('key_press_event',on_press)
    fig.canvas.mpl_connect('button_press_event', on_click)

    updatePlot(ax1,ax2,ax3,diff,diffheat,zippedList,idb,name,baseName)

    plt.show()

if __name__=="__main__":
    #showSolutions()

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

    allexceptnumeric = [exact_solutions,new,seeded,seededEndFace,seededFace,withFace,withComplicatedCenter,output,gigaSeeded,withConstrainedVoronoi]
        #allexceptnumeric = allexceptnumeric + [v for v in list.iterdir()]

    #compareSolutions(base=[v for v in seeded.iterdir() if len([w for w in out.iterdir() if v.name == w.name])>0],others=[v for v in out.iterdir()])
    compareSolutions(others=[new1,new2],base=allexceptnumeric)
    #compareSolutions(base=[v for v in seeded.iterdir()],others=[v for v in out.iterdir()])
