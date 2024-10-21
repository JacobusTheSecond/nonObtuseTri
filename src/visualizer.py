import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from cgshop2025_pyutils import InstanceDatabase,ZipSolutionIterator,verify

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

def updatePlot(ax1,ax2,ax3,diff,extremum,zippedList,idb,name,baseName):
    ax1.clear()
    ax2.clear()
    ax3.clear()
    global plot_counter
    ax1.imshow(diff, cmap='PiYG', interpolation='nearest', vmin=-extremum, vmax=extremum)
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

def compareSolutions(base,others):

    filepath = Path(__file__)
    idb = InstanceDatabase(filepath.parent.parent/"challenge_instances_cgshop25"/"zips"/"challenge_instances_cgshop25_rev1.zip")

    #sols1 = ZipSolutionIterator(zips/base)
    basesols = []
    for name in base:
        basesols.append([sol for sol in ZipSolutionIterator(name)])

    transformedOtherSols = [[] for sol in basesols[0]]
    #transform
    for i in range(len(basesols)):
        for j in range(len(basesols[0])):
            transformedOtherSols[j].append(basesols[i][j])
    basesols = transformedOtherSols


    othersols = []
    for name in others:
        othersols.append([sol for sol in ZipSolutionIterator(name)])

    transformedOtherSols = [[] for sol in othersols[0]]
    #transform
    for i in range(len(othersols)):
        for j in range(len(othersols[0])):
            transformedOtherSols[j].append(othersols[i][j])
    othersols = transformedOtherSols

    global plot_counter
    plot_counter = 0
    zippedList = []
    diff = []
    baseName = []
    name = []
    fullList = []

    for bases,other in zip(basesols,othersols):
        baseIdx = np.argmin([len(a.steiner_points_x) for a in bases])
        a = bases[baseIdx]
        idx = np.argmin([len(o.steiner_points_x) for o in other])
        #other = b if len(b.steiner_points_x) < len(c.steiner_points_x) else c
        assert a.instance_uid == other[idx].instance_uid
        zippedList.append([a,other[idx]])
        diff.append(len(a.steiner_points_x) - len(other[idx].steiner_points_x))
        #baseName.append(bases[baseIdx])
        #name.append(others[idx])
        fullList.append([[a,other[idx]],len(a.steiner_points_x) - len(other[idx].steiner_points_x),others[idx].name,base[baseIdx].name])

    fullList = sorted(fullList,key = lambda entry : str(entry[0][0].instance_uid))
    zippedList = [e[0] for e in fullList]
    diff = [e[1] for e in fullList]
    name = [e[2] for e in fullList]
    baseName = [e[3] for e  in fullList]


    minimum = min(diff)
    maximum = max(diff)
    extremum = max(abs(minimum),abs(maximum))

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
            updatePlot(ax1, ax2, ax3, diff, extremum, zippedList, idb, name, baseName)
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

        updatePlot(ax1, ax2, ax3, diff, extremum, zippedList, idb, name, baseName)
        plt.draw()



    fig.canvas.mpl_connect('key_press_event',on_press)
    fig.canvas.mpl_connect('button_press_event', on_click)

    updatePlot(ax1,ax2,ax3,diff,extremum,zippedList,idb,name,baseName)

    plt.show()

if __name__=="__main__":
    #showSolutions()

    filepath = Path(__file__)
    numeric_solutions = filepath.parent.parent/"instance_solutions" / "numeric_solutions"
    exact_solutions = filepath.parent.parent/"instance_solutions" / "exact_solutions"
    new = filepath.parent.parent/"instance_solutions" / "newestVersion"

    extractionnames = ["properEdgeContractPrepend.zip","properEdgeContract.zip"]

    compareSolutions(base=[v for v in numeric_solutions.iterdir()]+[v for v in exact_solutions.iterdir() ],others=[v for v in new.iterdir()])