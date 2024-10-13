import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from cgshop2025_pyutils import InstanceDatabase,ZipSolutionIterator,verify

from hacky_internal_visualization_stuff import plot_solution #internal ugly functions I dont want you to see


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
def compareSolutions(solname1="solutions.zip",solname2="new_solutions.zip"):
    filepath = Path(__file__)
    zips = filepath.parent.parent/"challenge_instances_cgshop25" / "zips"

    idb = InstanceDatabase(zips/"challenge_instances_cgshop25_rev1.zip")

    sols1 = ZipSolutionIterator(zips/solname1)
    sols2 = ZipSolutionIterator(zips/solname2)

    global plot_counter
    plot_counter = 0
    zippedList = []
    diff = []

    for a,b in zip(sols1,sols2):
        zippedList.append([a,b])
        instance = idb[a.instance_uid]
        diff.append(len(a.steiner_points_x) - len(b.steiner_points_x))

    minimum = min(diff)
    maximum = max(diff)
    extremum = max(abs(minimum),abs(maximum))

    diff = np.reshape(diff,(5,30))

    limit = len(zippedList)

    fig = plt.figure()
    gs = fig.add_gridspec(nrows=2,ncols=2,height_ratios=[1,3])
    ax1 = fig.add_subplot(gs[0,:])
    ax2 = fig.add_subplot(gs[1,0])
    ax3 = fig.add_subplot(gs[1,1])

    ax1.imshow(diff,cmap='PiYG',interpolation='nearest',vmin=-extremum,vmax=extremum)
    for i in range(len(diff)):
        for j in range(len(diff[i])):
            if i*30+j == plot_counter:
                ax1.text(j,i,diff[i,j],ha="center",va="center",color="blue",fontweight='bold')
            else:
                ax1.text(j,i,diff[i,j],ha="center",va="center",color="black")

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

        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax1.imshow(diff,cmap='PiYG',interpolation='nearest',vmin=-extremum,vmax=extremum)
        for i in range(len(diff)):
            for j in range(len(diff[i])):
                if i * 30 + j == plot_counter:
                    ax1.text(j, i, diff[i, j], ha="center", va="center", color="blue", fontweight='bold')
                else:
                    ax1.text(j, i, diff[i, j], ha="center", va="center", color="black")
        sol1 = zippedList[plot_counter][0]
        sol2 = zippedList[plot_counter][1]
        instance = idb[sol1.instance_uid]
        result1 = verify(instance, sol1)
        result2 = verify(instance, sol2)
        # print(f"{solution.instance_uid}: {result}")
        plot_solution(ax2, instance, sol1, result1,prefix="w/o cornerrule")
        plot_solution(ax3, instance, sol2, result2,prefix="w/ cornerrule")
        ax1.set_title(instance.instance_uid)
        plt.draw()



    fig.canvas.mpl_connect('key_press_event',on_press)

    sol1 = zippedList[plot_counter][0]
    sol2 = zippedList[plot_counter][1]

    instance = idb[sol1.instance_uid]
    result1 = verify(instance, sol1)
    result2 = verify(instance, sol2)
    # print(f"{solution.instance_uid}: {result}")
    plot_solution(ax2, instance, sol1, result1,prefix="w/o cornerrule")
    plot_solution(ax3, instance, sol2, result2,prefix="w/ cornerrule")
    ax1.set_title(instance.instance_uid)
    plt.show()

if __name__=="__main__":
    #showSolutions()
    compareSolutions()