import multiprocessing
import sys
from multiprocessing import Pool, Lock
import logging
import time
import matplotlib.pyplot as plt
import matplotlib

from pathlib import Path
import numpy as np


from cgshop2025_pyutils import InstanceDatabase, ZipSolutionIterator, ZipWriter, verify, Cgshop2025Instance
exact = True
if exact:
    #from exactTriangulation import Triangulation
    from Triangulation import Triangulation,QualityImprover

    def improveQuality(instance: Cgshop2025Instance, withShow=True, axs=None, verbosity=0,seed=None):
        # print("WORK IN PROGRESS. PROCEED WITH CARE.")
        triangulation = Triangulation(instance, withValidate=False,seed=None,axs=axs)
        qi = QualityImprover(triangulation,seed=seed)
        if (withShow):
            plt.ion()
        return qi.improve()
else:
    from QualityImprover import improveQuality

def verifyAll(solname="cur_solution.zip"):

    filepath = Path(__file__)
    solLoc = filepath.parent.parent/"instance_solutions"

    idb = InstanceDatabase(filepath.parent.parent/"challenge_instances_cgshop25"/"zips"/"challenge_instances_cgshop25_rev1.zip")

    for solution in ZipSolutionIterator(solLoc/solname):
        instance = idb[solution.instance_uid]
        result = verify(instance,solution)
        print(f"{solution.instance_uid}: {result}")
        assert not result.errors, "Expect no errors."

def solveEveryInstance(solname="cur_solution.zip"):
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S", level=logging.WARNING)

    filepath = Path(__file__)
    solLoc = filepath.parent.parent/"instance_solutions"

    idb = InstanceDatabase(filepath.parent.parent/"challenge_instances_cgshop25"/"zips"/"challenge_instances_cgshop25_rev1.zip")

    solutions = []
    i = 0
    axs = None
    debugSeed = 5#267012647
    debugIdx = None#7#8#88
    debugUID = None#"simple-polygon-exterior-20_10_8c4306da"#point-set_10_13860916"
    withShow = False#True#True#True#(debugIdx != None) or (debugUID != None)
    if withShow:
        matplotlib.use("TkAgg")
        fig = plt.figure()
        fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.2, hspace=0.2)
        gs = fig.add_gridspec(nrows=3, ncols=2, width_ratios=[2, 1],height_ratios=[1,2,2])
        fig.patch.set_facecolor('lightgray')
        ax1 = fig.add_subplot(gs[:, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = ax2.twinx()
        ax4 = fig.add_subplot(gs[1, 1])
        ax5 = fig.add_subplot(gs[2, 1])
        axs = [ax1,ax2,ax3,ax4,ax5]
        for ax in axs:
            ax.set_facecolor('lightgray')

        #fig, axs = plt.subplots(1, 1)
    for instance in idb:
        i+=1
        #if i > 3:
        #    continue
        if debugIdx != None and i != debugIdx:
            continue
        if debugUID != None and instance.instance_uid != debugUID:
            continue
        start = time.time()
        #try:
        print(i,":",instance.instance_uid,":...",end='')
        verbosity = 0 if (debugIdx is None) else 1
        solution = improveQuality(instance, withShow=withShow, axs=axs, seed=debugSeed, verbosity=verbosity)
        solutions.append(solution)
            #print("Fuck instance", instance.instance_uid, end='')
        end = time.time()
        print("#Steiner:",len(solution.steiner_points_x),f"Elapsed time: {end-start:0,.2f}")
        #except:
        #    print("Some error occured")

    if (solLoc/solname).exists():
        (solLoc/solname).unlink()
    #Write the solutions to a new zip file
    with ZipWriter(solLoc/solname) as zw:
        for solution in solutions:
            zw.add_solution(solution)

    verifyAll(solname)

class DataBase:
    def __init__(self,num):
        self.data = [[] for i in range(num)]


    def append(self,index,datum):
        self.data[index].append(datum)

    def printSizes(self):
        sizes = [[] for i in range(len(self.data))]
        for i in range(len(self.data)):
            for j in self.data[i]:
                sizes[i].append(len(j.steiner_points_x))
        return str(sizes)

    def is_full(self):
        if np.all(np.array([len(i) for i in self.data]) == 150):
            return True
        else:
            return False

def workerFunction(index):
    filepath = Path(__file__)
    idb = InstanceDatabase(
        filepath.parent.parent / "challenge_instances_cgshop25" / "zips" / "challenge_instances_cgshop25_rev1.zip")
    for instance in idb:
        tr = Triangulation(instance)
        qi = QualityImprover(tr,seed=seeds[index])
        sol = qi.improve()
        lock.acquire()
        returner[index].append(sol)
        progress = np.array([len(l) for l in returner])
        if np.sum(progress) % 5 == 0:
            #sys.stdout.write("\033[K")
            print(progress)
            sys.stdout.write("\033[F")
        lock.release()

def init_pool_processes(the_lock,the_returner,the_seeds):
    global lock
    lock = the_lock
    global returner
    returner = the_returner
    global seeds
    seeds = the_seeds


def seeded_Multi():
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S", level=logging.ERROR)
    numThreads = 192
    total = 384
    filepath = Path(__file__)
    idb = InstanceDatabase(
        filepath.parent.parent / "challenge_instances_cgshop25" / "zips" / "challenge_instances_cgshop25_rev1.zip")
    manager = multiprocessing.Manager()
    returner = manager.list([manager.list() for i in range(total)])
    np.random.seed(0)
    seeds = [np.random.randint(0,1000000000) for i in range(total)]
    print(seeds)
    #print(seeds[22])
    lock = manager.Lock()
    with Pool(processes=numThreads,initializer=init_pool_processes,initargs=(lock,returner,seeds)) as pool:
        result = pool.map_async(workerFunction,range(total),chunksize=1)

        result.wait()
        allSolutions = [[sol for sol in sols] for sols in returner]

        solLoc = filepath.parent.parent / "instance_solutions" / "out"

        for i in range(len(allSolutions)):
            solname = "seed"+str(seeds[i])+".zip"
            solutions = allSolutions[i]

            try:
                for solution in solutions:
                    instance = idb[solution.instance_uid]
                    result = verify(instance, solution)
                    print(f"{solution.instance_uid}: {result}")

                if (solLoc / solname).exists():
                    (solLoc / solname).unlink()
                # Write the solutions to a new zip file
                with ZipWriter(solLoc / solname) as zw:
                    for solution in solutions:
                        zw.add_solution(solution)
            except:
                print("some error occured")

if __name__=="__main__":
    seeded_Multi()
    #solveEveryInstance()
