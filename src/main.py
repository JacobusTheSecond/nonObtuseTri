#import concurrent.futures
import multiprocessing
import sys
from functools import partial
from multiprocessing import Pool, Lock
import logging
import threading
import time
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use("TkAgg")
from pathlib import Path

import numpy as np
from cgshop2025_pyutils import InstanceDatabase, ZipSolutionIterator, ZipWriter, verify, Cgshop2025Instance
exact = True
if exact:
    #from exactTriangulation import Triangulation
    from Triangulation import Triangulation,QualityImprover

    def improveQuality(instance: Cgshop2025Instance, withShow=True, axs=None, verbosity=0,seed=None):
        # print("WORK IN PROGRESS. PROCEED WITH CARE.")
        triangulation = Triangulation(instance, withValidate=False,seed=seed,axs=axs)
        qi = QualityImprover(triangulation)
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
    filepath = Path(__file__)
    solLoc = filepath.parent.parent/"instance_solutions"

    idb = InstanceDatabase(filepath.parent.parent/"challenge_instances_cgshop25"/"zips"/"challenge_instances_cgshop25_rev1.zip")

    solutions = []
    i = 0
    axs = None
    debugSeed = None#267012647
    debugIdx = None#7#8#88
    debugUID = None#"simple-polygon-exterior-20_10_8c4306da"#point-set_10_13860916"
    withShow = True#True#(debugIdx != None) or (debugUID != None)
    if withShow:
        fig, axs = plt.subplots(1, 1)
        fig.patch.set_facecolor('lightgray')
    for instance in idb:
        i+=1
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
        tr = Triangulation(instance, seed=seeds[index])
        sol = tr.improveQuality()
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
    numThreads = 8
    total = 8
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
    #seeded_Multi()
    solveEveryInstance()