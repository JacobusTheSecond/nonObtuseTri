import time
import matplotlib.pyplot as plt
from pathlib import Path
from cgshop2025_pyutils import InstanceDatabase,ZipSolutionIterator,ZipWriter,verify
from hacky_internal_visualization_stuff import plot_solution #internal ugly functions I dont want you to see
exact = False
if exact:
    from exactImprover import improveQuality
else:
    from QualityImprover import improveQuality

def verifyAll(solname="cur_solutions.zip"):

    filepath = Path(__file__)
    zips = filepath.parent.parent/"challenge_instances_cgshop25" / "zips"

    idb = InstanceDatabase(zips/"challenge_instances_cgshop25_rev1.zip")

    for solution in ZipSolutionIterator(zips/solname):
        instance = idb[solution.instance_uid]
        result = verify(instance,solution)
        print(f"{solution.instance_uid}: {result}")
        assert not result.errors, "Expect no errors."

def solveEveryInstance(solname="cur_solutions.zip"):
    filepath = Path(__file__)
    zips = filepath.parent.parent/"challenge_instances_cgshop25" / "zips"

    idb = InstanceDatabase(zips/"challenge_instances_cgshop25_rev1.zip")

    solutions = []
    i = 0
    axs = None
    debugIdx = None
    withShow = True#debugIdx != None
    if withShow:
        fig, axs = plt.subplots(1, 1)
    for instance in idb:
        i+=1
        if instance.instance_uid != "simple-polygon-exterior_150_1301b82e":#debugIdx != None and i != debugIdx:
            continue
        start = time.time()
        try:
            print(i,":",instance.instance_uid,":...",end='')
            solution = improveQuality(instance,withShow=withShow,axs=axs,verbosity=0 if debugIdx == None else 1)
            if solution != None:
                solutions.append(solution)
            else:
                print("Fuck instance", instance.instance_uid, end='')
            end = time.time()
            print("#Steiner:",len(solution.steiner_points_x),"Elapsed time:", end - start)
        except:
            print("Some error occured")

    if (zips/solname).exists():
        (zips/solname).unlink()
    #Write the solutions to a new zip file
    with ZipWriter(zips/solname) as zw:
        for solution in solutions:
            zw.add_solution(solution)

    verifyAll(solname)


if __name__=="__main__":
    solveEveryInstance()#"NoLimitRefine.zip"
