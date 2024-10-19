import time
import matplotlib.pyplot as plt
from pathlib import Path
from cgshop2025_pyutils import InstanceDatabase, ZipSolutionIterator, ZipWriter, verify, Cgshop2025Instance
exact = True
if exact:
    from exactTriangulation import Triangulation

    def improveQuality(instance: Cgshop2025Instance, withShow=True, axs=None, verbosity=0):
        # print("WORK IN PROGRESS. PROCEED WITH CARE.")
        triangulation = Triangulation(instance, withValidate=True)
        l = len(triangulation.triangles)
        if (withShow):
            plt.ion()
        return triangulation.improveQuality(axs, verbosity)
else:
    from QualityImprover import improveQuality

def verifyAll(solname="cur_solutions.zip"):

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
    debugIdx = None#88
    debugUID = None#"point-set_10_13860916"
    withShow = (debugIdx != None) or (debugUID != None)
    if withShow:
        fig, axs = plt.subplots(1, 1)
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
        solution = improveQuality(instance, withShow=withShow, axs=axs, verbosity=verbosity)
        if solution != None:
            solutions.append(solution)
        else:
            print("Fuck instance", instance.instance_uid, end='')
        end = time.time()
        print("#Steiner:",len(solution.steiner_points_x),"Elapsed time:", end - start)
        #except:
        #    print("Some error occured")

    if (solLoc/solname).exists():
        (solLoc/solname).unlink()
    #Write the solutions to a new zip file
    with ZipWriter(solLoc/solname) as zw:
        for solution in solutions:
            zw.add_solution(solution)

    verifyAll(solname)


if __name__=="__main__":
    solveEveryInstance()#"NoLimitRefine.zip"
