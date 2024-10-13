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

if __name__=="__main__":
    showSolutions()