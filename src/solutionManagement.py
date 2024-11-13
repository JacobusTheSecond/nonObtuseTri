import logging
import pickle
from pathlib import Path

from cgshop2025_pyutils import ZipSolutionIterator, ZipWriter


def loadSolutions(foldername):

    best = []
    #first unpickle
    fn = foldername.name
    summaryName = foldername.parent.parent/"solution_summaries"/(str(fn)+".zip")
    pickelName = foldername.parent.parent/"solution_summaries"/(str(fn)+".pkl")
    addedSolutions = False
    if summaryName.exists():
        i = 0
        names = pickle.load(open(pickelName, "rb"))

        for sol in ZipSolutionIterator(summaryName):
            if len(best) == i:
                best.append([sol,names[i]])
            elif len(sol.steiner_points_x) < len(best[i][0].steiner_points_x):
                assert sol.instance_uid == best[i][0].instance_uid
                best[i] = [sol,names[i]]
                #logging.info(str(names[i])+" is better at "+str(sol.instance_uid) +" with solution size "+str(len(sol.steiner_points_x)))
            i += 1

    #first build the list
    if foldername.exists():
        for solname in foldername.iterdir():
            i = 0
            logging.info("reading "+str(solname))
            for sol in ZipSolutionIterator(solname):
                if len(best) == i:
                    if addedSolutions == False:
                        addedSolutions = True
                    best.append([sol,solname])
                elif len(sol.steiner_points_x) < len(best[i][0].steiner_points_x):
                    if addedSolutions == False:
                        addedSolutions = True
                    assert sol.instance_uid == best[i][0].instance_uid
                    best[i] = [sol,solname]
                    logging.info(str(solname)+" is better at "+str(sol.instance_uid) +" with solution size "+str(len(sol.steiner_points_x)))
                i += 1

    if len(best) == 0:
        logging.error("No matching data found for folder "+str(foldername))
        return []

    if addedSolutions:
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

def updateSummaries():
    filepath = Path(__file__)
    for name in (filepath.parent.parent/"instance_solutions").iterdir():
        if name.is_dir():
            loadSolutions(name)
