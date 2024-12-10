import multiprocessing
import sys
from multiprocessing import Pool, Lock
import logging
import time
import matplotlib.pyplot as plt
import matplotlib
import argparse

from pathlib import Path
import numpy as np


from cgshop2025_pyutils import InstanceDatabase, ZipSolutionIterator, ZipWriter, verify, Cgshop2025Instance

from solutionManagement import updateSummaries

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
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S", level=logging.INFO)

    filepath = Path(__file__)
    solLoc = filepath.parent.parent/"instance_solutions"

    idb = InstanceDatabase(filepath.parent.parent/"challenge_instances_cgshop25"/"zips"/"challenge_instances_cgshop25_rev1.zip")

    solutions = []
    i = 0
    axs = None


    debugSeed = 580073460#754181797#267012647
    debugIdxs = None#[87]#[125]#[114]#64#7#8#88
    debugUID = None#"simple-polygon-exterior-20_10_8c4306da"#point-set_10_13860916"
    withShow = True#True#True#True#(debugIdx != None) or (debugUID != None)
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
        if debugIdxs != None and i not in debugIdxs:
            continue
        if debugUID != None and instance.instance_uid != debugUID:
            continue
        start = time.time()
        #try:
        print(i,":",instance.instance_uid,":...",end='')
        verbosity = 0 if (debugIdxs is None) else 1
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
    i = 0
    for instance in idb:
        lock.acquire()
        times[index] = time.time()
        lock.release()
        tr = Triangulation(instance)
        qi = QualityImprover(tr,seed=seeds[index])
        sol = qi.improve()
        lock.acquire()
        returner[index].append(sol)
        progress = np.array([len(l) for l in returner])
        best[i] = len(sol.steiner_points_x) if best[i] < 0 else min(best[i], len(sol.steiner_points_x))
        if np.sum(progress) % 5 == 0:
            #sys.stdout.write("\033[K")
            print("PROGRESS:")
            print(progress)
            print("QUALITY:")
            np.set_printoptions(linewidth=(4*30)+3,formatter={"all":lambda x: str(x).rjust(3)})
            print(np.array(best))
            print("TIMES:")
            np.set_printoptions(linewidth=4 * (96 // 2) + 3, formatter={"all": lambda x: str(x).rjust(3)})
            tts = np.array(times)
            print(np.where(tts != -1,np.array(np.full(tts.shape ,time.time()) - tts,dtype=int)//60,-1))
        lock.release()
        i += 1
    lock.acquire()
    times[index] = -1
    lock.release()


def init_pool_processes(the_lock,the_returner,the_seeds,the_best,the_times):
    global lock
    lock = the_lock
    global returner
    returner = the_returner
    global seeds
    seeds = the_seeds
    global best
    best = the_best
    global times
    times = the_times


def seeded_Multi():
    numThreads = 96
    np.set_printoptions(linewidth=4*(96//2)+3,formatter={"all":lambda x: str(x).rjust(3)})
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S", level=logging.ERROR)
    #numThreads = 96
    total = numThreads
    filepath = Path(__file__)
    idb = InstanceDatabase(
        filepath.parent.parent / "challenge_instances_cgshop25" / "zips" / "challenge_instances_cgshop25_rev1.zip")
    manager = multiprocessing.Manager()
    returner = manager.list([manager.list() for i in range(total)])
    best = manager.list([-1 for _ in idb])
    times = manager.list([-1 for _ in range(total)])
    np.random.seed(1)
    seeds = [np.random.randint(0,1000000000) for i in range(total)]
    #print(seeds[107])
    #print(seeds[98])
    lock = manager.Lock()
    with Pool(initializer=init_pool_processes,initargs=(lock,returner,seeds,best,times)) as pool:
        result = pool.map_async(workerFunction,range(total),chunksize=1)

        result.wait()
        allSolutions = [[sol for sol in sols] for sols in returner]

        solLoc = filepath.parent.parent / "instance_solutions" / "288CircleArr2InRNoSegs"
        solLoc.mkdir(parents=True, exist_ok=True)
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

        updateSummaries()

def pooledWorkerFunction(index):
    lock.acquire()
    logging.error(f"started thread {multiprocessing.current_process()}")
    lock.release()
    seedIdx = index // numInstances
    instanceIdx = index % numInstances
    seed = seeds[seedIdx]
    instance = instanceList[instanceIdx]
    lock.acquire()
    myId = -1
    for cIdx in range(len(currentInstance)):
        if currentInstance[cIdx] == -1:
            myIdx = cIdx
            break
    assert(myIdx != -1)

    logging.error(f"{multiprocessing.current_process()} ({myIdx}): working on instanceId {instanceIdx} of name {instance.instance_uid} with seed {seed}")

    times[myIdx] = time.time()
    currentInstance[myIdx] = instanceIdx
    lock.release()
    tri = Triangulation(instance)
    qi = QualityImprover(tri,seed=seed)
    sol = qi.improve()
    lock.acquire()
    returner[seedIdx][instanceIdx] = sol
    progress = np.array([len([v for v in l if v is not None]) for l in returner])
    best[instanceIdx] = len(sol.steiner_points_x) if best[instanceIdx] < 0 else min(best[instanceIdx], len(sol.steiner_points_x))
    if np.sum(progress) % 1 == 0:
        print("PROGRESS:")
        print(progress)
        print("QUALITY:")
        np.set_printoptions(linewidth=(4*30)+3,formatter={"all":lambda x: str(x).rjust(3)})
        print(np.array(best))
        print("TIMES:")
        np.set_printoptions(linewidth=4 * (96 // 2) + 3, formatter={"all": lambda x: str(x).rjust(3)})
        tts = np.array(times)
        print(np.where(tts != -1,np.array(np.full(tts.shape ,time.time()) - tts,dtype=int)//60,-1))
        print("CURRENT ID:")
        print(np.array(currentInstance))
    times[myIdx] = -1
    currentInstance[myIdx] = -1

    logging.error(f"{multiprocessing.current_process()} ({myIdx}): finished instanceId {instanceIdx} of name {instance.instance_uid} with seed {seed}")

    lock.release()

def init_real_pool_processes(the_lock,the_returner,the_seeds,the_best,the_times,the_progress,the_number,the_instances):
    global lock
    lock = the_lock
    global returner
    returner = the_returner
    global seeds
    seeds = the_seeds
    global best
    best = the_best
    global times
    times = the_times
    global currentInstance
    currentInstance = the_progress
    global numInstances
    numInstances = the_number
    global instanceList
    instanceList = the_instances

def seededPool():
    numThreads = 96
    numSeeds = 4
    np.set_printoptions(linewidth=4*(96//2)+3,formatter={"all":lambda x: str(x).rjust(3)})
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S", level=logging.ERROR)
    filepath = Path(__file__)
    idb = InstanceDatabase(filepath.parent.parent / "challenge_instances_cgshop25" / "zips" / "challenge_instances_cgshop25_rev1.zip")
    manager = multiprocessing.Manager()
    numInstances = len([None for _ in idb])
    logging.error(f"number of isntances: {numInstances}")
    returner = manager.list([manager.list([None for _ in idb]) for i in range(numSeeds)])
    best = manager.list([-1 for _ in idb])
    times = manager.list([-1 for _ in range(numThreads)])
    instanceNos = manager.list([-1 for _ in range(numThreads)])
    np.random.seed(1337)
    seeds = [np.random.randint(0,1000000000) for i in range(numSeeds)]
    lock = manager.Lock()
    logging.error("staring up pool...")
    with Pool(initializer=init_real_pool_processes,initargs=(lock,returner,seeds,best,times,instanceNos,numInstances,[ins for ins in idb])) as pool:
        result = pool.map_async(pooledWorkerFunction,range(numInstances*numSeeds),chunksize=1)

        result.wait()
        allSolutions = [[sol for sol in sols] for sols in returner]

        solLoc = filepath.parent.parent / "instance_solutions" / "288CircleArr2InRNoSegs"
        solLoc.mkdir(parents=True, exist_ok=True)
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

        updateSummaries()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel",type=bool,default=False)
    args = parser.parse_args()
    if args.parallel:
        seededPool()
    else:
    #seeds = [209652396, 398764591, 924231285, 404868288, 441365315, 463622907, 192771779, 417693031, 745841673, 530702035, 626610453, 577165042, 805680932, 204159575, 608910406, 243580376, 917674584, 97308044, 573126970, 977814209, 179207654, 267012647, 124102743, 987744430, 292249176, 613256017, 754181797, 369705497, 305097549, 375363656, 374217481, 636393364, 86837363, 507843536, 354849523, 889724613, 120932350, 602801999, 515448239, 515770816, 981908306, 960389219, 211134424, 218660017, 908296947, 87950109, 131121811, 768281747, 507984782, 947610023, 600956192, 352272321, 615697673, 160516793, 836096639, 37003808, 93837855, 454869706, 707217652, 960356503, 62515875, 800291326, 104082891, 885408951, 930076700, 293921570, 580757632, 80701568, 318433188, 505240629, 642848645, 481447462, 954863080, 502227700, 586215697, 832141647, 655405444, 780912233, 858778666, 470332858, 485603871, 803296120, 654332161, 848819521, 426405863, 258666409, 944072761, 716257571, 657731430, 732884087, 734051083, 903586222, 464510034, 553734235, 2946944, 281012622, 463129187, 488384053, 322325388, 301492857, 165035946, 810037332, 576702667, 897430278, 438279108, 656229423, 897118847, 580073460, 692819075, 657595236, 351544500, 14137320, 705957710, 929047695, 965068436, 530471866, 682769175, 377989839, 474057613, 750555509, 671430485, 288629296, 593491249, 121742917, 844314793, 857151139, 509920704, 698047583, 304913605, 58289758, 417046784, 527619953, 377720498, 745862436, 412739490, 953246512, 972635997, 550586755, 939099746, 41336362, 973931469, 239292784, 450308071, 863972245, 85846399, 168310272, 823392138, 140904818, 986067499, 516240314, 470060415, 198170647, 991680187, 509931650, 810293625, 291985506, 560389694, 705504163, 722565854, 275511419, 931243971, 826661458, 279984046, 286051043, 494361748, 66097361, 38521394, 99849627, 716069646, 403471401, 735116346, 326187713, 356965412, 770071306, 119050326, 294501419, 40186728, 719022091, 396606210, 566067507, 632108112, 929118240, 392500406, 169247872, 63265574, 786800915, 634885294, 371570218, 226866547, 86361272, 708413884, 339044840, 514571853, 20166942, 628962584, 763716221, 299008770, 85933571, 875044958, 165255019, 911342861, 787216240, 26228040, 327093267, 627487327, 395317064, 627230106, 310319554, 596661496, 958128308, 607971651, 870800364, 773332797, 846647054, 855547744, 414072068, 981704643, 856756973, 499083403, 203740156, 73116246, 274009549, 695012217, 629237357, 753505099, 563083960, 349064344, 271520669, 565799061, 925205614, 826916889, 379954640, 967459772, 685770617, 5688182, 142619765, 885122493, 990131795, 618433185, 622208872, 746701728, 510278884, 456955387, 965798651, 82434169, 945460550, 221512149, 928033055, 687940109, 163170119, 172132024, 573215896, 506872399, 767805714, 551697960, 581856694, 441794767, 207371019, 396742822, 300497007, 446877021, 390295545, 222549121, 319224543, 144418678, 657992492, 940433624, 653278589, 304930840, 779186243, 950315986, 629412847, 606723843, 504924394, 417728204, 798283750, 609556182, 44413950, 241351357, 159010501, 875694807, 850086443, 430471200, 427645963, 727922280, 439425236, 920159370, 941668022, 391443877, 641875581, 664207004, 507347567, 161316328, 697901826, 35714587, 496324563, 716604633, 531802775, 124773173, 420942131, 493634734, 246183060, 693193653, 296785437, 601177667, 675097123, 142689063, 297069961, 998901833, 847950121, 529529751, 874209155, 730468142, 953721562, 217274290, 499858284, 496051971, 968607519, 71416585, 454421898, 991030811, 50311597, 64045255, 472352167, 833754173, 987801941, 737143852, 383986466, 90351522, 233379873, 382188165, 858978536, 770904831, 79550500, 734381824, 187680203, 916764962, 961749189, 535036798, 409532341, 833737982, 764853328, 464942526, 877953160, 415131837, 136747137, 144500945, 707356016, 112931, 521745784, 216337755, 331694248, 855119732, 791193113, 74142019, 489480039, 769188348, 153041699, 490835058, 386160414, 3626013, 988366577, 921226529, 266051730, 232969489, 636604637, 912210260, 901272061, 955386370, 799692869, 939521381, 834823052, 298816044, 868051709, 945909359, 976738395, 775433557, 18710951, 366341671, 249233355, 242314902, 792063390, 265411270, 562660295, 843288850, 972399452, 548689963, 505336915, 771391921, 182530857, 105994333, 449311918, 288834965, 770486105, 981813746, 874871264, 760559579, 157106505, 459755036, 629948395, 922116305, 942733242, 931565440, 884826070, 700415551, 671810576, 57219178, 589357220, 88695390, 984489496, 35986036, 565155007, 300721347, 663250227, 573223048, 626140088, 379319131, 320760925, 348100475, 84658926, 553972823, 237728, 526780563, 265687624, 101112263, 753558615, 206359565, 581270842, 797300305, 772576376, 870983540, 885320631, 805256000, 925598428, 166384548, 847700249, 985445152, 991518122, 559623950, 16929238, 273556764, 916167956, 890994642, 78171460, 952415766, 110220515, 908745594, 891077203, 660112007, 750268360, 989741629, 533305997, 628200965, 917299456, 922905736, 118664399, 800831951, 372735925, 247305355, 489006084, 504795484, 750963629, 74642714, 115507786, 567228212, 751227886, 931405233, 878210267, 627321651, 915473123, 280982377, 388387602, 787180917, 879891746, 622116403, 708492958, 536153914, 453603243, 953829175, 817899004, 130794492, 65821458, 397697473, 660221254, 933942762, 358296629, 224164958, 995117115, 362821968, 195913370, 910259836, 898322595, 181022266, 158548913, 761340105, 430759212, 270747405, 70564729, 239011654, 771072550, 39685799, 729785992, 356645901, 150980600, 136279760, 874003448, 371128854, 780100088, 274685775, 53229529, 345517795, 977084166, 161242321, 543335879, 457308628, 378639833, 938033090, 908686908, 881193360, 995150904, 553491314, 856067821, 821960984, 925828100, 211323492, 560269617, 895707479, 533812766, 184770713, 406089492, 424959329, 834693477, 506276537, 100184471, 56851867, 963183645, 417614753, 482366835, 636260149, 130681463, 995699635, 980842998, 15331335, 17386481, 599029897, 83663542, 509316992, 689308499, 587982200, 765465508, 309743313, 205479060, 815390581, 649108464, 48614659, 518217854, 963434282, 990847553, 420238868, 670532388, 481858884, 297257936, 957431920, 402032559, 905527946, 166957293, 672400744, 693089246, 103282523, 469317163, 357105727, 642092959, 348327828, 222611910, 675345673, 108318140, 997438043, 756828783, 569030051, 323455315, 229467980, 871322898, 968920433, 59907914, 49080546, 143733985, 88393643, 234428035, 631131020, 570629348, 341544762, 2674561, 384842097, 761977984, 738939687, 259519512, 782319908, 732461411, 9167685, 246397062, 341889027, 474813666, 554195833, 975147688, 414677436, 87307681, 26369308, 564714308, 394858522, 237830880, 972712184, 221614445, 52047049, 895563189, 885366981, 105624481, 787187335, 824699517, 840204894, 440813640, 143962358, 787864404, 558032007, 330155715, 778037461, 40123867, 165551724, 370986307, 244160614, 687365972, 846096711, 887317075, 123245791, 417794304, 117714565, 778931547, 40469611, 766081950, 531773385, 300042195, 376237575, 41610389, 98019301, 610098652, 519058824, 573228715, 846341678, 125378560, 901324136, 752321203, 191608358, 924185689, 213871690, 330525410, 942738043, 80896343, 569982048, 243997523, 333044430, 706811936, 899509363, 747805126, 625219425, 967610796, 151880507, 144516921, 774822578, 242362541, 43022057, 686623876, 155381177, 617742429, 779074907, 641938833, 118524323, 168464066, 553451924, 325063597, 612787385, 631419855, 886505847, 686636708, 803721065, 555821758, 661356036, 338969073, 188145138, 396254413, 888677790, 689534317, 567113559, 356360930, 895947171, 216373833, 962792666, 546023351, 640058906, 401405302, 661615638, 757227980, 816151247, 958725474, 857370202, 344443955, 333806566, 928761646, 29114064, 116260413, 584377788, 993844015, 72857704, 689508992, 41946506, 199249867, 250337165, 898363719, 893949446, 14720685, 390774005, 874976670, 462665231, 248297641, 890017958, 418624821, 867044267, 522605138, 325348743, 485428444, 303942465, 677025449, 682228546, 281422684, 365076814, 684679909, 405693659, 553614582, 573841508, 637520287, 855976157, 269350066, 802070012, 747462830, 847733341, 37297778, 977948321, 425366284, 143042281, 244753232, 897573954, 514414024, 452555763, 832702589, 458302602, 608916080, 70130394, 136791195, 795566776, 409558136, 649654593, 155797952, 770050757, 774254424, 427842850, 468914127, 838813699, 422959804, 513199342, 395255461]
    #print(np.where(np.array(seeds) == 626610453))
    #print(seeds[58])
        solveEveryInstance()
