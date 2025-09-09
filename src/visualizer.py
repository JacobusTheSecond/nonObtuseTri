import logging

import matplotlib.pyplot as plt
import matplotlib

from sqlalchemy.testing import against

from src.Triangulation import Triangulation
from src.solutionManagement import updateSummaries

matplotlib.use("TkAgg")
import numpy as np
from pathlib import Path
from cgshop2025_pyutils import InstanceDatabase, ZipSolutionIterator, verify, ZipWriter

from hacky_internal_visualization_stuff import plot_solution, plot_instance
from solutionManagement import loadSolutions, triangulationFromSolution
from exact_geometry import dot
from constants import noneEdge

import seaborn as sns
import pandas as pd


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
    gain = 0.85
    ax1.imshow(diffHeat, cmap='PiYG', interpolation='nearest', norm=matplotlib.colors.SymLogNorm(linthresh=0.5,linscale=1,vmin=-gain*extremum,vmax=gain*extremum,base=2))
    for i in range(len(diff)):
        for j in range(len(diff[i])):
            if i * 30 + j == plot_counter:
                ax1.text(j, i, diff[i, j], ha="center", va="center", color="blue", fontweight='bold')
            else:
                if diff[i, j] != "0" and diff[i, j] != 0:
                    ax1.text(j, i, diff[i, j], ha="center", va="center", color="black")

    sol1 = zippedList[plot_counter][0]
    sol2 = zippedList[plot_counter][1]
    instance = idb[sol1.instance_uid]
    print(f"Instance with {len(instance.points_x)} points and {len(instance.region_boundary) + len(instance.additional_constraints)} constrained edges.")

    #sol1.plotTriangulation()
    #sol2.plotTriangulation()

    #ax1.set_title(instance.instance_uid)
    #return

    result1 = verify(instance, sol1)
    result2 = verify(instance, sol2)
    # print(f"{solution.instance_uid}: {result}")
    plot_solution(ax2, instance, sol1, result1,prefix=baseName[plot_counter])
    plot_solution(ax3, instance, sol2, result2,prefix=name[plot_counter])
    ax1.set_title(instance.instance_uid)

def plotByType(solutions):
    plt.rcParams["text.usetex"]=True
    plt.rc("font",**{"family":"serif","serif":["Computer Modern Roman"],"monospace":["Computer Modern Typewriter"],'size': 20})

    #plt.rcParams["monospace"]="Computer Modern Typewriter"
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S", level=logging.INFO)

    filepath = Path(__file__)
    idb = InstanceDatabase(filepath.parent.parent/"challenge_instances_cgshop25"/"zips"/"challenge_instances_cgshop25_rev1.zip")

    numCons1 = [len(instance.additional_constraints) for instance in idb]
    numCons2 = [len(instance.region_boundary)+len(instance.additional_constraints) for instance in idb]
    print(max(numCons1),max(numCons2))
    #for instance in idb:
    #    print(f"{instance.instance_uid}: {len(instance.points_x)},{len(instance.region_boundary)+len(instance.additional_constraints)}")

    bests = []
    for name in solutions:
        best = loadSolutions(name)
        i = 0
        for sol,solname in best:
            if len(bests) == i:
                bests.append([sol,solname])
            elif len(sol.steiner_points_x) < len(bests[i][0].steiner_points_x):
                assert sol.instance_uid == bests[i][0].instance_uid
                bests[i] = [sol,solname]
                #logging.info(str(solname)+" is better at "+str(sol.instance_uid) +" with solution size "+str(len(sol.steiner_points_x)))
            i += 1


    ids = list(range(150))
    #ids = [i for i in range(150) if "ortho" not in bests[i][0].instance_uid]
    triangulations = []
    anglelist = []
    constraintList = []
    i = 0
    for best,_ in bests:
        if i in ids:
            triangulations.append(triangulationFromSolution(idb[best.instance_uid],best,[None,None,None,None,None]))
        i += 1

    def primitiveAngle(a,b,c):
        dotprod = float(dot(a-b,c-b))
        anorm = np.sqrt(float(dot(a-b,a-b)))
        cnorm = np.sqrt(float(dot(c-b,c-b)))
        return 360 * np.arccos(dotprod / anorm / cnorm) / 2 / np.pi
        # a · b = | a | | b | cos θ

    for trig in triangulations:
        myAngleTrios = []
        myOnConstraint = []
        triIdx = 0
        for tri in trig.triangles:



            a = trig.point(tri[0])
            b = trig.point(tri[1])
            c = trig.point(tri[2])
            myAngleTrios.append([primitiveAngle(a,b,c),primitiveAngle(b,c,a),primitiveAngle(c,a,b)])
            myOnConstraint.append([trig.triangleMap[triIdx][0][2] != noneEdge or trig.triangleMap[triIdx][2][2] != noneEdge,trig.triangleMap[triIdx][0][2] != noneEdge or trig.triangleMap[triIdx][1][2] != noneEdge,trig.triangleMap[triIdx][1][2] != noneEdge or trig.triangleMap[triIdx][2][2] != noneEdge])


            triIdx += 1
        args = np.argsort(-np.array(myAngleTrios))
        anglelist.append(list(np.take_along_axis(np.array(myAngleTrios),args,axis=-1)))
        constraintList.append(list(np.take_along_axis(np.array(myOnConstraint),args,axis=-1)))

    #fig,ax = plt.subplots()


    xs = []
    ys = []
    xMask = []
    yMask = []
    for id in range(len(anglelist)):
        sortedArgs = np.argsort(np.array(anglelist[id])[:,0])
        angles = np.array(anglelist[id])[sortedArgs]
        constraints = np.array(constraintList[id])[sortedArgs]

        xs += list(angles[:,0])
        ys += list(angles[:,2])

    # Create a Figure, which doesn't have to be square.
    fig = plt.figure(layout='constrained')
    shared_ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
    shared_ax.set_aspect("equal")
    plotoffset = 0.025
    plotheight = 0.2
    step = 1/3

    H, xedges, yedges = np.histogram2d(ys, xs, density=True, bins=(np.arange(0, 60 + step, step), np.arange(60, 90 + step, step)))
    # H_normalized = H/float(az1.shape[0]) # the integral over the histogrm is 1
    H_normalized = H # * 100 / sum(sum(H))  # the max value of the histogrm is 1
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    im = shared_ax.imshow(H_normalized.T, extent=extent,interpolation='none', origin='lower',norm=matplotlib.colors.LogNorm(),cmap=sns.color_palette("rocket_r", as_cmap=True))
    #fig.colorbar(im, ax=axes[1])

    shared_ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:0.0f}°"))
    shared_ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:0.0f}°"))
    shared_ax.set_yticks(np.arange(60, 91, 5))
    shared_ax.set_xticks(np.arange(0, 61, 5))

    shared_ax.set_xlabel("Smallest angle")
    shared_ax.set_ylabel("Largest angle")

    xhist = np.histogram(ys,bins=np.arange(0,60+step,step))
    yhist = np.histogram(xs,bins=np.arange(60,90+step,step))
    frac = np.max(xhist[0])/np.max(yhist[0])

    ax_histx = shared_ax.inset_axes([0, 1 + (2*plotoffset), 1, 2*frac*plotheight], sharex=shared_ax)
    ax_histy = shared_ax.inset_axes([1 + plotoffset, 0, plotheight, 1], sharey=shared_ax)
    ax_histx.set_axis_off()
    ax_histy.set_axis_off()

    colorbarax = shared_ax.inset_axes([0,0,0.4,0.1])
    colorbarax.set_axis_off()

    def my_formatter(x, pos):
        """Format 1 as 1, 0 as 0, and all values whose absolute values is between
        0 and 1 without the leading "0." (e.g., 0.7 is formatted as .7 and -0.4 is
        formatted as -.4)."""
        val_str = None
        if x < 0.01:
            val_str = f'{100*x}\%'
        else:
            val_str = f'{int(100*x)}\%'
        return val_str


    b = fig.colorbar(im,aspect=10,ax=colorbarax,orientation="horizontal",location="top",fraction=1,format=matplotlib.ticker.FuncFormatter(my_formatter))

    xcounts,xbins,xbars = ax_histx.hist(ys,bins=np.arange(0,60+step,step),color=sns.color_palette("rocket").as_hex()[1],edgecolor="black")
    ycounts,ybins,ybars = ax_histy.hist(xs,bins=np.arange(60,90+step,step),orientation="horizontal",color=sns.color_palette("rocket").as_hex()[1],edgecolor="black")
    #ax_histx.set_ylim((0.0, ax_histy.get_xlim()[1]/2))
    maximum = np.max(ycounts)
    norm = matplotlib.colors.LogNorm(vmin=1,vmax=100)
    #norm = lambda x : x/maximum
    colorBars = False
    if colorBars:
        for i, (xcnt, xvalue, xbar) in enumerate(zip(xcounts, xbins, xbars)):
            if xcnt == 0:
                xbar.set_facecolor(sns.color_palette("rocket_r", as_cmap=True)(norm(0)))
            else:
                xbar.set_facecolor(sns.color_palette("rocket_r", as_cmap=True)(norm(1+(99*(xcnt-1)/maximum))))

        for i, (ycnt, yvalue, ybar) in enumerate(zip(ycounts, ybins, ybars)):
            if ycnt == 0:
                ybar.set_facecolor(sns.color_palette("rocket_r", as_cmap=True)(norm(0)))
            else:
                ybar.set_facecolor(sns.color_palette("rocket_r", as_cmap=True)(norm(1+(99*(ycnt-1)/maximum))))



    plt.show()

    ax.clear()
    plt.style.use('_mpl-gallery-nogrid')
    step = 1
    g = sns.jointplot(x=ys,y=xs,kind="hist",ratio=2,xlim=(0, 60), ylim=(60, 90),joint_kws=dict(bins=(np.arange(0, 60 + step, step), np.arange(60, 90 + step, step))))
    #ax.hist2d(ys, xs, bins=(np.arange(0, 60 + step, step), np.arange(60, 90 + step, step)))
    # ax.set(xlim=(0, 90), ylim=(0, 90))
    #ax.set_aspect("equal")
    g.ax_joint.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:0.0f}°"))
    g.ax_joint.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:0.0f}°"))
    g.ax_joint.set_yticks(np.arange(60, 91, 5))
    g.ax_joint.set_xticks(np.arange(0, 61, 5))
    g.ax_joint.set_xlabel("Smallest angle")
    g.ax_joint.set_ylabel("Largest angle")
    g.ax_marg_x.set_ylim((0.0, g.ax_marg_y.get_xlim()[1]/2))
    #g.ax_joint.set_aspect("equal")

    plt.show()

    showCalled = False
    for id in ids:
        sol,_ = bests[id]
        sortedArgs = np.argsort(np.array(anglelist[id])[:,0])
        angles = np.array(anglelist[id])[sortedArgs]
        constraints = np.array(constraintList[id])[sortedArgs]
        hatches = np.where(constraints,"//////","")

        ax.clear()
        ax.bar(range(len(angles)),angles[:,0],width=1.0,edgecolor="black",hatch=hatches[:,0],color=sns.color_palette("Set2",3).as_hex()[1],linewidth=0.5)
        ax.bar(range(len(angles)),angles[:,1],bottom=angles[:,0],width=1.0,edgecolor="black",hatch=hatches[:,1],color=sns.color_palette("Set2",3).as_hex()[0],linewidth=0.5)
        ax.bar(range(len(angles)),angles[:,2],bottom=angles[:,0]+angles[:,1],width=1.0,edgecolor="black",hatch=hatches[:,2],color=sns.color_palette("Set2",3).as_hex()[2],linewidth=0.5)
        ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:0.0f}°"))
        ax.set_yticks([0,60,90,120,180])
        ax.set_xticks([])
        ax.set_xlim([-0.5,len(angles)-0.5])
        ax.plot([-0.5,len(angles)-0.5],[90,90],c="red",linewidth=2)
        ax.set_ylim([0,180])
        matplotlib.rcParams['axes.spines.left'] = False
        matplotlib.rcParams['axes.spines.right'] = False
        matplotlib.rcParams['axes.spines.top'] = False
        matplotlib.rcParams['axes.spines.bottom'] = False
        fig.tight_layout()
        ax.get_yticklabels()[2].set_color("red")

        ax.clear()
        fig.canvas.manager.set_window_title(f"{sol.instance_uid}_histogram")

        constraintMaxAngle = []
        unconstraitMaxAngle = []
        for angleTrio,constraintTrio in zip(angles,constraints):
            argmax = np.argsort(angleTrio)[-1]
            if constraintTrio[argmax]:
                constraintMaxAngle.append(angleTrio[argmax])
            else:
                unconstraitMaxAngle.append(angleTrio[argmax])

        ax.hist([constraintMaxAngle,unconstraitMaxAngle], 30, range=(60, 90), stacked=True,color=["darkblue","blue"],edgecolor='k',rwidth=0.8)

        #ax.scatter(x=[item[0] for item in buckets[k]],y=[item[1] for item in buckets[k]])
        if not showCalled:
            plt.show()
            showCalled = True

        #ax.set_ylim((0,70))
        ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:0.0f}°"))
        plt.draw()
        fig.tight_layout()
        ax.clear()


        constraintMaxAngle = []
        unconstraitMaxAngle = []
        for angleTrio,constraintTrio in zip(angles,constraints):
            argmax = np.argsort(angleTrio)[-2]
            if constraintTrio[argmax]:
                constraintMaxAngle.append(angleTrio[argmax])
            else:
                unconstraitMaxAngle.append(angleTrio[argmax])

        ax.hist([constraintMaxAngle,unconstraitMaxAngle], 45, range=(45, 90), stacked=True,color=["darkgreen","green"],edgecolor='k',rwidth=0.8)


        #ax.set_ylim((0,70))
        plt.draw()
        fig.tight_layout()
        ax.clear()


        constraintMaxAngle = []
        unconstraitMaxAngle = []
        for angleTrio,constraintTrio in zip(angles,constraints):
            argmax = np.argsort(angleTrio)[-3]
            if constraintTrio[argmax]:
                constraintMaxAngle.append(angleTrio[argmax])
            else:
                unconstraitMaxAngle.append(angleTrio[argmax])

        ax.hist([constraintMaxAngle,unconstraitMaxAngle], 60, range=(0, 60), stacked=True,color=["darkred","red"],edgecolor='k',rwidth=0.8)


        #ax.set_ylim((0,70))
        plt.draw()

        #ax.hist([countsA,countsB], binsA,stacked=True)
        #ax.hist(binsA[:-1], binsA,weights=countsA+countsB,zorder=0)
        #bins[:-1], bins, weights = counts

        fig.tight_layout()

        plt.pause(0.01)





    buckets = dict()
    keys = ["simple-polygon-exterior","point-set","ortho","simple-polygon"]
    keyOffsets = {keys[0]:-3,keys[1]:-1,keys[2]:1,keys[3]:3}
    for k in keys:
        buckets[k] = []

    for sol,_ in bests:
        uid = sol.instance_uid
        x = len(idb[uid].points_x)
        y = len(sol.steiner_points_x)
        handled = False
        for k in keys:
            if k in uid:
                handled = True
                buckets[k].append((x,y))
            if handled:
                break
        if not handled:
            print("uh oh...")

    data = {'Number of points in instance':[],'Instance type':[],'Number of Steiner points':[]}
    for k in keys:
        for x,y in buckets[k]:
            data['Number of points in instance'].append(x)
            data['Number of Steiner points'].append(y)
            data['Instance type'].append(k)

    df = pd.DataFrame(data)


    fig,ax = plt.subplots()
    def formatter(s):
        print("formatter called")
        return rf"$\texttt{{{s}}}$"
    sns.boxplot(x='Number of points in instance',y='Number of Steiner points',data=df,hue='Instance type',palette=["white","white","white","white"],ax=ax,native_scale=True,showfliers=False,legend=False,zorder=1)
    sns.boxplot(x='Number of points in instance',y='Number of Steiner points',data=df,hue='Instance type',palette='Set2',ax=ax,native_scale=True,showfliers=False,boxprops={"alpha":0.5},zorder=1,formatter=formatter)
    sns.boxplot(x='Number of points in instance',y='Number of Steiner points',data=df,hue='Instance type',palette='Set2',ax=ax,native_scale=True,showfliers=False,boxprops={"fill":None},legend=False,zorder=3)
    offsets = []
    for type in df['Instance type']:
        offsets.append(keyOffsets[type])
    df['Number of points in instance'] += offsets
    sns.swarmplot(x='Number of points in instance',y='Number of Steiner points',data=df,hue='Instance type',palette='Set2',ax=ax,native_scale=True,legend=False,zorder=2)

    i = 0
    np.set_printoptions(precision=2)
    for k in keys:

        xs = [item[0] for item in buckets[k]]
        ys = [item[1] for item in buckets[k]]
        sign = "+" if np.poly1d(np.polyfit(xs,ys,1))[0] >= 0 else ""
        sns.lineplot(x=np.unique(xs)+keyOffsets[k],y=np.poly1d(np.polyfit(xs,ys,1))(np.unique(xs)),label=rf"${np.poly1d(np.polyfit(xs,ys,1))[1]:.2f} x {sign} {np.poly1d(np.polyfit(xs,ys,1))[0]:.2f}$",zorder=-1,color=sns.color_palette("Set2",4).as_hex()[i],linestyle="dashed")
        ax.set_xticks(np.unique(xs))
        i += 1
    #texts = plt.legend().texts
    L = plt.legend(ncol=2)
    for i in range(4):
        L.get_texts()[i].set_text(formatter(keys[i]))
    #plt.legend(ncol=2)
    fig.tight_layout()
    #ax.set_xscale("log")
    #ax.set_yscale("log")
    plt.show()
    for k in keys:

        ax.clear()
        fig.canvas.manager.set_window_title(f"{k}_plot")
        xdict = dict()
        for x,y in buckets[k]:
            xdict[x] = xdict.get(x,[]) + [y]
        print(xdict)
        for x in xdict.keys():
            ax.boxplot([xdict[x]],positions=[x],widths=[8],showfliers=False)
            newxs = np.random.normal(x,0.5,size=len(xdict[x]))
            ax.plot(newxs,xdict[x],"r.",alpha=0.2)

        xs = [item[0] for item in buckets[k]]
        ys = [item[1] for item in buckets[k]]
        ax.plot(np.unique(xs),np.poly1d(np.polyfit(xs,ys,1))(np.unique(xs)),label=str(np.poly1d(np.polyfit(xs,ys,1))))
        ax.legend(loc="upper left")
        ax.set_ylim(-5,230)
        ax.set_aspect("equal")

        fig.tight_layout()

        #ax.scatter(x=[item[0] for item in buckets[k]],y=[item[1] for item in buckets[k]])
        plt.draw()
        plt.pause(0.01)



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
                #logging.info(str(solname)+" is better at "+str(sol.instance_uid) +" with solution size "+str(len(sol.steiner_points_x)))
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
                #logging.info(str(solname)+" is better at "+str(sol.instance_uid) +" with solution size "+str(len(sol.steiner_points_x)))
            i += 1

    global plot_counter
    plot_counter = 0
    zippedList = []
    diff = []
    baseName = []
    name = []
    fullList = []

    fig = plt.figure()
    gs = fig.add_gridspec(nrows=2,ncols=2,height_ratios=[1,2])
    ax1 = fig.add_subplot(gs[0,:])
    ax2 = fig.add_subplot(gs[1,0])
    ax3 = fig.add_subplot(gs[1,1])

    fig2,otherAx = plt.subplots()
    fig3,otherotherAx = plt.subplots()

    avgImprovePercent = 0
    for base,other in zip(bestBases,bestOthers):
        #logging.info("identifying best for " + str(base[0].instance_uid))
        a = base[0]
        b = other[0]
        #idx = np.argmin([len(o.steiner_points_x) for o in other])
        #other = b if len(b.steiner_points_x) < len(c.steiner_points_x) else c
        assert a.instance_uid == b.instance_uid
        #zippedList.append([a,b])
        #diff.append(len(a.steiner_points_x) - len(b.steiner_points_x))
        #baseName.append(bases[baseIdx])
        #name.append(others[idx])
        #atrig = triangulationFromSolution(idb[a.instance_uid],a,[ax2,None,None,None,None])
        #btrig = triangulationFromSolution(idb[b.instance_uid],b,[ax3,None,None,None,None])
        asize = len(a.steiner_points_x)
        bsize = len(b.steiner_points_x)
        diff = len(a.steiner_points_x) - len(b.steiner_points_x)
        diffText = None
        if asize == bsize:
            diffText = "0"
        elif asize < bsize:
            avgImprovePercent -= 100*((bsize/asize)-1)
            diffText = "-"+str(int(100*((bsize/asize)-1)))+"%"
        else:
            avgImprovePercent += 100*(1-(bsize/asize))
            diffText = str(int(100*(1 - (bsize/asize))))+"%"
        fullList.append([[a,b],len(a.steiner_points_x),(diff,diffText),str(other[1].parent.name)+"/"+str(other[1].name),str(base[1].parent.name)+"/"+str(base[1].name),idb[a.instance_uid].num_points])
    avgImprovePercent/=len(bestBases)
    logging.info(f"average improvement in percent: {avgImprovePercent:3.1f}%")
    #fullList = sorted(fullList,key = lambda entry : str(entry[0][0].instance_uid))
    #fullList = sorted(fullList,key = lambda entry : entry[5])
    zippedList = [e[0] for e in fullList]
    base = [e[1] for e in fullList]
    diff = [e[2][0] for e in fullList]
    diffTexts = [e[2][1] for e in fullList]
    name = [e[3] for e in fullList]
    baseName = [e[4] for e  in fullList]


    minimum = min(diff)
    maximum = max(diff)
    extremum = max(abs(minimum),abs(maximum))

    diffheat = (((np.array(base) + np.array(diff))/np.array(base))-1)
    diffheat = np.reshape(diffheat,(5,30))

    diff = np.reshape(diff,(5,30))
    percentage = True
    if percentage:
        diffTexts = np.reshape(diffTexts,(5,30))
    else:
        diffTexts = diff

    limit = len(zippedList)

    def on_click(event):
        if event.inaxes==ax1:
            global plot_counter
            plot_counter = 30 * (min(max(0,int(event.ydata+0.5)),4)) + min(max(0,int(event.xdata+0.5)),29)
            plot_counter = min(max(0,plot_counter),149)
            updatePlot(ax1, ax2, ax3, diffTexts, diffheat, zippedList, idb, name, baseName)
            sol1 = zippedList[plot_counter][0]
            sol2 = zippedList[plot_counter][1]
            instance = idb[sol1.instance_uid]
            otherAx.clear()
            plot_instance(otherAx, instance)
            otherotherAx.clear()
            result2 = verify(instance, sol2)
            # print(f"{solution.instance_uid}: {result}")
            plot_solution(otherotherAx, instance, sol2, result2,withNames=False)

            otherAx.axis("off")
            fig2.tight_layout()
            otherotherAx.axis("off")
            fig3.tight_layout()
            fig2.canvas.manager.set_window_title(f"{sol1.instance_uid}_instance")
            fig3.canvas.manager.set_window_title(f"{sol1.instance_uid}_solution")

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

        updatePlot(ax1, ax2, ax3, diffTexts, diffheat, zippedList, idb, name, baseName)
        sol1 = zippedList[plot_counter][0]
        sol2 = zippedList[plot_counter][1]
        instance = idb[sol1.instance_uid]
        otherAx.clear()
        plot_instance(otherAx, instance)
        otherotherAx.clear()
        result2 = verify(instance, sol2)
        # print(f"{solution.instance_uid}: {result}")
        plot_solution(otherotherAx, instance, sol2, result2,withNames=False)
        otherAx.axis("off")
        fig2.tight_layout()
        otherotherAx.axis("off")
        fig3.tight_layout()

        fig2.canvas.manager.set_window_title(f"{sol1.instance_uid}_instance")
        fig3.canvas.manager.set_window_title(f"{sol1.instance_uid}_solution")

        plt.draw()



    fig.canvas.mpl_connect('key_press_event',on_press)
    fig.canvas.mpl_connect('button_press_event', on_click)

    updatePlot(ax1,ax2,ax3,diffTexts,diffheat,zippedList,idb,name,baseName)
    sol1 = zippedList[plot_counter][0]
    sol2 = zippedList[plot_counter][1]
    instance = idb[sol1.instance_uid]
    otherAx.clear()
    plot_instance(otherAx, instance)
    otherotherAx.clear()
    triangulation = Triangulation(instance, withValidate=False, seed=None, axs=[otherotherAx,None,None,None,None])
    sol2 = triangulation.solutionParse()
    result2 = verify(instance, sol2)
    # print(f"{solution.instance_uid}: {result}")
    triangulation.plotTriangulation()
    #plot_solution(otherotherAx, instance, sol2, result2,withNames=False)
    otherAx.axis("off")
    fig2.tight_layout()
    otherotherAx.axis("off")
    fig3.tight_layout()

    fig2.canvas.manager.set_window_title(f"{sol1.instance_uid}_instance")
    fig3.canvas.manager.set_window_title(f"{sol1.instance_uid}_solution")


    plt.show()

def plotHistory():
    filepath = Path(__file__)
    idb = InstanceDatabase(filepath.parent.parent/"challenge_instances_cgshop25"/"zips"/"challenge_instances_cgshop25_rev1.zip")
    fig,ax = plt.subplots()
    plt.show()
    history = filepath.parent.parent/"history"
    for i in range(len(list(history.iterdir()))):
        for solution in ZipSolutionIterator(history/f"{i}.zip"):
            tr = triangulationFromSolution(idb[solution.instance_uid], solution, [ax, None, None, None, None])
            tr.plotTriangulation()
            ax.axis("off")
            ax.title.set_text("")
            fig.canvas.manager.set_window_title(f"{i}")
            fig.tight_layout()
            plt.draw()
            print(len(tr.validVertIdxs()) - tr.instanceSize + 2 * len(tr.getNonSuperseededBadTris()) + 1.1*len(np.where(tr.badTris == True)[0]))
            plt.pause(0.1)

def resultSummary():
    result = """point-set_20_0c4009d9 	0 	21 	0 	21 	26 	0 	26 	0 	12 	72 	0 	28 	26 	0 	0 	232 	0 	110 	30 	0 	0 	79
point-set_10_d009159f 	0 	7 	0 	7 	8 	0 	8 	0 	0 	10 	0 	8 	8 	0 	0 	70 	0 	69 	9 	0 	0 	33
point-set_40_1b92b629 	0 	31 	0 	31 	40 	0 	40 	0 	0 	137 	0 	34 	40 	0 	0 	422 	0 	1062 	67 	0 	0 	385
point-set_60_2ff8f975 	0 	48 	0 	48 	72 	0 	72 	0 	69 	30 	0 	52 	72 	0 	0 	694 	0 	3031 	101 	0 	0 	388
point-set_20_34a047f7 	0 	17 	0 	17 	23 	0 	23 	0 	0 	80 	0 	20 	23 	0 	0 	191 	0 	166 	27 	0 	0 	328
point-set_100_05594822 	0 	80 	0 	80 	111 	0 	111 	0 	109 	89 	0 	96 	111 	0 	0 	1238 	0 	3688 			29 	493
point-set_10_f999dc7f 	0 	10 	0 	11 	10 	0 	10 	0 	0 	39 	0 	10 	10 	0 	0 	25 	0 	33 	11 	0 	0 	86
simple-polygon_10_272aa6ea 	0 	4 	0 	4 	4 	0 	4 	0 	0 	9 	0 	4 	4 	0 	0 	5 	0 	5 	8 	0 	0 	5
simple-polygon-exterior-20_60_8221c868 	0 	62 	0 	62 	73 	0 	73 	0 	69 	52 	0 	80 	73 	0 	0 	1200 	11 	336 	94 	0 	29 	483
point-set_10_13860916 	0 	8 	0 	8 	11 	0 	11 	0 	0 	13 	0 	9 	11 	0 	0 	20 	0 	20 	11 	0 	0 	13
point-set_20_54ab0b47 	0 	11 	0 	11 	17 	8 	18 	0 	0 	74 	0 	11 	18 	0 	0 	192 	0 	249 	26 	0 	0 	414
point-set_60_74409a1d 	0 	42 	0 	42 	59 	0 	59 	0 	38 	122 	0 	51 	59 	0 	0 	679 	0 	3331 	102 	0 	0 	372
simple-polygon-exterior-20_40_84415804 	0 	30 	0 	30 	36 	0 	36 	0 	24 	29 	0 	33 	36 	0 	0 	605 	0 	253 	52 	0 	30 	387
simple-polygon-exterior-20_60_28a85662 	0 	57 	0 	57 	76 	0 	76 	0 	72 	78 	0 	72 	76 	0 	0 	919 	6 	413 	93 	0 	26 	405
simple-polygon-exterior-20_80_889602ae 	0 	77 	0 	77 	86 	0 	86 	0 	76 	14 	0 	103 	86 	0 	0 	1558 	16 	493 	111 	0 	30 	405
simple-polygon_100_cb23308c 	0 	43 	0 	43 	74 	0 	74 	0 	71 	27 	0 	72 	74 	0 	0 	859 	0 	562 	78 	0 	37 	312
simple-polygon-exterior_20_87cff693 	0 	17 	0 	17 	20 	0 	20 	0 	19 	3 	0 	17 	20 	0 	0 	166 	0 	94 	23 	0 	0 	400
simple-polygon_20_0dda68ed 	0 	8 	0 	8 	13 	0 	13 	0 	0 	16 	0 	8 	13 	0 	0 	24 	0 	23 	12 	0 	0 	187
simple-polygon-exterior_100_f1740925 	0 	88 	0 	88 	101 	0 	101 	0 	89 	200 	0 	102 	101 	0 	0 	1989 	23 	699 	143 	0 	52 	445
simple-polygon-exterior_150_89d077ac 	0 	114 	0 	114 	141 	0 	141 	0 	126 	149 	0 	171 	141 	0 	0 	3132 	10 	21051 	238 	0 	90 	505
simple-polygon-exterior-20_10_c6728228 	0 	6 	0 	6 	8 	0 	8 	0 	0 	15 	0 	6 	8 	0 	0 	21 	0 	21 	10 	0 	0 	330
simple-polygon-exterior-20_10_15783346 	0 	9 	0 	9 	7 	0 	7 	0 	0 	22 	0 	9 	7 	0 	0 	29 	0 	60 	7 	0 	0 	332
simple-polygon-exterior_20_2a7302a0 	0 	17 	0 	17 	16 	0 	16 	0 	0 	57 	0 	17 	16 	0 	0 	73 	0 	56 	24 	0 	0 	311
point-set_60_27bc003d 	0 	43 	0 	43 	65 	0 	65 	0 	64 	3 	0 	73 	65 	0 	0 	654 	0 	1217 			41 	448
simple-polygon-exterior-20_100_8d1c2e30 	0 	55 	0 	55 	64 	0 	64 	0 	34 	115 	0 	60 	64 	0 	0 	1651 	0 	3944 	130 	0 	28 	466
point-set_80_d77fb670 	0 	43 	0 	43 	68 	0 	68 	0 	19 	103 	0 	43 	68 	0 	0 	868 	0 	258 			0 	510
simple-polygon-exterior_60_ba2c82c0 	0 	40 	0 	40 	47 	0 	47 	0 	22 	163 	0 	48 	47 	0 	0 	1047 	0 	3493 	75 	0 	21 	477
simple-polygon-exterior-20_40_317d3f6d 	0 	24 	0 	24 	30 	0 	30 	0 	0 	75 	0 	26 	30 	0 	0 	642 	0 	161 	53 	0 	0 	421
simple-polygon_150_743d6b9c 	0 	55 	0 	55 	85 	0 	85 	0 	5 	196 	0 	60 	85 	0 	0 	559 	0 	522 	121 	0 	15 	271
ortho_20_e2aff192 	0 	5 	0 	5 	6 	0 	6 	0 	0 	10 	0 	5 	6 	0 	0 	17 	0 	16 	0 	39 	0 	6
simple-polygon-exterior_20_ff791267 	0 	25 	0 	25 	25 	0 	25 	0 	9 	45 	0 	31 	25 	0 	0 	186 	0 	189 	29 	0 	25 	0
simple-polygon-exterior_40_ca26cbd4 	0 	42 	0 	42 	45 	0 	45 	0 	42 	2 	0 	50 	45 	0 	0 	867 	1 	219 	61 	0 	18 	391
simple-polygon_60_17af118a 	0 	14 	0 	14 	21 	0 	21 	0 	0 	42 	0 	15 	21 	0 	0 	84 	0 	75 	45 	0 	0 	221
point-set_20_72cd2066 	0 	16 	0 	16 	21 	0 	21 	0 	0 	76 	0 	19 	21 	0 	0 	148 	0 	234 	27 	0 	0 	320
simple-polygon-exterior-20_250_4441b4ca 	0 	187 	0 	187 	248 	0 	248 	0 	205 	280 	0 	330 	248 	0 	0 	5018 	0 	24420 	390 	0 	92 	492
simple-polygon-exterior-20_20_4ddfa00e 	0 	19 	0 	19 	21 	0 	21 	0 	8 	68 	0 	21 	21 	0 	0 	211 	1 	174 	25 	0 	0 	375
ortho_40_56a6f463 	0 	19 	0 	19 	23 	0 	23 	0 	0 	27 	0 	20 	23 	0 	0 	48 	0 	31 	25 	0 	0 	44
simple-polygon-exterior-20_60_6088b7a9 	0 	48 	0 	48 	58 	0 	58 	0 	44 	111 	0 	56 	58 	0 	0 	920 	4 	246 	87 	0 	12 	392
simple-polygon_20_35585ee3 	0 	7 	0 	7 	8 	0 	8 	0 	0 	14 	0 	7 	8 	0 	0 	22 	0 	22 	12 	0 	0 	354
simple-polygon_80_48c9df87 	0 	30 	0 	30 	42 	0 	42 	0 	0 	58 	0 	31 	42 	0 	0 	133 	0 	144 	66 	0 	0 	200
simple-polygon-exterior-20_100_8bfbe418 	0 	84 	0 	84 	105 	0 	105 	0 	78 	213 	0 	127 	105 	0 	0 	1747 	19 	1847 	151 	0 	63 	443
simple-polygon-exterior_40_11434792 	0 	35 	0 	35 	45 	0 	45 	0 	23 	115 	0 	44 	45 	0 	0 	723 	0 	155 	60 	0 	14 	365
simple-polygon-exterior-20_100_7ed1ca87 	0 	94 	0 	94 	113 	0 	113 	0 	97 	239 	0 	111 	113 	0 	0 	1814 	17 	815 	150 	0 	113 	0
point-set_60_ac318d72 	0 	43 	0 	43 	62 	52 	64 	0 	39 	64 	0 	49 	64 	0 	0 	696 	0 	1379 			0 	127
ortho_250_6e6a66c2 	0 	81 	0 	81 	122 	0 	122 	0 	0 	553 	0 	95 	122 	0 	0 	2105 	0 	750 	140 	0 	0 	1109
simple-polygon-exterior-20_20_1e719235 	0 	19 	0 	19 	25 	0 	25 	0 	20 	46 	0 	26 	25 	0 	0 	299 	4 	110 	26 	0 	0 	434
simple-polygon-exterior_10_310dc6c7 	0 	6 	0 	6 	5 	0 	5 	0 	0 	18 	0 	6 	5 	0 	0 	8 	0 	8 	8 	0 	0 	338
simple-polygon_10_297edd18 	0 	5 	0 	5 	6 	0 	6 	0 	0 	6 	0 	5 	6 	0 	0 	14 	0 	10 	6 	0 	0 	9
point-set_150_982c9ab3 	0 	151 	0 	151 	198 	0 	198 	0 	196 	9 	0 	235 	198 	0 	0 	1942 	0 	2155 			38 	4042
simple-polygon_150_f24b0f8e 	0 	32 	0 	32 	48 	0 	48 	0 	30 	96 	0 	40 	48 	0 	23 	213 	0 	277 	101 	0 	0 	388
simple-polygon-exterior-20_100_c256488f 	0 	89 	0 	89 	113 	0 	113 	0 	103 	249 	0 	119 	113 	0 	0 	1934 	17 	452 	157 	0 	43 	437
point-set_80_ff15444b 	0 	60 	0 	60 	86 	0 	86 	0 	49 	159 	0 	67 	86 	0 	0 	1056 	0 	6972 	137 	0 	0 	384
simple-polygon-exterior-20_100_8ff7a64d 	0 	90 	0 	90 	109 	0 	109 	0 	100 	23 	0 	111 	109 	0 	0 	1838 	13 	827 	149 	0 	109 	0
simple-polygon-exterior_150_1301b82e 	0 	119 	0 	119 	164 	0 	164 	0 	146 	128 	0 	149 	164 	0 	0 	3000 	30 	587 	247 	0 	35 	502
simple-polygon-exterior_10_8b098f5e 	0 	10 	0 	10 	9 	0 	9 	0 	0 	12 	0 	10 	9 	0 	0 	30 	0 	30 	10 	0 	9 	0
simple-polygon-exterior-20_10_ce9152de 	0 	7 	0 	7 	9 	0 	9 	0 	1 	24 	0 	10 	9 	0 	0 	53 	0 	54 	10 	0 	0 	246
simple-polygon-exterior_60_881cf585 	0 	61 	0 	61 	75 	0 	75 	0 	52 	143 	0 	75 	75 	0 	0 	1106 	7 	387 	93 	0 	10 	391
point-set_10_ae0fff93 	0 	8 	0 	8 	6 	0 	6 	0 	0 	8 	0 	8 	6 	0 	0 	10 	0 	10 	0 	31 	0 	10
simple-polygon_150_c0cf1e9c 	0 	47 	0 	47 	74 	0 	74 	0 	0 	106 	0 	48 	74 	0 	0 	345 	0 	422 	113 	0 	23 	149
simple-polygon-exterior-20_40_5faf6985 	0 	28 	0 	28 	34 	0 	34 	0 	7 	76 	0 	32 	34 	0 	0 	631 	0 	126 	50 	0 	13 	442
simple-polygon-exterior-20_20_c27f41dd 	0 	17 	0 	17 	20 	0 	20 	0 	16 	53 	0 	18 	20 	0 	0 	79 	0 	166 	24 	0 	11 	347
point-set_10_c04b0024 	0 	7 	0 	7 	4 	3 	9 	0 	0 	9 	0 	7 	9 	0 	0 	17 	0 	17 	12 	0 	0 	10
point-set_40_9451c229 	0 	27 	0 	27 	34 	0 	34 	0 	7 	121 	0 	30 	34 	0 	0 	404 	0 	76 			0 	100
ortho_10_d2723dcc 	0 	3 	0 	3 	8 	8 	4 	0 	0 	3 	0 	3 	4 	0 	0 	6 	0 	3 	0 	15 	0 	3
point-set_250_3c338713 	0 	159 	0 	159 	257 	0 	257 	0 	183 	494 	0 	274 	257 	0 	0 	3519 	0 	6074 			33 	9001
ortho_150_53eb4022 	0 	38 	0 	38 	56 	0 	56 	0 	0 	302 	0 	39 	56 	0 	0 	272 	0 	252 	83 	0 	0 	568
point-set_20_fa3fd7e0 	0 	13 	0 	13 	15 	0 	15 	0 	3 	36 	0 	16 	15 	0 	0 	68 	0 	294 	24 	0 	0 	404
simple-polygon-exterior_100_686dd044 	0 	95 	0 	95 	111 	0 	111 	0 	94 	192 	0 	119 	111 	0 	0 	1923 	38 	2582 	149 	0 	111 	0
simple-polygon-exterior-20_60_e6f13145 	0 	73 	0 	73 	82 	0 	82 	0 	78 	15 	0 	109 	82 	0 	0 	1075 	20 	369 			31 	535
simple-polygon-exterior_40_785575e7 	0 	30 	0 	30 	33 	0 	33 	0 	13 	77 	0 	31 	33 	0 	0 	774 	3 	148 	57 	0 	29 	419
point-set_10_4bcb7c21 	0 	6 	0 	6 	6 	0 	6 	0 	0 	7 	0 	7 	6 	0 	0 	13 	0 	13 	11 	0 	0 	9
simple-polygon_40_12969fc3 	0 	14 	0 	14 	23 	0 	23 	0 	0 	33 	0 	16 	23 	0 	0 	119 	0 	158 	34 	0 	0 	359
simple-polygon-exterior-20_20_0f96fb2c 	0 	18 	0 	18 	19 	0 	19 	0 	16 	16 	0 	19 	19 	0 	0 	164 	0 	73 	23 	0 	16 	443
simple-polygon-exterior-20_250_0c9fa44e 	0 	203 	0 	203 	289 	0 	289 	0 	263 	103 	0 	331 	289 	0 	0 	4698 	99 	1194 	400 	0 	113 	411
simple-polygon-exterior-20_20_eb63a5b6 	0 	21 	0 	21 	21 	0 	21 	0 	0 	35 	0 	22 	21 	0 	0 	65 	0 	61 	24 	0 	0 	388
simple-polygon-exterior_150_fb998503 	0 	130 	0 	130 	176 	0 	176 	0 	163 	132 	0 	182 	176 	0 	0 	2767 	66 	666 	224 	0 	176 	0
point-set_250_93dd622a 	0 	84 	0 	84 	121 	0 	121 	0 	116 	2 	0 	119 	121 	0 	0 	2745 	0 	476 			5 	592
simple-polygon-exterior-20_10_8c4306da 	0 	9 	0 	9 	8 	0 	8 	0 	0 	13 	0 	9 	8 	0 	0 	16 	0 	16 	9 	0 	0 	234
point-set_100_0245ce31 	0 	29 	0 	29 	32 	0 	32 	0 	30 	2 	0 	29 	32 	0 	0 	965 	0 	39 			0 	73
simple-polygon-exterior_10_40642b31 	0 	8 	0 	8 	7 	0 	7 	0 	0 	29 	0 	8 	7 	0 	0 	15 	0 	15 	9 	0 	0 	307
simple-polygon_10_f2c8d74a 	0 	5 	0 	5 	5 	0 	5 	0 	0 	11 	0 	5 	5 	0 	0 	12 	0 	12 	7 	0 	0 	224
simple-polygon_100_4ee8f447 	0 	33 	0 	33 	49 	0 	49 	0 	46 	2 	0 	40 	49 	0 	0 	798 	0 	619 	71 	0 	18 	185
simple-polygon-exterior-20_80_1c5fcde7 	0 	70 	0 	70 	87 	0 	87 	0 	45 	170 	0 	86 	87 	0 	0 	1298 	8 	379 	118 	0 	0 	402
simple-polygon-exterior_40_7685a35e 	0 	34 	0 	34 	37 	0 	37 	0 	22 	106 	0 	45 	37 	0 	0 	687 	2 	215 	55 	0 	6 	371
simple-polygon_20_4bd3c2e5 	0 	5 	0 	5 	9 	2 	10 	0 	0 	10 	0 	5 	10 	0 	0 	35 	0 	49 	11 	0 	0 	344
point-set_100_dd67678e 	0 	62 	0 	62 	89 	0 	89 	0 	58 	150 	0 	75 	89 	0 	0 	1263 	0 	1031 			17 	1273
ortho_20_5a9e8244 	0 	6 	0 	6 	9 	0 	9 	0 	0 	9 	0 	6 	9 	0 	0 	24 	0 	13 	0 	49 	0 	8
simple-polygon-exterior_100_37aaf06f 	0 	94 	0 	94 	121 	0 	121 	0 	98 	104 	0 	124 	121 	0 	0 	1912 	11 	537 	148 	0 	121 	0
simple-polygon-exterior_10_c5616894 	0 	14 	0 	14 	12 	0 	12 	0 	3 	28 	0 	15 	12 	0 	0 	35 	0 	33 	12 	0 	0 	269
point-set_150_1fb326cf 	0 	121 	0 	121 	164 	0 	164 	0 	163 	6 	0 	194 	164 	0 	0 	1840 	0 	2812 			73 	3165
point-set_80_9a8373fb 	0 	29 	0 	29 	35 	0 	35 	0 	23 	17 	0 	31 	35 	0 	0 	841 	0 	55 			0 	93
simple-polygon-exterior_250_3f7ba0d3 	0 	216 	0 	216 	280 	0 	280 	0 	226 	336 	0 	296 	280 	0 	0 	5159 	168 	3228 	389 	0 	104 	411
ortho_60_c423f527 	0 	16 	0 	16 	27 	0 	27 	0 	0 	56 	0 	17 	27 	0 	0 	136 	0 	77 	34 	0 	0 	124
simple-polygon-exterior-20_80_f6e462fb 	0 	75 	0 	75 	90 	0 	90 	0 	76 	145 	0 	101 	90 	0 	0 	1344 	10 	350 	131 	0 	16 	388
point-set_150_b26a2c80 	0 	95 	0 	95 	153 	0 	153 	0 	93 	315 	0 	129 	153 	0 	0 	2104 	0 	10530 			0 	447
simple-polygon-exterior_80_d87f15e8 	0 	71 	0 	71 	87 	0 	87 	0 	65 	183 	0 	87 	87 	0 	0 	1588 	9 	461 	112 	0 	87 	0
simple-polygon-exterior_20_7520a1da 	0 	15 	0 	15 	16 	0 	16 	0 	5 	37 	0 	15 	16 	0 	0 	60 	0 	54 	24 	0 	16 	0
ortho_80_06ee55d4 	0 	22 	0 	22 	36 	0 	36 	0 	0 	43 	0 	25 	36 	0 	0 	203 	0 	68 	39 	0 	0 	108
point-set_150_20bcb550 	0 	77 	0 	77 	118 	0 	118 	0 	93 	62 	0 	108 	118 	0 	0 	1842 	0 	534 			11 	638
point-set_80_8383fead 	0 	58 	0 	58 	86 	0 	86 	0 	44 	125 	0 	63 	86 	0 	0 	1056 	0 	3384 			0 	489
simple-polygon-exterior-20_20_e64ff8fc 	0 	20 	0 	20 	22 	0 	22 	0 	14 	21 	0 	24 	22 	0 	0 	130 	0 	137 	25 	0 	0 	351
simple-polygon-exterior_20_92dcd467 	0 	19 	0 	20 	15 	0 	15 	0 	1 	70 	0 	19 	15 	0 	0 	159 	0 	68 	24 	0 	15 	0
point-set_10_97578aae 	0 	5 	0 	5 	6 	0 	6 	0 	0 	11 	0 	5 	6 	0 	0 	13 	0 	14 	2 	115 	0 	22
simple-polygon-exterior-20_60_53ad6d23 	0 	32 	0 	32 	45 	0 	45 	0 	22 	102 	0 	38 	45 	0 	0 	967 	0 	3665 	82 	0 	30 	495
simple-polygon-exterior_80_22d34c7e 	0 	74 	0 	74 	89 	0 	89 	0 	64 	161 	0 	103 	89 	0 	0 	1539 	14 	488 	123 	0 	31 	430
point-set_40_8cbf31aa 	0 	24 	0 	24 	30 	0 	30 	0 	0 	74 	0 	27 	30 	0 	0 	406 	0 	257 			0 	433
simple-polygon-exterior-20_250_823ed1ae 	0 	207 	0 	207 	283 	0 	283 	0 	274 	101 	0 	308 	283 	0 	0 	4058 	104 	845 	400 	0 	119 	358
point-set_40_f511c8ce 	0 	32 	0 	32 	44 	0 	44 	0 	29 	35 	0 	39 	44 	0 	0 	468 	0 	1661 	65 	0 	0 	276
simple-polygon-exterior_40_ff947945 	0 	38 	0 	38 	49 	0 	49 	0 	35 	22 	0 	50 	49 	0 	0 	749 	2 	179 	59 	0 	49 	0
ortho_60_5c5796a0 	0 	20 	0 	20 	31 	0 	31 	0 	0 	37 	0 	20 	31 	0 	0 	112 	0 	57 	34 	0 	0 	85
simple-polygon-exterior-20_150_7768dd44 	0 	123 	0 	123 	161 	0 	161 	0 	137 	260 	0 	190 	161 	0 	0 	2715 	38 	464 	249 	0 	161 	0
simple-polygon-exterior-20_10_6fbd9669 	0 	6 	0 	6 	6 	0 	6 	0 	0 	14 	0 	6 	6 	0 	0 	11 	0 	11 	9 	0 	0 	239
point-set_10_7451a2a9 	0 	6 	0 	6 	5 	0 	5 	0 	0 	26 	0 	6 	5 	0 	0 	8 	0 	8 	1 	17 	0 	7
simple-polygon-exterior-20_10_868921c7 	0 	8 	0 	8 	7 	0 	7 	0 	0 	11 	0 	8 	7 	0 	0 	13 	0 	13 	12 	0 	0 	288
simple-polygon_150_b42a5724 	0 	84 	0 	84 	109 	0 	109 	0 	82 	123 	0 	102 	109 	0 	50 	495 	0 	2506 	129 	0 	48 	318
simple-polygon-exterior-20_40_b9ab4f03 	0 	31 	0 	31 	47 	0 	47 	0 	21 	70 	0 	35 	47 	0 	0 	546 	0 	182 	59 	0 	0 	398
simple-polygon_250_432b4814 	0 	67 	0 	67 	128 	0 	128 	0 	5 	215 	0 	72 	128 	0 	0 	1086 	0 	1225 	192 	0 	21 	179
simple-polygon_250_c02755d7 	0 	75 	0 	75 	119 	0 	119 	0 	19 	215 	0 	81 	119 	0 	0 	600 	0 	662 	202 	0 	38 	264
simple-polygon-exterior-20_250_eb5ab92f 	0 	180 	0 	180 	253 	0 	253 	0 	239 	25 	0 	310 	253 	0 	0 	4786 	0 	22817 	405 	0 	253 	0
simple-polygon-exterior-20_40_65de7236 	0 	35 	0 	35 	41 	0 	41 	0 	26 	95 	0 	41 	41 	0 	0 	694 	3 	138 	57 	0 	41 	0
point-set_20_41c48315 	0 	15 	0 	15 	20 	0 	20 	0 	11 	60 	0 	17 	20 	0 	0 	103 	0 	87 	19 	127 	0 	332
simple-polygon_100_6101abad 	0 	29 	0 	29 	37 	0 	37 	0 	0 	62 	0 	30 	37 	0 	0 	144 	0 	154 	66 	0 	7 	214
point-set_80_837b0f11 	0 	66 	0 	66 	103 	93 	105 	0 	100 	46 	0 	82 	105 	0 	0 	1098 	0 	1532 			0 	487
simple-polygon-exterior-20_20_52abeef5 	0 	19 	0 	19 	24 	0 	24 	0 	0 	60 	0 	20 	24 	0 	0 	311 	0 	184 	27 	0 	0 	323
ortho_60_f744490d 	0 	15 	0 	15 	23 	0 	23 	0 	0 	66 	0 	17 	23 	0 	0 	113 	0 	102 	31 	0 	0 	171
simple-polygon-exterior_250_c0a19392 	0 	225 	0 	225 	302 	0 	302 	0 	256 	420 	0 	330 	302 	0 	0 	4827 	90 	1278 	401 	0 	302 	0
simple-polygon-exterior-20_10_46c44a43 	0 	8 	0 	8 	8 	0 	8 	0 	0 	23 	0 	9 	8 	0 	0 	21 	0 	21 	9 	0 	0 	39
ortho_40_df58ce3b 	0 	11 	0 	11 	13 	0 	13 	0 	0 	17 	0 	11 	13 	0 	0 	36 	0 	37 	19 	0 	0 	71
point-set_20_5868538a 	0 	13 	0 	13 	15 	0 	15 	0 	7 	52 	0 	14 	15 	0 	0 	72 	0 	106 	26 	0 	0 	22
point-set_60_9fc02edd 	0 	46 	0 	46 	80 	0 	80 	0 	52 	79 	0 	48 	80 	0 	0 	747 	0 	291 			0 	453
simple-polygon_80_7b8f6c4c 	0 	20 	0 	22 	22 	0 	22 	0 	10 	59 	0 	20 	22 	0 	0 	132 	0 	128 	40 	0 	0 	370
simple-polygon_60_0347cd75 	0 	24 	0 	24 	33 	0 	33 	0 	0 	37 	0 	26 	33 	0 	0 	173 	0 	198 	47 	0 	0 	274
simple-polygon-exterior-20_60_57cd1db6 	0 	46 	0 	46 	64 	0 	64 	0 	56 	19 	0 	55 	64 	0 	0 	1156 	0 	411 	85 	0 	30 	327
simple-polygon-exterior-20_150_41e0c5f0 	0 	126 	0 	126 	165 	0 	165 	0 	153 	24 	0 	176 	165 	0 	0 	2777 	34 	725 	238 	0 	76 	380
point-set_40_ae33a7ea 	0 	31 	0 	31 	38 	0 	38 	0 	33 	19 	0 	34 	38 	0 	0 	482 	0 	1940 	64 	0 	0 	359
point-set_80_1675b331 	0 	38 	0 	38 	54 	0 	54 	0 	3 	132 	0 	40 	54 	0 	0 	864 	0 	1568 			0 	412
simple-polygon-exterior_60_494df14f 	0 	52 	0 	52 	69 	0 	69 	0 	50 	92 	0 	69 	69 	0 	0 	1071 	0 	17056 	86 	0 	69 	0
simple-polygon-exterior_250_a97729dd 	0 	203 	0 	203 	259 	0 	259 	0 	227 	200 	0 	362 	259 	0 	0 	5117 	0 	25398 	395 	0 	259 	0
simple-polygon-exterior_60_1a49dffa 	0 	52 	0 	52 	58 	0 	58 	0 	41 	134 	0 	59 	58 	0 	0 	1072 	2 	277 	75 	0 	18 	393
simple-polygon_100_4b4ba391 	0 	20 	0 	20 	23 	0 	23 	0 	0 	45 	0 	26 	23 	0 	0 	62 	0 	65 	61 	0 	0 	261
simple-polygon-exterior_10_74050e4d 	0 	7 	0 	7 	7 	0 	7 	0 	0 	15 	0 	8 	7 	0 	0 	22 	0 	22 	9 	0 	0 	12
simple-polygon-exterior-20_60_57858065 	0 	51 	0 	51 	66 	0 	66 	0 	52 	126 	0 	59 	66 	0 	0 	959 	2 	309 	96 	0 	20 	339
ortho_250_3b977f7e 	0 	76 	0 	76 	109 	0 	109 	0 	0 	257 	0 	82 	109 	0 	0 	567 	0 	374 	142 	0 	0 	593
simple-polygon-exterior-20_100_512f0fc4 	0 	54 	0 	54 	65 	0 	65 	0 	27 	153 	0 	61 	65 	0 	0 	1721 	0 	3767 	144 	0 	34 	441
simple-polygon_250_6e9d9c26 	0 	36 	0 	36 	51 	0 	51 	0 	37 	108 	0 	49 	51 	0 	24 	228 	0 	206 	183 	0 	39 	272
simple-polygon-exterior_10_a5f0f2fc 	0 	6 	0 	7 	7 	0 	7 	0 	0 	33 	0 	6 	7 	0 	0 	24 	0 	25 	8 	0 	7 	0
simple-polygon-exterior_20_c820ed5d 	0 	21 	0 	21 	21 	0 	21 	0 	13 	55 	0 	25 	21 	0 	0 	85 	0 	123 	25 	0 	0 	280
ortho_100_bd1e4a14 	0 	33 	0 	33 	43 	0 	43 	0 	0 	112 	0 	33 	43 	0 	0 	182 	0 	146 	60 	0 	0 	203
simple-polygon-exterior_60_8670ab75 	0 	52 	0 	52 	61 	0 	61 	0 	43 	74 	0 	57 	61 	0 	0 	1114 	0 	6065 	77 	0 	32 	503
simple-polygon-exterior-20_40_8ad14096 	0 	37 	0 	37 	39 	0 	39 	0 	34 	31 	0 	40 	39 	0 	0 	637 	3 	164 	60 	0 	0 	392"""
    db = [[line.split("\t")[i] for i in [0,4,12]] for line in result.split("\n")]
    print(db)
    worse = []
    equal = []
    better = []
    for name,our,other in db:
        if int(our) < int(other):
            better.append([name,int(our),int(other)])
        elif int(our) == int(other):
            equal.append([name,int(our),int(other)])
        else:
            worse.append([name,int(our),int(other)])
    print(f"proper best: {len(better)}")
    print(f"tied: {len(equal)}")
    print(f"worse: {len(worse)}")


if __name__=="__main__":



    resultSummary()

    #showSolutions()
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S", level=logging.INFO)

    updateSummaries()

    #plotHistory()

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
    new3Old = filepath.parent.parent/"instance_solutions"/"288CircleArrNoOutRNoSegsOld"
    new3 = filepath.parent.parent/"instance_solutions"/"288CircleArrNoOutRNoSegs"
    new4 = filepath.parent.parent/"instance_solutions"/"288CircleArr2OutRNoSegs"
    new5 = filepath.parent.parent/"instance_solutions"/"withDepth1Greedy"
    merged = filepath.parent.parent/"instance_solutions"/"merged_summaries"
    merged_3 = filepath.parent.parent/"instance_solutions"/"merged_summaries_3"
    merged_5 = filepath.parent.parent/"instance_solutions"/"merged_summaries_5"
    merged_bak = filepath.parent.parent/"instance_solutions"/"merged_summaries_bak"
    mergemerge = filepath.parent.parent/"out_merged_summaries"

    allexceptnumeric = [exact_solutions,new,seeded,seededEndFace,seededFace,withFace,withComplicatedCenter,output,gigaSeeded,withConstrainedVoronoi]
        #allexceptnumeric = allexceptnumeric + [v for v in list.iterdir()]

    #compareSolutions(base=[v for v in seeded.iterdir() if len([w for w in out.iterdir() if v.name == w.name])>0],others=[v for v in out.iterdir()])
    againstMergeBak = False
    #plotByType([mergemerge,merged,merged_3,merged_5])#[merged,merged_3,merged_5,merged_bak,mergemerge])
    if againstMergeBak:
        compareSolutions(others=[mergemerge],base=[new5])#
    else:
        compareSolutions(others=[new5],base=[new1])  #
        #compareSolutions(others=[new5, new4, new3, new3Old],base= allexceptnumeric)  #

    #compareSolutions(base=[v for v in seeded.iterdir()],others=[v for v in out.iterdir()])
