import copy

from cgshop2025_pyutils.geometry import FieldNumber, Point
import exact_geometry as eg
import numpy as np

class KDTreeNode:
    def __init__(self,left=None,right=None):
        self.left = left
        self.right = right

class KDTree:
    def __init__(self,points,keys,leafsize=10,depth=0):
        self.leafsize = leafsize
        if len(points)<= leafsize:
            self.root = None
            self.points = points
            self.keys = keys
            self.size = len(points)
            self.axis = depth%2
            self.depth = depth

            self.updateBB()
        else:
            axis = depth % 2
            args = points[:,axis].argsort()
            points = points[args]
            keys = keys[args]
            #points.sort(key=lambda p: p.x() if axis == 0 else p.y())
            median_index = len(points) // 2

            median = points[median_index]
            key = keys[median_index]

            left_subtree = KDTree(points[:median_index],keys[:median_index],leafsize,depth+1)
            right_subtree = KDTree(points[median_index:],keys[median_index:],leafsize,depth+1)

            self.root = KDTreeNode(left_subtree,right_subtree)
            self.points = None
            self.keys = None
            self.size = left_subtree.size + right_subtree.size
            self.axis = depth%2
            self.depth = depth

            self.updateBB()

    def isEmpty(self):
        return self.size == 0

    def isLeaf(self):
        return self.root is None and self.points is not None

    def isInner(self):
        return self.root is not None and self.points is None

    def isValid(self):
        return self.isLeaf() is not self.isInner()

    def distToBB(self,point):
        cx = max(self.minX,min(point.x(),self.maxX))
        cy = max(self.minY,min(point.y(),self.maxY))
        dx = point.x() - cx
        dy = point.y() - cy
        return (dx*dx)+(dy*dy)

    def queryKeys(self,point,distance):
        r = []
        if self.isEmpty():
            return r
        if self.distToBB(point) > distance:
            return r
        if self.isLeaf():
            for i in range(len(self.points)):
                if eg.distsq(point,Point(self.points[i,0],self.points[i,1])) <= distance:
                    r.append(self.keys[i])
            return r
        #inner
        return self.root.left.queryKeys(point,distance) + self.root.right.queryKeys(point,distance)

    def bbExtremes(self):
        return [Point(self.minX,self.minY),Point(self.minX,self.maxY),Point(self.maxX,self.minY),Point(self.maxX,self.maxY)]

    def updateBB(self):
        if self.isEmpty():
            self.minX = None
            self.maxX = None
            self.minY = None
            self.maxY = None
        elif self.isLeaf():
            self.minX = np.min(self.points[:, 0])
            self.maxX = np.max(self.points[:, 0])
            self.minY = np.min(self.points[:, 1])
            self.maxY = np.max(self.points[:, 1])
        elif self.isInner():
            self.minX = np.min([v for v in [self.root.left.minX, self.root.right.minX] if v is not None])
            self.maxX = np.max([v for v in [self.root.left.maxX, self.root.right.maxX] if v is not None])
            self.minY = np.min([v for v in [self.root.left.minY, self.root.right.minY] if v is not None])
            self.maxY = np.max([v for v in [self.root.left.maxY, self.root.right.maxY] if v is not None])
        else:
            assert(False)
        #self.validateBB()

    def validateBB(self):

        if self.isEmpty():
            return self.minX == None and self.maxX == None and self.minY == None and self.maxY == None

        if self.isLeaf():
            for p in self.points:
                if p[0] < self.minX or p[0] > self.maxX:
                    return False
                if p[1] < self.minY or p[1] > self.maxY:
                    return False
            return True

        if not self.root.left.isEmpty():
            for p in self.root.left.bbExtremes():
                if p[0] < self.minX or p[0] > self.maxX:
                    return False
                if p[1] < self.minY or p[1] > self.maxY:
                    return False

        if not self.root.right.isEmpty():
            for p in self.root.right.bbExtremes():
                if p[0] < self.minX or p[0] > self.maxX:
                    return False
                if p[1] < self.minY or p[1] > self.maxY:
                    return False

        return True

    def query(self,point,distance,segments=None,sides=None,validateTuple=None):

        #if not self.validateBB():
        #    assert(False)

        #if segments == None:
        #    return self.queryKeys(point,distance)

        r = []

        if self.isEmpty():
            return r

        if self.isLeaf():
            for i in range(len(self.points)):
                myPoint = Point(self.points[i,0],self.points[i,1])

                if eg.distsq(point,myPoint) > distance:
                    continue

                isOnWrongSide = False
                for seg,side in zip(segments,sides):
                    mySide = eg.onWhichSide(seg,myPoint)
                    if mySide != eg.COLINEAR and mySide != side:
                        isOnWrongSide = True
                        break
                if isOnWrongSide:
                    continue

                r.append(self.keys[i])
            return r

        #inner

        #intersection with bb check
        if self.distToBB(point) > distance:
            return r

        if len(segments) != 0:
            bbExs = self.bbExtremes()
            for seg,side in zip(segments,sides):
                allOnWrongSide = True
                for bbEx in bbExs:
                    mySide = eg.onWhichSide(seg,bbEx)
                    if mySide == eg.COLINEAR or mySide == side:
                        allOnWrongSide = False
                        break
                if allOnWrongSide:
                    return r

        if validateTuple is None:
            return self.root.left.query(point,distance,segments,sides,validateTuple) + self.root.right.query(point,distance,segments,sides,validateTuple)

        lq = self.root.left.query(point,distance,segments,sides,validateTuple)
        if validateTuple != None:
            if self.root.left.searchKey(validateTuple)[0]:
                assertMe = True
                for tri in lq:
                    if np.all(np.array(tri) == np.array(validateTuple)):
                        assertMe = False
                if assertMe:
                    assert(False)

        rq = self.root.right.query(point,distance,segments,sides,validateTuple)
        if validateTuple != None:
            if self.root.right.searchKey(validateTuple)[0]:
                assertMe = True
                for tri in rq:
                    if np.all(np.array(tri) == np.array(validateTuple)):
                        assertMe = False
                if assertMe:
                    assert(False)

        return lq + rq

    def searchKey(self,key):
        if self.isLeaf():
            for i in range(len(self.keys)):
                myKey = self.keys[i]
                if np.all(np.array(key) == np.array(myKey)):
                    return True, self.points[i]
            return False, None
        lq = self.root.left.searchKey(key)
        rq = self.root.right.searchKey(key)
        bans = lq[0] or rq[0]
        pans = None if bans == False else lq[1] if lq[0] == True else rq[1]
        return bans,pans

    def addPoint(self,point:Point,key):
        if self.isLeaf():
            if len(self.points) + 1 <= self.leafsize:
                self.root = None
                self.points = np.append(self.points,[point],axis=0)
                self.keys = np.append(self.keys,[key],axis=0)
                self.size += 1

                self.updateBB()
            else:
                #expand node
                points = np.append(self.points,[point],axis=0)
                keys = np.append(self.keys,[key],axis=0)

                self.points = None
                self.keys = None

                args = points[:, self.axis].argsort()
                points = points[args]
                keys = keys[args]
                # points.sort(key=lambda p: p.x() if axis == 0 else p.y())
                median_index = len(points) // 2

                median = points[median_index]
                key = keys[median_index]

                left_subtree = KDTree(points[:median_index], keys[:median_index], self.leafsize, self.depth + 1)
                right_subtree = KDTree(points[median_index:], keys[median_index:], self.leafsize, self.depth + 1)

                self.root = KDTreeNode(left_subtree, right_subtree)

                self.size = left_subtree.size + right_subtree.size

                self.updateBB()

        else:
            if not self.isValid():
                assert(False)
            if self.root is None:
                assert(False)
            mustLeft = False
            mustRight = False
            if self.axis == 0:
                #X
                if (not self.root.left.isEmpty()) and self.root.left.maxX > point.x():
                    mustLeft = True
                if (not self.root.right.isEmpty()) and self.root.right.minX < point.x():
                    mustRight = True
            else:
                #Y
                if (not self.root.left.isEmpty()) and self.root.left.maxY > point.y():
                    mustLeft = True
                if (not self.root.right.isEmpty()) and self.root.right.minY < point.y():
                    mustRight = True
            if mustLeft:
                self.root.left.addPoint(point,key)
                self.size += 1
            elif mustRight:
                self.root.right.addPoint(point,key)
                self.size += 1
            else:
                if self.root.left.size < self.root.right.size:
                    self.root.left.addPoint(point,key)
                else:
                    self.root.right.addPoint(point,key)
                self.size += 1

            self.updateBB()
        pass

    def removePoint(self,point:Point):

        if self.isEmpty():
            return False

        if not (self.minX <= point.x() <= self.maxX and self.minY <= point.y() <= self.maxY):
            return False

        if self.isLeaf():
            myIdx = None
            for i in range(len(self.points)):
                if self.points[i][0] == point.x() and self.points[i][1] == point.y():
                    myIdx = i
                    break

            if myIdx == None:
                return False

            self.points = np.delete(self.points,myIdx,0)
            self.keys = np.delete(self.keys,myIdx,0)

            self.size -= 1
            self.updateBB()
            return True

        else:
            deleted = self.root.left.removePoint(point) or self.root.right.removePoint(point)
            if not deleted:
                return False

            self.size = self.root.left.size + self.root.right.size

            if self.size <= self.leafsize:
                #both children are already leafs
                points = np.array(list(self.root.left.points) + list(self.root.right.points))
                keys = np.array(list(self.root.left.keys) + list(self.root.right.keys))

                self.root = None
                self.points = points
                self.keys = keys

                self.size = len(points)

                self.updateBB()
            else:

                self.updateBB()
            return True

def fromDictGetPoint(key,keyDict):
    return keyDict.getByKey(tuple([key[0],key[1]]))[key[2]]

def fromDictGetPoints(keys,keyDict):
    return [fromDictGetPoint(key,keyDict) for key in keys]

def inside(point,distance,segments,sides,qPoint):
    if eg.distsq(point,qPoint) > distance:
        return False
    for seg,side in zip(segments,sides):
        mySide = eg.onWhichSide(seg,qPoint)
        if mySide != eg.COLINEAR and mySide != side:
            return False
    return True

class combinatorialKDTree:
    def __init__(self,keys,keyDict,leafsize=10,depth=0,stashDepth=5):
        #consts:
        self.leafsize = leafsize
        self.stashDepth = stashDepth
        self.stashId = None
        self.depth = depth

        self.keys = keys
        self.root = None
        self.axis = None
        self.size = len(keys)

        if len(self.keys) > leafsize:
            self.expand(keyDict)

        #bouding box
        self.minXKey = None
        self.maxXKey = None
        self.minYKey = None
        self.maxYKey = None

        self.updateBB(keyDict)

    def expand(self,keyDict):

        points = np.array(fromDictGetPoints(self.keys, keyDict))
        xdiff = np.max(points[:,0]) - np.min(points[:,0])
        ydiff = np.max(points[:,1]) - np.min(points[:,1])
        self.axis = 0 if xdiff > ydiff else 1
        args = points[:, self.axis].argsort()
        points = points[args]
        keys = self.keys[args]
        # points.sort(key=lambda p: p.x() if axis == 0 else p.y())
        median_index = len(points) // 2

        if self.isAtStashDepth():
            left_subtree_id = keyDict.addObject(combinatorialKDTree(keys[:median_index], keyDict, self.leafsize, 1 if self.depth % 2 == 0 else 0, self.stashDepth))
            right_subtree_id = keyDict.addObject(combinatorialKDTree(keys[median_index:], keyDict, self.leafsize,  1 if self.depth % 2 == 0 else 0, self.stashDepth))

            self.root = KDTreeNode(left_subtree_id, right_subtree_id)
        else:
            left_subtree = combinatorialKDTree(keys[:median_index], keyDict, self.leafsize, self.depth + 1, self.stashDepth)
            right_subtree = combinatorialKDTree(keys[median_index:], keyDict, self.leafsize, self.depth + 1, self.stashDepth)

            self.root = KDTreeNode(left_subtree, right_subtree)
        self.keys = None

    def contract(self,keyDict):
        #size and bounding box should be unchanged
        if self.isAtStashDepth():
            self.keys = np.array(list(keyDict.getById(self.root.left).getKeys(keyDict)) + list(keyDict.getById(self.root.right).getKeys(keyDict)))
        else:
            self.keys = np.array(list(self.root.left.getKeys(keyDict)) +list(self.root.right.getKeys(keyDict)))
        self.root = None
        self.axis = None

    def getKeys(self,keyDict):
        if self.isEmpty():
            return []
        if self.isLeaf():
            return self.keys
        if self.isInner():
            if self.isAtStashDepth():
                return np.array(list(keyDict.getById(self.root.left).getKeys(keyDict)) + list(keyDict.getById(self.root.right).getKeys(keyDict)))
            else:
                return np.array(list(self.root.left.getKeys(keyDict)) +list(self.root.right.getKeys(keyDict)))

    def isEmpty(self):
        return self.size == 0

    def isLeaf(self):
        return (self.root is None and self.stashId is None) and self.keys is not None

    def isInner(self):
        return self.keys is None and (self.stashId is not None or self.root is not None)

    def isAtStashDepth(self):
        return self.depth == self.stashDepth

    def distToBB(self,point,keyDict):
        cx = max(fromDictGetPoint(self.minXKey,keyDict).x(),min(point.x(),fromDictGetPoint(self.maxXKey,keyDict).x()))
        cy = max(fromDictGetPoint(self.minYKey,keyDict).y(),min(point.y(),fromDictGetPoint(self.maxYKey,keyDict).y()))
        dx = point.x() - cx
        dy = point.y() - cy
        return (dx*dx)+(dy*dy)

    def bbExtremes(self,keyDict):
        return [Point(fromDictGetPoint(self.minXKey,keyDict).x(),fromDictGetPoint(self.minYKey,keyDict).y()),
                Point(fromDictGetPoint(self.minXKey,keyDict).x(),fromDictGetPoint(self.maxYKey,keyDict).y()),
                Point(fromDictGetPoint(self.maxXKey,keyDict).x(),fromDictGetPoint(self.minYKey,keyDict).y()),
                Point(fromDictGetPoint(self.maxXKey,keyDict).x(),fromDictGetPoint(self.maxYKey,keyDict).y())]

    def updateBB(self,keyDict):
        if self.isEmpty():
            self.minXKey = None
            self.maxXKey = None
            self.minYKey = None
            self.maxYKey = None

        elif self.isLeaf():
            points = np.array(fromDictGetPoints(self.keys,keyDict))
            if len(points) == 0:
                assert(False)
            self.minXKey = self.keys[np.argmin(points[:, 0])]
            self.maxXKey = self.keys[np.argmax(points[:, 0])]
            self.minYKey = self.keys[np.argmin(points[:, 1])]
            self.maxYKey = self.keys[np.argmax(points[:, 1])]

        elif self.isInner():
            if self.isAtStashDepth():
                if keyDict.getById(self.root.left).isEmpty():
                    self.minXKey = keyDict.getById(self.root.right).minXKey
                    self.maxXKey = keyDict.getById(self.root.right).maxXKey
                    self.minYKey = keyDict.getById(self.root.right).minYKey
                    self.maxYKey = keyDict.getById(self.root.right).maxYKey
                elif keyDict.getById(self.root.right).isEmpty():
                    self.minXKey = keyDict.getById(self.root.left).minXKey
                    self.maxXKey = keyDict.getById(self.root.left).maxXKey
                    self.minYKey = keyDict.getById(self.root.left).minYKey
                    self.maxYKey = keyDict.getById(self.root.left).maxYKey
                else:
                    self.minXKey = [keyDict.getById(self.root.left).minXKey,keyDict.getById(self.root.right).minXKey][np.argmin([fromDictGetPoint(keyDict.getById(self.root.left).minXKey,keyDict).x(),fromDictGetPoint(keyDict.getById(self.root.right).minXKey,keyDict).x()])]
                    self.maxXKey = [keyDict.getById(self.root.left).maxXKey,keyDict.getById(self.root.right).maxXKey][np.argmax([fromDictGetPoint(keyDict.getById(self.root.left).maxXKey,keyDict).x(),fromDictGetPoint(keyDict.getById(self.root.right).maxXKey,keyDict).x()])]
                    self.minYKey = [keyDict.getById(self.root.left).minYKey,keyDict.getById(self.root.right).minYKey][np.argmin([fromDictGetPoint(keyDict.getById(self.root.left).minYKey,keyDict).y(),fromDictGetPoint(keyDict.getById(self.root.right).minYKey,keyDict).y()])]
                    self.maxYKey = [keyDict.getById(self.root.left).maxYKey,keyDict.getById(self.root.right).maxYKey][np.argmax([fromDictGetPoint(keyDict.getById(self.root.left).maxYKey,keyDict).y(),fromDictGetPoint(keyDict.getById(self.root.right).maxYKey,keyDict).y()])]
            else:
                if self.root.left.isEmpty():
                    self.minXKey = self.root.right.minXKey
                    self.maxXKey = self.root.right.maxXKey
                    self.minYKey = self.root.right.minYKey
                    self.maxYKey = self.root.right.maxYKey
                elif self.root.right.isEmpty():
                    self.minXKey = self.root.left.minXKey
                    self.maxXKey = self.root.left.maxXKey
                    self.minYKey = self.root.left.minYKey
                    self.maxYKey = self.root.left.maxYKey
                else:
                    self.minXKey = [self.root.left.minXKey,self.root.right.minXKey][np.argmin([fromDictGetPoint(self.root.left.minXKey,keyDict).x(),fromDictGetPoint(self.root.right.minXKey,keyDict).x()])]
                    self.maxXKey = [self.root.left.maxXKey,self.root.right.maxXKey][np.argmax([fromDictGetPoint(self.root.left.maxXKey,keyDict).x(),fromDictGetPoint(self.root.right.maxXKey,keyDict).x()])]
                    self.minYKey = [self.root.left.minYKey,self.root.right.minYKey][np.argmin([fromDictGetPoint(self.root.left.minYKey,keyDict).y(),fromDictGetPoint(self.root.right.minYKey,keyDict).y()])]
                    self.maxYKey = [self.root.left.maxYKey,self.root.right.maxYKey][np.argmax([fromDictGetPoint(self.root.left.maxYKey,keyDict).y(),fromDictGetPoint(self.root.right.maxYKey,keyDict).y()])]
        else:
            assert(False)

    def query(self,point,distance,keyDict,segments=None,sides=None,validateTuple=None,passThrough = False):

        if passThrough:
            if self.isLeaf():
                return [k for k in self.keys]
            else:
                if self.isAtStashDepth():
                    keyDict.getById(self.root.left).query(point, distance, keyDict, segments, sides, validateTuple,
                                         True) + keyDict.getById(self.root.left).query(point, distance, keyDict, segments,
                                                                       sides, validateTuple, True)
                else:
                    return self.root.left.query(point, distance, keyDict, segments, sides, validateTuple,
                                            True) + self.root.right.query(point, distance, keyDict, segments,
                                                                               sides, validateTuple, True)

        if self.isEmpty():
            return []

        if self.isLeaf():
            return [key for key in self.keys if inside(point,distance,segments,sides,fromDictGetPoint(key,keyDict))]

        #inner

        #check if the bounding box is outside the circle
        if self.distToBB(point,keyDict) > distance:
            return []

        #check if the bounding box lies outside the polygon defined by segments and side
        bbExs = self.bbExtremes(keyDict)
        for seg,side in zip(segments,sides):
            allOnWrongSide = True
            for bbEx in bbExs:
                mySide = eg.onWhichSide(seg,bbEx)
                if mySide == side or mySide == eg.COLINEAR:
                    allOnWrongSide = False
                    break
            if allOnWrongSide:
                return []

        #check for passthrough
        for bbEx in bbExs:
            if not inside(point,distance,segments,sides,bbEx):
                #an extreme is not inside, thus we need to properly query the two children
                if self.isAtStashDepth():
                    lq = keyDict.getById(self.root.left).query(point, distance, keyDict, segments, sides, validateTuple, False)
                    rq = keyDict.getById(self.root.right).query(point, distance, keyDict, segments, sides, validateTuple, False)
                    return lq + rq
                else:
                    lq = self.root.left.query(point, distance, keyDict, segments, sides, validateTuple, False)
                    rq = self.root.right.query(point, distance, keyDict, segments, sides, validateTuple, False)
                    return lq + rq

        #all extremes are inside, we simply list everything...
        if self.isAtStashDepth():
            lq = keyDict.getById(self.root.left).query(point, distance, keyDict, segments, sides, validateTuple, True)
            rq = keyDict.getById(self.root.right).query(point, distance, keyDict, segments, sides, validateTuple, True)
            return lq + rq
        else:
            lq = self.root.left.query(point, distance, keyDict, segments, sides, validateTuple, True)
            rq = self.root.right.query(point, distance, keyDict, segments, sides, validateTuple, True)
            return lq + rq

    def searchKey(self,key,keyDict):
        if self.isLeaf():
            for i in range(len(self.keys)):
                myKey = self.keys[i]
                if np.all(np.array(key) == np.array(myKey)):
                    return True
            return False
        if self.isInner():
            if self.isAtStashDepth():
                return keyDict.getById(self.root.left).searchKey(key,keyDict) or keyDict.getById(self.root.right).searchKey(key,keyDict)
            else:
                return self.root.left.searchKey(key,keyDict) or self.root.right.searchKey(key,keyDict)

    def batchUpdate(self,addKeys,removeKeys,keyDict):

        if len(addKeys) == 0 and len(removeKeys) == 0:
            return

        if self.isLeaf():
            for key in removeKeys:
                for i in range(len(self.keys)):
                    if np.all(np.array(key) == self.keys[i]):
                        self.keys = np.delete(self.keys,i,axis=0)
                        break
            self.keys = np.array(list(self.keys) + addKeys)
            self.size = len(self.keys)
            if len(self.keys) > self.leafsize:
                self.expand(keyDict)

        elif self.isInner():
            addPoints = fromDictGetPoints(addKeys,keyDict)
            removePoints = fromDictGetPoints(removeKeys,keyDict)
            leftRemoveKeys = []
            rightRemoveKeys = []

            newLeftTree = None
            newRightTree = None

            for key,point in zip(removeKeys,removePoints):
                if self.isAtStashDepth():
                    if self.axis == 0:
                        #X
                        if (not keyDict.getById(self.root.left).isEmpty()) and point.x() <= fromDictGetPoint(keyDict.getById(self.root.left).maxXKey,keyDict).x():
                            leftRemoveKeys.append(key)
                        if (not keyDict.getById(self.root.right).isEmpty()) and fromDictGetPoint(keyDict.getById(self.root.right).minXKey, keyDict).x() <= point.x():
                            rightRemoveKeys.append(key)
                    else:
                        #Y
                        if (not keyDict.getById(self.root.left).isEmpty()) and point.y() <= fromDictGetPoint(keyDict.getById(self.root.left).maxYKey,keyDict).y():
                            leftRemoveKeys.append(key)
                        if (not keyDict.getById(self.root.right).isEmpty()) and fromDictGetPoint(keyDict.getById(self.root.right).minYKey, keyDict).y() <= point.y():
                            rightRemoveKeys.append(key)
                else:
                    if self.axis == 0:
                        #X
                        if (not self.root.left.isEmpty()) and point.x() <= fromDictGetPoint(self.root.left.maxXKey,keyDict).x():
                            leftRemoveKeys.append(key)
                        if (not self.root.right.isEmpty()) and fromDictGetPoint(self.root.right.minXKey, keyDict).x() <= point.x():
                            rightRemoveKeys.append(key)
                    else:
                        #Y
                        if (not self.root.left.isEmpty()) and point.y() <= fromDictGetPoint(self.root.left.maxYKey,keyDict).y():
                            leftRemoveKeys.append(key)
                        if (not self.root.right.isEmpty()) and fromDictGetPoint(self.root.right.minYKey, keyDict).y() <= point.y():
                            rightRemoveKeys.append(key)

            if self.isAtStashDepth():
                newLeftTree = copy.deepcopy(keyDict.getById(self.root.left))
                newRightTree = copy.deepcopy(keyDict.getById(self.root.right))
                newLeftTree.batchUpdate([],leftRemoveKeys,keyDict)
                newRightTree.batchUpdate([],rightRemoveKeys,keyDict)
                self.size = newLeftTree.size + newRightTree.size
            else:
                self.root.left.batchUpdate([],leftRemoveKeys,keyDict)
                self.root.right.batchUpdate([],rightRemoveKeys,keyDict)
                self.size = self.root.left.size + self.root.right.size

            leftAddKeys = []
            rightAddKeys = []
            maybeAddKeys = []

            if len(addPoints) > 0:
                args = np.array(addPoints)[:, self.axis].argsort()
                for i in args:
                    point = addPoints[i]
                    key = addKeys[i]

                    mustLeft = False
                    mustRight = False
                    if self.isAtStashDepth():
                        if self.axis == 0:
                            # X
                            if (not keyDict.getById(self.root.left).isEmpty()) and point.x() < fromDictGetPoint(keyDict.getById(self.root.left).maxXKey,keyDict).x():
                                mustLeft = True
                            if (not keyDict.getById(self.root.right).isEmpty()) and fromDictGetPoint(keyDict.getById(self.root.right).minXKey, keyDict).x() < point.x():
                                mustRight = True
                        else:
                            # Y
                            if (not keyDict.getById(self.root.left).isEmpty()) and point.y() < fromDictGetPoint(keyDict.getById(self.root.left).maxYKey,keyDict).y():
                                mustLeft = True
                            if (not keyDict.getById(self.root.right).isEmpty()) and fromDictGetPoint(keyDict.getById(self.root.right).minYKey, keyDict).y() < point.y():
                                mustRight = True
                    else:
                        if self.axis == 0:
                            # X
                            if (not self.root.left.isEmpty()) and fromDictGetPoint(self.root.left.maxXKey,
                                                                                   keyDict).x() > point.x():
                                mustLeft = True
                            if (not self.root.right.isEmpty()) and fromDictGetPoint(self.root.right.minXKey,
                                                                                    keyDict).x() < point.x():
                                mustRight = True
                        else:
                            # Y
                            if (not self.root.left.isEmpty()) and fromDictGetPoint(self.root.left.maxYKey,
                                                                                   keyDict).y() > point.y():
                                mustLeft = True
                            if (not self.root.right.isEmpty()) and fromDictGetPoint(self.root.right.minYKey,
                                                                                    keyDict).y() < point.y():
                                mustRight = True

                    if mustLeft:
                        leftAddKeys.append(key)
                    elif mustRight:
                        rightAddKeys.append(key)
                    else:
                        maybeAddKeys.append(key)

                if self.isAtStashDepth():

                    newLeftSize = newLeftTree.size + len(leftAddKeys)
                    newRightSize = newRightTree.size + len(rightAddKeys)
                    self.size = newLeftSize + newRightSize + len(maybeAddKeys)
                    perfectMedian = self.size // 2 - newLeftSize
                    median = min(len(maybeAddKeys), max(0,perfectMedian))

                    newLeftTree.batchUpdate(leftAddKeys + maybeAddKeys[:median],[],keyDict)
                    newRightTree.batchUpdate(rightAddKeys + maybeAddKeys[median:],[],keyDict)

                else:

                    newLeftSize = self.root.left.size + len(leftAddKeys)
                    newRightSize = self.root.right.size + len(rightAddKeys)
                    self.size = newLeftSize + newRightSize + len(maybeAddKeys)
                    perfectMedian = self.size // 2 - newLeftSize
                    median = min(len(maybeAddKeys), max(0,perfectMedian))

                    self.root.left.batchUpdate(leftAddKeys + maybeAddKeys[:median],[],keyDict)
                    self.root.right.batchUpdate(rightAddKeys + maybeAddKeys[median:],[],keyDict)

            if self.isAtStashDepth():
                if self.size <= self.leafsize:
                    self.keys = np.array(list(newLeftTree.getKeys(keyDict)) + list(newRightTree.getKeys(keyDict)))
                    self.root = None
                    self.axis = None
                else:
                    self.root.left = keyDict.addObject(newLeftTree)
                    self.root.right = keyDict.addObject(newRightTree)

            else:
                if self.size <= self.leafsize:
                    self.contract(keyDict)
        self.updateBB(keyDict)