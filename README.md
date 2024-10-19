# nonObtuseTri

Solver and visualizer for [CG:SHOP problem 2025](https://cgshop.ibr.cs.tu-bs.de/competition/cg-shop-2025/#problem-description)

![alt text](https://github.com/JacobusTheSecond/nonObtuseTri/blob/main/illustration.png?raw=true)

## The Visualizer
### Prerequisites
The Visualizer does not depend on the modified `triangle` library. Simply install the CG:SHOP pyutils:
```
pip install --verbose git+https://github.com/CG-SHOP/pyutils25
```
### Execution
Run `python src/visualizer.py`.
By default it will compare the solution `challenge_instances_cgshop25/zips/solutions.zip` to the best solution out of
- `challenge_instances_cgshop25/zips/cornerLimit10Drop.zip`
- `challenge_instances_cgshop25/zips/cornerLimit10DropRefine.zip`
- `challenge_instances_cgshop25/zips/cornerLimitDrop.zip`
- `challenge_instances_cgshop25/zips/cornerRule.zip`
- `challenge_instances_cgshop25/zips/cornerRuleLimit.zip`
- `challenge_instances_cgshop25/zips/cornerDynLimitDropRefine.zip`
- `challenge_instances_cgshop25/zips/cornerLimit20DropRefine.zip`
- `challenge_instances_cgshop25/zips/cornerNoLimitDropRefine.zip`

### The GUI
The GUI consists of three parts:
- The instance map: At the top is a list where each square represents one of the competition instances. The number inside the cell corresponds to the difference in number of Steiner points between the two solutions. Zeros are omitted.
- The base: on the left the solution on which the comparison is based is shown.
- The improvement: on the right the solution out of the solution set which is best is shown.

### Controls:
- Select an instance by either clicking on the square or navigating the matrix with the arrow keys. The currently selected instance is highlighted in blue.
- zoom and pan with the standard matplotlib controls (bottom left).
- If the visualization breaks, reselect an instance. It should fix everything...

## The Solver

### Prerequisites  
Install the modified `triangle` library and CG:SHOP pyutils:
```
cd trianglemod
pip install .
pip install --verbose git+https://github.com/CG-SHOP/pyutils25
```
### Execution
Run `src/python main.py`. The solution by default will be written to `challenge_instances_cgshop25/zips/cur_solutions.zip`
