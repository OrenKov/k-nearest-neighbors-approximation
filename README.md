# KNN Approximation
A fast approximation for KNN, for a given training dataset. Each data-sample in this dataset is a 3D point.<br>

## Overview
KNN (k-nearest neighbors) is an associative classification ML method that can be use both for classification and regression. During prediction it searches for the nearest neighbors and takes their majority vote / average as the prediction.<br>
There are many approches of implementing it, such as Brute-Force, Ball Tree and K-D Tree. more information can be found [here](https://towardsdatascience.com/k-nearest-neighbors-computational-complexity-502d2c440d5).
<br>
<details open="open">
  <summary><strong>Table of Contents</strong></summary>
  <ol>
    <li><a href="#requirements">Requirements</a></li>
<!--     <li><a href="#do-it-yourself">Do It Yourself</a></li> -->
    <li><a href="#design">Design</a></li>
    <li><a href="#how-to-use">How To Use</a></li>
    <li><a href="#performance-estimation">Performance Estimation</a></li>

  </ol>
</details>
<br>

## Requirements
The Assignment baselines and requirements can be found and downloaded [here](https://github.com/OrenKov/k-nearest-neighbors-approximation/blob/main/KNN%20Approximation.docx).
<br>
<br>

## Design
* After reading about usfull tree-based data structures for the KNN algorithm, such as k-d-tree and ball-tree - I decided to go with the <strong>k-d-tree with some modifications</strong>, in fit to the constraints.
* k-d-trees can decrease the amount of time for KNN dramatically compared to Brute-Force method, and are pretty straight-forward to study and implement. 
* The k-d-tree version I chose to implelment is basic. There are more 'clever' ways to improve the splitting of the space into segments (which are not implemented by me), such as using CART heuristic and max-variance splits.
* A list of given 3D points will be assigned into the tree leaves, by splitting the 3D space into distinct-parallel-to-the-axes segments, as follows:
  *  Generally speaking, divide the points of the current node by the median of `axis % points_dim`, where in each level, axis increase by 1.
  *  Stop splitting when the number of points in the node is less then K, and practically make it a leaf. Each leaf is a distinct retrievable segment.
* <strong> Constrains: </strong>
  * Each segment contains at most K points.
  * Each segment contains at least `K/2` points, making it 'easier' to traverse the tree and find large amount of approximated neighbors at once.
* I chose to use the euclidian distance to measure the distance between 2 points, but others can be used as well according to the needs.
<br>


## How To Use
* Download the KNN.py file, and import it to your python working-file.
* <strong>Initialize</strong>:
    ```sh
    $ my_tree = KDTree(list_of_points, K)
    ```
* <strong>Get the approximated KNN</strong>:
    ```sh
    $ my_tree.knn(my_point)
    ```
    where `my_point` is in `(x,y,z)` format.
* <strong>Get z-axis average in a given point segment </strong>:
    ```sh
    $ my_tree.get_z_avg(my_point)
    ```
<br>

## Performance Estimation
### get_z_avg method
* I wanted to see how 'fast' the function runs correlated to the number of points in the data set and K (the number of nearest neighbors).
* 
* The expectation is that the running time increases logarithmically to the amount of points that were used to build the tree, because running the function is a matter of searching in the tree, and then calculating average over O(K) instances.<br>
![equation](https://latex.codecogs.com/gif.latex?\textbf{O(f)}&space;=&space;O(log(n)-log(K)&space;&plus;&space;O(K))&space;=&space;O(log(\frac&space;nK)&space;&plus;&space;O(K)))
*  I ran few toy-checks (under my computer computational limitations) and found out that indeed - the time it takes to perform the function increases logarithmically to the amount of points used to build the tree (for a given K):
<p align="center">
  <img src="https://i.im.ge/2021/08/20/PG5ic.png">
</p>
<br>

