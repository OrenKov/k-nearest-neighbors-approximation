# KNN Approximation
A fast approximation for KNN, for a given training dataset. Each data-sample in this dataset is a 3D point.<br>

## Overview
KNN (k-nearest neighbors) is an associative classification ML method that can be use both for classification and regression. During prediction it searches for the nearest neighbors and takes their majority vote / average as the prediction.<br>
There are many approches of implementing it, such as Brute-Force, Ball Tree and K-D Tree. more information about each can be found [here](https://towardsdatascience.com/k-nearest-neighbors-computational-complexity-502d2c440d5).


<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#requirements">Requirements</a></li>
<!--     <li><a href="#do-it-yourself">Do It Yourself</a></li> -->
    <li><a href="#cerdits-and-resources">Cerdits And Resources</a></li>
  </ol>
</details>
<br/>

## Requirements

1. Design a tree-based data structure for storing the training data. This data structure should divide the space of points into retrievable segments.
2. <strong> Constraint: </strong> Each segment must contain at most K points.
3. Implement a function, that is given a dataset (array of 3D points), generates this data structure and fills it with the input data points.
4. Implement a method that given a point (x,y,z) in the data, returns its K approximated nearest neighbors.
5. Consider another helpful constraint for the structure, similar to #2 above.
6. 
7.  <strong> Attendance  Survey </strong>:
   * A link to a survey hosted by [Mailchimp](https://mailchimp.com/).
8. <strong> Payment Link </strong>:
   * A link to a payment group hosted by [Bit](https://www.poalimsites.co.il/bit/index.html).
   * Note that the payment link was available only a week before and a week after the party.   
  <br>
  
<!-- ## Do It Yourself
If you would like to make your own landing page. -->

## Cerdits And Resources
* I used [Bootstrap](https://getbootstrap.com/) with modifications, as a template for many of the CSS features being used in the website.
* I used [Mailchimp](https://mailchimp.com/) services to create the survey. Mailchimp inclues many marketing features, like forms and emails, and can be easily inserted into   your own website.
   

[here](https://orenkov.github.io/party-landing-page/).



