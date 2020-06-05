# Augmentation on the fly
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)

Library to perform augmentation of images for training deep learning models during the training stage. The 
transformations will be performed in the pre-processing step.

This library is an alternative to <a href=https://github.com/mdbloice/Augmentor> Augmentor </a>, which performs 
image augmentations to be stored in memory or creates an ready to use generator
for some of the main deep learning frameworks. However, there are cases where you do not want
to store the images in memory and still use your own generator.

## Quick start guide
The purpose of this repository is to create a small library to perform image augmentation
for training deep learning networks in a flexible manner.

There are two main steps:
1. Selection of the type of augmentation and parameters.
2. Call the run function to apply to a given image.


| Original image and mask<sup></sup>                                                                               | Augmented original and mask images                                                                               |
|---------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| ![OriginalMask](.image/original_image_and_mask.jpg) | ![AugmentedMask](.image/original_image_and_mask.jpg)   



<table>
  <tr>
    <td>Original image with mask</td>
     <td>Augmented image with mask</td>
  </tr>
  <tr>
    <td><p float="left"><img src="./images/0051.jpg" width=200 > <img src="./images/0051_gt.jpg" width=200></p></td>
    <td><p float="left"><img src="./images/0051.jpg" width=200 > <img src="./images/0051_gt.jpg" width=200></p></td>
  </tr>
 </table>

## Type of augmentations
1. 
2.  

## Example 