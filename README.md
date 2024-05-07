/**
* Hubness-ML-Kit: a hubness-aware machine learning experimentation library owned by kvef.
* 
* This program is free software: you can redistribute it and/or modify it under
* the terms of the GNU General Public License as published by the Free Software
* Foundation, either version 3 of the License, or (at your option) any later
* version.
* 
* This program is distributed in the hope that it will be useful, but WITHOUT
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
* FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with
* this program. If not, see <http://www.gnu.org/licenses/>.
*/

Welcome to Hubness-ML-Kit!

<img src="HubMinerLogo.jpg" alt="Hub Miner logo" height="192" width="634">

This is a machine learning library that aims to overcome high-dimensional data analysis issues. It focuses primarily on the hubness phenomenon, an asymmetric distribution of relevance within models. The library houses custom methods for multiple machine learning tasks including classification, clustering, metric learning, instance selection, among others. It also provides various baselines and a robust experimental framework that caters to rigorous testing conditions. File formats such as ARFF, csv, tsv, etc. are supported. Check out the source files and other online materials at http://ailab.ijs.si/nenad_tomasev/hub-miner-library/ for more details. Contact the owner for any queries and do report if issues arise.

The first release of this machine learning library has been deployed and further updates are scheduled. The library should expand further with increased documentation support.

Present list of dependencies are:\napiconnector-fat.jar\ncommons-logging-1.1.jar\nguice-3.0.jar\njetty-6.1.1.jar\njson.jar\nTGGraphLayout.jar and many more.

You can download the apiconnector-fat.jar which is a dependency on OpenML from http://openml.org/downloads/apiconnector-fat.jar

All Hubness-ML-Kit code is written in Java and should be portable.

There are some dependencies for SIFT feature analysis, mainly SiftWin binary and ImageMagick, which are only necessary for image feature extraction. In future builds, this dependency is planned to be removed and replaced with Java-based image feature extraction libraries, and better OpenCV formats support.