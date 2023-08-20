/**
* Hub Miner: a hubness-aware machine learning experimentation library.
* Copyright (C) 2014  Nenad Tomasev. Email: nenad.tomasev at gmail.com
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
package learning.unsupervised.methods;

import algref.Author;
import algref.ConferencePublication;
import algref.Publication;
import algref.Publisher;
import combinatorial.Permutation;
import data.neighbors.NeighborSetFinder;
import data.representation.DataInstance;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Stack;
import learning.unsupervised.Cluster;
import learning.unsupervised.ClusteringAlg;
import util.ArrayUtil;
import util.AuxSort;

/**
 * This class implements the well-known density based DBScan algorithm first
 * proposed in the following paper: Martin Ester, Hans-Peter Kriegel, Jörg
 * Sander, Xiaowei Xu (1996). "A density-based algorithm for discovering
 * clusters in large spatial databases with noise"
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class DBScan extends ClusteringAlg implements
        learning.supervised.interfaces.DistMatrixUserInterface,
        data.neighbors.NSFUserInterface {

    private float[][] distances = null;
    private int[] bestAssociations = null;
    // k is used to look for the proper epsilon, according to the kdistances.
    private int k = 10;
    private NeighborSetFinder nsf;
    // We keep an array of visited points.
    private boolean[] visited;
    // minPoints is the minimum number of points in a neighborhood for the point
    // not to be considered noise.
    private int minPoints;
    private float epsilonNeighborhoodDist = Float.MAX_VALUE;
    // Noise percentage should be carefully set.
    private float noisePerc = 0.15f;
    
    @Override
    public Publication getPublicationInfo() {
        ConferencePublication pub = new ConferencePublication();
        pub.setConferenceName("KDD");
        pub.addAuthor(new Author("Martin", "Ester"));
        pub.addAuthor(new Author("Hans-Peter", "Kriegel"));
        pub.addAuthor(new Author("Jorg", "Sander"));
   