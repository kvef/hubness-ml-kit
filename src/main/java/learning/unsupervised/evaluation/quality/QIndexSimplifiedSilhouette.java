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
package learning.unsupervised.evaluation.quality;

import data.representation.DataInstance;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import java.util.ArrayList;
import java.util.Arrays;
import learning.unsupervised.Cluster;
import statistics.HigherMoments;
import util.AuxSort;

/**
 * Assigns an index value close to 1 to the good configurations and a value
 * close to -1 to the bad ones. This particular implementation also keeps track
 * of the A and B values in hubs, anti-hubs and regular points. In this case,
 * the A values are the dissimilarities to own cluster centroid and B values
 * are the lowest dissimilarities to other cluster centroids. This is also the
 * main difference between this simplified version and the full Silhouette
 * index, as it avoid calculating all pairwise distances in order to compute the
 * average distances to points from other clusters. It is therefore more 
 * scalable.
 * 
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class QIndexSimplifiedSilhouette extends ClusteringQualityIndex {
    
    // Object that does distance calculations.
    private CombinedMetric cmet = null;
    // Number of clusters in the data.
    private int numClusters;
    // An array of cluster associations for all data points.
    private int[] clusterAssociations;
    // 'A' values for all data points.
    private float[] instanceAarray;
    // 'B' values for all data points.
    private float[] instanceBarray;
    // Silhouette index values for all data points.
    private float[] instanceSilhouetteArray;
    // Average Silhouette values for all clusters.
    private float[] clusterSilhouetteArray;
    // Neighbor occurrence frequencies for all data points.
    public int[] hubnessArray = null;
    // Total 'A' and 'B' Silhouette values, corresponding to average
    // within cluster distances and lowest average inter-cluster distances.
    public double ATOTAL = 0;
    public double BTOTAL = 0;
    // 'A' and 'B' values for hub points within the data.
    public double HATOTAL = 0;
    public double HBTOTAL = 0;
    // 'A' and 'B' values for anti-hubs.
    public double AHATOTAL = 0;
    public double AHBTOTAL = 0;
    // 'A' and 'B' values of regular points that are neither hubs nor anti-hubs.
    public double REGATOTAL = 0;
    public double REGBTOTAL = 0;


    /**
     * @param numClusters Number of clusters in the configuration.
     * @param clusterAssociations Clust