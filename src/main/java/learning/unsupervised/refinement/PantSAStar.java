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
package learning.unsupervised.refinement;

import data.representation.DataSet;
import distances.primary.CombinedMetric;
import java.util.ArrayList;
import learning.unsupervised.Cluster;
import learning.unsupervised.ClusterConfigurationCleaner;
import learning.unsupervised.evaluation.quality.QIndexSilhouette;
import util.AuxSort;

/**
 * This class implements an algorithm described in the paper: Diego Ingaramo,
 * Marcelo Errecalde and Paolo Rosso. "A general bio-inspired method to improve
 * the short-text clustering task" which was presented at the 2010 CICLING
 * conference.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class PantSAStar {

    private int numClusters;
    private int[] clusterAssociations;
    private float[] silhouetteArray;
    private int[] rearrangement;
    private DataSet dset;
    private CombinedMetric cmet;
    private int[] clusterElements;
    private int numChanges = 0;

    /**
     * @param numClusters Number of clusters.
     * @param clusterAssociations Array that contains cluster associations for
     * data points.
     * @param silhouetteArray Array containing Silhouette index values for data
     * points.
     * @param dset DataSet object.
     * @param cmet CombinedMetric object.
     */
    public PantSAStar(int numClusters, int[] clusterAssociations,
            float[] silhouetteArray, DataSet dset, CombinedMetric cmet) {
        this.numClusters = numClusters;
        this.clusterAssociations = clusterAssociations;
        this.silhouetteArray = silhouetteArray;
        this.dset = dset;
        this.cmet = cmet;
    }

    /**
     * @param numClusters Number of clusters.
     * @param clusterAssociations Array that contains cluster associations for
     * data points.
     * @param dset DataSet object.
     * @param cmet CombinedMetric object.
     */
    public PantSAStar(int numClusters, int[] clusterAssociations, DataSet dset,
            CombinedMetric cmet) {
        this.numClusters = numClusters;
        this.clusterAssociations = clusterAssociations;
        this.dset = dset;
        this.cmet = cmet;
    }

    /**
     * In case the Silhouette index value array hasn't already been provided.
     */
    public void calculateSilhouetteArray() throws Exception {
        QIndexSilhouette dsi =
                new QIndexSilhouette(numClusters, clusterAssociations, dset,
                cmet);
        try {
            dsi.validity();
        } catch (Exception e) {
            throw e;
        }
        silhouetteArray = dsi.getInstanceSilhouetteArray();
    }

    /**
     * Necessary to perform before the refinement, which doesn't support empty
     * clusters.
     */
    private void removeEmptyClusters() {
        ClusterConfigurationCleaner ccc =
                new ClusterConfigurationCleaner(dset, clusterAssociations,
                numClusters);
        ccc.removeEmptyClusters();
        numClusters = ccc.getNewNumberOfClusters();
        clusterAssociations = ccc.getNewAssociationArray();
    }

    /**
     * Performs cluster configuration refinement according to the PantSAStar
     * algorithm.
     *
     * @return An integer array containing the refined cluster associations.
     * @throws Exception
     */
    public int[] refine() throws Exception {
        removeEmptyClusters();
        rearrangement = AuxSort.sortIndexedValue(silhouetteArray, true);
        clusterElements = new int[numClusters];
        // Calculate how many elements there are in each original cluster.
        for (int i = 0; i < clusterAssociations.length; i++) {
            clusterElements[clusterAssociations[i]]++;
        }
        int[][] silhouetteSortedPerCluster = new int[numClusters][];
        Cluster[] refinedClusters = new Cluster[numClusters];
        // Determine the max number of elements in any original cluster.
        int maxClusterSize = 0;
        for (int c = 0; c < numClusters; c++) {
            silhouetteSortedPerCluster[c] = new int[clusterElements[c]];
            refinedClusters[c] = new Cluster(dset);
            refinedClusters[c].indexes =
                    new ArrayList<>(clusterElements[c]);
            if (clusterElements[c] > maxClusterSize) {
                maxClusterSize = clusterElements[c];
            }
        }
        // Counters for each cluster that are incremented when new values are
        // added to cluster-specific Silhouette subarrays.
        int[] currAnticipatingIndexes = new int[numClusters];
        // Now we create ordered subArrays of the silhouette ordered array for
        // each cluster separately
        for (int i = 0; i < silhouetteArray.length; i++) {
            silhouetteSortedPerCluster[clusterAssociations[rearrangement[i]]][
                   