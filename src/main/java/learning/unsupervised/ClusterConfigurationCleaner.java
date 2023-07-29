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
package learning.unsupervised;

import data.representation.DataSet;

/**
 * This class implements the functionality for cleaning the cluster
 * configuration of empty cluster in the array.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ClusterConfigurationCleaner {

    private DataSet dataContext;
    private int numClustersOld;
    private int numClustersNew;
    // Instance-to-cluster associations.
    private int[] associations;

    /**
     * Initialization.
     *
     * @param dset Dataset object.
     * @param associations Array indicating cluster associations.
     * @param numClusters Number of clusters in the array.
     */
    public ClusterConfigurationCleaner(DataSet dset, int[] associations,
            int numClusters) {
        this.dataContext = dset;
        this.associations = associations;
        this.numClustersOld = numClusters;
        this.numClustersNew = 0;
    }

    /**
     * Will re-index the associations array to new values, removing the "holes"
     * in the cluster index sequence.
     */
    public void removeEmptyClusters() {
        boolean[] representedClusters = new boolean[numClustersOld];
        for (int i = 0; i < associations.length; i++) {
            representedClusters[associations[i]] = true;
        }
        // Now count represented clusters.
        int[] rereferencer = new int[numClustersOld];
        for (int i = 0; i < numClustersOld; i++) {
            if (representedClusters[i]) {
                rereferencer[i] = numClustersNew++;
            } else {
                rereferencer[i] =