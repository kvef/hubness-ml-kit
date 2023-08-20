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

import data.representation.DataSet;
import distances.primary.CombinedMetric;
import java.util.Arrays;
import learning.unsupervised.Cluster;

/**
 * This class implements the Tau clustering quality index that is based on 
 * calculating concordant and discordant distance pairs. The larger the index,
 * the better the clustering configuration. It is expressed as the correlation
 * between the distance matrix and the indicator matrix that has 1 where an
 * instance pair is in the same cluster and 0 otherwise.
 * 
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class QIndexTau extends ClusteringQualityIndex {
    
    private CombinedMetric cmet = null;
    private int[] clusterAssociations;
    private float[][] distances;
    private boolean dGiven = false;
    
    /**
     * Initialization.
     * 
     * @param clusterAssociations Cluster association array for the points.
     * @param dset DataSet object.
     */
    public QIndexTau(int[] clusterAssociations,
            DataSet dset) {
        this.clusterAssociations = clusterAssociation