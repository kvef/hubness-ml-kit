
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
 * It calculates an index which measures how concordant are the distances with
 * respect to data clusters. Standardized categories are assumed. The index
 * takes a value betwen -1 and 1. (Nc - Nd) / (Nc + Nd), Nc - concordant, Nd -
 * discordant distances.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class QIndexGoodmanKruskal extends ClusteringQualityIndex {

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
    public QIndexGoodmanKruskal(int[] clusterAssociations,
            DataSet dset) {
        this.clusterAssociations = clusterAssociations;
        setDataSet(dset);
        cmet = CombinedMetric.EUCLIDEAN;
    }

    /**
     * Initialization.
     * 
     * @param clusterAssociations Cluster association array for the points.
     * @param dset DataSet object.
     * @param cmet CombinedMetric object for distance calculations.
     */
    public QIndexGoodmanKruskal(int[] clusterAssociations, DataSet dset,
            CombinedMetric cmet) {
        this.clusterAssociations = clusterAssociations;
        setDataSet(dset);
        this.cmet = cmet;
    }

    /**
     * @param distances float[][] representing the upper triangular distance 
     * matrix of the data.
     */
    public void setDistanceMatrix(float[][] distances) {
        this.distances = distances;
        this.dGiven = true;
    }

    @Override
    public float validity() throws Exception {
        float gkIndex;
        DataSet instances = getDataSet();
        if (!dGiven) {
            distances = instances.calculateDistMatrix(cmet);
        }

        Cluster[] clusterConfiguration =
                Cluster.getConfigurationFromAssociations(clusterAssociations,
                instances);
        int numIntraDists = 0;
        int numInterDists = 0;
        int numClusters = clusterConfiguration.length;
        if (numClusters < 2) {
            return 0;
        }
        for (int c1 = 0; c1 < numClusters; c1++) {
            if (clusterConfiguration[c1].size() > 0) {
                numIntraDists += clusterConfiguration[c1].size()
                        * (clusterConfiguration[c1].size() - 1) / 2;
                for (int c2 = 0; c2 < numClusters; c2++) {
                    if (clusterConfiguration[c2].size() > 0) {
                        numInterDists += clusterConfiguration[c1].size()
                                * clusterConfiguration[c2].size();
                    }
                }
            }
        }
        float[] intraDists = new float[numIntraDists];
        float[] interDists = new float[numInterDists];
        int intraIndex = -1;
        int interIndex = -1;
        for (int i = 0; i < distances.length; i++) {
            if (instances.getInstance(i).isNoise()) {
                continue;
            }
            for (int j = 0; j < distances[i].length; j++) {
                if (instances.getInstance(j).isNoise()) {
                    continue;
                }
                if (clusterAssociations[i] == clusterAssociations[i + j + 1]) {
                    intraDists[++intraIndex] = distances[i][j];
                } else {
                    interDists[++interIndex] = distances[i][j];
                }
            }
        }
        Arrays.sort(intraDists); //ascending sort
        Arrays.sort(interDists); //ascending sort
        intraIndex = 0;
        interIndex = 0;
        long totalDists = numIntraDists + numInterDists;
        long totalPairs = numIntraDists * numInterDists;
        // This is the sum of concordant and discordant pairs.
        long Nc; // Num concordant pairs.
        long Nd = 0; // Num discordant pairs.
        // We will only count the breaches.
        do {
            while (intraIndex < numIntraDists
                    && intraDists[intraIndex] < interDists[interIndex]) {
                intraIndex++;
            }
            if (intraIndex == numIntraDists) {
                break;
            }
            // Then the interDists[interIndex] is discordant with all
            // intraDists[i] for i >= intraIndex. There are numIntraDists -
            //intraIndex of them
            Nd += numIntraDists - intraIndex;
            // The inter-distance at interIndex has been processed, so it is
            // time to move on.
            interIndex++;
        } while (intraIndex < numIntraDists && interIndex < numInterDists);
        Nc = totalPairs - Nd;
        gkIndex = (float) (Nc - Nd) / (float) (Nc + Nd);
        return gkIndex;
    }
}