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