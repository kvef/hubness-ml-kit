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
package learning.unsupervised.outliers;

import java.util.ArrayList;

import data.neighbors.NeighborSetFinder;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import statistics.HigherMoments;

/**
 * This class implements the Local Correlation Integral approach to outlier
 * detection and removal, as described in the following paper: 'LOCI: Fast
 * Outlier Detection Using the Local Correlation Integral' by Spiros
 * Papadimitriou, Hiroyuki Kitagawa, Phillip B. Gibbons and Christos Faloutsos
 * that was presented at IEEE 19th International Conference on Data Engineering
 * (ICDE'03) in Bangalore, India.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class LOCI extends OutlierDetector {

    // The part of the neighborhood used for calculating MDEFs.
    private float alpha = 0.5f;
    // The multiple of standard deviations that defines outliers.
    private float ksigma = 3f;
    // The minimum number of neighbors in an r-neighborhood.
    private int minNeighbors = 20;
    private CombinedMetric cmet = CombinedMetric.FLOAT_EUCLIDEAN;

    /**
     * @param dataset Dataset to be analyzed.
     * @param cmet The metric object.
     */
    public LOCI(DataSet dataset, CombinedMetric cmet) {
        setDataSet(dataset);
        this.cmet = cmet;
    }

    /**
     * @param dataset Dataset to be analyzed.
     * @param cmet The metric object.
     * @param minNeighbors Minimal number of neighbors for a neighborhood.
     * @param alpha The part of the neighbo