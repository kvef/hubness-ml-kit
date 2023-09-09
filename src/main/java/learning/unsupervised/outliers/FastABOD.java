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

import data.neighbors.NSFUserInterface;
import data.neighbors.NeighborSetFinder;
import data.representation.DataInstance;
import data.representation.DataSet;
import data.representation.util.DataMineConstants;
import distances.primary.MinkowskiMetric;
import java.util.ArrayList;
import util.ArrayUtil;
import util.AuxSort;

/**
 * This class implements angle-based outlier detection, an outlier detection 
 * approach tailored specifically for high-dimensional data. It was originally 
 * proposed in the paper titled 'Angle-Based Outlier Detection in 
 * High-dimensional Data' that was presented at KDD in 2008. It implements the 
 * faster, kNN-based approximation.
 * 
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class FastABOD extends OutlierDetector implements NSFUserInterface {
    
    private int k = 50;
    private NeighborSetFinder nsf;
    private float outlierRatio;
    
    /**
     * Default empty constructor.
     */
    public FastABOD() {
    }
    
    /**
     * Initialization.
     * 
     * @param dset DataSet object to find outliers for.
     * @param nsf NeighborSet finder object holding the kNN sets.
     * @param k Integer that is the neighborhood size to use for calculations.
     * @param outlierRatio Float value corresponding to the proportion of points
     * to consider as outliers.
     */
    public FastABOD(DataSet dset, NeighborSetFinder nsf, int k,
            float outlierRatio) {
        setDataSet(dset);
        this.k = k;
        this.nsf = nsf;
        this.outlierRatio = outlierRatio;
    }
    
    @Override
    public void setNSF(NeighborSetFinder nsf) {
        this.nsf = nsf;
        if (nsf != null) {
            this.k = nsf.getCurrK();
        }
    }
    
    /**
     * @param outlierRatio Float value that it the outlier ratio to use, the
     * proportion of points to select as outliers.
     */
    public void setOutlierRatio(float outlierRatio) {
        this.outlierRatio = outlierRatio;
    }

    @Override
    public NeighborSetFinder getNSF() {
        return nsf;
    }

    @Override
    public void noRecalcs() {
    }
    
    @Override
    public int getNeighborhoodSize() {
        return k;
    }
    
    @Override
    public void detectOutliers() throws Exception {
        DataSet dset = getDataSet();
        // Check for trivial and ill-defined cases.
        if (dset == null || dset.isEmpty()) {
            return;
        }
        if (nsf == null) {
            return;
        }
        if (k <= 1) {
            return;
        }
        if (nsf.getCurrK() > k) {
            // Sub-sample the kNN sets, since they are incompatible with the
            // specified neighborhood size.
            nsf = nsf.getSubNSF(k);
        } else if (nsf.getCurrK() < k) {
            throw new Exception("Provided kNN sets do not correspond to the "
                    + "requested neighborhood size: " + nsf.getCurrK() + " "
                    + "compared to " + k);
        }
        int size = dset.size();
        int[][] kNeighbors = nsf.getKNeighbors();
        double[] abofScores = new double[size];
        double[] normalizedAngles;
        double dotProduct;
        // Get the number of float and integer features.
        int numFloats = dset.getNumFloatAttr();
        int numInts = dset.getNumIntAttr();
        MinkowskiMetric euc = new MinkowskiMetric(2);
        // Loop over all data points.
        for (int i = 0; i < size; i++) {
            DataInstance instance = dset.getInstance(i);
            normalizedAngles = new double[(k * (k - 1)) / 2];
            int counter = -1;
            // Loop over all pairs of neighbor points.
   