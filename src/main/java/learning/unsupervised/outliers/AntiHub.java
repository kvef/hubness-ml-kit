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
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

/**
 * This class implements the AntiHub outlier detection method proposed in the 
 * paper titled: "Reverse Nearest Neighbors in Unsupervised Distance-Based 
 * Outlier Detection" by Milos Radovanovic et al., that was published in IEEE 
 * Transactions on Knowledge and Data Engineering (TKDE) in 2014.
 * 
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class AntiHub extends OutlierDetector implements NSFUserInterface {
    
    // The parameter used for summing up the neighbor occurrence frequencies of
    // neighbor points for anti-hub estimation. It is automatically determined
    // within the method, so it does not need to be set manually by the users.
    private float alpha;
    public static final float DEFAULT_ALPHA_STEP = 0.05f;
    public static final float DEFAULT_OUTLIER_RATIO = 0.05f;
    // Step size while searching for the optimal alpha value.
    private float alphaStep = DEFAULT_ALPHA_STEP;
    // The object used for metric calculations, if necessary.
    private CombinedMetric cmet = CombinedMetric.FLOAT_EUCLIDEAN;
    // Distance matrix, if available.
    private float[][] dMat;
    // The object to calculate and/or hold the kNN sets.
    private NeighborSetFinder nsf;
    // Neighborhood size.
    private int k;
    private float outlierRatio = DEFAULT_OUTLIER_RATIO;
    
    /**
     * Default empty constructor.
     */
    public AntiHub() {
    }
    
    /**
     * Initialization.
     * 
     * @param dset DataSet object to find outliers for.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     */
    public AntiHub(DataSet dset, CombinedMetric cmet, int k) {
        setDataSet(dset);
        this.cmet = cmet;
        this.k = k;
    }
    
    /**
     * Initialization.
     * 
     * @param dset DataSet object to find outliers for.
     * @param dMat float[][] that is the upper triangular distance matrix.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     */
    public AntiHub(DataSet dset, float[][] dMat, CombinedMetric cmet,
            int k) {
        setDataSet(dset);
        this.cmet = cmet;
        this.k = k;
        this.dMat = dMat;
    }
    
    /**
     * Initialization.
     * 
     * @param dset DataSet object to find outliers for.
     * @param nsf NeighborSetFinder object that holds the calculated kNN sets.
     * @param k Integer that is the neighborhood size.
     */
    public AntiHub(DataSet dset, NeighborSetFinder nsf, int k) {
        setDataSet(dset);
        this.k = k;
        if (nsf != null) {
            this.cmet = nsf.getCombinedMetric();
            this.nsf = nsf;
        }
    }
    
    /**
     * Initialization.
     * 
     * @param dset DataSet object to find outliers for.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     * @param outlierRatio Float value that it the outlier ratio to use, the
     * proportion of points to select as outliers.
     */
    public AntiHub(DataSet dset, CombinedMetric cmet, int k,
            float outlierRatio) {
        setDataSet(dset);
        this.cmet = cmet;
        this.k = k;
        this.outlierRatio = outlierRatio;
    }
    
    /**
     * Initialization.
     * 
     * @param dset DataSet object to find outliers for.
     * @param dMat float[][] that is the upper triangular distance matrix.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     * @param outlierRatio Float value that it the outlier ratio to use, the
     * proportion of points to select as outliers.
     */
    public AntiHub(DataSet dset, float[][] dMat, CombinedMetric cmet,
            int k, float outlierRatio) {
        setDataSet(dset);
        this.cmet = cmet;
        this.k = k;
        this.dMat = dMat;
        this.outlierRatio = outlierRatio;
    }
    
    /**
     * Initialization.
     * 
     * @param dset DataSet object to find outliers for.
     * @param nsf NeighborSetFinder object that holds the calculated kNN sets.
     * @param k Integer that is the neighborhood size.
     * @param outlierRatio Float value that it the outlier ratio to use, the
     * proportion of points to select as outliers.
     */
    public AntiHub(DataSet dset, NeighborSetFinder nsf, int k,
            float outlierRatio) {
        setDataSet(dset);
        this.k = k;
        if 