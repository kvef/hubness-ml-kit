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
package configuration;

import com.google.gson.Gson;
import com.google.inject.TypeLiteral;
import distances.kernel.Kernel;
import distances.primary.CombinedMetric;
import distances.primary.DistanceMeasure;
import distances.sparse.SparseCombinedMetric;
import distances.sparse.SparseMetric;
import ioformat.FileUtil;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.HashMap;
import learning.supervised.evaluation.cv.BatchClassifierTester;
import learning.unsupervised.evaluation.BatchClusteringTester;
import util.ReaderToStringUtil;

/**
 * This class is a configuration class for batch clustering testing, which
 * allows the batch tester to be invoked from other parts of the code, as well
 * as allowing customizable file format for saving the configuration. In this
 * case, it supports JSON I/O, which makes it easy to automatically generate
 * clustering evaluation requests from external code.
 * 
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class BatchClusteringConfig {
    
    // Whether to calculate model perplexity as a quality measure, as it can be
    // very slow for larger and high-dimensional datasets, prohibitively so.
    public boolean calculatePerplexity = false;
    // Whether to use split testing (training/test data).
    public boolean splitTesting = false;
    public float splitPerc = 1;
    // Whether a specific number of desired clusters was specified.
    public boolean nClustSpecified = true;
    // The normalization type to actually use in the experiments.
    public BatchClusteringTester.Normalization normType =
            BatchClusteringTester.Normalization.STANDARDIZE;
    public boolean clustersAutoSet = false;
    // The number of times a clustering is performed on every single dataset.
    public int timesOnDataSet = 30;
    // The preferred number of iterations, where applicable.
    public int minIter;
    // Names of clustering algorithms to use.
    public ArrayList<String> clustererNames = new ArrayList<>(10);
    // Possible parameter value maps to use with certain algorithms.
    public HashMap<String, HashMap<String, Object>> algorithmParametrizationMap;
    // The experimental neighborhood range, with default values.
    public int kMin = 5, kMax = 5, kStep = 1;
    // Noise and mislabeling experimental ranges, with default values.
    public float noiseMin = 0, noiseMax = 0, noiseStep = 1, mlMin = 0,
            mlMax = 0, mlStep = 1;
    // Files pointing to the input and output.
    public File inDir, outDir, distancesDir, mlWeightsDir;
    // Paths to the tested datasets.
    public ArrayList<String> dsPaths = new ArrayList<>(100);
    // Paths to the corresponding metrics for the datasets.
    public ArrayList<CombinedMetric> dsMetric = new ArrayList<>(100);
    public Kernel ker = null;
    // Secondary distance specification.
    public BatchClassifierTester.SecondaryDistance secondaryDistanceType =
            BatchClassifierTester.SecondaryDistance.NONE;
    public int secondaryK = 50;
    // Clustering range.
    public int cluNumMin;
    public int cluNumMax;
    public int cluNumStep;
    // Aproximate kNN calculations specification.
    public float approximateNeighborsAlpha = 1f;
    public boolean approximateNeighbors = false;
    // The number of threads used for distance matrix and kNN set calculations.
    public int numCommonThreads = 8;
    
    /**
     * This method prints the clustering configuration to a Json string.
     * 
     * @return String that is the Json representation of 