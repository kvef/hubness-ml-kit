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
import data.neighbors.hubness.BatchHubnessAnalyzer.Normalization;
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
import java.util.ArrayList;
import learning.supervised.evaluation.cv.BatchClassifierTester.SecondaryDistance;
import util.ReaderToStringUtil;

/**
 * This class is a configuration class for batch hubness stats calculations,
 * which allows the batch tester to be invoked from other parts of the code, as
 * well as allowing customizable file format for saving the configuration. In
 * this case, it supports JSON I/O, which makes it easy to automatically
 * generate the hubness evaluation requests from external code.
 * 
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class BatchHubnessAnalysisConfig {
    
    public SecondaryDistance secondaryDistanceType;
    // Neighborhood size to use for secondary distances.
    public int secondaryDistanceK = 50;
    // Normalization types.
    // The normalization type to actually use in the experiments.
    public Normalization normType = Normalization.STANDARDIZE;
    // The upper limit on the neighborhood sizes to examine.
    public int kMax = 50;
    // Noise and mislabeling levels to vary, with default values.
    public float noiseMin = 0, noiseMax = 0, noiseStep = 1, mlMin = 0,
            mlMax = 0, mlStep 