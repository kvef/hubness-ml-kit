
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
package images.mining.clustering;

import data.representation.DataInstance;
import data.representation.DataSet;
import ioformat.FileUtil;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import optimization.stochastic.algorithms.SimulatedThermicAnnealingLeap;
import optimization.stochastic.fitness.SIFTSegmentationHomogeneity;
import optimization.stochastic.operators.onFloats.HomogenousTwoDevsFloatMutator;
import util.CommandLineParser;
import util.ImageUtil;

/**
 * This utility script runs the experiments for optimizing the detection of
 * intra-image SIFT feature clusters by means of modified k-means clustering
 * that uses similarity ranking and combines different aspects of similarity -
 * spatial proximity, descriptor similarity, scale similarity and angle
 * similarity for the individual image features. The optimization is performed
 * stochastically via simulated annealing. This code corresponds to the
 * experiments that were published in the paper "Two pass k-means algorithm for
 * finding SIFT clusters in an image" in 2010 at the Slovenian KDD conference
 * which is a part of the larger Information Society multiconference.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class OptimizeSIFTClustering {

    private File inImagesDir;
    private File inSIFTDir;
    private File inSegmentDir;
    private String logPath;
    private String[] nameList;
    private int minClusters, maxClusters;
    private int numIter;
    private boolean useRank;

    /**
     * This method runs the optimization procedure for SIFT clustering.
     *
     * @throws Exception
     */
    public void runOptimization() throws Exception {
        nameList = ImageUtil.getImageNamesArray(inImagesDir, "jpg");
        // Homogeneity of SIFT features from different clusters in segments of