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
package data.neighbors.hubness.visualization;

import data.generators.MultiDimensionalSphericGaussianGenerator;
import data.neighbors.NeighborSetFinder;
import data.representation.DataInstance;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import distances.primary.MinkowskiMetric;
import ioformat.FileUtil;
import java.awt.Point;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;
import util.CommandLineParser;

/**
 * This utility class can be used to generate the data for some neighbor
 * occurrence and co-occurrence distribution charts on synthetic Gaussian
 * mixture data of variable dimensionality.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class HubnessDistributionGraphsSynthetic {

    private static int k = 10;
    private static final int REPETITIONS = 200;
    private static final int DATA_SIZE = 1000;
    private int[] hubnessArray;
    private int[] numCoocPointsArray;
    private int[] pairOccurrences;

    /**
     * This method generates synthetic Gaussian data of the specified size and
     * dimensionality and then calculates the neighbor occurrence and
     * co-occurrence distributions.
     *
     * @param dsize Integer that is the desired data size.
     * @param dim Integer that is the desired dimensionality.
     * @throws Exception
     */
    private void generateResults(int dsize, int dim)
            throws Exception {
        float[] meanArray = new float[dim]; // Means are a zero-array.
        float[] stDevArray = new float[dim];
        Random randa = new Random();
        // Initialize standard deviations.
        for (int i = 0; i < dim; i++) {
            stDevArray[i] = randa.nextFloat();
        }
        // Set some upper and lower bounds.
        float[] lBounds = new float[dim];
        Arrays.fill(lBounds, -100);
        float[] uBounds = new float[dim];
        Arrays.fill(uBounds, 100);
        // Generate synthetic Gaussian data.
        MultiDimensionalSphericGaussianGenerator gen =
                new MultiDimensionalSphericGaussianGenerator(
                meanArray, stDevArray, lBounds, uBounds);
        DataSet genData = new DataSet();
        String[] floatAttrNames = new String[dim];
        for (int i = 0; i < dim; i++) {
            floatAttrNames[i] = "dim" + dim;
        }
        genData.fAttrNames = floatAttrNames;
        genData.data = new ArrayList<>(dsize);
        for (int i = 0; i < dsize; i++) {
            DataInstance instance = new DataInstance(genData);
            instance.fAttr = gen.generateFloat();
            genData.addDataInstance(instance);
        }
        // Find the kNN sets.
        CombinedMetric cmet = new CombinedMetric(
                null, new MinkowskiMetric(), CombinedMetric.DEFAULT);
        NeighborSetFinder nsf = new NeighborSetFinder(genData, cmet);
        nsf.calculateDistances();
        nsf.calculateNeighborSets(k);
        int[][] kneighbors = nsf.getKNeighbors();
        // Neighbor occurrence frequencies.
        hubnessArray = nsf.getNeighborFrequencies();
        // Number of points that a point co-occurs with.
        numCoocPointsArray = new int[hubnessArray.length];
        // A list of co-occuring pairs.
        ArrayList<Point> coOccurringPairs = new ArrayList<>(dsize);
        // HashMap that maps the pair frequency.
        HashMap<Long, Integer> coDependencyMaps = new HashMap<>(dsize);
        // Hashing by concatenatin the numbers.
        long concat;
        int min;
        long max;
        int currFreq;
        for (int i = 0; i < hubnessArray.length; i++) {
            for (int kInd1 = 0; kInd1 < k; kInd1++) {
                for (int kInd2 = kInd1 + 1; kInd2 < k; kInd2++) {
                    min = Math.min(kneighbors[i][kInd1], kneighbors[i][kInd2]);
                    max = Math.max(kneighbors[i][kInd1], kneighbors[i][kInd2]);
                    concat = (max << 32) | (min & 0XFFFFFFFFL);
                    if (!coDependencyMaps.containsKey(concat)) {
          