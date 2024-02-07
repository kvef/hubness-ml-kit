/**
 * Hub Miner: a hubness-aware machine learning experimentation library.
 * Copyright (C) 2014 Nenad Tomasev. Email: nenad.tomasev at gmail.com
 * 
* This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 * 
* This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 * 
* You should have received a copy of the GNU General Public License along with
 * this program. If not, see <http://www.gnu.org/licenses/>.
 */
package preprocessing.instance_selection;

import algref.Author;
import algref.BookChapterPublication;
import algref.Publication;
import data.neighbors.NSFUserInterface;
import data.neighbors.NeighborSetFinder;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Random;
import statistics.HigherMoments;

/**
 * This class implements the edited normalized RBF noise filter that can be used
 * for instance selection.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ENRBF extends InstanceSelector implements NSFUserInterface {

    public static final double DEFAULT_ALPHA_VALUE = 0.9;
    private double alpha = DEFAULT_ALPHA_VALUE;
    // The upper triangular distance matrix on the data.
    private float[][] distMat;
    // Object that holds the kNN sets.
    private NeighborSetFinder nsf;

    /**
     * Default constructor.
     */
    public ENRBF() {
    }

    /**
     * Initialization.
     *
     * @param dset DataSet to reduce.
     * @param distMat float[][] that is the upper triangular distance matrix on
     * the data.
     * @param alpha Double value that is the parameter to use for determining
     * which instances to keep.
     */
    public ENRBF(DataSet dset, float[][] distMat, double alpha) {
        this.alpha = alpha;
        this.distMat = distMat;
        setOriginalDataSet(dset);
    }

    @Override
    public void reduceDataSet() throws Exception {
        DataSet originalDataSet = getOriginalDataSet();
        int dataSize = originalDataSet.size();
        // Initialization.
        int numClasses = getNumClasses();
        // First estimate the sigma value for the kernal.
        int minIndex, maxIndex, firstChoice, secondChoice;
        float[] distSample = new float[50];
        Random randa = new Random();
        for (int i = 0; i < 50; i++) {
            firstChoice = randa.nextInt(dataSize);
            secondChoice = firstChoice;
            while (firstChoice == secondChoice) {
                secondChoice = randa.nextInt(dataSize);
            }
            minIndex = Math.min(firstChoice, secondChoice);
            maxIndex = Math.max(firstChoice, secondChoice);
            distSample[i] = distMat[minIndex][maxIndex - minIndex - 1];
        }
        float distMean = HigherMoments.calculateArrayMean(distSample);
        float distSigma =
                HigherMoments.calculateArrayStDev(distMean, distSample);
        // Calculate the RBF matrix.
        double[][] rbfMat = new double[distMat.length][];
        double[] pointTotals = new double[dataSize];
        for (int i = 0; i < distMat.length; i++) {
            rbfMat[i] = new double[distMat.length - i - 1];
            for (int j = 0; j < distMat[i].length; j++) {
                rbfMat[i][j] = Math.exp(-(distMat[i][j] * distMat[i][j])
                        / distSigma);
                pointTotals[i] += rbfMat[i][j];
                pointTotals[i + j + 1] += rbfMat[i][j];
            }
        }
        // Calculate the class probabilities in points based on the RBF 
        // estimate.
        double[][] pointClassProbs = new double[dataSize][numClasses];
        int firstLabel, secondLabel;
        for (int i = 0; i < distMat.length; i++) {
            firstLabel = originalDataSet.getLabelOf(i);
            for (int j = 0; j < distMat[i].length; j++) {
                secondLabel = originalDataSet.getLabelOf(i + j + 1);
                pointClassProbs[i][secondLabel] +=
                        rbfMat[i][j] / pointTotals[i];
                pointClassProbs[i + j + 1][firstLabel] +=
                        rbfMat[i][j] / pointTotals[i + j + 1];
            }
        }
        // Now perform the filtering.
        ArrayList<Integer> protoIndexes = new ArrayList<>(dataSize);
        int label;
        boolean acceptable;
        for (int i = 0; i < dataSize; i++) {
            label = originalDataSet.getLabelOf(i);
            acceptable = true;
            for (int c = 0; c < numClasses; c++) {
                if (c != label && point