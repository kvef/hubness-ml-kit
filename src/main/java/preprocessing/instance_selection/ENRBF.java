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
                if (c != label && pointClassProbs[i][label]
                        < alpha * pointClassProbs[i][c]) {
                    acceptable = false;
                    break;
                }
            }
            if (acceptable) {
                protoIndexes.add(i);
            }
        }
        // Check whether at least one instance of each class has been selected.
        int[] protoClassCounts = new int[numClasses];
        int numEmptyClasses = numClasses;
        for (int i = 0; i < protoIndexes.size(); i++) {
            label = originalDataSet.getLabelOf(protoIndexes.get(i));
            if (protoClassCounts[label] == 0) {
                numEmptyClasses--;
            }
            protoClassCounts[label]++;
        }
        if (numEmptyClasses > 0) {
            HashMap<Integer, Integer> tabuMap =
                    new HashMap<>(protoIndexes.size() * 2);
            for (int i = 0; i < protoIndexes.size(); i++) {
                tabuMap.put(protoIndexes.get(i), i);
            }
            for (int i = 0; i < originalDataSet.size(); i++) {
                label = originalDataSet.getLabelOf(i);
                if (!tabuMap.containsKey(i) && protoClassCounts[label] == 0) {
                    protoIndexes.add(i);
                    protoClassCounts[label]++;
                    numEmptyClasses--;
                }
                if (numEmptyClasses == 0) {
                    break;
                }
            }
        }
        // Set the selected prototype indexes and sort them.
        setPrototypeIndexes(protoIndexes);
        sortSelectedIndexes();
    }

    @Override
    public void reduceDataSet(int numPrototypes) throws Exception {
        // This method automatically determines the correct number of prototypes
        // and it is usually a small number, so there is no way to enforce the 
        // number of prototypes here. Automatic selection is performed instead.
        reduceDataSet();
    }

    @Override
    public Publication getPublicationInfo() {
        BookChapterPublication pub = new BookChapterPublication();
        pub.setTitle("Data regularization");
        pub.addAuthor(new Author("N.", "Jankowski"));
        pub.setBookName("Neural Networks and Soft Computing");
        pub.setYear(2000);
        pub.setStartPage(209);
        pub.setEndPage(214);
        return pub;
    }

    @Override
    public InstanceSelector copy() {
        return new ENRBF(getOriginalDataSet(), distMat, alpha);
    }

    @Override
    public void setNSF(NeighborSetFinder nsf) {
        this.nsf = nsf;
        distMat = nsf.getDistances();
    }

    @Override
    public NeighborSetFinder getNSF() {
        return nsf;
    }

    @Override
    public void noRecalcs() {
    }

    @Override
    public void calculatePrototypeHubness(int k) throws Exception {
        if (nsf != null) {
            // Here we have some prior neighbor occurrence information and we
            // can re-use it to speed-up the top k prototype search.
            this.setNeighborhoodSize(k);
            if (k <= 0) {
                return;
            }
            DataSet originalDataSet = getOriginalDataSet();
            // The original k-neighbor information is used in order to speed up
            // the top-k prototype calculations, in those cases where these
            // prototypes are already known to occur as neighbors.
            // These occurrences are re-used dynamically.
            int[][] kns = nsf.getKNeighbors();
            float[][] kd = nsf.getKDistances();
            // Array that holds the kneighbors where only prototypes are allowed
            // as neighbor points.
            int[][] kneighb