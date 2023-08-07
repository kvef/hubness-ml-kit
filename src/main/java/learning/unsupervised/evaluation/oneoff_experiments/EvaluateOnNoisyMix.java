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
package learning.unsupervised.evaluation.oneoff_experiments;

import data.generators.NoisyGaussianMix;
import data.representation.DataInstance;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import feature.evaluation.Info;
import ioformat.FileUtil;
import ioformat.IOARFF;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import learning.unsupervised.Cluster;
import learning.unsupervised.evaluation.quality.QIndexSilhouette;
import learning.unsupervised.methods.GHPC;
import learning.unsupervised.methods.GHPKM;
import learning.unsupervised.methods.GKH;
import learning.unsupervised.methods.KMeansPlusPlus;

/**
 * This class tests how hubness-based clustering methods perform in synthetic
 * high-dimensional scenarios where uniform noise is slowly introduced to the
 * data in form of uniformly drawn instances around the Gaussian clusters. This
 * was one of the experiments presented in the PAKDD 2011 paper titled: "The
 * Role of Hubness in Clustering High-dimensional Data". This class is not very
 * flexible and it should be re-worked for future experiments. In its current
 * form, though - it can be used to run the same experiments as in the original
 * paper, for comparisons.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class EvaluateOnNoisyMix {

    private static final int DATA_SIZE = 10000;
    private static final int MAX_NOISY_INSTANCES = DATA_SIZE;
    private static final int STEPS = 40;
    private static final int NOISE_INCREMENT = MAX_NOISY_INSTANCES / STEPS;
    private int numSec = 0;
    private javax.swing.Timer timeTimer;
    private DataSet dsetTest;
    private PrintWriter pwKM;
    private PrintWriter pwPGKH;
    private PrintWriter pwGKH;
    private PrintWriter pwMin;
    private PrintWriter pwHPKM;
    private File writerDir;
    public int dim = 50;
    public int hubnessK = 50;
    private float[] silScores;
    private float[] avgError;
    private float[] avgClusterEntropy;
    private float avgSil;
    private float avgErr;
    private float avgTime;
    private float avgEntropy;
    private CombinedMetric cmet;

    public EvaluateOnNoisyMix(DataSet testDC) {
        this.dsetTest = testDC;
    }

    public EvaluateOnNoisyMix() {
    }

    /**
     * Starts the timer.
     */
    public void startTimer() {
        timeTimer = new javax.swing.Timer(1000, timerListener);
        timeTimer.start();
    }
    ActionListener timerListener = new ActionListener() {
        @Override
        public void actionPerformed(ActionEvent e) {
            numSec++;
            try {
            } catch (Exception exc) {
            }
        }
    };

    /**
     * Stops the timer.
     */
    public void stopTimer() {
        timeTimer.stop();
        numSec = 0;
    }

    /**
     * Loads the test data from the specified path.
     *
     * @param path Path to load the data from.
     * @throws Exception
     */
    public void loadData(String path) throws Exception {
        IOARFF persister = new IOARFF();
        dsetTest = persister.load(path);
    }

    /**
     * Sets the writer directory to the specified path.
     *
     * @param path File path to set the writer directory to.
     * @throws Exception
     */
    public void setWriterDir(String path) throws Exception {
        writerDir = new File(path);
    }

    /**
     * @param numTimes Repetitions.
     * @param numClusters Number of Gaussian clusters to generate.
     * @throws Exception
     */
    public void clusterWithAlgorithmOnLabeledData(
            int numTimes, int numClusters) throws Exception {
        NoisyGaussianMix genMix = new NoisyGaussianMix(numClusters, dim,
                DATA_SIZE, false, 0);
        dsetTest = genMix.generateRandomDataSet();
        System.out.println("Random data generated.");
        for (int numNoisy = 0; numNoisy <= MAX_NOISY_INSTANCES;
                numNoisy += NOISE_INCREMENT) {
            System.out.println("noise level: " + numNoisy);
            GHPC clusterer = new GHPC();
            KMeansPlusPlus clustererKM = new KMeansPlusPlus();
            GKH clustererGKH = new GKH();
            GHPKM clustererHPKM = new GHPKM();
            if (numNoisy > 0) {
                genMix.addNoiseToCollection(dsetTest, 500);
            }
            silScores = new float[numTimes];
            avgError = new float[numTimes];
            avgClusterEntropy = new float[numTimes];
            avgSil = 0;
            avgErr = 0;
            avgEntropy = 0;
            float[] silKMScores = new float[numTimes];
            float[] avgKMError = new float[numTimes];
            float[] avgKMClusterEntropy = new float[numTimes];
            float avgKMSil = 0;
            float avgKMErr = 0;
            float avgKMEntropy = 0;
            float[] silHPKMScores = new float[numTimes];
            float[] avgHPKMError = new float[numTimes];
            float[] avgHPKMClusterEntropy = new float[numTimes];
            float avgHPKMSil = 0;
            float avgHPKMErr = 0;
            float avgHPKMEntropy = 0;
            float[] silGKHScores = new float[numTimes];
            float[] avgGKHError = new float[numTimes];
            float[] avgGKHClusterEntropy = new float[numTimes];
            float avgGKHSil = 0;
            float avgGKHErr = 0;
            float avgGKHEntropy = 0;
            float[] silMinScores = new float[numTimes];
            float[] avgMinError = new float[numTimes];
            float[] avgMinClusterEntropy = new float[numTimes];
            float avgMinSil = 0;
            float avgMinErr = 0;
            float avgMinEntropy = 0;
            File currPGKHOutFile = new File(writerDir,
                    "PGKH_N