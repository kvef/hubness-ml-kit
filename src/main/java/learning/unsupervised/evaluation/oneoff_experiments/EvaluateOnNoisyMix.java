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
    private flo