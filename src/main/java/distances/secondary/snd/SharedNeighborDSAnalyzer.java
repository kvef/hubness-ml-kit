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
package distances.secondary.snd;

import data.neighbors.NeighborSetFinder;
import data.neighbors.SharedNeighborFinder;
import data.neighbors.hubness.HubnessExtremesGrabber;
import data.neighbors.hubness.HubnessSkewAndKurtosisExplorer;
import data.neighbors.hubness.HubnessAboveThresholdExplorer;
import data.neighbors.hubness.HubnessVarianceExplorer;
import data.neighbors.hubness.KNeighborEntropyExplorer;
import data.neighbors.hubness.TopHubsClusterUtil;
import data.representation.DataSet;
import data.representation.discrete.DiscretizedDataSet;
import distances.primary.CombinedMetric;
import distances.primary.DistanceMeasure;
import distances.sparse.SparseCombinedMetric;
import distances.sparse.SparseMetric;
import ioformat.FileUtil;
import ioformat.IOARFF;
import ioformat.IOCSV;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import learning.unsupervised.evaluation.quality.QIndexSilhouette;
import util.BasicMathUtil;

/**
 * A utility batch analyzer for shared-neighbor distance effectiveness on a
 * specified list of datasets with the specified primary metrics.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class SharedNeighborDSAnalyzer {

    int kMax;
    // Noise and mislabeling levels to vary.
    float noiseMin, noiseMax, noiseStep, mlMin, mlMax, mlStep;
    // Directory structure for input and output.
    File inConfigFile, inDir, outDir, currOutDSDir;
    // Paths and metric objects.
    ArrayList<String> dsPaths = new ArrayList<>(100);
    ArrayList<CombinedMetric> dsMetric = new ArrayList<>(100);
    // Original dataset and the current one (after applying some modifications).
    DataSet originalDSet, currDSet;
    DiscretizedDataSet currDiscDset;
    // The current metric object.
    CombinedMetric currCmet;
    // Original label array.
    int[] originalLabels;
    // Number of categories in the data.
    int numCategories;
    // Shared-neighbor metric parameters.
    boolean hubnessWeightedSND = true;
    float thetaSimhub = 0;
    int kSND = 50;

    /**
     *
     * @param inConfigFile File that contains the experiment configuration.
     */
    public SharedNeighborDSAnalyzer(File inConfigFile) {
        this.inConfigFile = inConfigFile;
    }

    /**
     * This method runs all the experiments.
     *
     * @throws Exception
     */
    public void runAllTests() throws Exception {
        int counter = 0;
        // For each dataset.
        for (String dsPath : dsPaths) {
            // Load the data.
            File dsFile = new File(dsPath);
            // Currently it has to be specified whether the data is in sparse
            // format or not. If it is, a prefix of "sparse:" is prepended to
            // the specified path.
            if (dsPath.startsWith("sparse:")) {
                String trueDSPath = dsPath.substring(dsPath.indexOf(':') + 1,
                        dsPath.length());
                IOARFF pers = new IOARFF();
                originalDSet = pers.loadSparse(trueDSPath);
            } else {
                if (dsPath.endsWith(".csv")) {
                    IOCSV reader = new IOCSV(true, ",");
                    originalDSet = reader.readData(dsFile);
                } else if (dsPath.endsWith(".arff")) {
                    IOARFF persister = new IOARFF();
                    originalDSet = persister.load(dsPath);
                } else {
                    System.out.println("error, could not read: " + dsPath);
                    continue;
                }
            }
            // Inform the user of the dataset the current tests are running on.
            System.out.println("testing on: " + dsPath);
            originalDSet.standardizeCategories();
            originalLabels = originalDSet.obtainLabelArray();
            numCategories = originalDSet.countCategories();
            int memCleanCount = 0;
            // Go through all the noise and mislabeling levels that were
            // specified in the configuration file. No noise and no mislabeling
            // is also an option, a default one at that.
            for (float noise = noiseMin; noise <= noiseMax; noise +=
                    noiseStep) {
                for (float ml = mlMin; ml <= mlMax; ml += mlStep) {
                    if (++memCleanCount % 5 == 0) {
                        System.gc();
                    }
                    currDSet = originalDSet.copy();
                    if (ml > 0) {
                        currDSet.induceMislabeling(ml, numCategories);
                    }
                    if (noise > 0) {
                        currDSet.addGaussianNoiseToNormalizedCollection(
                                noise, 0.1f);
                    }
                    if (hubnessWeightedSND) {
                        currOutDSDir = new File(outDir,
                                dsFile.getName().substring(
                                0, dsFile.getName().lastIndexOf("."))
                                + "SNH" + this.thetaSimhub + File.separator
                                + "k" + kMax + File.separator + "ml" + ml
                                + File.separator + "noise" + noise);
                    } else {
                        currOutDSDir = new File(outDir,
                                dsFile.getName().substring(0,
                                dsFile.getName().lastIndexOf("."))
                                + "SN" + File.separator + "k" + kMax
                                + File.separator + "ml" + ml + File.separator
                                + "noise" + noise);
                    }
                    FileUtil.createDirectory(currOutDSDir);
                    currCmet = dsMetric.get(counter);
                    float[] cP = currDSet.getClassPriors();

                    // Perform initial calculations.

                    NeighborSetFinder nsfTemp = new NeighborSetFinder(
                            currDSet, currCmet);
                    nsfTemp.calculateDistances();
                    nsfTemp.calculateNeighborSetsMultiThr(kSND, 6);

                    float[][] dMatPrimary = nsfTemp.getDistances();
                    // Primary distance analysis.
                    float maxPrimary = 0;
                    float minPrimary = Float.MAX_VALUE;
                    for (int i = 0; i < dMatPrimary.length; i++) {
                        for (int j = 0; j < dMatPrimary[i].length; j++) {
                            maxPrimary = Math.max(maxPrimary,
                                    dMatPrimary[i][j]);
                            minPrimary = Math.min(minPrimary,
                                    dMatPrimary[i][j]);
                        }
                    }
                    double intraSumPrimary = 0;
                    double interSumPrimary = 0;
                    double intraNumPrimary = 0;
                    double interNumPrimary = 0;
                    // The distribution of intra- and inter-class primary
                    // distances, with 50 bins.
                    double[] intraDistrPrimary = new double[50];
                    double[] interDistrPrimary = new double[50];
                    for (int i = 0; i < dMatPrimary.length; i++) {
                        for (int j = 0; j < dMatPrimary[i].length; j++) {
                            // Re-scale the primary distances.
                            dMatPrimary[i][j] = (dMatPrimary[i][j]
                                    - minPrimary) / (maxPrimary - minPrimary);
                            if (currDSet.getLabelOf(i) == currDSet.
                                    getLabelOf(i + j + 1)) {
                                intraSumPrimary += dMatPrimary[i][j];
                                intraNumPrimary++;
                 