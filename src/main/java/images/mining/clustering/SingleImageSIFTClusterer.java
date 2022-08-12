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

import data.representation.DataSet;
import data.representation.images.sift.LFeatRepresentation;
import distances.primary.CombinedMetric;
import distances.primary.MinkowskiMetric;
import images.mining.display.SIFTDraw;
import ioformat.FileUtil;
import ioformat.IOARFF;
import ioformat.images.SiftUtil;
import java.awt.image.BufferedImage;
import java.io.File;
import learning.unsupervised.Cluster;
import learning.unsupervised.ClusteringAlg;
import learning.unsupervised.evaluation.quality.OptimalConfigurationFinder;
import learning.unsupervised.methods.KMeans;
import util.CommandLineParser;

/**
 * This utility class enables the users to cluster the SIFT features on a single
 * image and output the clusters in an ARFF format.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class SingleImageSIFTClusterer {

    private LFeatRepresentation features = null;
    private CombinedMetric cmet = null;
    private OptimalConfigurationFinder selector = null;
    private int minClusters;
    private int maxClusters;
    private static final int NUM_REPETITIONS = 10;

    /**
     * @return SIFTRepresentation object that holds the features on the image.
     */
    public LFeatRepresentation getFeatures() {
        return features;
    }

    /**
     * @param minClusters Integer that is the minimal number of clusters to try
     * and use.
     * @param maxClusters Integer that is the maximal number of clusters to try
     * and use.
     */
    public void setBounds(int minClusters, int maxClusters) {
        this.minClusters = minClusters;
        this.maxClusters = maxClusters;
    }

    /**
     * @param cmet CombinedMetric object for distance calculations.
     */
    public void setCombinedMetric(CombinedMetric cmet) {
        this.cmet = cmet;
    }

    /**
     * @param features SIFTRepresentation object that holds the features on the
     * image.
     */
    public void setRepresentation(LFeatRepresentation features) {
        this.features = features;
    }

    /**
     * @param selector OptimalConfigurationFinder for finding the optimal
     * cluster configuration among all generated configurations.
     */
    public void setConfigurationSelector(OptimalConfigurationFinder selector) {
        this.selector = selector;
    }

    /**
     * Loads the SIFT features of an image from an ARFF file.
     *
     * @param inPath String that is the path to the ARFF file containing the
     * image SIFT features.
     */
    private void loadFeatures(String inPath) {
        try {
            features = SiftUtil.importFeaturesFromArff(inPath);
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
    }

    /**
     */
    public SingleImageSIFTClusterer() {
    }

    /**
     * Initialization.
     *
     * @param inPath String that is the path to the ARFF file containing the
     * image SIFT features.
     * @param cmet CombinedMetric object for distance calculations.
     * @param selector OptimalConfigurationFinder for finding the optimal
     * cluster configuration among all generated configurations.
     * @param minClusters Integer that is the minimal number of clusters to try
     * and use.
     * @param maxClusters Integer that is the maximal number of clusters to try
     * and use.
     */
    public SingleImageSIFTClusterer(String inPath, CombinedMetric cmet,
            OptimalConfigurationFinder selector, int minClusters,
            int maxClusters) {
        loadFeatures(inPath);
        this.selector = selector;
        this.cmet = cmet;
        this.minClusters = minClusters;
        this.maxClusters = maxClusters;
    }

    /**
     * Perform a single clustering run for a fixed number of clusters on the
     * features loaded from an ARFF file and persist the clusters in an output
     * ARFF file.
     *
     * @param inPath String that is the path to the ARFF file containing the
     * image SIFT features.
     * @param outPath
     * @param numClust
     * @throws Exception
     */
    public static void clusterARFFToARFF(String inPath,
            String outPath, int numClust) throws Exception {
        File inFile = new File(inPath);
        if (numClust < 1) {
            throw new Exception("Number of clusters must be positive, not "
                    + numClust);
        }
        if (!inFile.exists() || !inFile.isFile()) {
            throw new Exception("Bad input path " + inPath);
        }
        Cluster[] clusterConfiguration;
        // Load the features.
        LFeatRepresentation features = SiftUtil.importFeaturesFromArff(inPath);
        CombinedMetric cmet = new CombinedMetric(null,
                new MinkowskiMetric(), CombinedMetric.DEFAULT);
        // Perform the clustering.
        KMeans clusterer = new KMeans(features, cmet, numClust);
        clusterer.cluster();
        clusterConfiguration = clusterer.getClusters();
        DataSet outDSet = new DataSet();
        outDSet.fAttrNames =
                clusterConfiguration[0].getDefinitionDataset().fAttrNames.
                clone();
        outDSet.iAttrNames = new String[1];
        outDSet.iAttrNames[0] = "Cluster";
        outDSet.sAttrNames = new String[1];
        outDSet.sAttrNames[0] = "ImageName";
        persistClusters(outPath, clusterConfiguration, outDSet);
    }

    /**
     * This method performs the clustering and finds the optimal cluster
     * configuration based on a product of different cluster validity indices.
     *
     * @return Cluster[] that is the cluster configuration produced by the
     * clustering and selected as the optimal one.
     * @throws Exception
     */
    public Cluster[] clusterImageByIndexProduct() throws Exception {
        Cluster[][] clusteringConfigurations =
                new Cluster[(maxClusters - minClusters + 1) *
                NUM_REPETITIONS][];
        Cluster[][] nonDuplicateConfigurationArray =
                new Cluster[(maxClusters - minClusters + 1)][];
        Cluster[][] repetitionsArray;
        selector.setDataSet(features);
        for (int nClusters = minClusters; nClusters <= maxClusters;
                nClusters++) {
            System.out.println("Clustering for : " + nClusters + " clusters.");
            repetitionsArray = new Cluster[NUM_REPETITIONS][];
            for (int rIndex = 0; rIndex < NUM_REPETITIONS; rIndex++) {
                KMeans clusterer = new KMeans(features, cmet, nClusters);
                clusterer.cluster();
                clusteringConfigurations[NUM_REPETITIONS
          