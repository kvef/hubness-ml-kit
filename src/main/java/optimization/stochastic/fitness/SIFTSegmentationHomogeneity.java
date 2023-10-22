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
package optimization.stochastic.fitness;

import data.representation.DataInstance;
import data.representation.images.sift.LFeatRepresentation;
import data.representation.images.sift.LFeatVector;
import images.mining.clustering.IntraImageKMeansAdapted;
import images.mining.clustering.IntraImageKMeansAdaptedScale;
import images.mining.clustering.IntraImageKMeansWeighted;
import ioformat.images.SegmentationIO;
import ioformat.images.SiftUtil;
import java.awt.image.BufferedImage;
import java.io.File;
import javax.imageio.ImageIO;
import learning.unsupervised.Cluster;
import learning.unsupervised.ClusteringAlg;
import util.BasicMathUtil;

/**
 * Evaluates the fitness of image segmentation by the homogeneity of SIFT
 * clusters.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class SIFTSegmentationHomogeneity implements FitnessEvaluator {

    private boolean useRank = false; // Use ranks instead of distances.
    private File inImagesDir;
    private File inSIFTDir;
    private File inSegmentDir;
    private String[] nameList;
    private BufferedImage[] images;
    private LFeatRepresentation[] siftReps;
    private int[][][] segmentations;
    private int[] numSegments;
    private int minClusters = 10;
    private int maxClusters = 15;
    private float avgTotalEnt;
    private float avgImgEnt = 0;
    private float[][] segmentClassDistr;
    private int[] segmentTotals;
    private int numNonEmptySegs;

    /**
     * @param inImagesDir Directory containing the target images.
     * @param inSIFTDir Directory containing the extracted SIFT descriptors.
     * @param inSegmentDir Directory containing the segmentations.
     * @param nameList List of image names.
     * @param minClusters Minimal number of clusters to try.
     * @param maxClusters Maximal number of clusters to try.
     * @param useRank Whether to use rank or distances directly.
     * @return An instance of the SIFTSegmentationHomogeneity.
     * @throws Exception
     */
    public static SIFTSegmentationHomogeneity newInstance(
            File inImagesDir,
            File inSIFTDir,
            File inSegmentDir,
            String[] nameList,
            int minClusters,
            int maxClusters,
            boolean useRank) throws Exception {
        SIFTSegmentationHomogeneity fe = new SIFTSegmentationHomogeneity(
                inImagesDir,
                inSIFTDir,
                inSegmentDir,
                nameList,
                minClusters,
                maxClusters,
                useRank);
        fe.populate();
        return fe;
    }

    /**
     * @param inImagesDir Directory containing the target images.
     * @param inSIFTDir Directory containing the extracted SIFT descriptors.
     * @param inSegmentDir Directory contain