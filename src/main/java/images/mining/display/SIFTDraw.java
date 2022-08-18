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
package images.mining.display;

import data.representation.DataInstance;
import data.representation.images.sift.LFeatRepresentation;
import data.representation.images.sift.LFeatVector;
import data.representation.images.sift.util.ClusteredSIFTRepresentation;
import draw.basic.BoxBlur;
import draw.basic.RotatedEllipse;
import ioformat.IOARFF;
import ioformat.images.SiftUtil;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.Random;
import javax.imageio.ImageIO;
import learning.unsupervised.Cluster;
import statistics.Variance2D;
import util.ImageUtil;

/**
 * This class enables a visualization of SIFT feature distributions on top of an
 * image.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class SIFTDraw {

    // Clusters of visual words correspond to object in an image.
    private Cluster[] visualObjectClusters = null;
    // The basis image.
    private BufferedImage image = null;

    /**
     *
     * @param visualObjectClusters Clusters of features in the image.
     * @param imagePath String that is the path to load the image from.
     */
    public SIFTDraw(Cluster[] visualObjectClusters, String imagePath) {
        this.visualObjectClusters = visualObjectClusters;
        try {
            image = ImageIO.read(new File(imagePath));
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
    }

    /**
     *
     * @param visualObjectClusters Clusters of features in the image.
     * @param image BufferedImage object that is the basis image.
     */
    public SIFTDraw(Cluster[] visualObjectClusters, BufferedImage image) {
        this.visualObjectClusters = visualObjectClusters;
        this.image = image;
    }

    /**
     * This method draws the ellipses that correspond to SIFT clusters on top of
     * an image.
     *
     * @param ellipses Array of RotatedEllipse objects corresponding to SIFT
     * clusters.
     * @param oldImage Image that the ellipses will be drawn on top of.
     * @param useGradientDraw Whether to use gradients when drawing the
     * ellipses.
     * @return BufferedImage object that is the image with ellipses drawn on
     * top.
     * @throws Exception
     */
    public static BufferedImage drawClusterEllipsesOnImage(
            RotatedEllipse[] ellipses, BufferedImage oldImage,
            boolean useGradientDraw) throws Exception {
        if (oldImage == null) {
            return null;
        }
        if (ellipses == null || ellipses.length == 0) {
            // For consistency, even if no ellipses are to be drawn, we create
            // a new image object.
            return ImageUtil.copyImage(oldImage);
        }
        BufferedImage newImage = ImageUtil.copyImage(oldImage);
        Color[] ellipseColors = new Color[ellipses.length];
        Graphics2D graphics = newImage.createGraphics();
        Random randa = new Random();
        // Assign random colors to the ellipses.
        for (int i = 0; i < ellipseColors.length; i++) {
            if (useGradientDraw) {
                ellipseColors[i] = new Color(randa.nextFloat(),
                        randa.nextFloat(), randa.nextFloat(), 0.75f);
            } else {
                ellipseColors[i] = new Color(randa.nextFloat(),
                        randa.nextFloat(), randa.nextFloat(), 0.5f);
            }
        }
        for (int i = 0; i < ellipses.length; i++) {
            ellipses[i].setColor(ellipseColors[i]);
            if (!useGradientDraw) {
                ellipses[i].drawOnGraphics(graphics);
            } else {
                ellipses[i].drawWithGradient(graphics);
            }
        }
        return newImage;
    }

    /**
     * This method draws the ellipses that correspond to SIFT clusters on top of
     * an image.
     *
     * @param features ClusteredSIFTRepresentation object representing clusters
     * of SIFT features.
     * @param image Image that the ellipses will be drawn on top of.
     * @param outImagePath String that is the path where the new image will be
     * persisted.
     * @param useGradientDraw Whether to use gradients when drawing the
     * ellipses.
     * @throws Exception
     */
    public static void drawClustersOnImageAsEllipses(
            ClusteredSIFTRepresentation features, BufferedImage image,
            String outImagePath, boolean useGradientDraw) throws Exception {
        if (image == null) {
            return;
        }
        if (features == null || features.isEmpty()) {
            return;
        }
        // Get the cluster configuration.
        Cluster[] clusters = features.representAsClusters();
        Color[] clusterColors = new Color[clusters.length];
        Graphics2D graphics = image.createGraphics();
        // Assign random colors to the clusters for display.
        Random randa = new Random();
        for (int i = 0; i < clusterColors.length; i++) {
            if (useGradientDraw) {
                clusterColors[i] = new Color(randa.nextFloat(),
                        randa.nextFloat(), randa.nextFloat(), 0.75f);
            } else {
                clusterColors[i] = new Color(randa.nextFloat(),
                        randa.nextFloat(), randa.nextFloat(), 0.5f);
            }
        }
        // Find the variance vectors.
        Variance2D var = new Variance2D();
        RotatedEllipse[] ellipses = var.
                findVarianceEllipseForSIFTCLusterConfiguration(clusters);
        for (int i = 0; i < clusters.length; i++) {
            ellipses[i].setColor(clusterColors[i]);
            if (!useGradientDraw) {
                ellipses[i].drawOnGraphics(graphics);
            } else {
                ellipses[i].drawWithGradient(graphics);
            }
        }
        // Persist the resulting image.
        File outImageFile = new File(outImagePath);
        ImageIO.write(image, "jpg", outImageFile);
    }

    /**
     * This method draws the ellipses that correspond to SIFT clusters on top of
     * an image.
     *
     * @param arffPath String that is the path to the .arff file containing the
     * ClusteredSIFTRepresentation that describes SIFT clusters on an image.
     * @param image Image that the ellipses will be drawn on top of.
     * @param outImagePath String that is the path where the new image will be
     * persisted.
     * @param useGradientDraw Whether to use gradients when drawing the
     * ellipses.
     * @throws Exception
     */
    public static void drawClustersOnImageAsEllipses(
            String arffPath, BufferedImage image,
            String outImagePath, boolean useGradientDraw) throws Exception {
        IOARFF arff = new IOARFF();
        ClusteredSIFTRepresentation features = new ClusteredSIFTRepresentation(
                new LFeatRepresentation(arff.load(arffPath)));
        drawClustersOnImageAsEllipses(features, image, outImagePath,
                useGradientDraw);
    }

    /**
     * This method draws the ellipses that correspond to SIFT clusters on top of
     * an image.
     *
     * @param arffPath String that is the path to the .arff file containing the
     * ClusteredSIFTRepresent