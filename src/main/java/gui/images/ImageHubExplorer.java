
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
package gui.images;

import data.neighbors.NeighborSetFinder;
import data.neighbors.SharedNeighborFinder;
import data.neighbors.hubness.BucketedOccDistributionGetter;
import data.neighbors.hubness.HubOrphanRegularPercentagesCalculator;
import data.neighbors.hubness.HubnessExtremesGrabber;
import data.neighbors.hubness.HubnessSkewAndKurtosisExplorer;
import data.neighbors.hubness.HubnessAboveThresholdExplorer;
import data.neighbors.hubness.KNeighborEntropyExplorer;
import data.representation.DataInstance;
import data.representation.DataSet;
import data.representation.images.color.ColorHistogramVector;
import data.representation.images.sift.LFeatRepresentation;
import distances.primary.CombinedMetric;
import distances.secondary.LocalScalingCalculator;
import distances.secondary.MutualProximityCalculator;
import distances.secondary.NICDMCalculator;
import distances.secondary.snd.SharedNeighborCalculator;
import draw.basic.BoxBlur;
import draw.basic.ColorPalette;
import draw.basic.ScreenImage;
import draw.charts.PieRenderer;
import edu.uci.ics.jung.algorithms.layout.CircleLayout;
import edu.uci.ics.jung.algorithms.layout.Layout;
import edu.uci.ics.jung.graph.DirectedGraph;
import edu.uci.ics.jung.graph.DirectedSparseMultigraph;
import edu.uci.ics.jung.graph.Graph;
import edu.uci.ics.jung.graph.util.Context;
import edu.uci.ics.jung.graph.util.EdgeType;
import edu.uci.ics.jung.visualization.VisualizationViewer;
import edu.uci.ics.jung.visualization.control.PickingGraphMousePlugin;
import edu.uci.ics.jung.visualization.control.PluggableGraphMouse;
import edu.uci.ics.jung.visualization.picking.PickedState;
import ioformat.FileUtil;
import ioformat.IOARFF;
import ioformat.images.ConvertJPGToPGM;
import ioformat.images.SiftUtil;
import java.awt.Color;
import java.awt.Component;
import java.awt.ComponentOrientation;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Paint;
import java.awt.Shape;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Vector;
import javax.imageio.ImageIO;
import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import javax.swing.table.DefaultTableCellRenderer;
import javax.swing.table.DefaultTableModel;
import javax.swing.table.TableModel;
import learning.supervised.Classifier;
import learning.supervised.interfaces.DistMatrixUserInterface;
import learning.supervised.interfaces.NeighborPointsQueryUserInterface;
import learning.supervised.methods.knn.AKNN;
import learning.supervised.methods.knn.DWHFNN;
import learning.supervised.methods.knn.FNN;
import learning.supervised.methods.knn.HwKNN;
import learning.supervised.methods.knn.HIKNN;
import learning.supervised.methods.knn.KNN;
import learning.supervised.methods.knn.NHBNN;
import data.neighbors.NSFUserInterface;
import images.mining.codebook.GenericCodeBook;
import ioformat.images.OpenCVFeatureIO;
import java.awt.HeadlessException;
import java.io.IOException;
import learning.supervised.methods.knn.NWKNN;
import mdsj.MDSJ;
import org.apache.commons.collections15.Predicate;
import org.apache.commons.collections15.Transformer;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.PiePlot3D;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.renderer.category.StackedBarRenderer;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.data.general.DefaultPieDataset;
import org.jfree.util.Rotation;
import util.ArrayUtil;
import util.AuxSort;
import util.BasicMathUtil;

/**
 * This GUI was made with the intention of helping with analyzing the hubness of
 * the data and image data in particular, in terms of the skewed distribution of
 * implied relevance stemming from a particular choice of metric and feature
 * representation. It offers the users a choice between many standard primary
 * metrics and a set of state-of-the-art secondary metrics for hubness-aware
 * metric learning in order to improve the semantic consistency of between-image
 * similarities. It is composed of many visualization components for different
 * types of data overviews, with an emphasis on kNN set structure and hub
 * analysis. When using it, make sure to cite the following paper: Image Hub
 * Explorer: Evaluating Representations and Metrics for Content-based Image
 * Retrieval and Object Recognition, Nenad Tomasev and Dunja Mladenic, 2013,
 * ECML/PKDD conference.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ImageHubExplorer extends javax.swing.JFrame {

    // The workspace directory.
    private File workspace;
    // The current working directory.
    private File currentDirectory;
    private static final int PRIMARY_METRIC = 0;
    private static final int SECONDARY_METRIC = 1;
    private ImageHubExplorer frameReference = this;
    // Codebook data structures for visual word analysis.
    private GenericCodeBook codebook = null;
    private double[][] codebookProfiles = null;
    private float[] codebookGoodness = null;
    // Whether to also load the secondary distances from the disk (if available)
    // or to calculate them inside Image Hub Explorer. If the latter is selected
    // then the metric object is also available for external searches.
    private boolean secondaryLoadFlag = false;
    // This flag informs the methods that the system is currently calculating
    // something, so the user is prevented from executing certain actions.
    private volatile boolean busyCalculating = false;
    // Image data that is being analyzed.
    private BufferedImage[] images;
    // Thumbnails of the images for visualization.
    private ArrayList<BufferedImage> thumbnails;
    // Reverse neighbor sets for all the images for all the neighborhood sizes.
    private ArrayList<Integer>[][] rnnSetsAllK;
    // Neighbor occurrence profiles for all neighborhood sizes.
    private float[][][] occurrenceProfilesAllK;
    // Quantized data representation, if a representation is available. It is
    // possible to operate based on the loaded distances alone, if that is
    // necessary in the context of analysis.
    private DataSet quantizedRepresentation;
    // Number of classes in the data.
    private int numClasses;
    // Colors to use for different classes.
    private Color[] classColors;
    // Names of different classes.
    private String[] classNames;
    // Files containing primary and secondary distance matrices.
    private File primaryDMatFile;
    private File secondaryDMatFile;
    // Primary distance matrix.
    private float[][] distMatrixPrimary;
    // kNN sets in the primary distance.
    private NeighborSetFinder nsfPrimary;
    // Secondary distance matrix.
    private float[][] distMatrixSecondary;
    // kNN sets in the secondary distance.
    private NeighborSetFinder nsfSecondary;
    // CombinedMetric objects for distance calculations.
    private CombinedMetric primaryCMet = null;
    private CombinedMetric secondaryCMet = null;
    // We keep track of the selection history here.
    private ArrayList<Integer> selectedImageHistory;
    // This is the index of the current selection in the history. If we are
    // going back and forth, we move this index and the user can easily browse
    // through the browsing history.
    private int selectedImageIndexInHistory = 0;
    // These maps map the paths to specific instance indexes.
    private HashMap<String, Integer> pathIndexMap = null;
    private HashMap<String, Integer> pathIndexMapThumbnail = null;
    // Lists of image paths and image thumbnail paths.
    private ArrayList<String> imgPaths = null;
    private ArrayList<String> imgThumbPaths = null;
    // The current neighborhood size.
    private volatile int neighborhoodSize = 5;
    // Statistics related to the kNN topology and the hubness of the data.
    // Percentage of points that occur at least once as neighbors, over all
    // neighborhood sizes.
    private float[] aboveZeroArray = null;
    // Neighbor occurrence distribution skewness, over all neighborhood sizes.
    private float[] skewArray = null;
    // Neighbor occurrence distribution kurtosis, over all neighborhood sizes.
    private float[] kurtosisArray = null;
    // Highest neighbor occurrence counts, over all neighborhood sizes.
    private float[][] highestHubnesses = null;
    // Indexes of top hubs in the data, over all neighborhood sizes.
    private int[][] highestHubIndexes = null;
    // kNN set entropies, over all neighborhood sizes.
    private float[] kEntropies = null;
    // Reverse kNN set entropies, over all neighborhood sizes.
    private float[] reverseKNNEntropies = null;
    // kNN set entropy skewness, over all neighborhood sizes.
    private float[] kEntropySkews = null;
    // Reverse kNN set entropy skews, over all neighborhood sizes.
    private float[] reverseKNNEntropySkews = null;
    // Label mismatch percentages in kNN sets, over all neighborhood sizes.
    private float[] badHubnessArray = null;
    // Global class to class hubness, over all neighborhood sizes.
    private float[][][] globalClassToClasshubness = null;
    // Percentages of points that are hubs, over all neighborhood sizes.
    private float[] hubPercs = null;
    // Percentages of points that are orphans, over all neighborhood sizes.
    private float[] orphanPercs = null;
    // Percentages of points that are regular points, over all neighborhood
    // sizes.
    private float[] regularPercs = null;
    // Occurrence distributions as histograms with a fixed bucket width.
    private int[][] bucketedOccurrenceDistributions = null;
    // The current query image.
    private BufferedImage queryImage;
    // The representation of the current query.
    private DataInstance queryImageRep;
    // Local image features of the current query, if available.
    private LFeatRepresentation queryImageLFeat;
    // Neighbors of the current query image.
    private int[] queryImageNeighbors;
    // Distances to the neighbors of the current query image.
    private float[] queryImageNeighborDists;
    // Neighborhood size used for the current query.
    private int kQuery = 10;
    // Whether the kNN stats have already been calculated or not.
    private boolean neighborStatsCalculated = false;
    // Whether the classifier models have already been trained or not.
    private boolean trainedModels = false;
    // A list of classifiers.
    private Classifier[] classifiers;
    // A list of classifier names for display, as these may differ from the
    // implementaiton names.
    private String[] classifierNameList = {"kNN", "FNN", "NWKNN", "AKNN",
        "hw-kNN", "h-FNN", "HIKNN", "NHBNN"};
    // Lists of top hub, good hub and bad hub indexes for each class for all
    // neighborhood sizes.
    private ArrayList<Integer>[][] classTopHubLists;
    private ArrayList<Integer>[][] classTopGoodHubsList;
    private ArrayList<Integer>[][] classTopBadHubsList;
    // Corresponding occurrence frequencies to the above lists.
    private ArrayList<Integer>[][] classHubnessArrValues;
    private ArrayList<Integer>[][] classHubnessArrGoodValues;
    private ArrayList<Integer>[][] classHubnessArrBadValues;
    // A list of image indexes that belongs to each particular class.
    private ArrayList<Integer>[] classImageIndexes = new ArrayList[numClasses];
    // Point type distributions for all classes.
    private float[][] classPTypes;
    // Per-class visualizations of the influence of hubness.
    private ClassHubsPanel[] classStatsOverviews;
    // Image coordinates for the MDS screen.
    private float[][] imageCoordinatesXY;
    // Number of images to show in the MDS screen.
    private int numImagesDrawn = 300;
    // Maximum and minimum display scale.
    private int maxImageScale = 80;
    private int minImageScale = 10;
    // Calculated MDS landscapes. Calculating a landscape takes some time, so
    // this helps with quickly changing the background when the neighborhood
    // size is changed, as previously calculated backgrounds are then shown
    // instead of being re-calculated.
    private BufferedImage[] mdsBackgrounds;
    // kNN graphs for all the k-values.
    private DirectedGraph<ImageNode, NeighborLink>[] neighborGraphs;
    private ArrayList<Integer> vertexIndexes;
    private ArrayList<ImageNode> vertices;
    private ArrayList<NeighborLink>[] edges;
    private HashMap<Integer, ImageNode> verticesHash;
    private HashMap<Integer, Integer> verticesNodeIndexHash;
    private VisualizationViewer[] graphVServers;
    // File containing the codebook profile for visual word utility assessment.
    File codebookProfileFile = null;
    // Panels for codebook vector profiles.
    CodebookVectorProfilePanel[] codebookProfPanels;

    /**
     * Nodes for kNN graph visualizations.
     */
    static class ImageNode {

        int id;
        ImageIcon icon;
        String thumbPath;

        /**
         * Initialization.
         *
         * @param id Integer that is the node ID and will correspond to the
         * index of the image in the representation.
         * @param icon ImageIcon that is the image thumbnail.
         * @param thumbPath String that is the thumbnail path.
         */
        public ImageNode(int id, ImageIcon icon, String thumbPath) {
            this.icon = icon;
            this.id = id;
            this.thumbPath = thumbPath;
        }

        /**
         * @return ImageIcon that is the image thumbnail.
         */
        public Icon getIcon() {
            return icon;
        }

        @Override
        public String toString() {
            return thumbPath;
        }
    }

    /**
     * Edges for kNN graph visualization.
     */
    static class NeighborLink {

        // The total number of edges.
        private static int edgeCount = 0;
        // Edge weight.
        double weight;
        // Edge ID.
        int id;
        // Source and target ImageNode that are connected due to the neighbor
        // relation.
        ImageNode source, target;

        /**
         * Initialization.
         *
         * @param weight Double that is the edge weight.
         * @param source ImageNode that is the source vertex for this edge.
         * @param target ImageNode that is the target vertex for this edge.
         */
        public NeighborLink(double weight, ImageNode source, ImageNode target) {
            this.id = edgeCount++;
            this.weight = weight;
            this.source = source;
            this.target = target;
        }

        @Override
        public String toString() {
            return " " + BasicMathUtil.makeADecimalCutOff(weight, 3);
        }

        /**
         * @return ImageNode that is the source vertex for this edge.
         */
        public ImageNode getSource() {
            return source;
        }

        /**
         * @return ImageNode that is the target vertex for this edge.
         */
        public ImageNode getTarget() {
            return target;
        }
    }

    /**
     * This method deletes and resets all the kNN graphs.
     */
    private void graphsDelete() {
        if (neighborGraphs != null) {
            for (int i = 0; i < neighborGraphs.length; i++) {
                neighborGraphs[i] = null;
                graphVServers[i] = null;
            }
        }
        neighborGraphs = null;
        vertexIndexes = null;
        vertices = null;
        edges = null;
        neighborGraphScrollPane.setViewportView(null);
        neighborGraphScrollPane.revalidate();
        neighborGraphScrollPane.repaint();
        System.gc();
    }

    /**
     * This method removes the currently selected image from the kNN graph
     * visualizations for all neighborhood sizes.
     */
    private void removeSelectedImageFromGraph() {
        if (neighborStatsCalculated && neighborGraphs != null
                && selectedImageHistory != null
                && selectedImageHistory.size() > 0) {
            int index = selectedImageHistory.get(selectedImageIndexInHistory);
            if (verticesHash.containsKey(index)) {
                ImageNode delVertex = verticesHash.get(index);
                // Create two lists of edges, those that need to be deleted and
                // those that will be retained. Create one such list for each
                // neighborhood size.
                ArrayList<NeighborLink>[] retainedEdges;
                ArrayList<NeighborLink>[] discardedEdges;
                retainedEdges = new ArrayList[50];
                discardedEdges = new ArrayList[50];
                for (int kTmp = 0; kTmp < 50; kTmp++) {
                    retainedEdges[kTmp] = new ArrayList<>(500);
                    discardedEdges[kTmp] = new ArrayList<>(500);
                    for (int i = 0; i < edges[kTmp].size(); i++) {
                        if (edges[kTmp].get(i) != null) {
                            if (delVertex.equals(edges[kTmp].get(i).getSource())
                                    || delVertex.equals(
                                    edges[kTmp].get(i).getTarget())) {
                                // Discard the edge.
                                discardedEdges[kTmp].add(edges[kTmp].get(i));
                            } else {
                                // Keep the edge.
                                retainedEdges[kTmp].add(edges[kTmp].get(i));
                            }
                        }
                    }
                    // Update the internal edge lists.
                    edges[kTmp] = retainedEdges[kTmp];
                    // Remove the removed edges from the graphs.
                    for (int i = 0; i < discardedEdges[kTmp].size(); i++) {
                        neighborGraphs[kTmp].removeEdge(
                                discardedEdges[kTmp].get(i));
                    }
                    neighborGraphs[kTmp].removeVertex(delVertex);
                    graphVServers[kTmp].revalidate();
                    graphVServers[kTmp].repaint();
                }
                verticesHash.remove(index);
            }
            // Update the graphical components.
            neighborGraphScrollPane.revalidate();
            neighborGraphScrollPane.repaint();
        }
    }

    /**
     * Add a list of images to the kNN graph visualizations.
     *
     * @param indexes ArrayList<Integer> of image indexes to insert.
     */
    private void addSelectedImagesToGraph(ArrayList<Integer> indexes) {
        int[] aIndexes = new int[0];
        if (indexes != null) {
            aIndexes = new int[indexes.size()];
            for (int i = 0; i < indexes.size(); i++) {
                aIndexes[i] = indexes.get(i);
            }
        }
        // Delegate to a method that operates on the index arrays.
        addSelectedImagesToGraph(aIndexes);
    }

    /**
     * Add an array of images to the kNN graph visualizations.
     *
     * @param indexes int[] of image indexes to insert.
     */
    private void addSelectedImagesToGraph(int[] indexes) {
        if (neighborStatsCalculated && neighborGraphs != null) {
            for (int index : indexes) {
                if (verticesHash.containsKey(index)) {
                    // If the image is already contained in the graphs, skip it.
                    continue;
                }
                // Create a new node to insert.
                ImageNode newVertex = new ImageNode(
                        index, new ImageIcon(imgThumbPaths.get(index)),
                        imgThumbPaths.get(index));
                // Add the node to the vertex set.
                vertexIndexes.add(index);
                vertices.add(newVertex);
                for (int kTmp = 0; kTmp < 50; kTmp++) {
                    neighborGraphs[kTmp].addVertex(newVertex);
                    verticesHash.put(index, newVertex);
                    verticesNodeIndexHash.put(index, vertices.size() - 1);
                    graphVServers[kTmp].revalidate();
                    graphVServers[kTmp].repaint();
                    neighborGraphScrollPane.revalidate();
                    neighborGraphScrollPane.repaint();
                }
            }
            // This might be improved. All edges are removed from the graphs
            // and then inserted anew.
            for (int kTmp = 0; kTmp < 50; kTmp++) {
                for (int i = 0; i < edges[kTmp].size(); i++) {
                    neighborGraphs[kTmp].removeEdge(edges[kTmp].get(i));
                }
                edges[kTmp] = new ArrayList<>(100);
                graphVServers[kTmp].revalidate();
                graphVServers[kTmp].repaint();
                neighborGraphScrollPane.revalidate();
                neighborGraphScrollPane.repaint();
            }
            NeighborSetFinder nsf = getNSF();
            int[][] kneighbors = nsf.getKNeighbors();
            float[][] kdistances = nsf.getKDistances();
            // For all the neighborhood sizes in the range.
            for (int kTmp = 0; kTmp < 50; kTmp++) {
                for (int i = 0; i < vertices.size(); i++) {
                    // For all the neighbors.
                    for (int kN = 0; kN < kTmp + 1; kN++) {
                        if (verticesHash.containsKey(
                                kneighbors[vertexIndexes.get(i)][kN])) {
                            NeighborLink newEdge =
                                    new NeighborLink(kdistances[
                                    vertexIndexes.get(i)][kN], vertices.get(i),
                                    verticesHash.get(kneighbors[
                                    vertexIndexes.get(i)][kN]));
                            neighborGraphs[kTmp].addEdge(newEdge,
                                    vertices.get(i), verticesHash.get(
                                    kneighbors[vertexIndexes.get(i)][kN]),
                                    EdgeType.DIRECTED);
                            edges[kTmp].add(newEdge);
                        }
                    }
                }
                graphVServers[kTmp].revalidate();
                graphVServers[kTmp].repaint();
            }
            // Determine how to display the nodes.
            Layout<ImageNode, NeighborLink> layout =
                    new CircleLayout(neighborGraphs[neighborhoodSize - 1]);
            layout.setSize(new Dimension(500, 500));
            layout.initialize();
            VisualizationViewer<ImageNode, NeighborLink> vv =
                    new VisualizationViewer<>(layout);
            vv.setPreferredSize(new Dimension(550, 550));
            vv.setMinimumSize(new Dimension(550, 550));
            vv.setDoubleBuffered(true);
            vv.setEnabled(true);
            graphVServers[neighborhoodSize - 1] = vv;
            vv.getRenderContext().setVertexIconTransformer(
                    new IconTransformer());
            vv.getRenderContext().setVertexShapeTransformer(
                    new ShapeTransformer());
            vv.getRenderContext().setEdgeArrowPredicate(
                    new DirectionDisplayPredicate());
            vv.getRenderContext().setEdgeLabelTransformer(
                    new Transformer() {
                @Override
                public String transform(Object e) {
                    return (e.toString());
                }
            });
            PluggableGraphMouse gm = new PluggableGraphMouse();
            gm.add(new PickingGraphMousePlugin());
            vv.setGraphMouse(gm);
            vv.setBackground(Color.WHITE);
            vv.setVisible(true);
            final PickedState<ImageNode> pickedState =
                    vv.getPickedVertexState();
            pickedState.addItemListener(new ItemListener() {
                @Override
                public void itemStateChanged(ItemEvent e) {
                    Object subject = e.getItem();
                    if (subject instanceof ImageNode) {
                        ImageNode vertex = (ImageNode) subject;
                        if (pickedState.isPicked(vertex)) {
                            setSelectedImageForIndex(vertex.id);
                        } else {
                        }
                    }
                }
            });
            vv.validate();
            vv.repaint();
            neighborGraphScrollPane.setViewportView(vv);
            neighborGraphScrollPane.revalidate();
            neighborGraphScrollPane.repaint();
        }
    }

    /**
     * Inserts the currently selected image into all kNN graphs for
     * visualization.
     */
    private void addSelectedImageToGraph() {
        if (neighborStatsCalculated && neighborGraphs != null
                && selectedImageHistory != null
                && selectedImageHistory.size() > 0) {
            int index = selectedImageHistory.get(selectedImageIndexInHistory);
            if (verticesHash.containsKey(index)) {
                // If it is already contained within the graphs, do nothing.
                return;
            }
            // Create a new node for the image.
            ImageNode newVertex = new ImageNode(index,
                    new ImageIcon(imgThumbPaths.get(index)),
                    imgThumbPaths.get(index));
            vertexIndexes.add(index);
            vertices.add(newVertex);
            NeighborSetFinder nsf = getNSF();
            int[][] kneighbors = nsf.getKNeighbors();
            float[][] kdistances = nsf.getKDistances();
            // For all relevant neighborhood sizes.
            for (int kTmp = 0; kTmp < 50; kTmp++) {
                neighborGraphs[kTmp].addVertex(newVertex);
                verticesHash.put(index, newVertex);
                verticesNodeIndexHash.put(index, vertices.size() - 1);
                // Re-draw all the edges.
                // This should be changed to only update the affected ones.
                for (int i = 0; i < edges[kTmp].size(); i++) {
                    neighborGraphs[kTmp].removeEdge(edges[kTmp].get(i));
                }
                for (int i = 0; i < vertices.size(); i++) {
                    for (int kN = 0; kN < kTmp; kN++) {
                        if (verticesHash.containsKey(
                                kneighbors[vertexIndexes.get(i)][kN])) {
                            NeighborLink newEdge =
                                    new NeighborLink(
                                    kdistances[vertexIndexes.get(i)][kN],
                                    vertices.get(i),
                                    verticesHash.get(
                                    kneighbors[vertexIndexes.get(i)][kN]));
                            neighborGraphs[kTmp].addEdge(newEdge,
                                    vertices.get(i),
                                    verticesHash.get(
                                    kneighbors[vertexIndexes.get(i)][kN]),
                                    EdgeType.DIRECTED);
                            edges[kTmp].add(newEdge);
                        }
                    }
                }
                graphVServers[kTmp].revalidate();
                graphVServers[kTmp].repaint();
            }
            // Set up how the nodes will be drawn.
            Layout<ImageNode, NeighborLink> layout = new CircleLayout(
                    neighborGraphs[neighborhoodSize - 1]);
            layout.setSize(new Dimension(500, 500));
            layout.initialize();
            VisualizationViewer<ImageNode, NeighborLink> vv =
                    new VisualizationViewer<>(layout);
            vv.setPreferredSize(new Dimension(550, 550));
            vv.setMinimumSize(new Dimension(550, 550));
            vv.setDoubleBuffered(true);
            vv.setEnabled(true);
            graphVServers[neighborhoodSize - 1] = vv;
            vv.getRenderContext().setVertexIconTransformer(
                    new IconTransformer());
            vv.getRenderContext().setVertexShapeTransformer(
                    new ShapeTransformer());
            vv.getRenderContext().setEdgeArrowPredicate(
                    new DirectionDisplayPredicate());
            vv.getRenderContext().setEdgeLabelTransformer(
                    new Transformer() {
                @Override
                public String transform(Object e) {
                    return (e.toString());
                }
            });
            PluggableGraphMouse gm = new PluggableGraphMouse();
            gm.add(new PickingGraphMousePlugin());
            vv.setGraphMouse(gm);
            vv.setBackground(Color.WHITE);
            vv.setVisible(true);
            final PickedState<ImageNode> pickedState =
                    vv.getPickedVertexState();
            pickedState.addItemListener(new ItemListener() {
                @Override
                public void itemStateChanged(ItemEvent e) {
                    Object subject = e.getItem();
                    if (subject instanceof ImageNode) {
                        ImageNode vertex = (ImageNode) subject;
                        if (pickedState.isPicked(vertex)) {
                            setSelectedImageForIndex(vertex.id);
                        }
                    }
                }
            });
            // Refresh all the display components.
            vv.revalidate();
            vv.repaint();
            neighborGraphScrollPane.setViewportView(
                    graphVServers[neighborhoodSize - 1]);
            neighborGraphScrollPane.revalidate();
            neighborGraphScrollPane.repaint();
        }
    }

    /**
     * IconTransformer class. These class is used for node visualization in kNN
     * graphs.
     */
    class IconTransformer
            implements Transformer<ImageNode, Icon> {

        public static final int ICON_SIZE = 30;

        /**
         * @return Integer that is the shape height.
         */
        public int getHeight() {
            return (ICON_SIZE);
        }

        /**
         * @return Integer that is the shape width.
         */
        public int getWidth() {
            return (ICON_SIZE);
        }

        @Override
        public Icon transform(ImageNode node) {
            if (node != null) {
                Icon icon = (Icon) (new ImageIcon(node.toString()));
                return icon;
            } else {
                return null;
            }
        }
    }

    /**
     * ShapeTransformer class. These class is used for node visualization in kNN
     * graphs.
     */
    class ShapeTransformer
            implements Transformer<ImageNode, Shape> {

        public static final int ICON_SIZE = 30;

        /**
         * @return Integer that is the shape height.
         */
        public int getHeight() {
            return (ICON_SIZE);
        }

        /**
         * @return Integer that is the shape width.
         */
        public int getWidth() {
            return (ICON_SIZE);
        }

        @Override
        public Shape transform(ImageNode node) {
            if (node != null) {
                ImageIcon icon = new ImageIcon(node.toString());
                int width = icon.getIconWidth();
                int height = icon.getIconHeight();
                Rectangle2D shape = new Rectangle2D.Float(
                        -width / 2, -height / 2, width, height);
                return (Shape) shape;
            } else {
                return null;
            }
        }
    }

    /**
     * Handler for directed and undirected edges for kNN graph visualization.
     *
     * @param <V> Vertex type.
     * @param <E> Edge type.
     */
    private final static class DirectionDisplayPredicate<V, E>
            implements Predicate<Context<Graph<V, E>, E>> {

        /**
         * The default constructor.
         */
        public DirectionDisplayPredicate() {
        }

        @Override
        public boolean evaluate(Context<Graph<V, E>, E> context) {
            Graph<V, E> graph = context.graph;
            E edge = context.element;
            if (graph.getEdgeType(edge) == EdgeType.DIRECTED) {
                return true;
            }
            if (graph.getEdgeType(edge) == EdgeType.UNDIRECTED) {
                return true;
            }
            return true;
        }
    }

    /**
     * Initialize all the kNN graphs for visualization.
     */
    private void graphsInit() {
        neighborGraphs = new DirectedGraph[50];
        graphVServers = new VisualizationViewer[50];
        edges = new ArrayList[50];
        // For all the relevant neighborhood sizes.
        for (int kTmp = 0; kTmp < 50; kTmp++) {
            // Create a new graph.
            DirectedGraph graph = new DirectedSparseMultigraph<>();
            neighborGraphs[kTmp] = graph;
            Layout<ImageNode, NeighborLink> layout = new CircleLayout(
                    neighborGraphs[kTmp]);
            layout.setSize(new Dimension(500, 500));
            // Set the rendering specification.
            VisualizationViewer<ImageNode, NeighborLink> vv =
                    new VisualizationViewer<>(layout);
            vv.setPreferredSize(new Dimension(550, 550));
            vv.setMinimumSize(new Dimension(550, 550));
            vv.setDoubleBuffered(true);
            vv.setEnabled(true);
            graphVServers[kTmp] = vv;
            vv.getRenderContext().setVertexIconTransformer(
                    new IconTransformer());
            vv.getRenderContext().setVertexShapeTransformer(
                    new ShapeTransformer());
            vv.getRenderContext().setEdgeArrowPredicate(
                    new DirectionDisplayPredicate());
            vv.getRenderContext().setEdgeLabelTransformer(new Transformer() {
                @Override
                public String transform(Object e) {
                    return (e.toString());
                }
            });
            PluggableGraphMouse gm = new PluggableGraphMouse();
            gm.add(new PickingGraphMousePlugin());
            vv.setGraphMouse(gm);
            vv.setBackground(Color.WHITE);
            vv.setVisible(true);
            final PickedState<ImageNode> pickedState =
                    vv.getPickedVertexState();
            // Add the selection listeners.
            pickedState.addItemListener(new ItemListener() {
                @Override
                public void itemStateChanged(ItemEvent e) {
                    Object subject = e.getItem();
                    if (subject instanceof ImageNode) {
                        ImageNode vertex = (ImageNode) subject;
                        if (pickedState.isPicked(vertex)) {
                            setSelectedImageForIndex(vertex.id);
                        }
                    }
                }
            });
        }
        verticesHash = new HashMap<>(500);
        verticesNodeIndexHash = new HashMap<>(500);
        vertexIndexes = new ArrayList<>(200);
        vertices = new ArrayList<>(200);
        edges = new ArrayList[50];
        for (int kTmp = 0; kTmp < 50; kTmp++) {
            edges[kTmp] = new ArrayList<>(500);
        }
        // Refresh the display.
        graphVServers[neighborhoodSize - 1].revalidate();
        graphVServers[neighborhoodSize - 1].repaint();
        neighborGraphScrollPane.setViewportView(
                graphVServers[neighborhoodSize - 1]);
        neighborGraphScrollPane.setVisible(true);
        neighborGraphScrollPane.revalidate();
        neighborGraphScrollPane.repaint();
    }

    /**
     * Train all the classifier models.
     */
    public void trainModels() {
        if (busyCalculating) {
            // If the system is already working on something, this call will be
            // ignored.
            return;
        }
        busyCalculating = true;
        try {
            trainedModels = true;
            classifiers = new Classifier[classifierNameList.length];
            // Get the current metric context.
            CombinedMetric cmet = getCombinedMetric();
            // Initialize all the classifiers.
            classifiers[0] = new KNN(kQuery, cmet);
            classifiers[1] = new FNN(kQuery, cmet, numClasses);
            classifiers[2] = new NWKNN(kQuery, cmet, numClasses);
            classifiers[3] = new AKNN(kQuery, cmet, numClasses);
            classifiers[4] = new HwKNN(numClasses, cmet, kQuery);
            classifiers[5] = new DWHFNN(kQuery, cmet,
                    numClasses);
            classifiers[6] = new HIKNN(kQuery, cmet, numClasses);
            classifiers[7] = new NHBNN(kQuery, cmet, numClasses);
            // Get the current distance matrix.
            float[][] distances = getDistances();
            // Get the current kNN sets.
            NeighborSetFinder nsf = getNSF();
            System.out.println("Training classifier models.");
            ArrayList<Integer> completeDataArray = new ArrayList<>(
                    quantizedRepresentation.size());
            for (int i = 0; i < quantizedRepresentation.size(); i++) {
                completeDataArray.add(i);
            }
            for (int i = 0; i < classifiers.length; i++) {
                // Set the data.
                classifiers[i].setDataIndexes(completeDataArray,
                        quantizedRepresentation);
                if (classifiers[i] instanceof DistMatrixUserInterface) {
                    // If the classifier requires the distance matrix, set the
                    // distance matrix.
                    ((DistMatrixUserInterface) (classifiers[i])).
                            setDistMatrix(distances);
                }
                if (classifiers[i] instanceof NSFUserInterface) {
                    // If the classifier requires kNN sets, set the kNN sets.
                    ((NSFUserInterface) (classifiers[i])).setNSF(nsf);
                }
                try {
                    classifiers[i].train();
                } catch (Exception e) {
                    System.err.println(e.getMessage());
                }
            }
            System.out.println("Models trained.");
            JOptionPane.showMessageDialog(frameReference, "Models trained.");
        } catch (Exception e) {
            System.err.println(e.getMessage());
        } finally {
            // Report that the calculations have been finished.
            busyCalculating = false;
        }
    }

    /**
     * Sets the currently selected image to be the query image in the query
     * panel.
     */
    public void setQueryImageFromCollection() {
        if (busyCalculating) {
            // If the system is already working on something, this call will be
            // ignored.
            return;
        }
        if (selectedImageHistory == null
                || selectedImageIndexInHistory >= selectedImageHistory.size()) {
            return;
        }
        try {
            int index = selectedImageHistory.get(selectedImageIndexInHistory);
            queryImage = getPhoto(index);
            // Set the image to the query image panel.
            queryImagePanel.setImage(queryImage);
            queryImagePanel.revalidate();
            queryImagePanel.repaint();
            if (quantizedRepresentation != null) {
                // Use the existing representation, if available.
                queryImageRep = quantizedRepresentation.getInstance(index);
            }
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
    }

    /**
     * This method queries the image dataset by a single specified image.
     */
    private void imageQuery() {
        if (busyCalculating) {
            // If the system is already working on something, this call will be
            // ignored.
            return;
        }
        try {
            busyCalculating = true;
            // Get the k-nearest neighbors.
            NeighborSetFinder nsf = getNSF();
            queryImageNeighbors = NeighborSetFinder.getIndexesOfNeighbors(
                    quantizedRepresentation, queryImageRep,
                    Math.min(kQuery, nsf.getKNeighbors()[0].length),
                    getCombinedMetric());
            CombinedMetric cmet = getCombinedMetric();
            queryImageNeighborDists = new float[queryImageNeighbors.length];
            // Get the distances to the neighbors.
            for (int i = 0; i < queryImageNeighbors.length; i++) {
                queryImageNeighborDists[i] = cmet.dist(queryImageRep,
                        quantizedRepresentation.getInstance(
                        queryImageNeighbors[i]));
            }
            queryNNPanel.removeAll();
            queryNNPanel.revalidate();
            queryNNPanel.repaint();
            // Add all the kNN-s of the query to the query panel.
            for (int i = 0; i < queryImageNeighbors.length; i++) {
                BufferedImage thumb = thumbnails.get(queryImageNeighbors[i]);
                ImagePanelWithClass imgPan =
                        new ImagePanelWithClass(classColors);
                imgPan.addMouseListener(new NeighborSelectionListener());
                imgPan.setImage(thumb,
                        quantizedRepresentation.getLabelOf(
                        queryImageNeighbors[i]),
                        queryImageNeighbors[i]);
                queryNNPanel.add(imgPan);
            }
            queryNNPanel.revalidate();
            queryNNPanel.repaint();
            // If the classifier models are available, get some predictions and
            // display them to the user.
            if (trainedModels) {
                System.out.println("Classifying.");
                classifierPredictionsPanel.removeAll();
                classifierPredictionsPanel.revalidate();
                classifierPredictionsPanel.repaint();
                // Calculate the distances to the remaining points.
                float[] trainingDists =
                        new float[quantizedRepresentation.size()];
                for (int i = 0; i < queryImageNeighbors.length; i++) {
                    trainingDists[queryImageNeighbors[i]] =
                            queryImageNeighborDists[i];
                }
                for (int i = 0; i < classifiers.length; i++) {
                    System.out.println("Classification by"
                            + classifierNameList[i]);
                    // Class affiliation prediction.
                    float[] prediction;
                    if (classifiers[i] instanceof
                            NeighborPointsQueryUserInterface) {
                        // Get the prediction
                        prediction =
                                ((NeighborPointsQueryUserInterface) (
                                classifiers[i])).classifyProbabilistically(
                                queryImageRep,
                                trainingDists,
                                queryImageNeighbors);
                    } else {
                        // Get the prediction.
                        prediction = classifiers[i].classifyProbabilistically(
                                queryImageRep);
                    }
                    ClassifierResultPanel cResPanel =
                            new ClassifierResultPanel();
                    cResPanel.setResults(prediction,
                            classifierNameList[i], classColors, classNames);
                    classifierPredictionsPanel.add(cResPanel);
                }
                classifierPredictionsPanel.revalidate();
                classifierPredictionsPanel.repaint();
            }
        } catch (Exception e) {
            System.err.println(e.getMessage());
        } finally {
            busyCalculating = false;
        }
    }

    /**
     * Sets all the neighbor stats for the currently selected k-value.
     *
     * @param currentK Integer that is the current neighborhood size.
     */
    public synchronized void setStatFieldsForK(int currentK) {
        // Percentage of elements that occur at least ones.
        if (aboveZeroArray != null) {
            percAboveLabelValue.setText((new Float(BasicMathUtil.
                    makeADecimalCutOff(
                    aboveZeroArray[currentK - 1], 2))).toString());
        } else {
            percAboveLabelValue.setText("...");
        }
        // Neighbor occurrence distribution skewness.
        if (skewArray != null) {
            skewnessLabelValue.setText((new Float(BasicMathUtil.
                    makeADecimalCutOff(
                    skewArray[currentK - 1], 2))).toString());
        } else {
            skewnessLabelValue.setText("...");
        }
        // Neighbor occurrence distribution kurtosis.
        if (kurtosisArray != null) {
            kurtosisLabelValue.setText(
                    (new Float(BasicMathUtil.makeADecimalCutOff(
                    kurtosisArray[currentK - 1], 2))).toString());
        } else {
            kurtosisLabelValue.setText("...");
        }
        // Highest neighbor occurrence frequencies.
        if (highestHubnesses != null) {
            majorDegLabelValue.setText(
                    (new Float(BasicMathUtil.makeADecimalCutOff(
                    highestHubnesses[currentK - 1][
                    highestHubnesses[currentK - 1].length - 1], 2))).
                    toString());
        } else {
            majorDegLabelValue.setText("...");
        }
        // kNN set entropies.
        if (kEntropies != null) {
            nkEntropySkewnessValues.setText(
                    (new Float(BasicMathUtil.makeADecimalCutOff(
                    kEntropySkews[currentK - 1], 2))).toString());
            nkEntropyLabelValue.setText((new Float(BasicMathUtil.
                    makeADecimalCutOff(
                    kEntropies[currentK - 1], 2))).toString());
        } else {
            nkEntropyLabelValue.setText("...");
            nkEntropySkewnessValues.setText("...");
        }
        // Reverse kNN set entropies.
        if (reverseKNNEntropies != null) {
            rnkEntropySkewnessValue.setText(
                    (new Float(BasicMathUtil.makeADecimalCutOff(
                    reverseKNNEntropySkews[currentK - 1], 2))).toString());
            rnkEntropyValue.setText((new Float(BasicMathUtil.makeADecimalCutOff(
                    reverseKNNEntropies[currentK - 1], 2))).toString());
        } else {
            rnkEntropyValue.setText("...");
            rnkEntropySkewnessValue.setText("...");
        }
        // Bad hubness percentages as percentages of label mismatches in kNN
        // sets.
        if (badHubnessArray != null) {
            badHubnessLabelValue.setText((new Float(BasicMathUtil.
                    makeADecimalCutOff(
                    badHubnessArray[currentK - 1], 2))).toString());
        } else {
            badHubnessLabelValue.setText("...");
        }
        // Percentage of points that are hubs.
        if (hubPercs != null) {
            hubsLabelValue.setText((new Float(BasicMathUtil.makeADecimalCutOff(
                    hubPercs[currentK - 1], 2))).toString());
        } else {
            hubsLabelValue.setText("...");
        }
        // Percentage of points that are orphans.
        if (orphanPercs != null) {
            orphansLabelValue.setText((new Float(BasicMathUtil.
                    makeADecimalCutOff(
                    orphanPercs[currentK - 1], 2))).toString());
        } else {
            orphansLabelValue.setText("...");
        }
        // Percentage of points that are regular points.
        if (regularPercs != null) {
            regularLabelValue.setText((new Float(BasicMathUtil.
                    makeADecimalCutOff(
                    regularPercs[currentK - 1], 2))).toString());
        } else {
            regularLabelValue.setText("...");
        }
        // Refresh the display.
        percAboveLabelValue.revalidate();
        percAboveLabelValue.repaint();
        skewnessLabelValue.revalidate();
        skewnessLabelValue.repaint();
        kurtosisLabelValue.revalidate();
        kurtosisLabelValue.repaint();
        majorDegLabelValue.revalidate();
        majorDegLabelValue.repaint();
        nkEntropySkewnessValues.revalidate();
        nkEntropySkewnessValues.repaint();
        nkEntropyLabelValue.revalidate();
        nkEntropyLabelValue.repaint();
        rnkEntropySkewnessValue.revalidate();
        rnkEntropySkewnessValue.repaint();
        rnkEntropyValue.revalidate();
        rnkEntropyValue.repaint();
        badHubnessLabelValue.revalidate();
        badHubnessLabelValue.repaint();
        hubsLabelValue.revalidate();
        hubsLabelValue.repaint();
        orphansLabelValue.revalidate();
        orphansLabelValue.repaint();
        regularLabelValue.revalidate();
        regularLabelValue.repaint();
        // Now generate the occurrence frequency distribution chart, discretized
        // to fixed-length buckets.
        DefaultCategoryDataset hDistDataset = new DefaultCategoryDataset();
        for (int i = 0; i < bucketedOccurrenceDistributions[
                    neighborhoodSize - 1].length; i++) {
            hDistDataset.addValue(
                    bucketedOccurrenceDistributions[
                         neighborhoodSize - 1][i], "Number of Examples",
                    i + "");
        }
        JFreeChart chart = ChartFactory.createBarChart(
                "Occurrence Frequency Distribution", "", "",
                hDistDataset, PlotOrientation.VERTICAL, false, true, false);
        ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new Dimension(440, 180));
        chartHoldingPanelOccDistribution.removeAll();
        chartHoldingPanelOccDistribution.add(chartPanel);
        chartHoldingPanelOccDistribution.revalidate();
        chartHoldingPanelOccDistribution.repaint();
        // Calculate class to class hubness.
        for (int c1 = 0; c1 < numClasses; c1++) {
            for (int c2 = 0; c2 < numClasses; c2++) {
                classHubnessTable.setValueAt(
                        globalClassToClasshubness[currentK - 1][c1][c2], c1,
                        c2);
            }
        }
        classHubnessTable.setDefaultRenderer(Object.class,
                new ClassToClassHubnessMatrixRenderer(
                globalClassToClasshubness[currentK - 1],
                numClasses));
    }

    /**
     * Neighbor selection listener.
     */
    class NeighborSelectionListener implements MouseListener {

        @Override
        public void mousePressed(MouseEvent e) {
            Component comp = e.getComponent();
            if (comp instanceof ImagePanelWithClass) {
                int index = ((ImagePanelWithClass) comp).getImageIndex();
                setSelectedImageForIndex(index);
            }
        }

        @Override
        public void mouseReleased(MouseEvent e) {
        }

        @Override
        public void mouseEntered(MouseEvent e) {
        }

        @Override
        public void mouseExited(MouseEvent e) {
        }

        @Override
        public void mouseClicked(MouseEvent e) {
            Component comp = e.getComponent();
            if (comp instanceof ImagePanelWithClass) {
                int index = ((ImagePanelWithClass) comp).getImageIndex();
                setSelectedImageForIndex(index);
            }
        }
    }

    /**
     * This class handles adding neighbors and reverse neighbors to their
     * panels.
     */
    private class SetImageNeighborsHelper implements Runnable {

        private BufferedImage thumb;
        private int label, index;
        private JPanel panel;

        /**
         * Initialization.
         *
         * @param panel Panel to add the image to.
         * @param thumb Thumbnail of the image to add.
         * @param label Label of the image to add.
         * @param index Integer that is the index of the image to add.
         */
        public SetImageNeighborsHelper(JPanel panel, BufferedImage thumb,
                int label, int index) {
            this.thumb = thumb;
            this.label = label;
            this.index = index;
            this.panel = panel;
        }

        @Override
        public void run() {
            ImagePanelWithClass rrneighbor =
                    new ImagePanelWithClass(classColors);
            rrneighbor.addMouseListener(new NeighborSelectionListener());
            rrneighbor.setImage(thumb, label, index);
            panel.add(rrneighbor);
        }
    }

    /**
     * This method sets the image of the specified index as the currently
     * selected image and updates all the views.
     *
     * @param index Integer that is the index of the image to select as the
     * current image.
     */
    private synchronized void setSelectedImageForIndex(int index) {
        try {
            // Update the selected image panels.
            BufferedImage photo = getPhoto(index);
            selectedImagePanelClassNeighborMain.setImage(photo);
            selectedImagePanelClassNeighbor.setImage(photo);
            selectedImagePanelClass.setImage(photo);
            selectedImagePanelSearch.setImage(photo);
            String shortPath = imgPaths.get(index).substring(
                    workspace.getPath().length(), imgPaths.get(index).length());
            // Update the labels with the new name.
            selectedImagePathLabelClassNeighborMain.setText(shortPath);
            selectedImagePathLabelClassNeighbor.setText(shortPath);
            selectedImagePathLabelClass.setText(shortPath);
            selectedImagePathLabelSearch.setText(shortPath);
            // Update the class colors.
            selectedImageLabelClassNeighborMain.setBackground(
                    classColors[quantizedRepresentation.getLabelOf(index)]);
            selectedImageLabelClassNeighbor.setBackground(
                    classColors[quantizedRepresentation.getLabelOf(index)]);
            selectedImageLabelClass.setBackground(
                    classColors[quantizedRepresentation.getLabelOf(index)]);
            selectedImageLabelSearch.setBackground(
                    classColors[quantizedRepresentation.getLabelOf(index)]);
            // Refresh the display.
            selectedImageLabelClassNeighborMain.setOpaque(true);
            selectedImageLabelClassNeighbor.setOpaque(true);
            selectedImageLabelClass.setOpaque(true);
            selectedImageLabelSearch.setOpaque(true);
            selectedImageLabelClassNeighborMain.repaint();
            selectedImageLabelClassNeighbor.repaint();
            selectedImageLabelClass.repaint();
            selectedImageLabelSearch.repaint();
            // Update the history.
            if (selectedImageHistory == null) {
                selectedImageHistory = new ArrayList<>(200);
                selectedImageIndexInHistory = -1;
            }
            // Discard the future history.
            if (selectedImageIndexInHistory < selectedImageHistory.size() - 1) {
                for (int i = selectedImageHistory.size() - 1;
                        i > selectedImageIndexInHistory; i--) {
                    selectedImageHistory.remove(i);
                }
            }
            selectedImageHistory.add(index);
            selectedImageIndexInHistory = selectedImageHistory.size() - 1;
            // Update the nearest neighbors and the reverse nearest neighbors.
            NeighborSetFinder nsf = getNSF();
            nnPanel.removeAll();
            rnnPanel.removeAll();
            nnPanel.revalidate();
            nnPanel.repaint();
            rnnPanel.revalidate();
            rnnPanel.repaint();
            int[][] kneighbors = nsf.getKNeighbors();
            for (int neighborIndex = 0; neighborIndex < neighborhoodSize;
                    neighborIndex++) {
                // Insert all the nearest neighbors to their panel.
                BufferedImage thumb =
                        thumbnails.get(kneighbors[index][neighborIndex]);
                try {
                    Thread t = new Thread(
                            new SetImageNeighborsHelper(
                            nnPanel, thumb,
                            quantizedRepresentation.getLabelOf(
                            kneighbors[index][neighborIndex]),
                            kneighbors[index][neighborIndex]));
                    t.start();
                    t.join(500);
                    if (t.isAlive()) {
                        t.interrupt();
                    }
                } catch (Throwable thr) {
                    System.err.println(thr.getMessage());
                }
            }
            // Insert all the reverse nearest neighbors to their panel.
            ArrayList<Integer>[] rrns = null;
            if (rnnSetsAllK != null) {
                rrns = rnnSetsAllK[neighborhoodSize - 1];
            }
            if (rrns != null && rrns[index] != null && rrns[index].size() > 0) {
                for (int i = 0; i < rrns[index].size(); i++) {
                    BufferedImage thumb = thumbnails.get(rrns[index].get(i));
                    try {
                        Thread t = new Thread(
                                new SetImageNeighborsHelper(
                                rnnPanel, thumb,
                                quantizedRepresentation.getLabelOf(
                                rrns[index].get(i)), rrns[index].get(i)));
                        t.start();
                        t.join(500);
                        if (t.isAlive()) {
                            t.interrupt();
                        }
                    } catch (Throwable thr) {
                        System.err.println(thr.getMessage());
                    }
                }
            }
            // Refresh the neighbor and reverse neighbor panels.
            nnPanel.revalidate();
            nnPanel.repaint();
            rnnPanel.revalidate();
            rnnPanel.repaint();
            // Visualize the neighbor occurrence profile of the selected image.
            DefaultPieDataset pieData = new DefaultPieDataset();
            for (int c = 0; c < numClasses; c++) {
                pieData.setValue(classNames[c],
                        occurrenceProfilesAllK[neighborhoodSize - 1][index][c]);
            }
            JFreeChart chart = ChartFactory.createPieChart3D("occurrence "
                    + "profile", pieData, true, true, false);
            PiePlot3D plot = (PiePlot3D) chart.getPlot();
            plot.setStartAngle(290);
            plot.setDirection(Rotation.CLOCKWISE);
            plot.setForegroundAlpha(0.5f);
            PieRenderer prend = new PieRenderer(classColors);
            prend.setColor(plot, pieData);
            ChartPanel chartPanel = new ChartPanel(chart);
            chartPanel.setPreferredSize(new Dimension(240, 200));
            occProfileChartHolder.removeAll();
            occProfileChartHolder.add(chartPanel);
            occProfileChartHolder.revalidate();
            occProfileChartHolder.repaint();
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
    }

    /**
     * This method sets the image of the specified history index as the
     * currently selected image and updates all the views.
     *
     * @param index Integer that is the history index of the image to select as
     * the current image.
     */
    private synchronized void setSelectedImageForHistoryIndex(
            int historyIndex) {
        // Update the selected image panels.
        BufferedImage photo = getPhoto(selectedImageHistory.get(historyIndex));
        selectedImagePanelClassNeighborMain.setImage(photo);
        selectedImagePanelClassNeighbor.setImage(photo);
        selectedImagePanelClass.setImage(photo);
        selectedImagePanelSearch.setImage(photo);
        int index = selectedImageHistory.get(historyIndex);
        // Update the labels with the new name.
        String shortPath = imgPaths.get(index).substring(
                workspace.getPath().length(), imgPaths.get(index).length());
        selectedImagePathLabelClassNeighborMain.setText(shortPath);
        selectedImagePathLabelClassNeighbor.setText(shortPath);
        selectedImagePathLabelClass.setText(shortPath);
        selectedImagePathLabelSearch.setText(shortPath);
        // Update the class colors.
        selectedImageLabelClassNeighborMain.setBackground(
                classColors[quantizedRepresentation.getLabelOf(index)]);
        selectedImageLabelClassNeighbor.setBackground(
                classColors[quantizedRepresentation.getLabelOf(index)]);
        selectedImageLabelClass.setBackground(
                classColors[quantizedRepresentation.getLabelOf(index)]);
        selectedImageLabelSearch.setBackground(
                classColors[quantizedRepresentation.getLabelOf(index)]);
        // Refresh the display.
        selectedImageLabelClassNeighborMain.setOpaque(true);
        selectedImageLabelClassNeighbor.setOpaque(true);
        selectedImageLabelClass.setOpaque(true);
        selectedImageLabelSearch.setOpaque(true);
        selectedImageLabelClassNeighborMain.repaint();
        selectedImageLabelClassNeighbor.repaint();
        selectedImageLabelClass.repaint();
        selectedImageLabelSearch.repaint();
        // Update the nearest neighbors and the reverse nearest neighbors.
        NeighborSetFinder nsf = getNSF();
        nnPanel.removeAll();
        rnnPanel.removeAll();
        nnPanel.revalidate();
        nnPanel.repaint();
        rnnPanel.revalidate();
        rnnPanel.repaint();
        int[][] kneighbors = nsf.getKNeighbors();
        for (int neighborIndex = 0; neighborIndex < neighborhoodSize;
                neighborIndex++) {
            BufferedImage thumb = thumbnails.get(
                    kneighbors[index][neighborIndex]);
            try {
                Thread t = new Thread(new SetImageNeighborsHelper(
                        nnPanel, thumb, quantizedRepresentation.getLabelOf(
                        kneighbors[index][neighborIndex]),
                        kneighbors[index][neighborIndex]));
                t.start();
                t.join(500);
                if (t.isAlive()) {
                    t.interrupt();
                }
            } catch (Throwable thr) {
                System.err.println(thr.getMessage());
            }
        }
        ArrayList<Integer>[] rrns = rnnSetsAllK[neighborhoodSize - 1];
        if (rrns[index] != null && rrns[index].size() > 0) {
            for (int i = 0; i < rrns[index].size(); i++) {
                BufferedImage thumb = thumbnails.get(rrns[index].get(i));
                try {
                    Thread t = new Thread(
                            new SetImageNeighborsHelper(
                            rnnPanel, thumb,
                            quantizedRepresentation.getLabelOf(
                            rrns[index].get(i)), rrns[index].get(i)));
                    t.start();
                    t.join(500);
                    if (t.isAlive()) {
                        t.interrupt();
                    }
                } catch (Throwable thr) {
                    System.err.println(thr.getMessage());
                }
            }
        }
        // Refresh the neighbor and reverse neighbor panels.
        nnPanel.revalidate();
        nnPanel.repaint();
        rnnPanel.revalidate();
        rnnPanel.repaint();
        // Visualize the neighbor occurrence profile of the selected image.
        DefaultPieDataset pieData = new DefaultPieDataset();
        for (int c = 0; c < numClasses; c++) {
            pieData.setValue(classNames[c],
                    occurrenceProfilesAllK[neighborhoodSize - 1][index][c]);
        }
        JFreeChart chart = ChartFactory.createPieChart3D("occurrence profile",
                pieData, true, true, false);
        PiePlot3D plot = (PiePlot3D) chart.getPlot();
        plot.setStartAngle(290);
        plot.setDirection(Rotation.CLOCKWISE);
        plot.setForegroundAlpha(0.5f);
        PieRenderer prend = new PieRenderer(classColors);
        prend.setColor(plot, pieData);
        ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new Dimension(240, 200));
        occProfileChartHolder.removeAll();
        occProfileChartHolder.add(chartPanel);
        occProfileChartHolder.revalidate();
        occProfileChartHolder.repaint();
    }

    /**
     * This method gets the photo for the provided image index.
     *
     * @param index Integer that is the image index in the data.
     * @return BufferedImage corresponding to the image index.
     */
    private BufferedImage getPhoto(int index) {
        if (images[index] == null) {
            int pathFeatureIndex =
                    quantizedRepresentation.getIndexForAttributeName(
                    "relative_path");
            File inImageFile = new File(workspace, "photos"
                    + quantizedRepresentation.getInstance(index).sAttr[
                    pathFeatureIndex]);
            try {
                images[index] = ImageIO.read(inImageFile);
            } catch (Exception e) {
                System.err.println(e.getMessage());
            }
        }
        return images[index];
    }

    /**
     * This method returns the distances that are currently in use. If there is
     * a secondary distance matrix, it returns that one. If there is no
     * secondary distance matrix, it returns the primary matrix instead.
     *
     * @return float[][] that is the currently used distance matrix.
     */
    private float[][] getDistances() {
        if (distMatrixSecondary != null) {
            return distMatrixSecondary;
        } else {
            return distMatrixPrimary;
        }
    }

    /**
     * This method returns the currently employed CombinedMetric object for
     * distance calculations. If there are secondary distances in use, it
     * returns the secondary CombinedMetric object. If not, the primary one.
     *
     * @return CombinedMetric object that is currently in use.
     */
    private CombinedMetric getCombinedMetric() {
        if (secondaryCMet != null) {
            return secondaryCMet;
        } else {
            return primaryCMet;
        }
    }

    /**
     * This method returns the current NeighborSetFinder.
     *
     * @return NeighborSet finder that is currently in use by the system.
     */
    private NeighborSetFinder getNSF() {
        if (nsfSecondary != null) {
            return nsfSecondary;
        } else {
            return nsfPrimary;
        }
    }

    /**
     * Creates new form HubExplorer
     */
    public ImageHubExplorer() {
        initComponents();
        additionalInit();
    }

    /**
     * Initialization.
     */
    private void additionalInit() {
        // Initialize kNN and reverse neighbor panels.
        nnPanel.setLayout(new FlowLayout());
        rnnPanel.setLayout(new FlowLayout());
        nnPanel.setComponentOrientation(
                ComponentOrientation.LEFT_TO_RIGHT);
        rnnPanel.setComponentOrientation(
                ComponentOrientation.LEFT_TO_RIGHT);
        rnnPanel.setMaximumSize(new Dimension(60000, 400));
        // Initialize the neighborhood size selection slider.
        kSelectionSlider.setMajorTickSpacing(5);
        kSelectionSlider.setMinorTickSpacing(1);
        kSelectionSlider.setPaintLabels(true);
        kSelectionSlider.setPaintTicks(true);
        kSelectionSlider.addChangeListener(new SliderChanger());
        // Initialize various chart panels.
        classColorAndNamesPanel.setLayout(new VerticalFlowLayout());
        occProfileChartHolder.setLayout(new FlowLayout());
        classDistributionHolder.setLayout(new FlowLayout());
        chartHoldingPanelOccDistribution.setLayout(new FlowLayout());
        chartHoldingPanelOccDistribution.setPreferredSize(
                new Dimension(497, 191));
        queryNNPanel.setLayout(new VerticalFlowLayout());
        classifierPredictionsPanel.setLayout(new VerticalFlowLayout());
        // Initialize the scroll panes.
        prClassScrollPane.setPreferredSize(new Dimension(237, 432));
        prClassScrollPane.setMinimumSize(new Dimension(237, 432));
        prClassScrollPane.setMaximumSize(new Dimension(237, 432));
        queryNNScrollPane.setPreferredSize(new Dimension(207, 455));
        queryNNScrollPane.setMinimumSize(new Dimension(207, 455));
        queryNNScrollPane.setMaximumSize(new Dimension(207, 455));
        classesScrollPanel.setLayout(new VerticalFlowLayout());
        classesScrollPanel.setPreferredSize(new Dimension(760, 1168));
        classesScrollPanel.setMaximumSize(new Dimension(760, 1168));
        classesScrollPanel.setMinimumSize(new Dimension(760, 1168));
        classesScrollPane.setPreferredSize(new Dimension(760, 308));
        classesScrollPane.setMaximumSize(new Dimension(760, 308));
        classesScrollPane.setMinimumSize(new Dimension(760, 308));
        // Initialize the MDS component.
        mdsCollectionPanel.setPreferredSize(new Dimension(1500, 1500));
        mdsCollectionPanel.setMaximumSize(new Dimension(1500, 1500));
        mdsCollectionPanel.setMinimumSize(new Dimension(1500, 1500));
        // Initialize the kNN graph visualization component.
        neighborGraphScrollPane.setPreferredSize(new Dimension(550, 550));
        neighborGraphScrollPane.setMinimumSize(new Dimension(550, 550));
        neighborGraphScrollPane.setMaximumSize(new Dimension(550, 550));
        neighborGraphScrollPane.setVisible(true);
        selectedImagePathLabelClassNeighborMain.setPreferredSize(
                new Dimension(30, 16));
        selectedImagePathLabelClassNeighborMain.setMaximumSize(
                new Dimension(30, 16));
        selectedImagePathLabelClassNeighborMain.setMinimumSize(
                new Dimension(30, 16));
        jScrollPane1.setPreferredSize(new Dimension(30, 16));
        jScrollPane1.setMinimumSize(new Dimension(30, 16));
        jScrollPane1.setMaximumSize(new Dimension(30, 16));
    }

    /**
     * This listener handles the changes in the current neighborhood size by
     * moving the k-slider.
     */
    class SliderChanger implements ChangeListener {

        @Override
        public void stateChanged(ChangeEvent e) {
            if (!neighborStatsCalculated) {
                // If the kNN stats haven't been calculated, there is no need to
                // do anything.
                return;
            }
            Object src = e.getSource();
            if (src instanceof JSlider) {
                // Get the selected neighborhood size.
                neighborhoodSize = Math.max(((JSlider) src).getValue(), 1);
                // Get the index of the currently selected image.
                int index = -1;
                if (selectedImageHistory != null
                        && selectedImageHistory.size() > 0) {
                    index = selectedImageHistory.get(
                            selectedImageIndexInHistory);
                }
                // Get the object holding the kNN sets.
                NeighborSetFinder nsf = getNSF();
                // Reinitialize the panels.
                nnPanel.removeAll();
                rnnPanel.removeAll();
                nnPanel.revalidate();
                nnPanel.repaint();
                rnnPanel.revalidate();
                rnnPanel.repaint();
                if (index != -1) {
                    // Get the kNN sets.
                    int[][] kneighbors = nsf.getKNeighbors();
                    // Refresh the neighbor list.
                    for (int i = 0; i < neighborhoodSize; i++) {
                        ImagePanelWithClass neighbor =
                                new ImagePanelWithClass(classColors);
                        neighbor.addMouseListener(
                                new NeighborSelectionListener());
                        neighbor.setImage(
                                thumbnails.get(kneighbors[index][i]),
                                quantizedRepresentation.getLabelOf(
                                kneighbors[index][i]), kneighbors[index][i]);
                        nnPanel.add(neighbor);
                    }
                    // Refresh the reverse nearest neighbor list.
                    ArrayList<Integer>[] rrns =
                            rnnSetsAllK[neighborhoodSize - 1];
                    if (rrns[index] != null && rrns[index].size() > 0) {
                        for (int i = 0; i < rrns[index].size(); i++) {
                            ImagePanelWithClass rrneighbor =
                                    new ImagePanelWithClass(classColors);
                            rrneighbor.setImage(
                                    thumbnails.get(rrns[index].get(i)),
                                    quantizedRepresentation.getLabelOf(
                                    rrns[index].get(i)), rrns[index].get(i));
                            rrneighbor.addMouseListener(
                                    new NeighborSelectionListener());
                            rnnPanel.add(rrneighbor);
                        }
                    }
                    // Refresh the occurrence profile of the current image.
                    DefaultPieDataset pieData = new DefaultPieDataset();
                    for (int c = 0; c < numClasses; c++) {
                        pieData.setValue(classNames[c],
                                occurrenceProfilesAllK[neighborhoodSize - 1][
                                index][c]);
                    }
                    JFreeChart chart = ChartFactory.createPieChart3D(
                            "occurrence profile", pieData, true, true, false);
                    PiePlot3D plot = (PiePlot3D) chart.getPlot();
                    plot.setStartAngle(290);
                    plot.setDirection(Rotation.CLOCKWISE);
                    plot.setForegroundAlpha(0.5f);
                    PieRenderer prend = new PieRenderer(classColors);
                    prend.setColor(plot, pieData);
                    ChartPanel chartPanel = new ChartPanel(chart);
                    chartPanel.setPreferredSize(new Dimension(240, 200));
                    // Refresh the display.
                    occProfileChartHolder.removeAll();
                    occProfileChartHolder.add(chartPanel);
                    occProfileChartHolder.revalidate();
                    occProfileChartHolder.repaint();
                    nnPanel.revalidate();
                    nnPanel.repaint();
                    rnnPanel.revalidate();
                    rnnPanel.repaint();
                    // Refresh the kNN graph visualizations, as new edges might
                    // need to be inserted.
                    graphVServers[neighborhoodSize - 1].setPreferredSize(
                            new Dimension(500, 500));
                    graphVServers[neighborhoodSize - 1].setMinimumSize(
                            new Dimension(500, 500));
                    Layout<ImageNode, NeighborLink> layout =
                            new CircleLayout(
                            neighborGraphs[neighborhoodSize - 1]);
                    layout.setSize(new Dimension(500, 500));
                    layout.initialize();
                    VisualizationViewer<ImageNode, NeighborLink> vv =
                            new VisualizationViewer<>(layout);
                    vv.setPreferredSize(new Dimension(550, 550));
                    vv.setMinimumSize(new Dimension(550, 550));
                    vv.setDoubleBuffered(true);
                    vv.setEnabled(true);
                    graphVServers[neighborhoodSize - 1] = vv;
                    vv.getRenderContext().setVertexIconTransformer(
                            new IconTransformer());
                    vv.getRenderContext().setVertexShapeTransformer(
                            new ShapeTransformer());
                    vv.getRenderContext().setEdgeArrowPredicate(
                            new DirectionDisplayPredicate());
                    vv.getRenderContext().setEdgeLabelTransformer(
                            new Transformer() {
                        @Override
                        public String transform(Object e) {
                            return (e.toString());
                        }
                    });
                    PluggableGraphMouse gm = new PluggableGraphMouse();
                    gm.add(new PickingGraphMousePlugin());
                    vv.setGraphMouse(gm);
                    vv.setBackground(Color.WHITE);
                    vv.setVisible(true);
                    final PickedState<ImageNode> pickedState =
                            vv.getPickedVertexState();
                    pickedState.addItemListener(new ItemListener() {
                        @Override
                        public void itemStateChanged(ItemEvent e) {
                            Object subject = e.getItem();
                            if (subject instanceof ImageNode) {
                                ImageNode vertex = (ImageNode) subject;
                                if (pickedState.isPicked(vertex)) {
                                    setSelectedImageForIndex(vertex.id);
                                } else {
                                }
                            }
                        }
                    });
                    // Refresh the graph displays.
                    vv.validate();
                    neighborGraphScrollPane.setViewportView(
                            graphVServers[neighborhoodSize - 1]);
                    neighborGraphScrollPane.revalidate();
                    neighborGraphScrollPane.repaint();
                }
                // Refresh the kNN stats in the main screen.
                setStatFieldsForK(neighborhoodSize);
                classesScrollPanel.removeAll();
                // Refresh the class summary panels
                for (int c = 0; c < numClasses; c++) {
                    ClassHubsPanel chp =
                            new ClassHubsPanel(classColors[c], classNames[c]);
                    classStatsOverviews[c] = chp;
                    chp.setPointTypeDistribution(classPTypes[c]);
                    chp.revalidate();
                    chp.repaint();
                    // Lists of hubs, good hubs and bad hubs for the class.
                    JPanel hubsPanel = chp.getHubsPanel();
                    JPanel hubsPanelGood = chp.getGoodHubsPanel();
                    JPanel hubsPanelBad = chp.getBadHubsPanel();
                    hubsPanel.removeAll();
                    for (int i = 0; i < Math.min(
                            50, classImageIndexes[c].size()); i++) {
                        BufferedImage thumb = thumbnails.get(
                                classTopHubLists[
                                neighborhoodSize - 1][c].get(i));
                        ImagePanelWithClass imgPan =
                                new ImagePanelWithClass(classColors);
                        imgPan.addMouseListener(
                                new NeighborSelectionListener());
                        imgPan.setImage(thumb,
                                quantizedRepresentation.getLabelOf(
                                classTopHubLists[neighborhoodSize - 1][c].
                                get(i)), classTopHubLists[neighborhoodSize - 1][
                                c].get(i));
                        hubsPanel.add(imgPan);
                    }
                    hubsPanel.revalidate();
                    hubsPanel.repaint();
                    hubsPanelGood.removeAll();
                    for (int i = 0; i < Math.min(50,
                            classImageIndexes[c].size()); i++) {
                        BufferedImage thumb = thumbnails.get(
                                classTopGoodHubsList[neighborhoodSize - 1][c].
                                get(i));
                        ImagePanelWithClass imgPan =
                                new ImagePanelWithClass(classColors);
                        imgPan.addMouseListener(
                                new NeighborSelectionListener());
                        imgPan.setImage(thumb,
                                quantizedRepresentation.getLabelOf(
                                classTopGoodHubsList[neighborhoodSize - 1][c].
                                get(i)), classTopGoodHubsList[
                                neighborhoodSize - 1][c].get(i));
                        hubsPanelGood.add(imgPan);
                    }
                    hubsPanelGood.revalidate();
                    hubsPanelGood.repaint();
                    hubsPanelBad.removeAll();
                    for (int i = 0; i < Math.min(50,
                            classImageIndexes[c].size()); i++) {
                        BufferedImage thumb = thumbnails.get(
                                classTopBadHubsList[neighborhoodSize - 1][c].
                                get(i));
                        ImagePanelWithClass imgPan =
                                new ImagePanelWithClass(classColors);
                        imgPan.addMouseListener(
                                new NeighborSelectionListener());
                        imgPan.setImage(thumb,
                                quantizedRepresentation.getLabelOf(
                                classTopBadHubsList[neighborhoodSize - 1][c].
                                get(i)), classTopBadHubsList[
                                neighborhoodSize - 1][c].get(i));
                        hubsPanelBad.add(imgPan);
                    }
                    hubsPanelBad.revalidate();
                    hubsPanelBad.repaint();
                    chp.revalidate();
                    chp.repaint();
                    classesScrollPanel.add(chp);
                }
                classesScrollPanel.revalidate();
                classesScrollPanel.repaint();
                classesScrollPane.revalidate();
                classesScrollPane.repaint();
                // Handle the data visualization in the MDS screen.
                if (imageCoordinatesXY != null) {
                    // In case some of the thumbnails crosses the bounding box
                    // of the MDS panel, offsets are set to compensate and to
                    // ensure all images are visible in their entirety.
                    float offX, offY;
                    if (highestHubnesses != null) {
                        float maxOccurrenceFrequency =
                                ArrayUtil.max(highestHubnesses[
                                neighborhoodSize - 1]);
                        float[] thumbSizes = new float[highestHubnesses[
                                neighborhoodSize - 1].length];
                        ArrayList<Rectangle2D> bounds =
                                new ArrayList<>(thumbSizes.length);
                        ArrayList<ImagePanelWithClass> imgsMDS =
                                new ArrayList<>(thumbSizes.length);
                        for (int i = 0; i < thumbSizes.length; i++) {
                            // Calculate the thumbnail size.
                            try {
                                thumbSizes[i] = pointScale(
                                        highestHubnesses[
                                        neighborhoodSize - 1][i],
                                        maxOccurrenceFrequency,
                                        minImageScale,
                                        maxImageScale);
                            } catch (Exception eSecond) {
                                System.err.println(eSecond.getMessage());
                            }
                            if (imageCoordinatesXY[
                                    highestHubIndexes[
                                    neighborhoodSize - 1][i]][0]
                                    + thumbSizes[i] / 2
                                    > mdsCollectionPanel.getWidth()) {
                                offX = (thumbSizes[i] / 2
                                        - (mdsCollectionPanel.getWidth()
                                        - imageCoordinatesXY[highestHubIndexes[
                                        neighborhoodSize - 1][i]][0]));
                            } else if (imageCoordinatesXY[highestHubIndexes[
                                    neighborhoodSize - 1][i]][0]
                                    - thumbSizes[i] / 2 < 0) {
                                offX = imageCoordinatesXY[highestHubIndexes[
                                        neighborhoodSize - 1][i]][0];
                            } else {
                                offX = thumbSizes[i] / 2;
                            }
                            if (imageCoordinatesXY[highestHubIndexes[
                                    neighborhoodSize - 1][i]][1]
                                    + thumbSizes[i] / 2
                                    > mdsCollectionPanel.getHeight()) {
                                offY = (thumbSizes[i] / 2
                                        - (mdsCollectionPanel.getHeight()
                                        - imageCoordinatesXY[highestHubIndexes[
                                        neighborhoodSize - 1][i]][1]));
                            } else if (imageCoordinatesXY[highestHubIndexes[
                                    neighborhoodSize - 1][i]][1]
                                    - thumbSizes[i] / 2 < 0) {
                                offY = imageCoordinatesXY[highestHubIndexes[
                                        neighborhoodSize - 1][i]][1];
                            } else {
                                offY = thumbSizes[i] / 2;
                            }
                            // Get the image thumbnail to show.
                            BufferedImage thumb = thumbnails.get(
                                    highestHubIndexes[neighborhoodSize - 1][i]);
                            ImagePanelWithClass imgPan =
                                    new ImagePanelWithClass(classColors);
                            imgPan.addMouseListener(
                                    new NeighborSelectionListener());
                            imgPan.setImage(thumb,
                                    quantizedRepresentation.getLabelOf(
                                    highestHubIndexes[neighborhoodSize - 1][i]),
                                    highestHubIndexes[neighborhoodSize - 1][i]);
                            imgsMDS.add(imgPan);
                            // Set the bounding rectangle.
                            bounds.add(new Rectangle2D.Float(
                                    imageCoordinatesXY[highestHubIndexes[
                                    neighborhoodSize - 1][i]][0] - offX,
                                    imageCoordinatesXY[highestHubIndexes[
                                    neighborhoodSize - 1][i]][1] - offY,
                                    thumbSizes[i], thumbSizes[i]));
                        }
                        // Set the images for display in the MDS overview panel.
                        mdsCollectionPanel.setImageSet(imgsMDS, bounds);
                        setMDSBackground();
                        // Refresh the display.
                        mdsCollectionPanel.revalidate();
                        mdsCollectionPanel.repaint();
                    }
                }
            }
        }
    }

    /**
     * This method loads the distance matrix from a file.
     *
     * @param dMatFile File that holds the distance matrix.
     * @return float[][] that is the distance matrix.
     * @throws Exception
     */
    public float[][] loadDMatFromFile(File dMatFile) throws Exception {
        BufferedReader br = new BufferedReader(new InputStreamReader(
                new FileInputStream(dMatFile)));
        float[][] dMatLoaded = null;
        String line;
        String[] lineItems;
        try {
            int size = Integer.parseInt(br.readLine());
            dMatLoaded = new float[size][];
            for (int i = 0; i < size - 1; i++) {
                dMatLoaded[i] = new float[size - i - 1];
                line = br.readLine();
                lineItems = line.split(",");
                for (int j = 0; j < lineItems.length; j++) {
                    dMatLoaded[i][j] = Float.parseFloat(lineItems[j]);
                }
            }
            dMatLoaded[size - 1] = new float[0];
        } catch (IOException | NumberFormatException e) {
            throw e;
        } finally {
            br.close();
        }
        return dMatLoaded;
    }

    /**
     * This method prints a distance matrix to a file.
     *
     * @param distMat float[][] that is the distance matrix.
     * @param dMatFile File to write the matrix to.