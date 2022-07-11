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

import data.representation.images.sift.LFeatRepresentation;
import data.representation.images.sift.LFeatVector;
import distances.primary.CombinedMetric;
import distances.primary.LocalImageFeatureMetric;
import images.mining.codebook.GenericCodeBook;
import images.mining.display.SIFTDraw;
import java.awt.Color;
import java.awt.Component;
import java.awt.FlowLayout;
import java.awt.Graphics2D;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Arrays;
import javax.imageio.ImageIO;
import javax.swing.JFileChooser;
import javax.swing.JFrame;

/**
 * This frame is used for visual word utility assessment in Image Hub Explorer.
 * It shows a list of visual words with their class-conditional occurrence
 * profiles and it shows the utility landscape on top of the current image. It
 * is possible to examine the total image region utility, as well as to
 * visualize each visual word separately.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class QuantizedImageViewer extends javax.swing.JFrame {

    // The relevant data sources. Not all of them are directly used from within
    // this frame in the current implementation, but the functionality of this
    // frame is open to many extensions and it will be extended soon. Therefore,
    // a bit more information was included to begin with.
    // Feature representation for the image in question.
    private LFeatRepresentation imageFRep;
    private float[] codebookGoodness;
    // An object that represents the visual word definitions.
    private GenericCodeBook codebook;
    private double[][] codebookProfiles;
    // Class colors for display.
    private Color[] classColors;
    // Class names for display.
    private String[] classNames;
    // Currently examined image.
    private BufferedImage originalImage;
    // An image of the overall utility of different regions in the original
    // image.
    private BufferedImage goodnessSIFTImage;
    // Each image in the array corresponds to a visualization of a single
    // visual word.
    private BufferedImage[] codebookVisualizationImages;
    // This array holds the index of the closest codebook feature for each
    // feature in the currently examined image.
    private int[] codebookAssignments;
    // One partial representation for each codebook feature.
    private LFeatRepresentation[] partialReps;
    // Black and white image.
    private BufferedImage bwImage;
    private File currentDirectory = new File(".");
    // Visual word index of the currently examined codebook feature.
    private int selectedCodebookFeatureIndex = -1;

    /**
     * Creates new form QuantizedImageViewer
     */
    public QuantizedImageViewer() {
        initComponents();
    }

    /**
     * Initialization.
     *
     * @param originalImage BufferedImage that is the current image.
     * @param imageFRep Image feature representation.
     * @param codebookGoodness Float array of codebook goodness scores.
     * @param codebook GenericCodeBook object that holds the visual word
     *