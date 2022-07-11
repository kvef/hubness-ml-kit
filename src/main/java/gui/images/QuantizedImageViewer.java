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
     * definitions.
     * @param codebookProfiles double[][] that represents the class-conditional
     * occurrence profiles for all the codebooks.
     * @param cProfPanels CodebookVectorProfilePanel[] of the codebook profile
     * panels.
     * @param classColors Color[] of class colors.
     * @param classNames String[] of class names.
     */
    public QuantizedImageViewer(
            BufferedImage originalImage,
            LFeatRepresentation imageFRep,
            float[] codebookGoodness,
            GenericCodeBook codebook,
            double[][] codebookProfiles,
            CodebookVectorProfilePanel[] cProfPanels,
            Color[] classColors,
            String[] classNames) {
        initComponents();
        codebookProfilesPanel.setLayout(new FlowLayout());
        this.imageFRep = imageFRep;
        this.codebookGoodness = codebookGoodness;
        this.codebook = codebook;
        this.codebookProfiles = codebookProfiles;
        this.classColors = classColors;
        this.classNames = classNames;
        this.originalImage = originalImage;
        // Get a proper black-white image to draw on.
        bwImage = new BufferedImage(originalImage.getWidth(),
                originalImage.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g2d = bwImage.createGraphics();
        g2d.drawImage(originalImage, 0, 0, originalImage.getWidth(),
                originalImage.getHeight(), null);
        BufferedImage bwImageTmp = new BufferedImage(originalImage.getWidth(),
                originalImage.getHeight(), BufferedImage.TYPE_INT_ARGB);
        g2d = bwImageTmp.createGraphics();
        g2d.drawImage(bwImage, 0, 0, originalImage.getWidth(),
                originalImage.getHeight(), null);
        bwImage = bwImageTmp;
        partialReps = new LFeatRepresentation[codebook.getSize()];
        codebookVisualizationImages = new BufferedImage[codebook.getSize()];
        // Insert all the individual codebook profile visualization panels.
        for (int cInd = 0; cInd < codebook.getSize(); cInd++) {
            codebookProfilesPanel.add(cProfPanels[cInd]);
            cProfPanels[cInd].addMouseListener(new CodebookSelectionListener());
            partialReps[cInd] = new LFeatRepresentation();
        }
        codebookProfilesPanel.revalidate();
        codebookProfilesPanel.repaint();
        codebookScrollPane.revalidate();
        codebookScrollPane.repaint();
        originalImagePanel.setImage(originalImage);
        // Calculate the closest codebook feature for each feature in the
        // original image.
        float[] featureGoodness = new float[imageFRep.size()];
        codebookAssignments = new int[imageFRep.size()];
        for (int i = 0; i < imageFRep.size(); i++) {
            LFeatVector sv = (LFeatVector) (imageFRep.getInstance(i));
            try {
                codebookAssignments[i] = codebook.getIndexOfClosestCodebook(sv);
                partialReps[codebookAssignments[i]].addDataInstance(sv);
            } catch (Exception e) {
                System.err.println("Quantization error.");
                System.err.println(e);
            }
            featureGoodness[i] = codebookGoodness[codebookAssignments[i]];
        }
        // Determine the best visual word and visualize it first by default.
        int maxRepIndex = 0;
        int maxSize = 0;
        for (int cInd = 0; cInd < codebook.getSize(); cInd++) {
            if (partialReps[cInd].size() > maxSize) {
                maxRepIndex = cInd;
                maxSize = partialReps[cInd].size();
            }
        }
        visualizeVisualWordUtility(maxRepIndex);
        // Visualize the overall utility of different image regions.
        try {
            goodnessSIFTImage = SIFTDraw.drawSIFTGoodnessOnImage(imageFRep,
                    featureGoodness, bwImage);
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
        allQuantizedPanel.setImage(goodnessSIFTImage);
        setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
    }

    /**
     * Visualizes the utility of a particular visual word.
     *
     * @param codebookfeatureIndex Integer that is the index of the particular
     * codebook feature.
     */
    private void visualizeVisualWordUtility(int codebookfeatureIndex) {
        occCountValueLabel.setText(""
                + partialReps[codebookfeatureIndex].size());
        selectedCodebookFeatureIndex = codebookfeatureIndex;
        cvectLabel.setText("Observing codebook vector: "
                + codebookfeatureIndex);
        if (codebookVisualizationImages[codebookfeatureIndex] != null) {
            // If the visualization has already been calculated, just move to
            // the appropriate object in-memory.
            selectedCodebookPanel.setImage(
                    codebookVisualizationImages[codebookfeatureIndex]);
        } else {
            // Calculate a new visualization.
            if (partialReps[codebookfeatureIndex].isEmpty()) {
                // If there are no matches for this codebook feature, just show
                // the grayscale image with no features on top.
                selectedCodebookPanel.setImage(bwImage);
                return;
            }
            // Set the local image feature goodness for each feature in this
            // partial view that contains only the matches to the current
            // visual word to be the goodness of that visual word.
            float[] featureGoodness = new float[partialReps[
                    codebookfeatureIndex].size()];
            Arrays.fill(featureGoodness, 0, featureGoodness.length,
                    codebookGoodness[codebookfeatureIndex]);
            try {
                codebookVisualizationImages[codebookfeatureIndex] =
                        SIFTDraw.drawSIFTGoodnessOnImage(
                        partialReps[codebookfeatureIndex],
                        featureGoodness, bwImage);
            } catch (Exception e) {
                System.err.println(e.getMessage());
            }
            selectedCodebookPanel.setImage(
                    codebookVisualizationImages[codebookfeatureIndex]);
        }
    }

    /**
     * Listener for visual word selections.
     */
    class CodebookSelectionListener implements MouseListener {

        @Override
        public void mousePressed(MouseEvent e) {
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
            System.out.println("selection made");
            if (comp instanceof CodebookVectorProfilePanel) {
                int index = ((CodebookVectorProfilePanel) comp).
                        getCodebookIndex();
                System.out.println("selected index " + index);
                visualizeVisualWordUtility(index);
            } else if (comp.getParent() != null && comp.getParent() instanceof
                    CodebookVectorProfilePanel) {
                int index = ((CodebookVectorProfilePanel) comp.getParent()).
                        getCodebookIndex();
                System.out.println("selected index " + index);
           