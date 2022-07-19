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
package gui.synthetic;

import data.neighbors.NeighborSetFinder;
import data.representation.DataInstance;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import distances.primary.MinkowskiMetric;
import draw.basic.ColorPalette;
import ioformat.IOARFF;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.Toolkit;
import java.awt.event.KeyEvent;
import java.awt.image.BufferedImage;
import java.awt.image.MemoryImageSource;
import java.io.File;
import java.util.ArrayList;
import java.util.Random;
import javax.imageio.ImageIO;
import javax.swing.JColorChooser;
import javax.swing.JFileChooser;
import javax.swing.JOptionPane;
import javax.swing.JRadioButton;
import learning.supervised.Classifier;
import learning.supervised.methods.knn.DWHFNN;
import learning.supervised.methods.knn.HFNN;
import learning.supervised.methods.knn.HwKNN;
import learning.supervised.methods.knn.HIKNN;
import learning.supervised.methods.knn.HIKNNNonDW;
import learning.supervised.methods.knn.KNN;
import learning.supervised.methods.knn.NHBNN;
import util.BasicMathUtil;

/**
 * This class is a GUI for creating synthetic 2D data by inserting either
 * individual points or data distributions at the specified coordinates. It
 * supports multi-class data and can also generate visualizations of the kNN and
 * the reverse kNN topology, as well as classification landscapes that
 * correspond to a set of kNN methods that are implemented in this library. It
 * is possible to insert uniform noise into the data. The current implementation
 * has an upper limit on the number of possible classes to insert the instances
 * for (which can be changed), though this GUI is meant primarily for making
 * illustrative and toy examples, meaning that inserting hundreds or thousands
 * of classes would make little sense from the user perspective.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class Visual2DdataGenerator extends javax.swing.JFrame {

    private JRadioButton[] classChoosers;
    // Display colors for different classes.
    private Color[] classColors;
    private int numVisibleClasses = 1;
    private File currentDirectory = new File(".");
    private File currentInFile = null;
    private File currentOutFile = null;
    // Whether to insert a point at the specified coordinates or a given
    // Gaussian distribution. Users can switch from one mode to the other.
    private boolean gaussianInsertionMode = false;

    /**
     * Creates new form Visual2DdataGenerator
     */
    public Visual2DdataGenerator() {
        initComponents();
        initialization();
    }

    /**
     * Initialize all the components.
     */
    public final void initialization() {
        // Initialize the class chooser buttons.
        classChoosers = new JRadioButton[15];
        classChoosers[0] = class0Radio;
        classChoosers[1] = class1Radio;
        classChoosers[2] = class2Radio;
        classChoosers[3] = class3Radio;
        classChoosers[4] = class4Radio;
        classChoosers[5] = class5Radio;
        classChoosers[6] = class6Radio;
        classChoosers[7] = class7Radio;
        classChoosers[8] = class8Radio;
        classChoosers[9] = class9Radio;
        classChoosers[10] = class10Radio;
        classChoosers[11] = class11Radio;
        classChoosers[12] = class12Radio;
        classChoosers[13] = class13Radio;
        classChoosers[14] = class14Radio;
        classChoosers[0].setVisible(true);
        for (int i = 1; i < classChoosers.length; i++) {
            classChoosers[i].setVisible(false);
        }
        // Initialize class colors.
        classColors = new Color[15];
        ColorPalette palette = new ColorPalette(0.4);
        classColors[0] = palette.FIREBRICK_RED;
        classColors[1] = palette.MEDIUM_SPRING_GREEN;
        classColors[2] = palette.SLATE_BLUE;
        classColors[3] = palette.DARK_WOOD;
        classColors[4] = palette.DARK_OLIVE_GREEN;
        classColors[5] = palette.ORANGE;
        classColors[6] = palette.DIM_GREY;
        classColors[7] = palette.MAROON;
        classColors[8] = palette.MEDIUM_AQUAMARINE;
        classColors[9] = palette.YELLOW_GREEN;
        classColors[10] = palette.KHAKI;
        classColors[11] = palette.PLUM;
        classColors[12] = palette.PEACH_PUFF;
        classColors[13] = palette.VIOLET;
        classColors[14] = palette.OLD_GOLD;
        for (int i = 0; i < classChoosers.length; i++) {
            classChoosers[i].setBackground(classColors[i]);
        }
        // Initialize the drawing panel for the points themselves.
        drawDSPanel.classColors = classColors;
        drawDSPanel.totalWidth = drawDS