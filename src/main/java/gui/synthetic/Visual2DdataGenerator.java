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
        drawDSPanel.totalWidth = drawDSPanel.getWidth();
        drawDSPanel.totalHeight = drawDSPanel.getHeight();
        drawDSPanel.endX = drawDSPanel.getWidth();
        drawDSPanel.endY = drawDSPanel.getHeight();
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        classSelectButtons = new javax.swing.ButtonGroup();
        class0Radio = new javax.swing.JRadioButton();
        class1Radio = new javax.swing.JRadioButton();
        class2Radio = new javax.swing.JRadioButton();
        class3Radio = new javax.swing.JRadioButton();
        class4Radio = new javax.swing.JRadioButton();
        class5Radio = new javax.swing.JRadioButton();
        class6Radio = new javax.swing.JRadioButton();
        class7Radio = new javax.swing.JRadioButton();
        class8Radio = new javax.swing.JRadioButton();
        class9Radio = new javax.swing.JRadioButton();
        class10Radio = new javax.swing.JRadioButton();
        class11Radio = new javax.swing.JRadioButton();
        class12Radio = new javax.swing.JRadioButton();
        class13Radio = new javax.swing.JRadioButton();
        class14Radio = new javax.swing.JRadioButton();
        addClassButton = new javax.swing.JButton();
        xNameLabel = new javax.swing.JLabel();
        yNameLabel = new javax.swing.JLabel();
        xLabel = new javax.swing.JLabel();
        yLabel = new javax.swing.JLabel();
        scalingNameLabel = new javax.swing.JLabel();
        scaleTextField = new javax.swing.JTextField();
        drawDSPanel = new gui.synthetic.DatasetDrawingPanel();
        menuBar = new javax.swing.JMenuBar();
        fileMenu = new javax.swing.JMenu();
        fileMenu.setMnemonic(KeyEvent.VK_F);
        newItem = new javax.swing.JMenuItem();
        newItem.setMnemonic(KeyEvent.VK_N);
        openItem = new javax.swing.JMenuItem();
        openItem.setMnemonic(KeyEvent.VK_O);
        saveItem = new javax.swing.JMenuItem();
        saveItem.setMnemonic(KeyEvent.VK_S);
        closeItem = new javax.swing.JMenuItem();
        closeItem.setMnemonic(KeyEvent.VK_C);
        editMenu = new javax.swing.JMenu();
        editMenu.setMnemonic(KeyEvent.VK_E);
        noiseItem = new javax.swing.JMenuItem();
        noiseItem.setMnemonic(KeyEvent.VK_N);
        mislabelItem = new javax.swing.JMenuItem();
        mislabelItem.setMnemonic(KeyEvent.VK_M);
        rotateItem = new javax.swing.JMenuItem();
        rotateItem.setMnemonic(KeyEvent.VK_R);
        undoItem = new javax.swing.JMenuItem();
        undoItem.setMnemonic(KeyEvent.VK_U);
        insertGaussianDataitem = new javax.swing.JMenuItem();
        insertGaussianDataitem.setMnemonic(KeyEvent.VK_I);
        toolsMenu = new javax.swing.JMenu();
        toolsMenu.setMnemonic(KeyEvent.VK_T);
        propertiesSubMenu = new javax.swing.JMenu();
        propertiesSubMenu.setMnemonic(KeyEvent.VK_P);
        bgColorItem = new javax.swing.JMenuItem();
        bgColorItem.setMnemonic(KeyEvent.VK_B);
        imageExportItem = new javax.swing.JMenuItem();
        imageExportItem.setMnemonic(KeyEvent.VK_E);
        knnMenu = new javax.swing.JMenu();
        knnMenu.setMnemonic(KeyEvent.VK_K);
        hubnessItem = new javax.swing.JMenu();
        hubnessLandscapeItem = new javax.swing.JMenuItem();
        hubnessLandscapeItem.setMnemonic(KeyEvent.VK_H);
        HubnessEntropyLandscapeItem = new javax.swing.JMenuItem();
        HubnessEntropyLandscapeItem.setMnemonic(KeyEvent.VK_E);
        badHubnessInterpolatedItem = new javax.swing.JMenuItem();
        badHubnessInterpolatedItem.setMnemonic(KeyEvent.VK_B);
        badHubnessKNNItem = new javax.swing.JMenuItem();
        classMapsMenu = new javax.swing.JMenu();
        knnDensityMenuItem = new javax.swing.JMenuItem();
        knnDensityMenuItem.setMnemonic(KeyEvent.VK_K);
        nhbnnProbMenuItem = new javax.swing.JMenuItem();
        nhbnnProbMenuItem.setMnemonic(KeyEvent.VK_N);
        hiknnInformationMapItem = new javax.swing.JMenuItem();
        hiknnInformationMapItem.setMnemonic(KeyEvent.VK_H);
        hiknnNonWeightedInfoItem = new javax.swing.JMenuItem();
        hwKNNDensityMenuItem = new javax.swing.JMenuItem();
        hFNNDensityMenuItem = new javax.swing.JMenuItem();
        helpMenu = new javax.swing.JMenu();
        helpMenu.setMnemonic(KeyEvent.VK_H);
        aboutItem = new javax.swing.JMenuItem();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);

        classSelectButtons.add(class0Radio);
        class0Radio.setText("Class 0");

        classSelectButtons.add(class1Radio);
        class1Radio.setText("Class 1");

        classSelectButtons.add(class2Radio);
        class2Radio.setText("Class 2");

        classSelectButtons.add(class3Radio);
        class3Radio.setText("Class 3");

        classSelectButtons.add(class4Radio);
        class4Radio.setText("Class 4");

        classSelectButtons.add(class5Radio);
        class5Radio.setText("Class 5");

        classSelectButtons.add(class6Radio);
        class6Radio.setText("Class 6");

        classSelectButtons.add(class7Radio);
        class7Radio.setText("Class 7");

        classSelectButtons.add(class8Radio);
        class8Radio.setText("Class 8");

        classSelectButtons.add(class9Radio);
        class9Radio.setText("Class 9");

        classSelectButtons.add(class10Radio);
        class10Radio.setText("Class 10");

        classSelectButtons.add(class11Radio);
        class11Radio.setText("Class 11");

        classSelectButtons.add(class12Radio);
        class12Radio.setText("Class 12");

        classSelectButtons.add(class13Radio);
        class13Radio.setText("Class 13");

        classSelectButtons.add(class14Radio);
        class14Radio.setText("Class 14");

        addClassButton.setText("Add Class");
        addClassButton.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mouseClicked(java.awt.event.MouseEvent evt) {
                addClassButtonMouseClicked(evt);
            }
        });

        xNameLabel.setText("X:");

        yNameLabel.setText("Y:");

        xLabel.setText("0");

        yLabel.setText("0");

        scalingNameLabel.setText("setScaling factor:");

        drawDSPanel.setBackground(new java.awt.Color(255, 255, 255));
        drawDSPanel.setBorder(new javax.swing.border.MatteBorder(null));
        drawDSPanel.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mouseClicked(java.awt.event.MouseEvent evt) {
                drawDSPanelMouseClicked(evt);
            }
        });
        drawDSPanel.addMouseMotionListener(new java.awt.event.MouseMotionAdapter() {
            public void mouseMoved(java.awt.event.MouseEvent evt) {
                drawDSPanelMouseMoved(evt);
            }
        });

        javax.swing.GroupLayout drawDSPanelLayout = new javax.swing.GroupLayout(drawDSPanel);
        drawDSPanel.setLayout(drawDSPanelLayout);
        drawDSPanelLayout.setHorizontalGroup(
            drawDSPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 868, Short.MAX_VALUE)
        );
        drawDSPanelLayout.setVerticalGroup(
            drawDSPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 511, Short.MAX_VALUE)
        );

        fileMenu.setText("<html><u>F</u>ile");

        newItem.setText("<html><u>N</u>ew");
        newItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                newItemActionPerformed(evt);
            }
        });
        fileMenu.add(newItem);

        openItem.setText("<html><u>O</u>pen");
        openItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                openItemActionPerformed(evt);
            }
        });
        fileMenu.add(openItem);

        saveItem.setText("<html><u>S</u>ave");
        saveItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                saveItemActionPerformed(evt);
            }
        });
        fileMenu.add(saveItem);

        closeItem.setText("<html><u>Q</u>uit");
        closeItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                closeItemActionPerformed(evt);
            }
        });
        fileMenu.add(closeItem);

        menuBar.add(fileMenu);

        editMenu.setText("<html><u>E</u>dit");

        noiseItem.setText("<html>Add Gaussian <u>N</u>oise</html>");
        noiseItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                noiseItemActionPerformed(evt);
            }
        });
        editMenu.add(noiseItem);

        mislabelItem.setText("<html>Induce <u>M</u>islabeling</html>");
        mislabelItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                mislabelItemActionPerformed(evt);
            }
        });
        editMenu.add(mislabelItem);

        rotateItem.setText("<html><u>R</u>otate</html>");
        rotateItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                rotateItemActionPerformed(evt);
            }
        });
        editMenu.add(rotateItem);

        undoItem.setText("<html><u>U</u>ndo</html>");
        undoItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                undoItemActionPerformed(evt);
            }
        });
        editMenu.add(undoItem);

        insertGaussianDataitem.setText("<html><u>I</u>nsert Gaussian Data");
        insertGaussianDataitem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                insertGaussianDataitemActionPerformed(evt);
            }
        });
        editMenu.add(insertGaussianDataitem);

        menuBar.add(editMenu);

        toolsMenu.setText("<html><u>T</u>ools");

        propertiesSubMenu.setText("<html><u>P</u>roperties");

        bgColorItem.setText("<html><u>B</u>ackground color</html>");
        bgColorItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                bgColorItemActionPerformed(evt);
            }
        });
        propertiesSubMenu.add(bgColorItem);

        toolsMenu.add(propertiesSubMenu);

        imageExportItem.setText("<html><u>E</u>xport image</html>");
        imageExportItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                imageExportItemActionPerformed(evt);
            }
        });
        toolsMenu.add(imageExportItem);

        menuBar.add(toolsMenu);

        knnMenu.setText("<html><u>K</u>NN");

        hubnessItem.setText("hubness");

        hubnessLandscapeItem.setText("<html><u>H</u>ubness landscape");
        hubnessLandscapeItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                hubnessLandscapeItemActionPerformed(evt);
            }
        });
        hubnessItem.add(hubnessLandscapeItem);

        HubnessEntropyLandscapeItem.setText("<html>Hubness <u>E</u>ntropy landscape");
        HubnessEntropyLandscapeItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                HubnessEntropyLandscapeItemActionPerformed(evt);
            }
        });
        hubnessItem.add(HubnessEntropyLandscapeItem);

        badHubnessInterpolatedItem.setText("<html><u>B</u>ad hubness interpolated");
        badHubnessInterpolatedItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                badHubnessInterpolatedItemActionPerformed(evt);
            }
        });
        hubnessItem.add(badHubnessInterpolatedItem);

        badHubnessKNNItem.setText("<html>Bad Hubness kNN");
        badHubnessKNNItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                badHubnessKNNItemActionPerformed(evt);
            }
        });
        hubnessItem.add(badHubnessKNNItem);

        knnMenu.add(hubnessItem);

        classMapsMenu.setText("classification maps");

        knnDensityMenuItem.setText("<html><u>K</u>NN probability map");
        knnDensityMenuItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                knnDensityMenuItemActionPerformed(evt);
            }
        });
        classMapsMenu.add(knnDensityMenuItem);

        nhbnnProbMenuItem.setText("<html><u>N</u>HBNN probability map");
        nhbnnProbMenuItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                nhbnnProbMenuItemActionPerformed(evt);
            }
        });
        classMapsMenu.add(nhbnnProbMenuItem);

        hiknnInformationMapItem.setText("<html><u>H</u>IKNN information map");
        hiknnInformationMapItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                hiknnInformationMapItemActionPerformed(evt);
            }
        });
        classMapsMenu.add(hiknnInformationMapItem);

        hiknnNonWeightedInfoItem.setText("<html>Non-Weighted HIKNN information map");
        hiknnNonWeightedInfoItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                hiknnNonWeightedInfoItemActionPerformed(evt);
            }
        });
        classMapsMenu.add(hiknnNonWeightedInfoItem);

        hwKNNDensityMenuItem.setText("hw-KNN probability map");
        hwKNNDensityMenuItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                hwKNNDensityMenuItemActionPerformed(evt);
            }
        });
        classMapsMenu.add(hwKNNDensityMenuItem);

        hFNNDensityMenuItem.setText("h-FNN probability map");
        hFNNDensityMenuItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                hFNNDensityMenuItemActionPerformed(evt);
            }
        });
        classMapsMenu.add(hFNNDensityMenuItem);

        knnMenu.add(classMapsMenu);

        menuBar.add(knnMenu);

        helpMenu.setText("<html><u>H</u>elp");

        aboutItem.setText("<html><u>A</u>bout");
        helpMenu.add(aboutItem);

        menuBar.add(helpMenu);

        setJMenuBar(menuBar);

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(yNameLabel)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(yLabel))
                    .addGroup(layout.createSequentialGroup()
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addGroup(layout.createSequentialGroup()
                                .addComponent(xNameLabel)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(xLabel)
                                .addGap(231, 231, 231)
                                .addComponent(scalingNameLabel)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(scaleTextField, javax.swing.GroupLayout.PREFERRED_SIZE, 92, javax.swing.GroupLayout.PREFERRED_SIZE))
                            .addComponent(drawDSPanel, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                        .addGap(19, 19, 19)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(addClassButton)
                            .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
                                .addComponent(class0Radio)
                                .addComponent(class1Radio)
                                .addComponent(class2Radio)
                                .addComponent(class3Radio)
                                .addComponent(class4Radio)
                                .addComponent(class5Radio)
                                .addComponent(class6Radio)
                                .addComponent(class7Radio)
                                .addComponent(class8Radio))
                            .addComponent(class9Radio)
                            .addComponent(class10Radio)
                            .addComponent(class11Radio)
                            .addComponent(class12Radio)
                            .addComponent(class13Radio)
          