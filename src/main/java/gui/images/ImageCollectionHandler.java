
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

import data.representation.DataInstance;
import data.representation.DataSet;
import data.representation.images.color.ColorHistogramVector;
import data.representation.images.quantized.QuantizedImageDistribution;
import data.representation.images.quantized.QuantizedImageDistributionDataSet;
import data.representation.images.quantized.QuantizedImageHistogram;
import data.representation.images.quantized.QuantizedImageHistogramDataSet;
import data.representation.images.sift.LFeatRepresentation;
import distances.primary.ColorsAndCodebooksMetric;
import distances.primary.CombinedMetric;
import distances.primary.Manhattan;
import images.mining.codebook.SIFTCodeBook;
import images.mining.codebook.SIFTCodebookMaker;
import ioformat.FileUtil;
import ioformat.IOARFF;
import ioformat.images.ConvertJPGToPGM;
import ioformat.images.SiftUtil;
import ioformat.images.ThumbnailMaker;
import java.awt.Color;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Timer;
import java.util.TimerTask;
import javax.imageio.ImageIO;
import javax.swing.ButtonGroup;
import javax.swing.JFileChooser;
import javax.swing.JOptionPane;
import javax.swing.JProgressBar;
import javax.swing.JRadioButton;
import learning.unsupervised.Cluster;
import learning.unsupervised.ClusterConfigurationCleaner;
import learning.unsupervised.ClusteringAlg;
import learning.unsupervised.evaluation.quality.OptimalConfigurationFinder;
import learning.unsupervised.methods.FastKMeans;
import util.FileCounter;

/**
 * This GUI allows the users to do some batch image processing and SIFT feature
 * extraction via SiftWin. More types of features should be included in future
 * versions. It can also cluster images and show a summary of clustering
 * results, as well as generate quantized feature representations via feature
 * clustering and codebook assignments. Apart from SIFT features, this class can
 * also extract the color histogram information from images, combine the two
 * representations into one - and make image thumbnails for later visualization.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ImageCollectionHandler extends javax.swing.JFrame {

    private File currentDirectory = new File(".");