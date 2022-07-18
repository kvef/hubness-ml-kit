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

import java.awt.Graphics;
import java.awt.Color;
import data.representation.DataSet;
import data.representation.DataInstance;
import java.util.ArrayList;

/**
 * This class represents a panel for drawing synthetic 2D datasets manually.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class DatasetDrawingPanel extends javax.swing.JPanel {

    // Action constants.
    public static final int INSTANCE_ADD = 0;
    public static final int DATASET_CHANGE = 1;
    public static final int BG_COLOR_CHANGE = 2;
    // There is an undo option here, so we keep track of the history through
    // these variables.
    public ArrayList<DataSet> allDSets = new ArrayList<>(200);
    public ArrayList<Color> prevColors = new ArrayList<>(10);
    public ArrayList<Integer> actionHistory = new ArrayList<>(1000);
    // Different colors for different classes.
    public Color[] classColors;
    // Float values are in the range [0,1], they are re-scaled afterwards.
    public DataSet dset = null;
    public int circlePointRadius = 3;
    public int currClass = 0;
    // Additional variables to support zooming.
    public int startX = 0;
    public int endX = getWidth();
    public int startY = 0;
    public int endY = getHeight();
    // Height and width of the bounding rectangle.
    public int totalHeight = getHeight();
    public int totalWidth = getWidth();

    /**
     * Creates new form DatasetDrawingPanel
     */
    public DatasetDrawingPanel() {
        initComponents();
    }

    /**
     * Undo the last action.
     */
    public void undoLast() {
        if (actionHistory.size() > 0) {
            int lastAction = actionHistory.get(actionHistory.size() - 1);
            actionHistory.remove(actionHistory.size() - 1);
            if (lastAction == INSTANCE_ADD) {
                dset.data.remove(dset.size() - 1);
            } else if (lastAction == BG_COLOR_CHANGE) {
                setBackground(prevColors.get(prevColors.size() - 1));
                prevColors.remove(prevColors.size() - 1);
            } else if (lastAction == DATASET_CHANGE) {
                dset = allDSets.get(allDSets.size() - 1);
               