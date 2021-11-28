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
package data.representation.images.sift;

import data.representation.DataSet;
import java.awt.geom.Point2D;
import java.util.ArrayList;
import learning.unsupervised.Cluster;

/**
 * This class implements a container for individual local features that can be
 * used as a feature representation of an individual image.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class LFeatRepresentation extends DataSet {

    private String imagePath;
    public static final int DEFAULT_DESCRIPTOR_LENGTH = 128;

    /**
     * @param imagePath String that is the image path.
     */
    public void setPath(String imagePath) {
        this.imagePath = imagePath;
    }

    /**
     * @return String that is the corresponding image path.
     */
    public String getPath() {
        return imagePath;
    }

    /**
     */
    public LFeatRepresentation() {
        // Assumes standard descriptor length
        fAttrNames = new String[DEFAULT_DESCRIPTOR_LENGTH + 4];
        fAttrNames[0] = "Y";
        fAttrNames[1] = "X";
        fAttrNames[2] = "Scale";
        fAttrNames[3] = "Angle";
        for (int i = 0; i < 128; i++) {
            fAttrNames[i + 4] = "desc" + i;
        }
        data = new ArrayList<>(100);
    }

    /**
     * Generates a SIFT data definition based on the descriptor length.
     *
     * @param descLength The length of SIFT descriptors.
     */
    public LFeatRepresentation(int descLength) {
        fAttrNames = new String[descLength + 4];
        fAttrNames[0] = "Y";
        fAttrNames[1] = "X";
        fAttrNames[2] = "Scale";
        fAttrNames[3] = "Angle";
        for (int i = 0; i < descLength; i++) {
            fAttrNames[i + 4] = "desc" + i;
        }
        data = new ArrayList<>(100);
    }

    /**
     * @param initsize Initial size of the data array.
     * @param increment Increment.
     */
    public LFeatRepresentation(int initsize, int increment) {
        // Assumes standard descriptor length
        fAttrNames = new String[DEFAULT_DESCRIPTOR_LENGTH + 4];
        fAttrNames[0] = "Y";
        fAttrNames[1] = "X";
        fAttrNames[2] = "Scale";
        fAttrNames[3] = "Angle";
        for (int i = 0; i < 128; i++) {
            fAttrNames[i + 4] = "desc" + i