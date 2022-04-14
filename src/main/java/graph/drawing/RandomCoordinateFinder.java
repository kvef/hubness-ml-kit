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
package graph.drawing;

import graph.basic.DMGraph;
import graph.basic.VertexInstance;
import graph.io.JGraphConverter;
import java.util.Random;

/**
 * This class sets the random coordinates to the graph noces.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class RandomCoordinateFinder implements CoordinateFinderInterface {

    private DMGraph g = null;
    private double progress = 0d;
    // Whether to automatically update the associated JGraph object.
    private boolean updateJG = false;
    private boolean isRunning = true;
    private double width;
    private double height;

    /**
     * Initialization.
     *
     * @param g DMGraph object to set the vertex coordinates for.
     * @param width Double that is the width of the canvas.
     * @param height Double that is the height of the canvas.
     */
    public RandomCoordinateFinder(DMGraph g, double width, double height) {
        this.g = g;
        this.width = width;
        this.height = height;
    }

    @Override
    public void run() {
        try {
            findCoordinates();
        } catch (Exception e) {
        }
    }

    @Override
    public void setAutoJGUpdate(boolean updateJG) {
        this.updateJG = updateJG;
    }

    @O