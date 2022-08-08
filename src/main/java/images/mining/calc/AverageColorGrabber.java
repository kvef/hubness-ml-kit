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
package images.mining.calc;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.awt.image.PixelGrabber;

/**
 * This class implements the functionality for getting an average color in an
 * neighborhood of a point in an array of pixels.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class AverageColorGrabber {

    // Must be an odd number.
    public static final int WINDOW_WIDTH = 7;
    BufferedImage image;

    /**
     * @param image BufferedImage that is to be analyzed.
     */
    public AverageColorGrabber(BufferedImage image) {
        this.image = image;
    }

    /**
     * Get the average color in a neighborhood of a point.
     *
     * @param x Float that is the x coordinate.
     * @param y Float that is the y coordinate.
     * @return Integer array of red, green and blue color values of the
     * calculated average color.
     */
    public int[] getAverageColorInArray(float x, float y) {
        // Lower and upper bounds.
        int x_l