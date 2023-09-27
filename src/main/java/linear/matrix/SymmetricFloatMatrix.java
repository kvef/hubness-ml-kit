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
package linear.matrix;

import ioformat.FileUtil;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;

/**
 * In this variant diagonal elements do exist (they are not equal to zero).
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class SymmetricFloatMatrix implements DataMatrixInterface {

    private float[][] data = null;

    /**
     * @return Two-dimensional float array that is the row representation of the
     * symmetric float matrix.
     */
    public float[][] getMatrix2DArray() {
        return data;
    }

    public SymmetricFloatMatrix() {
    }

    /**
     * @param dim Number of rows and columns.
     */
    public SymmetricFloatMatrix(int dim) {
        data = new float[dim][];
        for (int i = 0; i < dim; i++) {
           