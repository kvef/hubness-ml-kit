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
package statistics.tests;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import statistics.tests.constants.TConst;

/**
 * This class performs the t-test by checking if the score exceeds the critical
 * values. The output of these methods is to be interpreted as follows below: 0:
 * no significance, 1: .05 level 2: .01 level
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class TTests {

    public static final int NO_SIGNIFICANCE = 0;
    public static final int SIGNIFICANCE_5 = 1;
    public static final int SIGNIFICANCE_1 = 2;
    // This object contains the information on the critical values.
    TConst critTable = null;

    public TTests() {
        critTable = new TConst();
    }

    /**
     * Paired two-tailed t-test.
     *
     * @param resA First float value array.
     * @param resB Second float value array.
     * @return 0: no significance, 1: .05 level 2: .01 level
     */
    public int pairedTwoTailed(float[] resA, float[] resB) {
        if (resA == null || resB == null) {
            return 0;
        }
        float numSamples = resA.length;
        // This is Student's t-value.
        float t;
        float difsSUM = 0;
        float difsSQSUM = 0;
        float difsMean;
        float[] difs = new float[resA.length];
        for (int i = 0; i < difs.length; i++) {
            difs[i] = resA[i] - resB[i];
            difsSUM += difs