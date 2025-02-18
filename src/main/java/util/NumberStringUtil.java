
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
package util;

/**
 * Utility class for formatting numbers for printout.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class NumberStringUtil {

    /**
     * Prepend zeroes to a number, so that all numbers can have the same length
     * in their String representation.
     *
     * @param numChars Number of characters to extend the number to.
     * @param number Integer value to extend.
     * @return
     */
    public static String getNumberString(int numChars, int number) {
        number = number % (int) (Math.pow(10, numChars));
        int numberChars;
        if (number > 0) {
            numberChars = (int) Math.log10(number) + 1;
        } else {
            numberChars = 1;
        }
        String s = "";
        for (int i = 0; i < numChars - numberChars; i++) {
            s += "0";
        }
        s += number;
        return s;
    }
}