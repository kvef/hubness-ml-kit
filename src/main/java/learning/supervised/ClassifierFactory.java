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
package learning.supervised;

import distances.primary.CombinedMetric;
import java.util.ArrayList;
import learning.supervised.evaluation.ValidateableInterface;
import learning.supervised.meta.boosting.AdaBoostM2;
import learning.supervised.meta.boosting.baselearners.DWHFNNBoostable;
import learning.supervised.meta.boosting.baselearners.HFNNBoostable;
import learning.supervised.meta.boosting.baselearners.HIKNNBoostable;
import learning.supervised.meta.boosting.baselearners.HwKNNBoostable;
import learning.supervised.methods.RSLVQ;
import learning.supervised.methods.discrete.DNaiveBayes;
import learning.supervised.methods.discrete.DOneRule;
import learning.supervised.methods.discrete.DWeightedNaiveBayes;
import learning.supervised.methods.discrete.DZeroRule;
import learning.supervised.methods.discrete.KNNNB;
import learning.supervised.methods.discrete.LWNB;
import learning.supervised.methods.discrete.trees.DCT_ID3;
import learning.supervised.methods.knn.AKNN;
import learning.supervised.methods.knn.ANHBNN;
import learning.supervised.methods.knn.CBWkNN;
import learning.supervised.methods.knn.DWHFNN;
import learning.supervised.methods.knn.DWKNN;
import learning.supervised.methods.knn.FNN;
import learning.supervised.methods.knn.HFNN;
import learning.supervised.methods.knn.HIKNN;
import learning.supervised.methods.knn.HIKNNNonDW;
import learning.supervised.methods.knn.HwKNN;
import learning.supervised.methods.knn.KNN;
import learning.supervised.methods.knn.NHBNN;
import learning.supervised.methods.knn.NWKNN;
import learning.supervised.methods.knn.RRKNN;

/**
 * This class is used for obtaining initial classifier instances for the
 * specified neighborhood size, metric and number of classes in the data.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ClassifierFactory {

    /**
     * This method generates an initial classifier instance for the specified
     * classifier name, neighborhood size, metric and number of classes.
     *
     * @param classifierName String that is the classifier name.
     * @param numCategories Integer that is the number of categories in the
     * data.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     * @return ValidateableInterface object corresponding to the parameter
     * specification.
     */
    public ValidateableInterface getClassifierForName(String classifierName,
            int numCategories, CombinedMetric cmet, int k) {
        ValidateableInterface classAlg;
        if (classifierName.equalsIgnoreCase("DNaiveBayes")
                || classifierName.equalsIgnoreCase("learning.supervised."
                + "methods.discrete.DNaiveBayes")) {
            classAlg = new DNaiveBayes();
        } else if (classifierName.equalsIgnoreCase("DWeightedNaiveBayes")
                || classifierName.equalsIgnoreCase("learning.supervised."
                + "methods.discrete.DWeightedNaiveBayes")) {
            classAlg = new DWeightedNaiveBayes();
        } else if (classifierName.equalsIgnoreCase("KNN")
                || classifierName.equalsIgnoreCase("learning.supervised."
                + "methods.knn.KNN")) {
            classAlg = new KNN(k, cmet);
        } else if (classifierName.equalsIgnoreCase("dwKNN")
                || classifierName.equalsIgnoreCase("learning.supervised."
                + "methods.knn.DWKNN")) {
            classAlg = new DWKNN(k, cmet);
        } else if (classifierName.equalsIgnoreCase("AKNN")
                || classifierName.equalsIgnoreCase("learning.supervised."
                + "methods.knn.AKNN")) {
            classAlg = new AKNN(k, cmet, numCategories);
        } else if (classifierName.equalsIgnoreCase("NWKNN")
                || classifierName.equalsIgnoreCase("learning.supervised."
                + "methods.knn.NWKNN")) {
            classAlg = new NWKNN(k, cmet, numCategories);
        } else if (classifierName.equalsIgnoreCase("hwKNN")
                || classifierName.equalsIgnoreCase("learning.supervised."
                + "methods.knn.HwKNN")) {
            classAlg = new HwKNN(numCategories, cmet, k);
        } else if (classifierName.equalsIgnoreCase("CBWkNN")
                || classifierName.equalsIgnoreCase("learning.supervised."
                + "methods.knn.CBWkNN")) {
            classAlg = new CBWkNN(numCategories, cmet, k);
        } else if (classifierName.equalsIgnoreCase("DWHFNN")
                || classifierName.equalsIgnoreCase("DWH-FNN")
                || classifierName.equalsIgnoreCase("learning.supervised."
                + "methods.knn.DWHFNN")) {
            classAlg = new DWHFNN(k, cmet, numCategories);
            ((DWHFNN) classAlg).noRecalcs();
        } else if (classifierName.equalsIgnoreCase("hFNN")
                || classifierName.equalsIgnoreCase("h-FNN")
                || classifierName.equalsIgnoreCase("learning.supervised."
                + "methods.knn.HFNN")) {
            classAlg = new HFNN(k, cmet, numCategories);
            ((HFNN) classAlg).noRecalcs();
        } else if (classifierName.equalsIgnoreCase("B1dwh-FNN")) {
            classAlg = new AdaBoostM2(new DWHFNNBoostable(
                    k, cmet, numCategories, DWHFNNBoostable.B1));
        } else if (classifierName.equalsIgnoreCase("B2dwh-FNN")) {
            classAlg = new AdaBoostM2(new DWHFNNBoostable(
                    k, cmet, numCategories, DWHFNNBoostable.B2));
        } else if (classifierName.equalsIgnoreCase("B1h-FNN")) {
            classAlg = new AdaBoostM2(new HFNNBoostable(
                    k, cmet, numCategories, HFNNBoostable.B1));
        } else if (classifierName.equalsIgnoreCase("B2h-FNN")) {
            classAlg = new AdaBoostM2(new HFNNBoostable(
                    k, cmet, numCategories, HFNNBoostable.B2));
        } else if (classifierName.equalsIgnoreCase("B1HIKNN")) {
            classAlg = new AdaBoostM2(new HIKNNBoostable(
                    k, cmet, numCategories, HIKNNBoostable.B1));
        } else if (classifierName.equalsIgnoreCase("B2HIKNN")) {
            classAlg = new AdaBoostM2(new HIKNNBoostable(
                    k, cmet, numCategories, HIKNNBoostable.B2));
        } else if (classifierName.equalsIgnoreCase("B1HWKNN")) {
            classAlg = new AdaBoostM2(new HwKNNBoostable(
                    numCategories, cmet, k, HwKNNBoostable.B1));
        } else if (classifierName.equalsIgnoreCase("B2HWKNN")) {
            classAlg = new AdaBoostM2(new HwKNNBoostable(
                    numCategories, cmet, k, HwKNNBoostable.B2));
        } else if (classifierName.equalsIgnoreCase("NHBNN")
                || classifierName.equalsIgnoreCase("learning.supervised."
                + "methods.knn.NHBNN")) {
            classAlg = new NHBNN(k, cmet, numCategories);
            ((NHBNN) classAlg).noRecalcs();
        } else if (classifierName.equalsIgnoreCase("ANHBNN")
                || classifierName.equalsIgnoreCase("learning.supervised."
                + "methods.knn.ANHBNN")) {
            classAlg = new ANHBNN(k, cmet, numCategories);
            ((ANHBNN) classAlg).noRecalcs();
        } else if (classifierName.equalsIgnoreCase("HIKNN")
                || classifierName.equalsIgnoreCase("learning.supervised."
                + "methods.knn.HIKNN")) {
            classAlg = new HIKNN(k, cmet, numCategories);
            ((HIKNN) classAlg).noRecalcs();
        } else if (classifierName.equalsIgnoreCase("HIKNNnondw")
                || classifierName.equalsIgnoreCase("learning.supervised."
                + "methods.knn.HIKNNNonDw")) {
            classAlg = new HIKNNNonDW(k, cmet, numCategories);
            ((HIKNNNonDW) classAl