
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
package ioformat.images;

import data.representation.images.sift.LFeatRepresentation;
import data.representation.images.sift.LFeatVector;
import ioformat.FileUtil;
import ioformat.IOARFF;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.io.PrintWriter;

/**
 * This class handles the import from SiftWin SIFT keyfiles and the related
 * format conversions..
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class SiftUtil {

    /**
     * Imports the features from a SiftWin keyfile into a SIFTRepresentation
     * object.
     *
     * @param keyFile File that contains the SIFT features in the SiftWin
     * format.
     * @return SIFTRepresentation that holds the loaded features.
     * @throws Exception
     */
    public static LFeatRepresentation importFeaturesFromSift(File keyFile)
            throws Exception {
        LFeatRepresentation rep = null;
        if (keyFile.exists() && keyFile.isFile()) {
            BufferedReader br = new BufferedReader(new InputStreamReader(
                    new FileInputStream(keyFile)));
            try {
                String line = br.readLine();
                String[] header = line.split(" ");
                int numFeatures = Integer.parseInt(header[0]);
                rep = new LFeatRepresentation(numFeatures, 10);
                LFeatVector siftVect;
                int index;
                // The header will contain the coordinates, the scale and the
                // angle.
                String[] siftHeader;
                // The descriptor components.
                String[] siftComponents;
                for (int featureIndex = 0; featureIndex < numFeatures;
                        featureIndex++) {
                    siftVect = new LFeatVector(rep);
                    line = br.readLine();
                    // Contains y, x, scale and angle.
                    line = line.trim();
                    siftHeader = line.split(" ");
                    siftVect.setY(Float.parseFloat(siftHeader[0]));
                    siftVect.setX(Float.parseFloat(siftHeader[1]));
                    siftVect.setScale(Float.parseFloat(siftHeader[2]));
                    siftVect.setAngle(Float.parseFloat(siftHeader[3]));
                    index = 4;
                    for (int j = 0; j < 7; j++) {
                        // There are 7 lines with SIFT vector components,
                        // first 6 containing 20, the last one 8.
                        line = br.readLine();
                        line = line.trim();
                        siftComponents = line.split(" ");
                        for (int k = 0; k < siftComponents.length; k++) {
                            siftVect.fAttr[index++] =
                                    Float.parseFloat(siftComponents[k]);
                        }
                    }
                    rep.addDataInstance(siftVect);
                }
            } catch (Exception e) {
                System.err.println(e.getMessage());
            } finally {
                br.close();
            }
            return rep;
        } else {
            return new LFeatRepresentation();
        }
    }

    /**
     * Loads the SIFTRepresentation from an ARFF file.
     *
     * @param inPath String that is the path to the data.
     * @return SIFTRepresentation object.
     * @throws Exception
     */
    public static LFeatRepresentation importFeaturesFromArff(String inPath)
            throws Exception {
        File featureFile = new File(inPath);
        LFeatRepresentation rep;
        if (featureFile.exists() && featureFile.isFile()) {
            IOARFF arff = new IOARFF();
            rep = new LFeatRepresentation(arff.load(inPath));
            return rep;
        } else {
            throw new Exception("File " + inPath + " does not exist");
        }
    }

    /**
     * This method does a batch conversion as it reads a directory of SiftWin
     * keyfiles and outputs a directory of ARFF files containing the SIFT
     * representations.
     *
     * @param inFolderPath String that is the path to the input directory.
     * @param outFolderPath String that is the path to the output directory.
     * @throws Exception
     */
    public static void siftToArffFolder(String inFolderPath,
            String outFolderPath) throws Exception {
        File inDirectory = new File(inFolderPath);
        File outDirectory = new File(outFolderPath);
        if (inDirectory.exists() && inDirectory.isDirectory()) {
            File[] children;
            FileUtil.createDirectory(outDirectory);
            children = inDirectory.listFiles();
            for (int i = 0; i < children.length; i++) {
                if (children[i].isFile()) {
                    if ((children[i].getName().substring(
                            children[i].getName().length() - 3,
                            children[i].getName().length())).
                            equalsIgnoreCase("key")) {
                        siftToArffFile(children[i].getPath(),
                                outFolderPath + File.separator
                                + children[i].getName().substring(0,
                                children[i].getName().length() - 3) + "arff");
                    }
                } else {
                    siftToArffFolder(children[i].getPath(),
                            outFolderPath + File.separator
                            + children[i].getName());
                }
            }
        } else {
            throw new Exception("Bad directory path " + inDirectory.getPath());
        }
    }

    /**
     * This method transforms one SiftWin keyfile to an ARFF file in the
     * specified location.
     *
     * @param inFilePath String that is the path to the keyfile in SiftWin
     * format.
     * @param outFilePath String that is the path to the output ARFF feature
     * file.
     * @throws Exception
     */
    public static void siftToArffFile(String inFilePath,
            String outFilePath) throws Exception {
        File inFile = new File(inFilePath);
        File outFile = new File(outFilePath);
        if (inFile.exists() && inFile.isFile()) {
            FileUtil.createFile(outFile);
            PrintWriter pw = new PrintWriter(new FileWriter(outFile));
            BufferedReader br = new BufferedReader(new InputStreamReader(
                    new FileInputStream(inFile)));
            String line;
            try {
                pw.println("%Title " + inFile.getName().substring(0,
                        inFile.getName().length() - 4)
                        + " image SIFT features");
                line = br.readLine();
                line = line.trim();
                int sepIndex = line.indexOf(' ');
                if (sepIndex == -1) {
                    throw new Exception("Wrong file format for file "
                            + inFile.getName());
                }
                String numFeatString = line.substring(0, sepIndex);
                String descriptorSizeString = line.substring(sepIndex + 1,
                        line.length());
                int numSIFT = Integer.parseInt(numFeatString);
                int descSize = Integer.parseInt(descriptorSizeString);
                pw.println("@ATTRIBUTE y NUMERIC");
                pw.println("@ATTRIBUTE x NUMERIC");
                pw.println("@ATTRIBUTE scale NUMERIC");
                pw.println("@ATTRIBUTE rotation NUMERIC");
                for (int i = 1; i <= descSize; i++) {
                    pw.println("@ATTRIBUTE desc" + i + " NUMERIC");
                }
                pw.println();
                pw.println("@DATA");
                for (int i = 0; i < numSIFT; i++) {
                    line = br.readLine();
                    line = line.trim();
                    sepIndex = line.indexOf(' ');
                    pw.print(line.substring(0, sepIndex));
                    line = line.substring(sepIndex + 1, line.length());
                    sepIndex = line.indexOf(' ');
                    pw.print("," + line.substring(0, sepIndex));
                    line = line.substring(sepIndex + 1, line.length());
                    sepIndex = line.indexOf(' ');
                    pw.print("," + line.substring(0, sepIndex));
                    pw.print("," + line.substring(sepIndex + 1, line.length()));
                    line = null;
                    for (int j = 0; j < descSize; j++) {
                        if (line == null) {
                            line = br.readLine();
                            line = line.trim();
                        }
                        sepIndex = line.indexOf(' ');
                        if (sepIndex != -1) {
                            pw.print("," + line.substring(0, sepIndex));
                            line = line.substring(sepIndex + 1, line.length());
                        } else {
                            pw.print("," + line);
                            line = null;
                        }
                    }
                    pw.println();
                }
            } catch (Exception e) {
                throw e;
            } finally {
                pw.close();
                br.close();
            }
        } else {
            throw new Exception("SIFT key file " + inFile.getPath()
                    + " does not exist");
        }
    }

    /**
     * This method takes the features from a directory of SiftWin keyfiles and
     * loads them all into a single representation that is persisted in the ARFF
     * format at the final destination.
     *
     * @param inPath String that is the input path for the SiftWin keyfile
     * directory.
     * @param outPath String that is the output path for the ARFF file.
     * @throws Exception
     */
    public static void siftFolderToOneArffFile(String inPath,
            String outPath) throws Exception {
        // The default SIFT feature descriptor size.
        int descSize = 128;
        File inDirectory = new File(inPath);
        File outFile = new File(outPath);
        if (!(inDirectory.exists() && inDirectory.isDirectory())) {