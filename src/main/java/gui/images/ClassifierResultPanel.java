
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

import draw.charts.PieRenderer;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.FlowLayout;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PiePlot;
import org.jfree.data.general.DefaultPieDataset;
import org.jfree.util.Rotation;

/**
 * This panel represents classification results as a pie charts for the tested
 * classifiers.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ClassifierResultPanel extends javax.swing.JPanel {

    /**
     * Initialization.
     */
    public ClassifierResultPanel() {
        initComponents();
        resultChartPanel.setLayout(new FlowLayout());
    }

    /**
     * Sets the new results for display.
     *
     * @param prediction Float array corresponding to the classifier prediction.
     * @param classifierName String that is the classifier name.
     * @param classColors Color array representing the class color.
     * @param classNames String array representing the class names.
     */
    public void setResults(float[] prediction, String classifierName,
            Color[] classColors, String[] classNames) {
        int numClasses = classNames.length;
        DefaultPieDataset pieData = new DefaultPieDataset();
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            pieData.setValue(classNames[cIndex], prediction[cIndex]);
        }
        JFreeChart chart = ChartFactory.createPieChart3D(classifierName
                + " prediction", pieData, true, true, false);
        PiePlot plot = (PiePlot) chart.getPlot();
        plot.setDirection(Rotation.CLOCKWISE);
        plot.setForegroundAlpha(0.5f);
        PieRenderer prend = new PieRenderer(classColors);
        prend.setColor(plot, pieData);
        ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new Dimension(140, 140));
        resultChartPanel.removeAll();
        resultChartPanel.add(chartPanel);
        resultChartPanel.revalidate();
        resultChartPanel.repaint();
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        resultChartPanel = new javax.swing.JPanel();

        javax.swing.GroupLayout resultChartPanelLayout = new javax.swing.GroupLayout(resultChartPanel);
        resultChartPanel.setLayout(resultChartPanelLayout);
        resultChartPanelLayout.setHorizontalGroup(
            resultChartPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 223, Short.MAX_VALUE)
        );
        resultChartPanelLayout.setVerticalGroup(
            resultChartPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 216, Short.MAX_VALUE)
        );

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(this);
        this.setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(resultChartPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(resultChartPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
        );
    }// </editor-fold>//GEN-END:initComponents
    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JPanel resultChartPanel;
    // End of variables declaration//GEN-END:variables
}