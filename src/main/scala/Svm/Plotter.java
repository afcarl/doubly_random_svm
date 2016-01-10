package Svm;

import org.knowm.xchart.Chart;
import org.knowm.xchart.StyleManager;
import org.knowm.xchart.SwingWrapper;

/**
 * Created by owner on 1/7/16.
 */
public class /*Harry*/ Plotter {

    public static void scatterPlot(double[][] data){
        Chart chart = new Chart(800,600);

        double[] xData = new double[data[0].length];
        double[] yData = new double[data[0].length];

        for (int i = 0; i < xData.length; i++) {
            xData[i] = data[0][i];
            yData[i] = data[1][i];
        }
        chart.getStyleManager().setChartType(StyleManager.ChartType.Scatter);

        // Customize Chart
        chart.getStyleManager().setChartTitleVisible(false);
        chart.getStyleManager().setLegendPosition(StyleManager.LegendPosition.InsideSW);
        chart.getStyleManager().setMarkerSize(16);

        // Series

        chart.addSeries("Gaussian Blob", xData, yData);

        new SwingWrapper(chart).displayChart();
    }
    /**
     * Gaussian Blob
     * <p>
     * Demonstrates the following:
     * <ul>
     * <li>ChartType.Scatter
     * <li>Series data as a Set
     * <li>Setting marker size
     * <li>Formatting of negative numbers with large magnitude but small differences
     */
//    public class ScatterChart01 implements ExampleChart {
//
//        public static void main(String[] args) {
//
//            ExampleChart exampleChart = new ScatterChart01();
//            Chart chart = exampleChart.getChart();
//            new SwingWrapper(chart).displayChart();
//        }
//
//        @Override
//        public Chart getChart() {
//
//            Set<Double> xData = new HashSet<Double>();
//            Set<Double> yData = new HashSet<Double>();
//            Random random = new Random();
//            int size = 1000;
//            for (int i = 0; i < size; i++) {
//                xData.add(random.nextGaussian() / 1000);
//                yData.add(-1000000 + random.nextGaussian());
//            }
//
//            // Create Chart
//            Chart chart = new Chart(800, 600);
//            chart.getStyleManager().setChartType(ChartType.Scatter);
//
//            // Customize Chart
//            chart.getStyleManager().setChartTitleVisible(false);
//            chart.getStyleManager().setLegendPosition(LegendPosition.InsideSW);
//            chart.getStyleManager().setMarkerSize(16);
//
//            // Series
//            chart.addSeries("Gaussian Blob", xData, yData);
//
//            return chart;
//        }
//
//    }
}
