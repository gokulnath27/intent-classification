
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.classification.NaiveBayesModel;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public class NaiveBayesClassifierModelCreation {

    private static final Logger LOGGER = LoggerFactory
            .getLogger(NaiveBayesClassifierModelCreation.class);

    public static void main(String[] args) {

        SparkSession sparkSession = SparkUtils.createSparkSession();

        String path = "/Users/gokul-6650/Downloads/intent-classification/finalDataset.csv";

        Dataset<Row> df = SparkUtils.readDataSetAsCSV(path, sparkSession);

        LOGGER.info("Total row count in csv file is:" + df.count());

        Dataset<Row> filteredDataSet = df.filter("message is not null");

        Dataset<Row> tokenizedDF = SparkUtils.tokenizeWords(filteredDataSet, "message", "tokenized-words");

        Dataset<Row> finalSet = SparkUtils.createNGrams(tokenizedDF, "tokenized-words", "ngrams");

        HashingTF hashingTF = new HashingTF().setInputCol("ngrams").setOutputCol("features");
        Dataset<Row> featurizedData = hashingTF.transform(finalSet);

        /**
         * Try without IDF First then try with IDF.
         */
//        IDF idf = new IDF().setInputCol("rawFeatures").setOutputCol("features");
//        IDFModel idfModel = idf.fit(featurizedData);
//
//        Dataset<Row> rescaledData = idfModel.transform(featurizedData);


        Dataset<Row> finalDF = SparkUtils.castColumnTo(featurizedData, "label", DataTypes.DoubleType);
        Dataset<Row> naRemovedDF = finalDF.na().drop();

        LOGGER.info("Total row count after pre processing is:" + naRemovedDF.count());

        NaiveBayesModel model = new NaiveBayes().fit(naRemovedDF);

        LOGGER.info("Model is successfully saved");

        try {
            model.write().overwrite().save("/Users/gokul-6650/Downloads/intent-classification");
        } catch (IOException e) {
            LOGGER.info("Problem with storing model locally");
        }


    }
}
