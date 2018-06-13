import org.apache.spark.ml.classification.NaiveBayesModel;
import org.apache.spark.ml.classification.OneVsRestModel;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.NGram;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.sql.*;
import java.util.ArrayList;
import java.util.List;

public class predictSVC {
    public List<String> predictIntent(List<String> intentRequests, SparkSession sparkSession, OneVsRestModel  model) {
        List<String> responseList = new ArrayList<String>();
        String intent = new String();
        Dataset<String> intentDS = sparkSession.createDataset(intentRequests, Encoders.STRING());
        intentDS.show();
        Dataset<Row> predictionDataset = model.transform(PreProcessingUtil.tokenizeDF(intentDS.toDF(), sparkSession, "value", "features"));
        predictionDataset.show();
        List<Row> predictions = predictionDataset.select("prediction").collectAsList();
        for(int i=0; i < predictions.size(); i++) {
            Row row = predictions.get(i);
            switch(row.get(0).toString()) {
                case "0.0":
                    intent = "show_schedule_next_previous";
                    break;
                case "1.0":
                    intent = "show_schedule_free_slot";
                    break;
                case "2.0":
                    intent = "Move_activities";
                    break;
                case "3.0":
                    intent = "show_schedule";
                    break;
                case "4.0":
                    intent = "show_schedule_upcoming";
                    break;
                case "5.0":
                    intent = "Delete_activities";
                    break;
            }
            responseList.add(intent);
        }


        MulticlassMetrics metrics = new MulticlassMetrics(predictionDataset);
        System.out.println(metrics.confusionMatrix());
        return responseList;
    }
}
