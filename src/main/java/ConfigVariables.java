//import java.io.InputStream;
//import java.util.Properties;
//
//public class ConfigVariables {
//
//    static Properties prop = new Properties();
//    static InputStream input = null;
//
//    static {
//        try {
//            input = Thread.currentThread().getContextClassLoader().getResourceAsStream("config.properties");
//            prop.load(input);
//        }
//        catch (Exception e){
//            e.printStackTrace();
//        }
//    }
//
//    public static final Long LANGUAGE_TOOL_RESULT_CACHE = Long.parseLong(prop.getProperty("LANGUAGE_TOOL_RESULT_CACHE"));
//
//    public static final Long LANGUAGE_TOOL_MAX_TEXT_LENGTH = Long.parseLong(prop.getProperty("LANGUAGE_TOOL_MAX_TEXT_LENGTH"));
//
//    public static final Long LANGUAGE_TOOL_MAX_TIMEOUT_MILLIS = Long.parseLong(prop.getProperty("LANGUAGE_TOOL_MAX_TIMEOUT_MILLIS"));;
//
//    public static final String SERVER_URI = prop.getProperty("SERVER_URI");
//
//    public static final Boolean DEV_MODE = Boolean.parseBoolean(prop.getProperty("DEV_MODE"));
//
//    public static final String SPARK_WAREHOUSE_PATH = "data/";
//
//    public static final String SPARK_HOST = "local";
//
//    public static final String DFS_UTIL_PATH = prop.getProperty("ZFS_PATH");
//
//    public static final String DFS_TRAINING_DATA_FILE_PATH = prop.getProperty("ZFS_TRAINING_DATA_FILE_PATH");
//
//    public static final String MAILSENTIMENT_NAIVEBAYES_PROD_DATASET_PATH = "data/mailsentiment_training.csv";
//
//    public static final String MAILSENTIMENT_NAIVEBAYES_DATASET_PATH = "data/mailsentiment_training.csv";
//
//    public static final String SIGNATURE_REMVOAL_MLP_PROD_DATASET_PATH = "data/datasetForSignature.csv";
//
//    public static final String FEEDBACK = "data/feedback.csv";
//
//    public static final String PUBLIC_KEY = prop.getProperty("PUBLIC_KEY");
//
//    public static final String ZFS_PRIVATE_KEY = prop.getProperty("ZFS_PRIVATE_KEY");
//
//    public static final Boolean CHATBOT_TEST_MODE = Boolean.parseBoolean(prop.getProperty("CHATBOT_TEST_MODE"));
//
//    public static final String CHATBOT_BASEURL_MACHINE_KEY = prop.getProperty("CHATBOT_BASEURL_MACHINE_KEY");
//
//}