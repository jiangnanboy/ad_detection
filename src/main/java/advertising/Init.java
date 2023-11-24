package advertising;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import util.CollectionUtil;
import util.PropertiesReader;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/**
 * @author sy
 * @date 2023/11/24 22:57
 */
public class Init {
    public static OrtSession session;
    public static OrtEnvironment env;
    public static Map<String, Long> dict;
    public static Set<String> stopWords;


    static {
        try {
            initModel(PropertiesReader.get("ad_model"));
            initDict(PropertiesReader.get("dict_path"));
            initStopWords(PropertiesReader.get("stop_words_path"));
        } catch (OrtException e) {
            e.printStackTrace();
        }
    }

//    static {
//        try {
//            String modelPath = Init.class.getResource("/model/pred.onnx").getPath();
//            String dictPath = Init.class.getResource("/model/dict.txt").getPath();
//            if(modelPath.startsWith("/")) {
//                modelPath = modelPath.substring(1);
//            }
//            if(dictPath.startsWith("/")) {
//                dictPath = dictPath.substring(1);
//            }
//            initModel(modelPath);
//            initDict(dictPath);
//        } catch (OrtException e) {
//            e.printStackTrace();
//        }
//    }

    /**
     * @param modelPath
     * @throws OrtException
     */
    public static void initModel(String modelPath) throws OrtException {
        System.out.println("init model...");
        env = OrtEnvironment.getEnvironment();
        session = env.createSession(modelPath, new OrtSession.SessionOptions());
    }

    public static void closeModel() {
        System.out.println("close model...");
        if(Optional.ofNullable(session).isPresent()) {
            try {
                session.close();
            } catch (OrtException e) {
                e.printStackTrace();
            }
        }
        if(Optional.ofNullable(env).isPresent()) {
            env.close();
        }
    }

    /**
     * @param dictPath
     */
    public static void initDict(String dictPath) {
        System.out.println("init dict...");
        try(BufferedReader br = Files.newBufferedReader(Paths.get(dictPath), StandardCharsets.UTF_8)) {
            dict = CollectionUtil.newHashMap();
            String line;
            while ((line = br.readLine()) != null) {
                String[] strs = line.split(" ");
                String word = strs[0].trim();
                long index = Long.valueOf(strs[1].trim());
                dict.put(word, index);
            }
        } catch(IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * @param stopWordsPath
     */
    public static void initStopWords(String stopWordsPath) {
       System.out.println("init stop words...");
       try(BufferedReader br = Files.newBufferedReader(Paths.get(stopWordsPath), StandardCharsets.UTF_8)) {
           stopWords = CollectionUtil.newHashset();
           String line;
           while ((line = br.readLine()) != null) {
               line = line.trim();
               stopWords.add(line);
           }
       } catch (IOException e) {
           e.printStackTrace();
       }
    }

}


