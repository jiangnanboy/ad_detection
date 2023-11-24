package advertising;

import ai.onnxruntime.*;
import util.CollectionUtil;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * @author sy
 * @date 2023/11/24 22:25
 */
public class AdDetection {

    public static void main(String...args) {
        AdDetection adDetection = new AdDetection();
        String line = "据中新经纬援引华尔街日报中文网报道，法国检察官正在调查法国亿万富翁、LVMH集团董事长伯纳德·阿尔诺和一名俄罗斯商人之间可能存在的洗钱交易。";
        System.out.println(adDetection.isAd(line));
    }

    /**
     * @param sent
     * @return
     */
    public boolean isAd(String sent) {
        boolean flag = false;
        try {
            Map<String, OnnxTensor> inputMap = parse(sent);
            double prob = infer(inputMap);
            if(prob > 0.5) {
                flag = true;
            }
        } catch (OrtException e) {
            e.printStackTrace();
        }
        return flag;
    }

    /**
     * @param sent
     * @return
     * @throws OrtException
     */
    public Map<String, OnnxTensor> parse(String sent) throws OrtException {
        return parse(sent, 500);
    }

    /**
     * @param sent
     * @param maxLength
     * @return
     * @throws OrtException
     */
    public Map<String, OnnxTensor> parse(String sent, int maxLength) throws OrtException {
        List<String> tokenList = CollectionUtil.newArrayList();
        for(int i=0; i<sent.length(); i++) {
            tokenList.add(sent.substring(i, i+1));
        }
//        tokenList = tokenList.stream().filter(token -> !Init.stopWords.contains(token)).collect(Collectors.toList());
        if(tokenList.size() > maxLength) {
            tokenList = tokenList.subList(0, maxLength - 1);
        } else if(tokenList.size() < maxLength) {
            int range = maxLength - tokenList.size();
            for(int i=0; i<range; i++) {
                tokenList.add("<pad>");
            }
        }
        List<Long> tokenIds = tokenList.stream().map(token -> Init.dict.getOrDefault(token, 0L)).collect(Collectors.toList());
        long[] inputIds = new long[tokenIds.size()];
        for(int i=0; i<tokenIds.size(); i++) {
            inputIds[i] = tokenIds.get(i);
        }
        long[] shape = new long[]{1, inputIds.length};
        Object ObjInputIds = OrtUtil.reshape(inputIds, shape);
        OnnxTensor inputOnnx = OnnxTensor.createTensor(Init.env, ObjInputIds);
        Map<String, OnnxTensor> inputMap = CollectionUtil.newHashMap();
        inputMap.put("input_1", inputOnnx);
        return inputMap;
    }

    /**
     * @param inputs
     * @return
     */
    public double infer(Map<String, OnnxTensor> inputs) {
        double prob = 0;
        try (OrtSession.Result result = Init.session.run(inputs)) {
            OnnxValue onnxValue = result.get(0);
            float[][] labels = (float[][])onnxValue.getValue();
            float[] resultLabels = labels[0];
            double[] softmaxResults = softmax(resultLabels);
            prob = getProb(softmaxResults);
        } catch (OrtException e) {
            e.printStackTrace();
        }
        return prob;
    }

    /**
     * @param probabilities
     * @return
     */
    public double getProb(double[] probabilities) {
        double prob = probabilities[1];
        return prob;
    }

    /**
     * @param input
     * @return
     */
    private double[] softmax(float[] input) {
        List<Float> inputList = CollectionUtil.newArrayList();
        for(int i=0; i<input.length; i++) {
            inputList.add(input[i]);
        }
        double inputSum = inputList.stream().mapToDouble(Math::exp).sum();
        double[] output = new double[input.length];
        for (int i=0; i<input.length; i++) {
            output[i] = (Math.exp(input[i]) / inputSum);
        }
        return output;
    }

}


