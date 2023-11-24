### advertising detection
广告语检测

### step
1.利用textcnn训练一个二分类，0是正常，1是广告，语料见resources下的train.csv与test.csv

2.将textcnn模型转为onnx

3.利用onnxruntime进行推理预测

### example
(检测是否是广告，可用于广告过滤)

predict：src/main/java/advertising
```
AdDetection adDetection = new AdDetection();
String line = "据中新经纬援引华尔街日报中文网报道，法国检察官正在调查法国亿万富翁、LVMH集团董事长伯纳德·阿尔诺和一名俄罗斯商人之间可能存在的洗钱交易。";
System.out.println(adDetection.isAd(line));
```

### contact

1、github：https://github.com/jiangnanboy

2、blog：https://www.cnblogs.com/little-horse/

3、e-mail:2229029156@qq.com


