using OpenCvSharp;

// ============================================================
// 第四课：二值化与形态学操作 —— 处理"区域"的数学
// ============================================================
// 理论核心：
// 1. 二值化：灰度图 → 只有 0/255 两值的图，是"区域分析"的前提
//    Otsu 法：让计算机自动找最佳阈值（类间方差最大化）
// 2. 形态学：用小核（结构元素）扫描二值图，但运算不是加权求和，
//    而是"取最值"——本质仍是卷积框架的变体
//    - 腐蚀 Erode：邻域内取最小值 → 白色区域"缩"（细节被啃掉）
//    - 膨胀 Dilate：邻域内取最大值 → 白色区域"胀"（小洞被填上）
//    - 开运算 Open：先腐蚀再膨胀 → 去掉白色小噪点（小于核的都被抹掉）
//    - 闭运算 Close：先膨胀再腐蚀 → 填补白色区域内部小黑洞
// ============================================================

// ---------- 1. 读取并灰度化 ----------
Mat src = Cv2.ImRead(@"3.jpg", ImreadModes.Color);
if (src.Empty())
{
    Console.WriteLine("读取失败：请确认 3.jpg 在项目输出目录（bin/Debug/net8.0）中");
    return;
}
Mat gray = new Mat();
Cv2.CvtColor(src, gray, ColorConversionCodes.BGR2GRAY);
int height = gray.Height;
int width = gray.Width;

// ---------- 2. 固定阈值 vs Otsu 自动阈值 ----------
// 固定阈值 127：一半经验值，实际很难猜准（暗图/亮图差异巨大）
Mat binFixed = new Mat();
Cv2.Threshold(gray, binFixed, 127, 255, ThresholdTypes.Binary);

// Otsu：遍历所有可能阈值，找"前景/背景两类分得最开"的那个（类间方差最大）
// 加 Otsu 标志后，阈值参数(127)会被忽略，由算法计算并返回
Mat binOtsu = new Mat();
double otsuValue = Cv2.Threshold(gray, binOtsu, 127, 255, ThresholdTypes.Binary | ThresholdTypes.Otsu);
Console.WriteLine($"Otsu 自动选择的阈值: {otsuValue:F1}（对比我们瞎猜的127）");
Console.WriteLine("观察:如果图片偏暗/偏亮，固定127会切得很离谱，Otsu永远切在两类之间");

// ---------- 3. 结构元素：形态学的"卷积核" ----------
// 与普通卷积核的区别：形状有意义（矩形/十字/椭圆），权重无意义（只看覆盖范围）
Mat kernel5 = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(5, 5));
// Rect: 实心 5x5 方块 | Cross: 十字形（只连上下左右）| Ellipse: 椭圆（边缘更圆润）
// 核越大，腐蚀/膨胀的效果越猛

// ---------- 4. 手写腐蚀：理解"邻域取最小" ----------
// 腐蚀的规则：核覆盖的范围内只要有一个黑点(0)，中心就变黑
// 效果：白色区域被"啃瘦"——细的白色笔画会直接消失
Mat manualErode = new Mat(height, width, MatType.CV_8UC1, new Scalar(0));
for (int y = 2; y < height - 2; y++)       // 5x5核，边界留2像素
{
    for (int x = 2; x < width - 2; x++)
    {
        byte minVal = 255;
        for (int dy = -2; dy <= 2; dy++)
        {
            for (int dx = -2; dx <= 2; dx++)
            {
                byte v = binOtsu.At<byte>(y + dy, x + dx);
                if (v < minVal) minVal = v;     // 邻域找最小
            }
        }
        manualErode.Set(y, x, minVal);
    }
}
Console.WriteLine("手写 5x5 腐蚀完成");

// ---------- 5. 内置腐蚀/膨胀 ----------
Mat eroded = new Mat();
Cv2.Erode(binOtsu, eroded, kernel5);        // 白区缩小
Mat dilated = new Mat();
Cv2.Dilate(binOtsu, dilated, kernel5);      // 白区扩大
Console.WriteLine("腐蚀/膨胀完成");

// ---------- 6. 开运算与闭运算：形态学的实用主力 ----------
// 开 = 先腐蚀再膨胀：腐蚀阶段小噪点直接消失（小于核的活不下来），
//                      膨胀阶段把幸存的大区域恢复原大小 → 净效果=去白噪点
Mat opened = new Mat();
Cv2.MorphologyEx(binOtsu, opened, MorphTypes.Open, kernel5);

// 闭 = 先膨胀再腐蚀：膨胀阶段白色小黑洞被填平，
//                      腐蚀阶段恢复外形 → 净效果=填黑洞/愈合断裂
Mat closed = new Mat();
Cv2.MorphologyEx(binOtsu, closed, MorphTypes.Close, kernel5);
Console.WriteLine("开/闭运算完成");

// ---------- 7. 综合实验：给二值图撒噪点，用形态学清理 ----------
// 模拟真实场景：二值化后总有杂点（第三课的椒盐噪声教训）
Mat noisyBin = binOtsu.Clone();
Random rand = new Random(42);
for (int i = 0; i < 3000; i++)
{
    int y = rand.Next(height);
    int x = rand.Next(width);
    // 在黑白两色中随机取，制造"黑底白噪点 + 白区黑麻点"混合污染
    noisyBin.Set(y, x, (byte)(rand.Next(2) * 255));
}

// 一步清理：先闭(填黑麻点)再开(去白噪点)
Mat cleaned = new Mat();
Cv2.MorphologyEx(noisyBin, cleaned, MorphTypes.Close, kernel5);
Cv2.MorphologyEx(cleaned, cleaned, MorphTypes.Open, kernel5);
Console.WriteLine("噪点清理完成");

// ---------- 8. 形态学梯度：膨胀 - 腐蚀 = 区域轮廓 ----------
// 膨胀后的白区比原大，腐蚀后的比原小，两者相减：
// 中间重叠区抵消为0，只剩边缘一圈 → 直接得到"区域轮廓"
Mat gradient = new Mat();
Cv2.MorphologyEx(binOtsu, gradient, MorphTypes.Gradient, kernel5);
Console.WriteLine("形态学梯度完成（对比第三课的边缘检测）");

// ---------- 9. 展示全部结果 ----------
Cv2.ImShow("1-灰度原图", gray);
Cv2.ImShow("2-固定阈值127", binFixed);
Cv2.ImShow($"3-Otsu自动阈值({otsuValue:F0})", binOtsu);
Cv2.ImShow("4-手写5x5腐蚀", manualErode);
Cv2.ImShow("5-内置腐蚀(白区缩小)", eroded);
Cv2.ImShow("6-内置膨胀(白区扩大)", dilated);
Cv2.ImShow("7-开运算(去白噪点)", opened);
Cv2.ImShow("8-闭运算(填黑洞)", closed);
Cv2.ImShow("9-污染的二值图", noisyBin);
Cv2.ImShow("10-闭+开清理后", cleaned);
Cv2.ImShow("11-形态学梯度(轮廓)", gradient);
Cv2.WaitKey(0);
Cv2.DestroyAllWindows();

// ============================================================
// 本课小结：
// 1. Otsu 自动阈值告别手调127；ThresholdTypes 可按需反转
// 2. 腐蚀=邻域取最小(白区缩)，膨胀=邻域取最大(白区胀)
// 3. 开=先腐后胀(去白噪点)，闭=先胀后腐(填黑洞)
// 4. 形态学梯度=膨胀-腐蚀，一步提取区域轮廓
// 5. 形态学与卷积同框架：小核扫全图，只是"加权求和"换成"取最值"
// 下一课预告：轮廓提取 FindContours —— 在干净的二值图上数物体
// ============================================================
