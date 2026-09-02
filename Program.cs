using OpenCvSharp;

// ============================================================
// 第五课：轮廓提取与物体计数 —— 前四课的总装项目
// ============================================================
// 完整流水线（数物体项目的标准打法）：
//   灰度化 → 二值化(Otsu) → 形态学清理(开闭) → 找轮廓 → 过滤计数
//
// FindContours 原理：在二值图上"沿着白色区块的边界行走"，
// 把每个连通的白色区块的边界点按顺序串成一条闭合曲线（轮廓）。
// 因为它是沿着"实心区块"的边缘走的，所以轮廓天然闭合、天然不断线
// ——这正是第三课 Canny 苦苦追求的两点（NMS细线化+滞后连接）
//
// 关键认知：轮廓不是"边缘检测的另一种方法"，而是"区域分析的副产品"
// ——先有面（二值区块），再沿着面的边界走出线（轮廓）
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

// ---------- 2. Otsu 二值化（第四课） ----------
Mat binary = new Mat();
Cv2.Threshold(gray, binary, 127, 255, ThresholdTypes.Binary | ThresholdTypes.Otsu);
Console.WriteLine("二值化完成");

// ---------- 3. 形态学清理（第四课）：噪点会被当成"物体"数出来！ ----------
// 如果跳过这一步，FindContours 会把每个噪点都算一个"物体"（多计）
// 先闭后开：填黑麻点 + 去白噪渣
Mat kernel5 = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(5, 5));
Mat cleaned = new Mat();
Cv2.MorphologyEx(binary, cleaned, MorphTypes.Close, kernel5);
Cv2.MorphologyEx(cleaned, cleaned, MorphTypes.Open, kernel5);
Console.WriteLine("形态学清理完成（不清理的话噪点会被数成物体）");

// ---------- 4. FindContours：核心一步 ----------
// RetrievalModes.External: 只取最外层轮廓（物体内部的洞不要）
//                          （若要嵌套轮廓/洞，用 RetrievalModes.Tree）
// ContourApproximationModes.ApproxSimple: 压缩轮廓点（直线段只存端点，
//                          省内存；要逐像素完整边界用 ApproxNone）
// 返回 Point[][]: 每条轮廓是一个点数组
Point[][] contours = Cv2.FindContoursAsArray(cleaned, RetrievalModes.External,
                                             ContourApproximationModes.ApproxSimple);

int total = contours.Length;
Console.WriteLine($"找到 {total} 个轮廓（含噪点和碎块，需要过滤）");

// ---------- 5. 轮廓过滤：面积门槛（计数项目的灵魂） ----------
// 直接数轮廓数不靠谱：残余噪点、图像边角的碎块都会混进来。
// 真实物体有"合理大小"，用面积卡门槛把假的踢掉。
// 技巧：门槛取最大轮廓面积的 5%（自适应，不用手调绝对值）
double maxArea = 0;
for (int i = 0; i < total; i++)
{
    double area = Cv2.ContourArea(contours[i]);
    if (area > maxArea) maxArea = area;
}
double minValidArea = maxArea * 0.05; // 门槛 = 最大面积的5%，可调

// ---------- 6. 在原图上标注通过过滤的"物体" ----------
Mat result = src.Clone(); // 画在彩色原图上（不是二值图），观感更好
int validCount = 0;
for (int i = 0; i < total; i++)
{
    double area = Cv2.ContourArea(contours[i]);

    if (area >= minValidArea)
    {
        validCount++;
        // 画轮廓线：绿色 2 像素宽
        Cv2.DrawContours(result, contours, i, new Scalar(0, 255, 0), 2);

        // 包围盒：把物体框起来（OpenCV坐标，Scalar顺序是BGR）
        Rect box = Cv2.BoundingRect(contours[i]);
        Cv2.Rectangle(result, box, new Scalar(0, 200, 255), 2);

        // 标注编号和面积在物体上方
        Cv2.PutText(result, $"#{validCount} area={area:F0}",
                    new Point(box.X, box.Y - 5),
                    HersheyFonts.HersheySimplex, 0.5,
                    new Scalar(0, 255, 255), 1);
    }
}
Console.WriteLine($"过滤后有效物体数: {validCount}（门槛面积 {minValidArea:F0}）");

// ---------- 7. 展示流水线各阶段 ----------
Cv2.ImShow("1-原图", src);
Cv2.ImShow("2-Otsu二值化", binary);
Cv2.ImShow("3-形态学清理后", cleaned);
Cv2.ImShow($"4-计数结果: {validCount} 个物体", result);
Cv2.WaitKey(0);
Cv2.DestroyAllWindows();

// ============================================================
// 本课小结：
// 1. 流水线: 灰度→Otsu二值→开闭清理→FindContours→面积过滤→计数
// 2. FindContours 沿"实心区块边界"行走，轮廓天然闭合不断线
// 3. External 模式只要外轮廓，Tree 模式含嵌套（洞的边界也算）
// 4. 面积过滤是计数项目的灵魂：自适应门槛 = 最大轮廓×5%
// 5. DrawContours/Rectangle/PutText 是结果可视化的三板斧
// 本图(3.jpg)不是标准计数场景，效果取决于图片内容；
// 练习建议：找一张"多个物品在纯色桌面"的照片效果最佳
// ============================================================
