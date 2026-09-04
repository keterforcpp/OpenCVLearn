using OpenCvSharp;

// ============================================================
// 第九课：距离变换与分水岭 —— 粘连目标的分割与计数
// ============================================================
// 第五课实战问题二：物体粘连 → FindContours 把两个数成一个
// 第四课伏笔回收："腐蚀分不开的粘连，上分水岭"
//
// 核心思想（地质比喻）：
//   距离变换: 二值图上每个白点到最近黑边的距离
//     → 物体中心离边界最远 → 距离图上每个物体一座"山丘"
//     → 两个粘连物体 = 两座山共用一条坡（粘连颈是山脊上的鞍部）
//   分水岭: 从每个山丘顶部(种子)向外"漫水"
//     → 两边的水在鞍部相遇 → 相遇线就是切割线
//   一句话: 独立物体各有"最胖处"，从最胖处向外认领地盘，
//           接壤处自然就是分界 —— 这正是人眼分辨粘连物体的方式
// ============================================================

// ---------- 0. 数字实例：距离变换就是"量到边有多远" ----------
// 一行二值图（0=黑边, 1=物体）:  0 0 1 1 1 1 1 0 0
// 每个白点到最近黑边的距离:     0 0 1 2 3 2 1 0 0
//                                     ↑
//                          离两边一样远的位置 = 物体"中线"
// 单个物体 → 一座单峰山；两个粘连物体 → 双峰，鞍部在粘连颈
Console.WriteLine("距离变换: 每个白点标上'到最近黑边的距离'");
Console.WriteLine("例: [0 0 1 1 1 1 1 0 0] → [0 0 1 2 3 2 1 0 0]（峰在中线）\n");

// ---------- 1. 合成粘连图：两个重叠的圆 ----------
// 用合成图而非照片：粘连程度可控，保证演示效果稳定
// 两圆 r=70、圆心距 100 < 140 → 重叠粘连，FindContours 只见 1 个轮廓
Mat mask = new Mat(400, 500, MatType.CV_8UC1, new Scalar(0));
Cv2.Circle(mask, new Point(200, 200), 70, new Scalar(255), -1);
Cv2.Circle(mask, new Point(300, 200), 70, new Scalar(255), -1);
int h = mask.Height, w = mask.Width;
Console.WriteLine($"合成粘连图: 两圆 r=70, 圆心距 100（重叠 40 像素）");

// ---------- 2. 病理展示：老流水线计数 = 1 ----------
Point[][] contours = Cv2.FindContoursAsArray(mask, RetrievalModes.External,
                                             ContourApproximationModes.ApproxSimple);
Console.WriteLine($"\n老流水线(FindContours): 数出 {contours.Length} 个物体（真值 2）← 病");

// 腐蚀能救吗？（第四课的老工具）
// 粘连颈宽约 98 像素，腐蚀每次只把边界往里吃 ~2 像素
//   → 吃到颈断开需要 ~25 次，但那时圆(r=70)也被吃得只剩壳
//   → "腐蚀分不开，分开时物体也没了" —— 第四课埋的伏笔，本课验证
Mat eroded5 = new Mat();
Mat kernel5 = Cv2.GetStructuringElement(MorphShapes.Ellipse, new Size(5, 5));
Cv2.Erode(mask, eroded5, kernel5, null, 5);   // 迭代 5 次
Point[][] c5 = Cv2.FindContoursAsArray(eroded5, RetrievalModes.External,
                                       ContourApproximationModes.ApproxSimple);
Console.WriteLine($"腐蚀 5 次后: 仍 {c5.Length} 个（颈太肥，吃不动）");
Console.WriteLine("结论: 粘连靠腐蚀无解 → 需要分水岭");

// ---------- 3. 手写距离变换（小图暴力版）vs API 对照 ----------
// 原理暴力版: 每个白点，扫描全图找最近的黑点，算欧氏距离
//   O(N²) 只在演示小图上可行 —— 大图必须用 OpenCV 的两遍扫描法
// 小图 14x6，两个粘连的矩形块
Mat small = new Mat(6, 14, MatType.CV_8UC1, new Scalar(0));
Cv2.Rectangle(small, new Rect(2, 1, 4, 4), new Scalar(255), -1);
Cv2.Rectangle(small, new Rect(8, 1, 4, 4), new Scalar(255), -1);
// 注意: 两块横向间隔 2 像素、不粘连 → 距离图应是两座独立小山
if (!small.GetArray(out byte[] sp))
{
    Console.WriteLine("GetArray 失败");
    return;
}
int sh = small.Height, sw = small.Width;
float[,] manual = new float[sh, sw];
for (int y = 0; y < sh; y++)
{
    for (int x = 0; x < sw; x++)
    {
        if (sp[y * sw + x] == 0) { manual[y, x] = 0; continue; }
        float best = float.MaxValue;
        for (int yy = 0; yy < sh; yy++)          // 暴力: 扫全图找最近黑点
            for (int xx = 0; xx < sw; xx++)
            {
                if (sp[yy * sw + xx] != 0) continue;
                float d = MathF.Sqrt((yy - y) * (yy - y) + (xx - x) * (xx - x));
                if (d < best) best = d;
            }
        manual[y, x] = best;
    }
}
Mat distSmall = new Mat();
Cv2.DistanceTransform(small, distSmall, DistanceTypes.L2, DistanceTransformMasks.Mask3);
float maxErr = 0;
for (int y = 0; y < sh; y++)
    for (int x = 0; x < sw; x++)
    {
        float err = MathF.Abs(manual[y, x] - distSmall.At<float>(y, x));
        if (err > maxErr) maxErr = err;
    }
Console.WriteLine($"\n手写暴力距离变换 vs Cv2.DistanceTransform 最大差 = {maxErr:F3}（应≈0）");
Console.WriteLine("距离图(手写, 每行14列):");
for (int y = 0; y < sh; y++)
{
    Console.Write("  ");
    for (int x = 0; x < sw; x++) Console.Write($"{manual[y, x],4:F0}");
    Console.WriteLine();
}
Console.WriteLine("→ 两座小山各一个峰 = 两个独立物体（若粘连则鞍部相连）\n");

// ---------- 4. 大图距离变换 ----------
// 输入 8UC1 二值图, 输出 32FC1 距离图（At<float> 读！）
// DistL2 = 欧氏距离（还有 DistL1 曼哈顿/ DistC 棋盘等近似，L2 最准）
Mat dist = new Mat();
Cv2.DistanceTransform(mask, dist, DistanceTypes.L2, DistanceTransformMasks.Mask3);
Cv2.MinMaxIdx(dist, out _, out double maxDist);
Console.WriteLine($"大图距离变换: 最大距离 = {maxDist:F1}（≈圆半径，山最高的地方）");
// 显示: 32F 距离图归一化到 0~255 才能看（越亮=离边越远=越靠物体中心）
Mat distShow = new Mat();
Cv2.Normalize(dist, distShow, 0, 255, NormTypes.MinMax);
Mat dist8u = new Mat();
Cv2.ConvertScaleAbs(distShow, dist8u);

// ---------- 5. 构造种子和 markers（分水岭的"发令枪"） ----------
// 种子 = 距离图的高地（> maxDist×0.5）→ 每个物体"最胖处"的一小块
// 相对阈值（第五课老规矩）: 门槛跟最大距离走，不用手调绝对值
Mat seedsF = new Mat();
Cv2.Threshold(dist, seedsF, maxDist * 0.5, 255, ThresholdTypes.Binary);
Mat seeds = new Mat();
seedsF.ConvertTo(seeds, MatType.CV_8UC1);
Point[][] seedContours = Cv2.FindContoursAsArray(seeds, RetrievalModes.External,
                                                 ContourApproximationModes.ApproxSimple);
Console.WriteLine($"\n种子提取: 距离 > {maxDist * 0.5:F0} 的高地 → {seedContours.Length} 颗种子（每物体一颗）");

// markers: 32SC1 整数标签图（分水岭的输入输出）
//   0     = 未知区域（水还没漫到，交给算法判决）
//   1     = 背景（图像边框一圈 —— 背景也需要种子，否则会被物体吞并）
//   2,3.. = 各物体的种子（DrawContours 填充写编号）
Mat markers = new Mat(h, w, MatType.CV_32SC1, new Scalar(0));   // 不清零老坑: 显式给 0
Cv2.Rectangle(markers, new Rect(0, 0, w, h), new Scalar(1), 3); // 边框线标背景=1
for (int i = 0; i < seedContours.Length; i++)
    Cv2.DrawContours(markers, seedContours, i, new Scalar(i + 2), -1);  // 种子区标 2,3..

// ---------- 6. 分水岭：漫水、相遇、划界 ----------
Mat maskBgr = new Mat();
Cv2.CvtColor(mask, maskBgr, ColorConversionCodes.GRAY2BGR);   // Watershed 要 8UC3 输入
Cv2.Watershed(maskBgr, markers);   // markers 原地改写: 每像素=归属编号, 边界=-1

// 可视化 + 计数（GetArray 整块读 32S 标签，循环里纯内存）
if (!markers.GetArray(out int[] labels))
{
    Console.WriteLine("GetArray 失败: markers 不是 32SC1");
    return;
}
Scalar[] palette =                                  // 每个编号配一个颜色（BGR）
{
    new Scalar(80, 80, 80), new Scalar(0, 200, 255), new Scalar(0, 255, 0),
    new Scalar(255, 200, 0), new Scalar(255, 0, 200), new Scalar(200, 0, 255),
};
Mat result = new Mat(h, w, MatType.CV_8UC3, new Scalar(0, 0, 0));
HashSet<int> objects = new HashSet<int>();
for (int y = 0; y < h; y++)
{
    for (int x = 0; x < w; x++)
    {
        int lab = labels[y * w + x];
        if (lab == -1)                    // 分水岭划出的边界线
            Cv2.Circle(result, new Point(x, y), 1, new Scalar(255, 255, 255), -1);
        else if (lab >= 2)                // 物体区域: 按编号上色并记账
        {
            objects.Add(lab);
            Cv2.Circle(result, new Point(x, y), 1, palette[(lab - 2) % palette.Length], -1);
        }
        else if (lab == 1)                // 背景
            Cv2.Circle(result, new Point(x, y), 1, new Scalar(40, 40, 40), -1);
    }
}
Console.WriteLine($"\n分水岭结果: {objects.Count} 个物体（老流水线数 1，真值 2）✓");
Console.WriteLine("白色细线 = 两股水相遇的鞍部 = 自动切开的粘连颈");

// ---------- 7. 参数实验：种子阈值 0.3 vs 0.5 vs 0.7 ----------
// 阈值低(0.3): 种子大 → 两颗种子可能通过粘连颈连成一颗 → 又数成 1
// 阈值高(0.7): 种子小 → 瘦物体的峰不够高 → 整个物体没有种子 → 漏数
// 0.5 居中: 种子分离且每个物体都有一颗 —— 这个实验揭示分水岭的命门:
//   种子选不对，分水岭也无能为力（垃圾进垃圾出）
RunWatershed(mask, 0.3, "A-种子阈值0.3(种子过大,粘连)");
RunWatershed(mask, 0.7, "B-种子阈值0.7(种子过小)");

// ---------- 8. 展示 ----------
Cv2.ImShow("1-粘连二值图(真值2个)", mask);
Cv2.ImShow("2-腐蚀5次(仍连着)", eroded5);
Cv2.ImShow("3-距离图(亮=离边远=中心)", dist8u);
Cv2.ImShow("4-种子(距离高地)", seeds);
Cv2.ImShow($"5-分水岭: {objects.Count} 个物体", result);
Cv2.WaitKey(0);
Cv2.DestroyAllWindows();

// ============================================================
// 本课小结：
// 1. 距离变换: 白点到最近黑边的距离 → 每个物体一座山，山顶=最胖处
// 2. 粘连物体 = 双峰山，鞍部在粘连颈 → 腐蚀吃不断(第四课伏笔验证)
// 3. 分水岭: 从种子(山顶)漫水，相遇处划界 → 粘连颈被自动切开
// 4. markers 协议: 0=未知, 1=背景, ≥2=物体种子; 输出边界=-1
// 5. 种子质量决定成败: 阈值低种子粘连、阈值高瘦物体漏种 —— 垃圾进垃圾出
// 6. 距离图峰值思想和 TopHat"相减留差"同构: 都是"和周围比出特征"
// 练习建议: 把两圆圆心距改成 60/140, 观察种子阈值窗口何时失效
// ============================================================

// ---------- 工具函数 ----------
// 用指定种子阈值跑一遍完整分水岭（第 7 节参数实验用，流程同第 5~6 节）
static void RunWatershed(Mat mask, double ratio, string winName)
{
    int hh = mask.Height, ww = mask.Width;
    Mat d = new Mat();
    Cv2.DistanceTransform(mask, d, DistanceTypes.L2, DistanceTransformMasks.Mask3);
    Cv2.MinMaxIdx(d, out _, out double maxD);
    Mat sF = new Mat();
    Cv2.Threshold(d, sF, maxD * ratio, 255, ThresholdTypes.Binary);
    Mat s = new Mat();
    sF.ConvertTo(s, MatType.CV_8UC1);
    Point[][] sc = Cv2.FindContoursAsArray(s, RetrievalModes.External,
                                           ContourApproximationModes.ApproxSimple);
    Mat mk = new Mat(hh, ww, MatType.CV_32SC1, new Scalar(0));
    Cv2.Rectangle(mk, new Rect(0, 0, ww, hh), new Scalar(1), 3);
    for (int i = 0; i < sc.Length; i++)
        Cv2.DrawContours(mk, sc, i, new Scalar(i + 2), -1);

    Mat bgr = new Mat();
    Cv2.CvtColor(mask, bgr, ColorConversionCodes.GRAY2BGR);
    Cv2.Watershed(bgr, mk);

    // 可视化: 种子数量即计数结果
    if (!mk.GetArray(out int[] lab))
        return;
    Mat vis = new Mat(hh, ww, MatType.CV_8UC3, new Scalar(0, 0, 0));
    HashSet<int> objs = new HashSet<int>();
    for (int y = 0; y < hh; y++)
        for (int x = 0; x < ww; x++)
        {
            int v = lab[y * ww + x];
            if (v == -1)
                vis.Set(y, x, new Vec3b(255, 255, 255));
            else if (v >= 2)
            {
                objs.Add(v);
                vis.Set(y, x, new Vec3b(0, 200, 255));
            }
            else
                vis.Set(y, x, new Vec3b(40, 40, 40));
        }
    Cv2.PutText(vis, $"seeds={sc.Length} objects={objs.Count}", new Point(15, 30),
                HersheyFonts.HersheySimplex, 0.7, new Scalar(0, 255, 0), 2);
    Cv2.ImShow(winName, vis);
}
