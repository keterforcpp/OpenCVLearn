using OpenCvSharp;

// ============================================================
// 第十三课：多目标定位与 NMS —— 一个模板找出所有硬币
// ============================================================
// 第十二课只找了"最像的一个位置"(MinMaxIdx 取全局峰值)
// 本课问题: 图里有约 10 枚同款硬币, 一个模板要全部找出来
//
// 核心难点: 全局峰值只能给一个答案 → 多目标要"反复取峰值"
// 新问题随之而来: 同一枚硬币周围有大量"次高分配"(模板只差1像素
//   的位置得分也很高) → 相邻一堆框指向同一枚硬币 → 需要 NMS
//
// NMS (Non-Maximum Suppression 非极大值抑制):
//   "邻域内只留得分最高的框, 其余抑制掉"
//   和 Canny 的 NMS(第三课)同名同思想: 山头上只留尖峰, 山腰全压平
//   流程: 按得分排序 → 取最高 → 删掉与它重叠的邻居 → 重复
//
// 重叠度量 IoU (Intersection over Union 交并比):
//   两框交集面积 / 并集面积, [0,1]: 0=不相干, 1=完全重合
//   IoU > 阈值(0.3~0.5) 判为"同一目标的重复框" → 抑制
//   这个概念是目标检测的通用货币(第五课NMS、YOLO全用它)
// ============================================================

// ---------- 0. 数字实例: 为什么必须有 NMS ----------
// 假设得分地图某局部: 模板最优位 (100,200) 得分 0.97
//   (101,200) 0.94 | (100,201) 0.93 | (99,200) 0.91 ...
//   —— 同一枚硬币, 一堆位置都"很匹配"(模板挪1像素还能对上大半)
// 不做 NMS: 阈值一过滤, 同枚硬币出 5 个框 → 计数翻倍
// NMS 后:   (100,200) 是局部最高 → 留; 邻居与它重叠 → 删
//   —— 每枚硬币只剩一个代表框
Console.WriteLine("多目标 = 反复取峰值 + 每次取完抹平邻域(NMS)\n");

// ---------- 1. 读取: yb.jpg 大图 + ybmb.png 模板 ----------
Mat src = Cv2.ImRead(@"yb.jpg", ImreadModes.Color);
Mat tpl = Cv2.ImRead(@"ybmb.png", ImreadModes.Color);
if (src.Empty() || tpl.Empty())
{
    Console.WriteLine("读取失败：请确认 yb.jpg / ybmb.png 在项目输出目录（bin/Debug/net8.0）中");
    return;
}
int h = src.Height, w = src.Width;
Console.WriteLine($"大图 {w}x{h}, 模板 {tpl.Width}x{tpl.Height}");
// 健壮性: 模板必须小于大图, 且不能是死平图(Normed 的 NaN 隐患, 第十二课)
if (tpl.Width >= w || tpl.Height >= h)
{
    Console.WriteLine("模板比大图大, 无法匹配");
    return;
}

// ---------- 2. 匹配: 模板比大图拍得"更大" → 多尺度问题 ----------
// 现实坑: 模板是从另一张照片裁的, 拍摄距离不同 → 同一硬币像素尺寸不同
// 直接匹配得分会很低 → 先试原始尺度看得分, 不够再缩放模板重试
double TryMatch(Mat image, Mat template, out int bx, out int by)
{
    Mat res = new Mat();
    Cv2.MatchTemplate(image, template, res, TemplateMatchModes.CCoeffNormed);
    int[] minL = new int[2], maxL = new int[2];
    Cv2.MinMaxIdx(res, out _, out double maxV, minL, maxL);
    bx = maxL[1]; by = maxL[0];        // [1]=x [0]=y (第十二课的坑)
    return maxV;
}
int bx0, by0;
double s0 = TryMatch(src, tpl, out bx0, out by0);
Console.WriteLine($"\n原始尺度: 最高得分 {s0:F3}");
if (s0 < 0.7)
{
    Console.WriteLine("得分偏低 → 目标尺寸与模板不一致, 需要缩放模板(多尺度匹配)");
    // 多尺度策略: 模板按 0.5~1.5 倍逐档试, 取全局最高分
    // (工程升级版是"图像金字塔"—— 图缩模板不缩, 更快, 思想相同)
    double bestScale = 1.0, bestScore = 0; Mat bestTpl = tpl;
    for (double k = 0.5; k <= 1.5; k += 0.1)
    {
        Mat scaled = new Mat();
        Cv2.Resize(tpl, scaled, new Size(0, 0), k, k, InterpolationFlags.Area);
        if (scaled.Width >= w || scaled.Height >= h) continue;
        int sxx, syy;
        double sc = TryMatch(src, scaled, out sxx, out syy);
        if (sc > bestScore) { bestScore = sc; bestScale = k; bestTpl = scaled; }
    }
    Console.WriteLine($"  最佳缩放 {bestScale:F1}x, 得分 {bestScore:F3}");
    tpl = bestTpl;
}

// ---------- 3. 多目标提取: 循环"取峰值 → 阈值判 → NMS 抹邻域" ----------
Mat result = new Mat();
Cv2.MatchTemplate(src, tpl, result, TemplateMatchModes.CCoeffNormed);
if (!result.GetArray(out float[] scores))
{
    Console.WriteLine("GetArray 失败: result 不是 32FC1");
    return;
}
int rh = result.Height, rw = result.Width;
double threshold = 0.65;    // 命中门槛: 得分低于此不算目标(第十二课分数带的下沿)
Console.WriteLine($"\n开始多目标提取(阈值 {threshold}):");

// 检测框列表
List<(Rect box, double score)> detections = new();
int[] dx4 = { -1, 1, 0, 0, -1, -1, 1, 1 };   // 8邻域偏移
int[] dy4 = { 0, 0, -1, 1, -1, 1, -1, 1 };
// 手写"贪心峰值提取 + NMS"一体版:
//   复制一份得分图 → 循环: 找最大 → 过阈值则收下 → 邻域抹零(局部NMS)
//   直到最大值 < 阈值 → 所有目标提取完毕
float[] work = (float[])scores.Clone();
int suppressR = Math.Max(tpl.Width, tpl.Height) / 2;   // 抑制半径=半个模板
while (true)
{
    // 找当前最大值(手写 MinMaxIdx, 练一遍纯数组版)
    int bestI = 0; float bestV = float.MinValue;
    for (int i = 0; i < work.Length; i++)
        if (work[i] > bestV) { bestV = work[i]; bestI = i; }
    if (bestV < threshold) break;               // 峰值都不达标 → 收工

    int px = bestI % rw, py = bestI / rw;       // 一维下标 → (x,y)
    detections.Add((new Rect(px, py, tpl.Width, tpl.Height), bestV));

    // 局部抑制: 以峰值为中心的方形邻域整体清零
    // (贪心版 NMS: 不算 IoU, 直接"峰值周围一臂范围内全压平")
    int x0 = Math.Max(0, px - suppressR), x1 = Math.Min(rw - 1, px + suppressR);
    int y0 = Math.Max(0, py - suppressR), y1 = Math.Min(rh - 1, py + suppressR);
    for (int yy = y0; yy <= y1; yy++)
        for (int xx = x0; xx <= x1; xx++)
            work[yy * rw + xx] = 0f;
}
Console.WriteLine($"贪心峰值+NMS: 检出 {detections.Count} 个目标");

// ---------- 4. IoU 版 NMS: 更标准的后处理(与贪心版对照) ----------
// 贪心抑制的"一臂范围"是拍脑袋半径; 标准 NMS 用 IoU 度量重叠:
//   IoU = 交集面积 / 并集面积, > iouTh 的邻居才是"重复框"
static double IoU(Rect a, Rect b)
{
    int ix = Math.Max(a.X, b.X), iy = Math.Max(a.Y, b.Y);
    int ix2 = Math.Min(a.Right, b.Right), iy2 = Math.Min(a.Bottom, b.Bottom);
    int iw = Math.Max(0, ix2 - ix), ih = Math.Max(0, iy2 - iy);
    int inter = iw * ih;                          // 交集
    int uni = a.Width * a.Height + b.Width * b.Height - inter;  // 并集=和-交集
    return uni == 0 ? 0 : (double)inter / uni;
}
// 把所有阈值之上的候选全收(不做贪心), 再按 IoU 抑制 —— 标准两段式
List<(Rect box, double score)> candidates = new();
for (int y = 0; y < rh; y++)
    for (int x = 0; x < rw; x++)
        if (scores[y * rw + x] >= threshold)
            candidates.Add((new Rect(x, y, tpl.Width, tpl.Height), scores[y * rw + x]));
// 按得分降序 → 逐个收下 → 删掉与已收框 IoU 超标的
List<(Rect box, double score)> kept = new();
foreach (var cand in candidates.OrderByDescending(c => c.score))
{
    bool overlapped = false;
    foreach (var k in kept)
        if (IoU(cand.box, k.box) > 0.3) { overlapped = true; break; }
    if (!overlapped) kept.Add(cand);
}
Console.WriteLine($"标准NMS: {candidates.Count} 个候选 → 抑制后 {kept.Count} 个(IoU阈值 0.3)");

// ---------- 5. 可视化 + 计数 ----------
Mat draw1 = src.Clone(), draw2 = src.Clone();
foreach (var (box, score) in detections)
{
    Cv2.Rectangle(draw1, box, new Scalar(0, 255, 0), 2);
    Cv2.PutText(draw1, $"{score:F2}", new Point(box.X, box.Y - 4),
                HersheyFonts.HersheySimplex, 0.4, new Scalar(0, 255, 0), 1);
}
foreach (var (box, score) in kept)
{
    Cv2.Rectangle(draw2, box, new Scalar(0, 0, 255), 2);
    Cv2.PutText(draw2, $"{score:F2}", new Point(box.X, box.Y - 4),
                HersheyFonts.HersheySimplex, 0.4, new Scalar(0, 0, 255), 1);
}
Console.WriteLine($"\n结果: 贪心版 {detections.Count} 个 vs 标准版 {kept.Count} 个(应接近)");
Console.WriteLine("注: 图中共 19 枚硬币, 同款(华盛顿头像 25 分)约 10 枚 ——");
Console.WriteLine("    检出数对不对, 以你图里实际同款数量为准; 误检/漏检原因见小结");

// 结果图可视化(显示套路同第十二课)
Mat resultShow = new Mat();
Cv2.Normalize(result, resultShow, 0, 255, NormTypes.MinMax);
Mat result8u = new Mat();
Cv2.ConvertScaleAbs(resultShow, result8u);

// ---------- 6. 参数实验: 阈值 0.5 / 0.65 / 0.8 的检出数变化 ----------
// 阈值低: 检出多但混入误检(其他硬币也长得像)
// 阈值高: 干净但漏检(磨损严重的同款硬币得分低)
Console.WriteLine("\n参数实验: 修改第 3 节 threshold 观察检出数(0.5 多而杂 / 0.8 少而准)");

// ---------- 7. 展示 ----------
Cv2.ImShow("1-大图", src);
Cv2.ImShow("2-模板", tpl);
Cv2.ImShow("3-得分地图", result8u);
Cv2.ImShow($"4-贪心峰值NMS: {detections.Count} 个", draw1);
Cv2.ImShow($"5-标准IoU-NMS: {kept.Count} 个", draw2);
Cv2.WaitKey(0);
Cv2.DestroyAllWindows();

// ============================================================
// 本课小结：
// 1. 多目标 = 反复取峰值: 收下最高 → 邻域压制 → 再取次高 ...
// 2. NMS: 邻域内只留最高分(与 Canny 的 NMS 同思想, 山头只留尖峰)
// 3. 贪心版: 峰值周围固定半径抹零(简单快); 标准版: IoU 度量重叠
// 4. IoU = 交集/并集 ∈ [0,1], 目标检测通用货币, 阈值常取 0.3~0.5
// 5. 模板与目标尺度不一致 → 多尺度匹配(0.5~1.5 逐档试取最高)
// 6. 阈值是纯度-完整度旋钮(同 Canny/H 容差/param2 家族)
// 练习建议: 阈值改 0.5/0.8 各跑一遍数检出数; 换磨损硬币当模板试漏检
// ============================================================
