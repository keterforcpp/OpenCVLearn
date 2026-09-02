using OpenCvSharp;

// ============================================================
// 第六课：直方图与均衡化 —— 图像的"体检报告"
// ============================================================
// 直方图 = 灰度值的"计票表"：统计每个灰度级(0~255)各有多少个像素
//   横轴：灰度值 0(黑) → 255(白)
//   纵轴：该灰度值的像素个数
//
// 为什么它是"诊断工具"（本课最重要观念）：
//   双峰直方图（暗的一堆 + 亮的一堆）→ Otsu 在谷底切一刀，切得干净
//   单峰/宽峰（灰度挤在一起）       → 一刀切必失败 → 等第八课自适应阈值来救
//
// 均衡化 = 把挤在一起的灰度"拉开"：按累积分布函数 CDF 重映射灰度级
//   挤在 [100,160] 的暗淡图 → 拉伸到接近 [0,255]，对比度大增
// ============================================================

// ---------- 0. 数字实例：直方图就是"计票" ----------
// 6 个像素，灰度值分别是 50,52,50,200,205,50：
//   bin[50]=3  bin[52]=1  bin[200]=1  bin[205]=1，其余 252 个 bin 全是 0
// 直方图统计只"计票"不做任何邻域运算——对照第二课卷积的滑窗加权求和，
// 它连窗口都不需要，是像素级的纯统计
int[] demo = { 50, 52, 50, 200, 205, 50 };
int[] demoBins = new int[256];
foreach (int v in demo) demoBins[v]++;
Console.WriteLine($"数字实例: 6个像素中 灰度50出现{demoBins[50]}次, 52出现{demoBins[52]}次, 200出现{demoBins[200]}次");

// ---------- 1. 读取并灰度化 ----------
Mat src = Cv2.ImRead(@"3.jpg", ImreadModes.Color);
if (src.Empty())
{
    Console.WriteLine("读取失败：请确认 3.jpg 在项目输出目录（bin/Debug/net8.0）中");
    return;
}
Mat gray = new Mat();
Cv2.CvtColor(src, gray, ColorConversionCodes.BGR2GRAY);
Console.WriteLine($"\n图像尺寸 {gray.Width}x{gray.Height}, 总像素 {(long)gray.Width * gray.Height}");

// ---------- 2. 手写直方图统计 ----------
int[] bins = new int[256];            // 256 个票箱，数组下标 = 灰度值
int h = gray.Height, w = gray.Width;  // 缓存属性到局部变量（P/Invoke 性能坑）
for (int y = 0; y < h; y++)
{
    for (int x = 0; x < w; x++)
    {
        bins[gray.At<byte>(y, x)]++;  // 灰度值直接当数组下标，一次计票
    }
}
// 票箱用 int 够用：1920x1080=207万像素全落一个箱也只有 207万 << int上限21亿
// 但下面凡是要做乘法的地方先转 long——int*int 会先溢出再提升，C/C++ 老坑 C# 一样有

// ---------- 3. 内置 API：Cv2.CalcHist ----------
// 参数逐个解释：
//   channels = {0}: 统计第 0 通道（灰度图仅 1 通道；彩色图可选 B/G/R 之一）
//   histSize = {256}: 每维箱子数，一值一箱
//   ranges = [0,256): 取值范围——上界是开区间！写成 255 会丢掉所有灰度 255 的像素
Mat histMat = new Mat();  // 输出: 256x1 的 CV_32FC1（坑：必须用 At<float> 读，不是 At<byte>）
Cv2.CalcHist(new[] { gray }, new[] { 0 }, null, histMat, 1,
             new[] { 256 }, new Rangef[] { new Rangef(0, 256) });

// 验证手写与内置结果完全一致
long diffCount = 0;
for (int i = 0; i < 256; i++)
{
    diffCount += Math.Abs((long)histMat.At<float>(i, 0) - bins[i]);
}
Console.WriteLine($"手写 vs CalcHist 总差 = {diffCount}（应为 0：计数值存 float 无精度损失）");

// ---------- 4. 把直方图画成图（诊断报告可视化） ----------
Mat histImg = DrawHist(bins, "Gray Histogram");
Console.WriteLine("\n看窗口2的直方图：峰在哪里、挤不挤，直接决定 Otsu 的生死");

// ---------- 5. 直方图诊断：预测 Otsu 的成败 ----------
// 复用第 2 节的 bins 算四段占比（不用重新扫图——计票表一次算好反复用）
long total = (long)h * w;
long c1 = 0, c2 = 0, c3 = 0;
for (int i = 0; i < 64; i++) c1 += bins[i];
for (int i = 64; i < 128; i++) c2 += bins[i];
for (int i = 128; i < 192; i++) c3 += bins[i];
long c4 = total - c1 - c2 - c3;
Console.WriteLine($"灰度分布: 暗部[0,63]={c1 * 100.0 / total:F1}%  中暗[64,127]={c2 * 100.0 / total:F1}%  "
                  + $"中亮[128,191]={c3 * 100.0 / total:F1}%  亮部[192,255]={c4 * 100.0 / total:F1}%");

// 回收第四课：让 Otsu 自己报阈值（只要这个数，二值结果图不用）
double otsuTh = Cv2.Threshold(gray, new Mat(), 0, 255, ThresholdTypes.Binary | ThresholdTypes.Otsu);
Console.WriteLine($"Otsu 阈值 = {otsuTh:F0}");
// 诊断口诀：
//   双峰明显、谷底深    → 全局阈值切得干净（第五课流水线成立的前提）
//   单峰 / 宽峰 / 挤中间 → 一刀下去必丢细节 → 第八课自适应阈值正面解决

// ---------- 6. 均衡化：手写 CDF 重映射 vs Cv2.EqualizeHist ----------
// 原理（数字实例）：4 个像素 {50,50,100,150}，CDF 为累积占比
//   CDF(50)=2/4=0.5  CDF(100)=3/4=0.75  CDF(150)=4/4=1.0
//   重映射（简化式 newVal = CDF*255）：
//     50→128   100→191   150→255
//   原来挤在 [50,150] 窄区间 → 拉开到 [128,255]：间隔从 50 变 63，对比度变大
//   这就是均衡化的全部秘密：按 CDF 拉伸灰度间距，占像素多的灰度段分到更宽的区间
//
// 完整公式还要减 cdfMin（第一个非零箱的 CDF），保证最暗的有效灰度映射到 0

// 手写：先建 256 项查找表 LUT，再整表套用——"先算好答案再查表"的工程套路
long cdf = 0, cdfMin = -1;
byte[] lut = new byte[256];
for (int i = 0; i < 256; i++)
{
    cdf += bins[i];
    if (cdfMin < 0 && bins[i] > 0) cdfMin = cdf;  // 第一个非零箱的累积值
    if (cdfMin >= 0)  // 比第一个非零灰度还暗的值图中不存在，LUT 留 0 即可
        lut[i] = (byte)Math.Round((cdf - cdfMin) / (double)(total - cdfMin) * 255,
                                  MidpointRounding.AwayFromZero);  // 对齐 C 的四舍五入
}
Mat lutMat = new Mat(1, 256, MatType.CV_8UC1);
for (int i = 0; i < 256; i++) lutMat.Set(0, i, lut[i]);
Mat eqManual = new Mat();
Cv2.LUT(gray, lutMat, eqManual);  // LUT 整块查表，比手写逐像素循环快

// 内置 API（只收 8UC1 单通道，传彩色图直接抛异常——常见坑）
Mat eq = new Mat();
Cv2.EqualizeHist(gray, eq);

// 对照实验：两者应逐像素一致（OpenCV 内部就是这套 CDF 公式）
Mat diffMat = new Mat();
Cv2.Absdiff(eqManual, eq, diffMat);
Cv2.MinMaxIdx(diffMat, out _, out double maxDiff);
Console.WriteLine($"\n手写 CDF 均衡化 vs EqualizeHist 最大像素差 = {maxDiff}（应为 0）");

// 均衡化后的直方图：注意不是"完美平坦"！
// 常见误解：均衡化 = 直方图变均匀。实际只能"拉开间距"——
// 像素仍集中在原来那几个灰度附近，只是彼此隔得更开，柱子间出现空隙
int[] binsEq = CountHist(eq);
Mat histEqImg = DrawHist(binsEq, "After Equalization");

// ---------- 7. CLAHE：限制对比度自适应均衡 ----------
// 全局均衡化的问题：整张图共用一张映射表，占比大的暗部被强行拉亮
//   → 暗部噪声跟着放大、天空等亮部过曝（看窗口3的暗处颗粒感）
// CLAHE 思路：
//   1) 把图切成 8x8 块小瓷砖，每块独立均衡——各管各的亮度底色
//   2) clipLimit 给每块直方图"限高"：超高的峰先削顶，削下来的票均分给别的箱
//      → 单个灰度值不能霸占映射区间，噪声放大被限制住
Mat clahe2 = new Mat(), clahe6 = new Mat();
using (CLAHE claA = Cv2.CreateCLAHE(2.0, new Size(8, 8))) claA.Apply(gray, clahe2);
using (CLAHE claB = Cv2.CreateCLAHE(6.0, new Size(8, 8))) claB.Apply(gray, clahe6);
// 参数实验：clipLimit 越大对比度拉得越狠，也越接近全局均衡化的"用力过猛"
// 把 2.0 / 6.0 / 20.0 各跑一遍，找暗部细节与噪声之间的平衡点

// ---------- 8. 展示 ----------
Cv2.ImShow("1-原图(灰度)", gray);
Cv2.ImShow("2-直方图诊断", histImg);
Cv2.ImShow("3-全局均衡化", eq);
Cv2.ImShow("4-均衡化后直方图", histEqImg);
Cv2.ImShow("5-CLAHE clip=2(温和)", clahe2);
Cv2.ImShow("6-CLAHE clip=6(激进)", clahe6);
Cv2.WaitKey(0);
Cv2.DestroyAllWindows();

// ============================================================
// 本课小结：
// 1. 直方图 = 每个灰度值的计票表，纯统计无邻域；CalcHist 上界 256 开区间
// 2. 直方图是诊断工具：双峰→Otsu 可切；单峰/宽峰→全局阈值必失败（第八课伏笔）
// 3. 均衡化 = 按 CDF 重映射灰度：像素多的灰度段分到更宽的区间，拉对比度
// 4. 均衡化后直方图不是平坦的，只是"拉开间距"——柱间有空隙才对
// 5. 全局均衡化放大暗部噪声 → CLAHE 分块均衡 + clipLimit 限高，兼顾细节与噪声
// 练习建议：换一张逆光/雾蒙蒙的低对比度照片跑本课代码，均衡化效果会更震撼
// ============================================================

// ---------- 工具函数 ----------
// 统计一张 8UC1 图的直方图（第 2 节逻辑的复用封装）
static int[] CountHist(Mat img)
{
    int[] b = new int[256];
    int hh = img.Height, ww = img.Width;
    for (int y = 0; y < hh; y++)
        for (int x = 0; x < ww; x++)
            b[img.At<byte>(y, x)]++;
    return b;
}

// 把 256 个计票箱画成柱状图（宽 512 = 每箱 2 像素；高 300 = 底部留 30 写刻度）
static Mat DrawHist(int[] b, string title)
{
    // 坑：new Mat(高,宽,类型) 不带 Scalar 的重载不清零！必须显式给初值
    Mat canvas = new Mat(300, 512, MatType.CV_8UC3, new Scalar(40, 40, 40));
    int maxBin = b.Max();
    if (maxBin == 0) return canvas;
    for (int i = 0; i < 256; i++)
    {
        // (long) 先转再乘：b[i] 可达百万级，先乘 270 会逼近 int 上限
        int barH = (int)((long)b[i] * (300 - 40) / maxBin);
        Cv2.Rectangle(canvas, new Rect(i * 2, 300 - 30 - barH, 2, barH),
                      new Scalar(200, 200, 200), -1);  // 灰色实心柱，-1=填充
    }
    // 刻度：0(黑) — 128 — 255(白)，帮助定位峰偏暗侧还是亮侧
    // 注意：PutText 的 Hershey 字体不支持中文，图内文字只能用 ASCII
    Cv2.PutText(canvas, "0", new Point(2, 295), HersheyFonts.HersheySimplex, 0.4, new Scalar(180, 180, 180), 1);
    Cv2.PutText(canvas, "128", new Point(248, 295), HersheyFonts.HersheySimplex, 0.4, new Scalar(180, 180, 180), 1);
    Cv2.PutText(canvas, "255", new Point(488, 295), HersheyFonts.HersheySimplex, 0.4, new Scalar(180, 180, 180), 1);
    Cv2.PutText(canvas, title, new Point(5, 20), HersheyFonts.HersheySimplex, 0.5, new Scalar(0, 255, 255), 1);
    return canvas;
}
