using OpenCvSharp;

// ============================================================
// 第八课：自适应阈值与光照不均 —— 让阈值跟着亮度走
// ============================================================
// 第五课实战的病根：光照不均时同一物体半亮半暗，
// 全局阈值（含 Otsu）一刀切必然顾此失彼 —— 本课正面解决
//
// 核心思想：全局阈值假设"全图共用一个标准"，光照不均打破假设
//   → 对策：每个像素的阈值 = 它周围邻域的平均亮度 − C
//     亮的区域标准自动高、暗的区域标准自动低 —— 阈值跟着光照走
//
// 两条技术路线：
//   路线一 AdaptiveThreshold：直接算局部阈值（本课主角，手写+API+验证）
//   路线二 平场校正：大核模糊估计光照场 → 除掉光照 → 还原均匀图再 Otsu
// ============================================================

// ---------- 0. 数字实例：一刀切为什么两头都错 ----------
// 设全局阈值 = 120（Otsu 在光照不均图上算出的"折中值"）
//   亮区（灯照到）: 背景 220, 物体 150 → 都 >120 → 全白，物体融进背景
//   暗区（阴影里）: 背景  80, 物体  30 → 都 <120 → 全黑，物体又融进背景
//   → 一刀切在图的两端各错一次（第五课实战的"融成一片"）
// 自适应判决（阈值 = 邻域均值 − C, C=10）:
//   亮区邻域均值≈215 → 当地阈值 205: 背景 220>205 白, 物体 150<205 黑 → 分开
//   暗区邻域均值≈ 75 → 当地阈值  65: 背景  80> 65 白, 物体  30< 65 黑 → 分开
//   → "比当地平均亮多少"这个标准，在亮区暗区同时成立
Console.WriteLine("全局阈值 120: 亮区(220,150)全白融合, 暗区(80,30)全黑融合");
Console.WriteLine("自适应阈值(均值-10): 亮区判 205, 暗区判 65, 两边都分开\n");

// ---------- 1. 读取并灰度化 ----------
Mat src = Cv2.ImRead(@"3.jpg", ImreadModes.Color);
if (src.Empty())
{
    Console.WriteLine("读取失败：请确认 3.jpg 在项目输出目录（bin/Debug/net8.0）中");
    return;
}
Mat gray = new Mat();
Cv2.CvtColor(src, gray, ColorConversionCodes.BGR2GRAY);
int h = gray.Height, w = gray.Width;   // 缓存属性（P/Invoke 老规矩）

// ---------- 2. 人造光照不均：左亮右暗渐变 ----------
// 光照的物理模型是"乘性"的：光照减弱 k 倍 → 拍到的亮度同乘 k
// 用逐列系数 1.0 → 0.35 乘整图，模拟"左边有灯、右边背光"
// 性能技巧：GetArray 把整个 Mat 一次性拷进 C# 托管数组，循环里纯内存操作
//   —— 比循环内逐次 At<byte>（每次都跨 P/Invoke 边界）快百倍
if (!gray.GetArray(out byte[] px))
{
    Console.WriteLine("GetArray 失败：gray 不是 8UC1");
    return;
}
byte[] pxUneven = new byte[px.Length];
for (int y = 0; y < h; y++)
{
    int row = y * w;
    for (int x = 0; x < w; x++)
    {
        double k = 1.0 - 0.65 * x / (w - 1);   // 列位置 → 光照系数
        pxUneven[row + x] = (byte)Math.Round(px[row + x] * k);
    }
}
Mat uneven = new Mat(h, w, MatType.CV_8UC1, new Scalar(0));
uneven.SetArray(pxUneven);                     // 托管数组一次性写回 Mat
Console.WriteLine("已构造光照不均图：左端亮度 100%，右端 35%");

// ---------- 3. 病理展示：全局 Otsu 的失败现场 ----------
Mat otsuFail = new Mat();
double otsuTh = Cv2.Threshold(uneven, otsuFail, 0, 255,
                              ThresholdTypes.Binary | ThresholdTypes.Otsu);
Console.WriteLine($"\n全局 Otsu 阈值 = {otsuTh:F0}（折中值：照顾亮区就丢暗区，反之亦然）");
Console.WriteLine("看窗口2: 一刀切的结果 —— 亮半边糊成一片白, 暗半边糊成一片黑");

// ---------- 4. 手写自适应阈值（积分图 + 邻域均值 − C） ----------
// 算法：对每个像素取 blockSize×blockSize 邻域的均值 mean，
//       判决 gray > mean − C ? 255 : 0
// 难点在性能：若每个像素都重扫一遍窗口（51×51=2601 格），
//   100 万像素 × 2601 = 26 亿次加法 —— 不可行
// 积分图（summed-area table）把它降到 O(1)/像素：
//   sum(y,x) = (0,0)~(y,x) 矩形的像素总和，构建一遍 O(N)
//   之后任意矩形和 = 4 个角各查一次加减：
//     矩形(y1..y2, x1..x2)和 = S(y2+1,x2+1) − S(y1,x2+1) − S(y2+1,x1) + S(y1,x1)
//   数字实例: 图 [1 2]   积分图 [0 0  0 ]
//                  [3 4]         [0 1  3 ]
//                                [0 4 10]
//     全图和 = 10−0−0+0 = 10 = 1+2+3+4 ✓；右下单格 = 10−3−4+1 = 4 ✓
const int BlockSize = 51;   // 邻域边长（必须奇数，中心才唯一）
const double CC = 10;       // 从均值中减去的余量（判决余量，滤掉缓变浮动）
Mat adaptManual = AdaptiveMeanManual(uneven, BlockSize, CC);
Console.WriteLine("\n手写自适应阈值完成（积分图加速，全图线性扫描）");

// ---------- 5. Cv2.AdaptiveThreshold + 内部区域对照验证 ----------
// 参数逐个：
//   maxValue=255: 命中时输出的值
//   MeanC    : 局部阈值 = 邻域算术均值 − C
//   GaussianC: 邻域高斯加权均值 − C（近处像素权重大，阈值更平滑）
//   Binary   : 高于阈值→白（此 API 只支持 Binary/BinaryInv 两种）
//   blockSize: 邻域边长，必须奇数
//   C=10     : 判决标准是"比当地平均亮 10 以上"才算目标
Mat adaptMean = new Mat();
Cv2.AdaptiveThreshold(uneven, adaptMean, 255, AdaptiveThresholdTypes.MeanC,
                      ThresholdTypes.Binary, BlockSize, CC);
Mat adaptGauss = new Mat();
Cv2.AdaptiveThreshold(uneven, adaptGauss, 255, AdaptiveThresholdTypes.GaussianC,
                      ThresholdTypes.Binary, BlockSize, CC);

// 对照验证（第六七课套路）：手写 vs API 应逐像素一致
// 细节：图像边界一圈（宽 blockSize/2）两者语义不同 ——
//   OpenCV 用"复制边缘像素"补窗口（分母固定 blockSize²）
//   手写版只统计界内像素（分母随窗口缩小）
//   → 只验证"窗口不出界"的内部区域；边界差异属预期，不影响主体
int r = BlockSize / 2;
Rect inner = new Rect(r, r, w - 2 * r, h - 2 * r);
Mat diffMat = new Mat();
Cv2.Absdiff(adaptMean.SubMat(inner), adaptManual.SubMat(inner), diffMat);
Cv2.MinMaxIdx(diffMat, out _, out double maxDiff);
Console.WriteLine($"手写 vs AdaptiveThreshold(MeanC) 内部区域最大像素差 = {maxDiff}（应为 0）");

// ---------- 6. 参数实验：blockSize 11 vs 51 vs 101 ----------
// blockSize 决定"局部"有多大：
//   11  : 窗口小 → 阈值紧贴局部细节 → 小噪声也能抬高当地均值 → mask 偏碎
//   51  : 适中 → 阈值跟光照走、不被小细节带偏
//   101 : 窗口大 → 邻域均值越来越像全图均值 → 趋近"全局均值−C"的固定阈值
//         （ blockSize ≥ 图尺寸时完全退化为全局，Otsu 的老毛病回归 ）
Mat adapt11 = new Mat(), adapt101 = new Mat();
Cv2.AdaptiveThreshold(uneven, adapt11, 255, AdaptiveThresholdTypes.MeanC,
                      ThresholdTypes.Binary, 11, CC);
Cv2.AdaptiveThreshold(uneven, adapt101, 255, AdaptiveThresholdTypes.MeanC,
                      ThresholdTypes.Binary, 101, CC);
// C 的直觉：C 越大 → 判决越严（要"比当地平均亮更多"才算目标）→ 白区精瘦
//          C 太小 → 判决贴着当地平均 → 缓变区域自己跟自己比 → 大片误白
Console.WriteLine("\n参数实验: 窗口4(51) vs 窗口6(11,碎) vs 窗口7(101,钝)");

// ---------- 7. 路线二：平场校正（估计光照 → 除掉 → 再 Otsu） ----------
// 物理模型: 拍到的图 = 物体反射率 × 光照强度
//   物体是小形状（高频），光照是缓渐变（低频）
//   大核高斯模糊 → 小物体被抹掉 → 剩下的就是光照场的估计
//   原图 ÷ 光照 → 还原"反射率图"（等效把光照拉均匀）→ 全局 Otsu 复活
Mat lightEst = new Mat();
Cv2.GaussianBlur(uneven, lightEst, new Size(101, 101), 0);
Mat f32 = new Mat(), light32 = new Mat(), ratio = new Mat();
uneven.ConvertTo(f32, MatType.CV_32F);
lightEst.ConvertTo(light32, MatType.CV_32F);
light32 = light32 + new Scalar(1);       // 防零：光照估计为 0 的暗区除法会爆
Cv2.Divide(f32, light32, ratio, 255.0);  // scale=255: 商×255 映回 0~255 量程
Mat corrected = new Mat();
Cv2.ConvertScaleAbs(ratio, corrected);   // 32F → 8U（宽算窄显，老规矩）
Mat flatOtsu = new Mat();
double flatTh = Cv2.Threshold(corrected, flatOtsu, 0, 255,
                              ThresholdTypes.Binary | ThresholdTypes.Otsu);
Console.WriteLine($"\n平场校正后 Otsu 阈值 = {flatTh:F0}（光照已除掉，一刀又能切好了）");
// 适用边界：物体必须明显小于模糊核（101）—— 物体若比核还大，
// 模糊抹不掉它 → 光照估计被物体污染 → 校正失败（和第九课分水岭的"分不开就上大杀器"同款预警）

// ---------- 8. 展示 ----------
Cv2.ImShow("1-光照不均图(左亮右暗)", uneven);
Cv2.ImShow("2-全局Otsu-失败现场", otsuFail);
Cv2.ImShow("3-手写自适应(积分图)", adaptManual);
Cv2.ImShow($"4-API MeanC 窗口{BlockSize}", adaptMean);
Cv2.ImShow($"5-API GaussianC 窗口{BlockSize}", adaptGauss);
Cv2.ImShow("6-blockSize=11(碎)", adapt11);
Cv2.ImShow("7-blockSize=101(钝,趋全局)", adapt101);
Cv2.ImShow("8-平场校正-反射率图", corrected);
Cv2.ImShow("9-平场+Otsu", flatOtsu);
Cv2.WaitKey(0);
Cv2.DestroyAllWindows();

// ============================================================
// 本课小结：
// 1. 光照不均打破全局阈值的"全图一个标准"假设 → 一刀切两端各错一次
// 2. 自适应阈值: 每像素阈值 = 邻域均值 − C，标准跟着当地亮度走
// 3. 积分图: 一遍线性构建，任意矩形和 4 角加减 O(1) 查询 —— 性能大杀器
// 4. blockSize 控制跟多"局部": 太小贴噪声, 太大退化为全局; C 是判决余量
// 5. 平场校正: 大核模糊估光照 → 相除还原反射率 → Otsu 复活（物体须小于核）
// 6. GetArray/SetArray 整块搬运像素，绕开逐次 At 的 P/Invoke 开销
// 练习建议: 把第 2 节渐变改成上下方向、或系数 0.65 改 0.9，观察各法鲁棒性
// ============================================================

// ---------- 工具函数 ----------
// 手写自适应阈值（MeanC 语义）：阈值 = 邻域均值 − C，gray > 阈值 → 白
// 边界处理：窗口越界只统计界内像素（与 OpenCV 的复制边缘补窗略不同，见第 5 节）
static Mat AdaptiveMeanManual(Mat img, int blockSize, double C)
{
    int hh = img.Height, ww = img.Width;
    if (!img.GetArray(out byte[] p))
        throw new ArgumentException("AdaptiveMeanManual: img 必须是 8UC1");

    // 积分图: (hh+1)×(ww+1)，第 0 行/列全 0 —— 多垫一圈零，查询时免边界 if
    long[] s = new long[(hh + 1) * (ww + 1)];
    for (int y = 0; y < hh; y++)
    {
        long rowSum = 0;
        int rowIdx = (y + 1) * (ww + 1), prevIdx = y * (ww + 1);
        for (int x = 0; x < ww; x++)
        {
            rowSum += p[y * ww + x];                          // 当行累积
            s[rowIdx + x + 1] = s[prevIdx + x + 1] + rowSum;  // 上方累积 + 当行
        }
    }

    int r = blockSize / 2;
    byte[] outPx = new byte[p.Length];
    for (int y = 0; y < hh; y++)
    {
        int y1 = Math.Max(0, y - r), y2 = Math.Min(hh - 1, y + r);
        for (int x = 0; x < ww; x++)
        {
            int x1 = Math.Max(0, x - r), x2 = Math.Min(ww - 1, x + r);
            int cnt = (y2 - y1 + 1) * (x2 - x1 + 1);   // 界内实际像素数
            // 4 角加减取窗口和（分母用界内像素数 → 边界是"缩小窗口"语义）
            long rectSum = s[(y2 + 1) * (ww + 1) + x2 + 1]
                         - s[y1 * (ww + 1) + x2 + 1]
                         - s[(y2 + 1) * (ww + 1) + x1]
                         + s[y1 * (ww + 1) + x1];
            // 对齐 OpenCV：其内部 boxFilter 输出 8U，均值先舍入成整数再判
            int meanI = (int)Math.Round((double)rectSum / cnt, MidpointRounding.AwayFromZero);
            if (p[y * ww + x] > meanI - C) outPx[y * ww + x] = 255;
        }
    }
    Mat dst = new Mat(hh, ww, MatType.CV_8UC1, new Scalar(0));
    dst.SetArray(outPx);
    return dst;
}
