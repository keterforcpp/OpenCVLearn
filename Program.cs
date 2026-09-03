using OpenCvSharp;

// ============================================================
// 第七课：HSV 颜色空间与颜色分割 —— 换一套坐标看颜色
// ============================================================
// BGR 的困境：三个数描述"蓝绿红配比"，人读不懂，光照一变全变
//   例：纯红(0,0,255) 光照减半 → (0,0,128)，三个数全变了
// HSV 的思路：把颜色拆成人能理解的三个独立问题
//   H 色相(Hue)   —— "是什么颜色"（红橙黄绿青蓝紫转一圈 0~360°）
//   S 饱和度(Sat) —— "颜色有多浓"（0=灰（没颜色），255=浓烈）
//   V 明度(Val)   —— "有多亮"（0=黑，255=亮）
// 核心价值：光照变化主要打击 V，H 基本不动 → 按颜色分割比按灰度分割抗光照
//
// OpenCV 两大坑（本课主角）：
//   1. 8U 图的 H 范围是 0~179（角度÷2 存进 byte），不是 0~255 也不是 0~360
//   2. 红色横跨 H=0 两侧（350°~10°），InRange 抓红色要分两段再 OR 合并
// ============================================================

// ---------- 0. 数字实例：光照杀死灰度，杀不死色相 ----------
// 纯红 BGR(0,0,255)：灰度 = 0.299×255 ≈ 76
// 暗红 BGR(0,0,128)：灰度 = 0.299×128 ≈ 38 —— 灰度掉一半，"红"在灰度轴上搬家了
// 换 HSV：两色的 H 都是 0°、S 都是 255，只有 V 从 255 掉到 128
// → 按灰度找物体：光照一变阈值就废（第五课实战的问题一）
//   按色相找物体：光照只动 V，掩膜基本不受影响 —— 本课的立足点
Console.WriteLine("光照实验：纯红(0,0,255) vs 暗红(0,0,128)");
Console.WriteLine("  灰度轴: 76 → 38（掉一半，物体和背景的相对位置全变）");
Console.WriteLine("  HSV轴 : H=0→0, S=255→255（纹丝不动），仅 V=255→128\n");

// ---------- 1. 读取 ----------
Mat src = Cv2.ImRead(@"3.jpg", ImreadModes.Color);
if (src.Empty())
{
    Console.WriteLine("读取失败：请确认 3.jpg 在项目输出目录（bin/Debug/net8.0）中");
    return;
}
Mat gray = new Mat();
Cv2.CvtColor(src, gray, ColorConversionCodes.BGR2GRAY);

// ---------- 2. 手写 BGR→HSV 转换公式（几个样例色） ----------
// 公式（先按数学定义算 0~360° 的角度，再缩放）：
//   V = max(B,G,R)                          —— 最亮的通道就是明度
//   S = (max-min)/max × 255                 —— 最大最小差越多，颜色越浓
//   H = 谁最大看谁：R大→60°×(G-B)/diff；G大→60°×((B-R)/diff+2)；B大→60°×((R-G)/diff+4)
//   （算出负数 +360°；OpenCV 存 byte，所以最后 ÷2 变 0~179）
(string name, byte B, byte G, byte R)[] samples =
{
    ("纯红", 0, 0, 255), ("暗红", 0, 0, 128), ("绿", 0, 255, 0),
    ("蓝", 255, 0, 0), ("黄", 0, 255, 255), ("灰", 128, 128, 128),
};
// 同一组颜色喂给官方 CvtColor，和手写公式对照
Mat bgrRow = new Mat(samples.Length, 1, MatType.CV_8UC3, new Scalar(0, 0, 0));
for (int i = 0; i < samples.Length; i++)
    bgrRow.Set(i, 0, new Vec3b(samples[i].B, samples[i].G, samples[i].R));
Mat hsvRow = new Mat();
Cv2.CvtColor(bgrRow, hsvRow, ColorConversionCodes.BGR2HSV);

Console.WriteLine("颜色  手写(H,S,V)   OpenCV(H,S,V)   H角度含义");
for (int i = 0; i < samples.Length; i++)
{
    var m = Bgr2Hsv(samples[i].B, samples[i].G, samples[i].R);
    Vec3b o = hsvRow.At<Vec3b>(i, 0);   // Item0=H, Item1=S, Item2=V（通道顺序）
    Console.WriteLine($"{samples[i].name,-4} ({m.h,3},{m.s,3},{m.v,3})   ({o.Item0,3},{o.Item1,3},{o.Item2,3})   H={m.h * 2}°");
}
Console.WriteLine("（注意灰色的 H=0 毫无意义：S=0 时根本没有颜色可谈 → 过滤灰色靠 S 门槛）\n");

// ---------- 3. 整图转 HSV + 拆通道看 ----------
Mat hsv = new Mat();
Cv2.CvtColor(src, hsv, ColorConversionCodes.BGR2HSV);
Mat[] ch = Cv2.Split(hsv);   // ch[0]=H, ch[1]=S, ch[2]=V
// H 通道坑：值域只有 0~179，直接显示偏暗 → ×1.4 拉到接近 0~255 才好看
// （ConvertScaleAbs 就是第三课用过的"宽算窄显"工具：src×alpha+beta 再饱和回 8U）
Mat hShow = new Mat();
Cv2.ConvertScaleAbs(ch[0], hShow, 1.4);

// ---------- 4. 自动选目标色：全图找 S 最高的像素 ----------
// 思路：S 最高 = 全图最"彩"的像素，拿它的 H 当分割目标，任何图都能演示
// （比写死"抓蓝色"通用；想抓别的颜色，把这里换成固定值即可）
int hh = hsv.Height, ww = hsv.Width;   // 缓存属性（P/Invoke 老规矩）
int bestS = -1, bx = 0, by = 0;
for (int y = 0; y < hh; y++)
{
    for (int x = 0; x < ww; x++)
    {
        Vec3b p = hsv.At<Vec3b>(y, x);
        if (p.Item2 > 40 && p.Item1 > bestS)   // V>40：太暗的像素不可信，跳过
        {
            bestS = p.Item1; bx = x; by = y;
        }
    }
}
byte targetH = hsv.At<Vec3b>(by, bx).Item0;
Console.WriteLine($"目标色: 像素({bx},{by}) H={targetH}(={targetH * 2}°) S={bestS} V={hsv.At<Vec3b>(by, bx).Item2}");
Console.WriteLine("  H 速查: 0红 30黄 60绿 90青 120蓝 150紫（±15 内算同色）\n");
if (bestS < 30)
    Console.WriteLine("警告: 全图饱和度都很低（接近黑白照片），颜色分割效果有限\n");

// ---------- 5. Cv2.InRange + 手写对照验证（第六课的验证套路） ----------
// InRange: 三个通道同时落在 [lower, upper] 内 → 输出 255（白），否则 0（黑）
// 上下界都是闭区间。Scalar 顺序 = HSV 顺序
int tol = 10;                                          // H 容差：±10（约±20°）
int hLo = Math.Max(0, targetH - tol), hHi = Math.Min(179, targetH + tol);
int sLo = 60;   // S 门槛：低于 60 = 接近灰色，H 不可信，直接排除
int vLo = 40;   // V 门槛：太暗的像素噪声大，排除
Mat mask = new Mat();
Cv2.InRange(hsv, new Scalar(hLo, sLo, vLo), new Scalar(hHi, 255, 255), mask);

// 手写版：InRange 的黑盒里就是这段逐像素三通道比较
Mat maskManual = new Mat(hh, ww, MatType.CV_8UC1, new Scalar(0));  // new Mat 不清零老坑
for (int y = 0; y < hh; y++)
{
    for (int x = 0; x < ww; x++)
    {
        Vec3b p = hsv.At<Vec3b>(y, x);
        bool inside = p.Item0 >= hLo && p.Item0 <= hHi &&
                      p.Item1 >= sLo && p.Item2 >= vLo;
        maskManual.At<byte>(y, x) = (byte)(inside ? 255 : 0);
    }
}
Mat diffMat = new Mat();
Cv2.Absdiff(mask, maskManual, diffMat);
Cv2.MinMaxIdx(diffMat, out _, out double maxDiff);
Console.WriteLine($"手写 InRange vs Cv2.InRange 最大像素差 = {maxDiff}（应为 0）");

// ---------- 6. 完整流水线：mask → 形态学 → 轮廓 → 计数（回收第四五课） ----------
// 红色跨界补刀：红色在 H=0 两侧（350°~10°），若目标色贴边，主 mask 外再补一段
if (targetH - tol < 0)        // 目标贴 0 这一侧 → 补 179 那一侧
{
    Mat extra = new Mat();
    Cv2.InRange(hsv, new Scalar(180 + targetH - tol, sLo, vLo),
                     new Scalar(179, 255, 255), extra);
    Cv2.BitwiseOr(mask, extra, mask);   // 两个掩膜求并集
}
else if (targetH + tol > 179) // 目标贴 179 这一侧 → 补 0 那一侧
{
    Mat extra = new Mat();
    Cv2.InRange(hsv, new Scalar(0, sLo, vLo),
                     new Scalar(targetH + tol - 180, 255, 255), extra);
    Cv2.BitwiseOr(mask, extra, mask);
}

Mat kernel5 = Cv2.GetStructuringElement(MorphShapes.Ellipse, new Size(5, 5));
Cv2.MorphologyEx(mask, mask, MorphTypes.Open, kernel5);   // 开运算去白噪渣（第四课）

Point[] [] contours = Cv2.FindContoursAsArray(mask, RetrievalModes.External,
                                              ContourApproximationModes.ApproxSimple);
double maxArea = 0;
for (int i = 0; i < contours.Length; i++)
{
    double a = Cv2.ContourArea(contours[i]);
    if (a > maxArea) maxArea = a;
}
Mat result = src.Clone();
int count = 0;
for (int i = 0; i < contours.Length; i++)
{
    if (Cv2.ContourArea(contours[i]) >= maxArea * 0.05)   // 相对面积门槛（第五课）
    {
        count++;
        Rect box = Cv2.BoundingRect(contours[i]);
        Cv2.DrawContours(result, contours, i, new Scalar(0, 255, 0), 2);
        Cv2.PutText(result, $"#{count}", new Point(box.X, box.Y - 5),
                    HersheyFonts.HersheySimplex, 0.7, new Scalar(0, 255, 0), 2);
    }
}
Console.WriteLine($"\nHSV 分割流水线: 找到 {contours.Length} 个轮廓，过滤后 {count} 个目标色区域");

// ---------- 7. 对照实验一：灰度 Otsu vs HSV 颜色分割 ----------
// 同一张图两条路：灰度轴一刀切 vs 色相轴按区间抓
// 看窗口5和窗口6谁把目标抠得干净 —— 灰度分不开但颜色分得开的场景，差距巨大
Mat otsuMask = new Mat();
Cv2.Threshold(gray, otsuMask, 0, 255, ThresholdTypes.Binary | ThresholdTypes.Otsu);
Console.WriteLine("对照: 窗口5(Otsu灰度掩膜) vs 窗口6(HSV颜色掩膜) —— 谁的目标更完整、背景更干净？");

// ---------- 8. 参数实验二：H 容差 ±10 vs ±30 ----------
// 容差小：抠得纯但可能漏（目标颜色稍有深浅就漏抓）
// 容差大：抓得全但可能误（把相邻颜色也抓进来）
int wLo = Math.Max(0, targetH - 30), wHi = Math.Min(179, targetH + 30);
Mat maskWide = new Mat();
Cv2.InRange(hsv, new Scalar(wLo, sLo, vLo), new Scalar(wHi, 255, 255), maskWide);
// （红色目标跨界时此处同样需要第 6 节的补刀，演示从简）
Console.WriteLine("参数实验: 窗口6(H±10) vs 窗口7(H±30) —— 纯度与完整度的权衡\n");

// ---------- 9. 展示 ----------
Cv2.ImShow("1-原图", src);
Cv2.ImShow("2-H色相通道(×1.4显示)", hShow);
Cv2.ImShow("3-S饱和度通道", ch[1]);
Cv2.ImShow("4-V明度通道", ch[2]);
Cv2.ImShow("5-对照-灰度Otsu掩膜", otsuMask);
Cv2.ImShow($"6-HSV掩膜 H={targetH}±10", mask);
Cv2.ImShow("7-HSV掩膜 H±30", maskWide);
Cv2.ImShow($"8-计数结果: {count} 个目标区域", result);
Cv2.WaitKey(0);
Cv2.DestroyAllWindows();

// ============================================================
// 本课小结：
// 1. HSV 把颜色拆成 H(是什么色) S(多浓) V(多亮)，光照变化主要打击 V
// 2. OpenCV 8U 图的 H 是 0~179（角度÷2）；红色横跨 0 两侧要两段 InRange 再 OR
// 3. InRange 三通道同落区间→白，输出 mask 直接接第五课流水线（形态学→轮廓→计数）
// 4. S 门槛过滤灰色像素（S 低时 H 无意义）、V 门槛过滤暗部噪声
// 5. 生成二值图的三条路会师：阈值切割(第四课)、边缘(第三课)、颜色分割(本课)
// 6. H 容差是纯度与完整度的权衡：±10 纯、±30 全，按场景调
// 练习建议：把第 4 节目标色换成图中第二种颜色，再看 ±10/±30 的差异
// ============================================================

// ---------- 工具函数 ----------
// 手写 BGR→HSV（8U 版）：返回 OpenCV 刻度（H:0~179, S/V:0~255）
static (int h, int s, int v) Bgr2Hsv(byte B, byte G, byte R)
{
    int max = Math.Max(B, Math.Max(G, R));
    int min = Math.Min(B, Math.Min(G, R));
    int v = max;
    int s = max == 0 ? 0 : (max - min) * 255 / max;   // 全黑像素 S 定义为 0，防除零
    double hDeg;
    if (max == min) hDeg = 0;                          // 灰色无色相，规定为 0
    else if (max == R) hDeg = 60.0 * (G - B) / (max - min);        // 红区段，可为负
    else if (max == G) hDeg = 60.0 * ((B - R) / (double)(max - min) + 2); // 绿区段
    else              hDeg = 60.0 * ((R - G) / (double)(max - min) + 4);  // 蓝区段
    if (hDeg < 0) hDeg += 360;                         // 红区段负角拉回正半圈
    return ((int)Math.Round(hDeg / 2), s, v);          // ÷2 存进 0~179
}
