using OpenCvSharp;

// ============================================================
// 第十课：几何变换 —— 移动、旋转、缩放像素的"坐标搬运术"
// ============================================================
// 核心反转：几何变换不是"搬像素"，而是"对新图每个位置问：
//   我该去原图哪个坐标取色？" —— 反向映射（backward mapping）
//
//   为什么反向？正向（旧像素→新位置）会留下"洞"：
//     旋转后两个旧像素可能落到同一格（重叠），某些新格没人落（洞）
//   反向保证：新图每格都有且仅有一次取值机会 → 无洞、无重叠
//
// 三种变换由一个矩阵统一（仿射矩阵 2x3）：
//   平移: [1 0 tx]   缩放: [sx 0 0 ]   旋转: [cosθ -sinθ 0]
//         [0 1 ty]         [0  sy 0]         [sinθ  cosθ 0]
//   WarpAffine 一个函数全包 —— 换矩阵 = 换变换
//
// 插值：坐标搬运后落在"格子之间"（如 (3.7, 5.2)），取色要插值
//   最近邻: 抄最近的格 → 快但锯齿      （教学：手写感受它）
//   双线性: 四邻域加权 → 慢一点但平滑  （工程默认）
//   INTER_NEAREST 放大二值图/标签图必用（插值会造出新灰度值）
// ============================================================

// ---------- 0. 数字实例：反向映射怎么算 ----------
// 旋转 90° 时新图 (x', y') 该去原图哪取？
//   反算公式 x = y', y = w'-1-x'（旋转的逆）
// 例：新图 (0,0) ← 原图 (0, 2)（3 宽小图）
// 更一般地，仿射用矩阵逆：原图坐标 = M⁻¹ × 新图坐标
// WarpAffine 内部就是"对每个新像素，套逆矩阵，去原图取色"
Console.WriteLine("反向映射: 新图每格反问'我来自原图哪里' → 无洞无重叠");
Console.WriteLine("插值: 反算出的坐标是(3.7,5.2)这种小数 → 用周围格子估算\n");

// ---------- 1. 读取 ----------
Mat src = Cv2.ImRead(@"3.jpg", ImreadModes.Color);
if (src.Empty())
{
    Console.WriteLine("读取失败：请确认 3.jpg 在项目输出目录（bin/Debug/net8.0）中");
    return;
}
int h = src.Height, w = src.Width;
Console.WriteLine($"原图 {w}x{h}");

// ---------- 2. 缩放 Resize + 插值方法对比 ----------
// 缩到 1/4 再放回原尺寸 —— 模拟"低分辨率损失"
// 近期邻(左): 像素块感、锯齿 —— 每格抄一个邻居，无新颜色产生
// 双线性(右): 平滑 —— 四邻域按距离加权，会"造出"原图没有的中间色
Mat smallN = new Mat(), smallB = new Mat();
Cv2.Resize(src, smallN, new Size(w / 4, h / 4), 0, 0, InterpolationFlags.Nearest);
Cv2.Resize(src, smallB, new Size(w / 4, h / 4), 0, 0, InterpolationFlags.Linear);
Mat backN = new Mat(), backB = new Mat();
Cv2.Resize(smallN, backN, new Size(w, h), 0, 0, InterpolationFlags.Nearest);
Cv2.Resize(smallB, backB, new Size(w, h), 0, 0, InterpolationFlags.Linear);
Console.WriteLine("缩小再放大: 左=最近邻(块状锯齿) vs 右=双线性(平滑发糊)");

// ---------- 3. 手写最近邻缩放（3x 放大） ----------
// 亲手实现"反向映射 + 最近邻"：对放大图每格，反算原图坐标，抄最近格
// 缩放的反向映射是除法: 原图坐标 = 新图坐标 / 放大倍数
// 例: 放大3倍后新图 x=7 → 原图 7/3=2.33 → 最近邻取 2
int scale = 3;
int hw = w * scale, hh = h * scale;
Mat bigManual = new Mat(hh, hw, MatType.CV_8UC3, new Scalar(0, 0, 0));
if (!src.GetArray(out Vec3b[] srcPx))
{
    Console.WriteLine("GetArray 失败");
    return;
}
for (int y = 0; y < hh; y++)
{
    int sy = Math.Min(h - 1, (int)Math.Round(y / (double)scale, MidpointRounding.AwayFromZero));
    for (int x = 0; x < hw; x++)
    {
        int sx = Math.Min(w - 1, (int)Math.Round(x / (double)scale, MidpointRounding.AwayFromZero));
        // 最近邻 = 抄 (sy,sx) 一个格子（无插值、无新颜色）
        bigManual.Set(y, x, srcPx[sy * w + sx]);
    }
}
Mat bigApi = new Mat();
Cv2.Resize(src, bigApi, new Size(hw, hh), 0, 0, InterpolationFlags.Nearest);
Mat diffMat = new Mat();
Cv2.Absdiff(bigManual, bigApi, diffMat);
Cv2.MinMaxIdx(diffMat, out _, out double maxDiff);
Console.WriteLine($"\n手写最近邻放大 vs Resize(Nearest) 最大像素差 = {maxDiff}（应为 0）");

// ---------- 4. 平移：最简单的仿射 ----------
// 仿射矩阵 2x3: [1 0 tx; 0 1 ty] —— 不旋转不缩放，只挪 (tx,ty)
// WarpAffine(src, dst, M, 输出尺寸): M 的 C# 形态是 2x3 Mat（CV_64F）
// 坑: Mat.FromArray(1,0,100,0,1,50) 传的是一维数组 → 造出 6x1 单列矩阵
//     WarpAffine 断言 rows==2 && cols==3 直接抛异常
//     必须传二维数组 double[2,3] 才是 2x3
Mat tMat = Mat.FromArray<double>(new double[,] { { 1, 0, 100 }, { 0, 1, 50 } });   // 右移100 下移50
Mat shifted = new Mat();
Cv2.WarpAffine(src, shifted, tMat, src.Size());
Console.WriteLine("\n平移: 右移100下移50 —— 挪出去的部分丢失，留进来的部分是黑");
Console.WriteLine("（黑 = new Mat 打底色，反向映射取不到原图的地方填 Scalar 默认值）");

// ---------- 5. 旋转：GetRotationMatrix2D 生成矩阵 ----------
// 参数: (旋转中心, 角度(度,正=逆时针), 缩放系数)
// 生成的矩阵 = 平移到中心 → 旋转 → 平移回去 的复合（绕指定点转，不是绕原点）
// 中心取图像中心 → 旋转后内容大致还在画面里
double angle = 30;
Mat rotMat = Cv2.GetRotationMatrix2D(new Point2f(w / 2f, h / 2f), angle, 1.0);
Mat rotated = new Mat();
Cv2.WarpAffine(src, rotated, rotMat, src.Size());
Console.WriteLine($"\n旋转 {angle}°: 四角出现黑边（原图是矩形，转完矩形超出画面）");

// ---------- 6. 旋转矫正：算"转正后不裁边"的新画布 ----------
// 绕中心旋转仍用原尺寸画布 → 四角被裁（窗口5的黑角）
// 工程做法：按旋转后的外接矩形尺寸，把中心平移到新画布中心
// 新宽 = |w·cosθ| + |h·sinθ|，新高 = |w·sinθ| + |h·cosθ|
//   —— 原图四个角旋转后的横坐标极值差 = 新宽（包围盒思想）
double rad = angle * Math.PI / 180.0;
double cos = Math.Abs(Math.Cos(rad)), sin = Math.Abs(Math.Sin(rad));
int nw = (int)Math.Round(w * cos + h * sin);
int nh = (int)Math.Round(w * sin + h * cos);
// 在旋转矩阵上追加平移: 把旋转中心从旧图中心挪到新图中心
rotMat.At<double>(0, 2) += (nw - w) / 2.0;
rotMat.At<double>(1, 2) += (nh - h) / 2.0;
Mat rotatedFit = new Mat();
Cv2.WarpAffine(src, rotatedFit, rotMat, new Size(nw, nh));
Console.WriteLine($"旋转+扩画布: 新尺寸 {nw}x{nh}（外接矩形，四角不裁）");

// ---------- 7. 参数实验：插值方法对旋转的影响 ----------
// 旋转(非90°倍数)后反算坐标全是小数 → 插值方法直接影响画质
// Nearest: 边缘锯齿台阶   Linear: 边缘平滑（多1次运算×4邻域）
Mat rotNear = new Mat();
Mat rotMat2 = Cv2.GetRotationMatrix2D(new Point2f(w / 2f, h / 2f), angle, 1.0);
Cv2.WarpAffine(src, rotNear, rotMat2, src.Size(), InterpolationFlags.Nearest);
Console.WriteLine("\n参数实验: 窗口5(双线性) vs 窗口8(最近邻) —— 盯边缘看锯齿差异");

// ---------- 8. 二值图/标签图的坑：必须 Nearest ----------
// 用双线性缩放二值图 = 造出 0~255 之间的新灰度 → 二值图变"灰值图"
// 例: 原图 0 和 255 相邻, 双线性插出 128 —— mask 被污染
Mat bin = new Mat();
Cv2.CvtColor(src, bin, ColorConversionCodes.BGR2GRAY);
Cv2.Threshold(bin, bin, 0, 255, ThresholdTypes.Binary | ThresholdTypes.Otsu);
Mat binLinear = new Mat(), binNear = new Mat();
Cv2.Resize(bin, binLinear, new Size(w / 3, h / 3), 0, 0, InterpolationFlags.Linear);
Cv2.Resize(bin, binNear, new Size(w / 3, h / 3), 0, 0, InterpolationFlags.Nearest);
int grayCount = 0;
if (binLinear.GetArray(out byte[] bp))
{
    for (int i = 0; i < bp.Length; i++)
        if (bp[i] != 0 && bp[i] != 255) grayCount++;   // 数"中间灰度"像素
}
Console.WriteLine($"\n二值图用双线性缩小: {grayCount} 个像素被插成中间灰度（mask 已污染）");
Console.WriteLine("结论: mask/标签图缩放永远用 Nearest（第九课 markers 同理）");

// ---------- 9. 展示 ----------
Cv2.ImShow("1-原图", src);
Cv2.ImShow("2-缩放对比-最近邻(块状)", backN);
Cv2.ImShow("3-缩放对比-双线性(平滑)", backB);
Cv2.ImShow("4-手写最近邻放大3x", bigManual);
Cv2.ImShow("5-平移(100,50)", shifted);
Cv2.ImShow($"6-旋转{angle}°-原画布(裁角)", rotated);
Cv2.ImShow($"7-旋转{angle}°-扩画布{nw}x{nh}", rotatedFit);
Cv2.ImShow("8-旋转-最近邻(锯齿)", rotNear);
Cv2.ImShow("9-二值缩放-左Linear右Nearest拼图", HConcat(binLinear, binNear));
Cv2.WaitKey(0);
Cv2.DestroyAllWindows();

// ============================================================
// 本课小结：
// 1. 几何变换 = 反向映射: 新图每格反问"去原图哪取色" → 无洞无重叠
// 2. 仿射矩阵 2x3 统一平移/缩放/旋转; WarpAffine 换矩阵即换变换
// 3. 坐标落格间 → 插值: 最近邻(快/锯齿) vs 双线性(慢/平滑)
// 4. 旋转不裁边: 新画布 = 旋转外接矩形 + 中心平移修正
// 5. 二值/标签图缩放必须 Nearest —— Linear 会插出中间灰度污染 mask
// 6. 平移丢失的部分补黑 = 反向映射取不到原图时的默认填充
// 练习建议: 把角度改成 90/45/-30, 观察外接矩形尺寸与黑角的变化
// ============================================================

// ---------- 工具函数 ----------
// 水平拼接两张单通道图（拼图对比用；尺寸不同时以第一张为准裁剪）
static Mat HConcat(Mat a, Mat b)
{
    int hh = Math.Min(a.Height, b.Height);
    int w1 = Math.Min(a.Width, b.Width);
    Mat ra = a.SubMat(new Rect(0, 0, w1, hh));
    Mat rb = b.SubMat(new Rect(0, 0, Math.Min(b.Width, w1), hh));
    Mat outMat = new Mat();
    Cv2.HConcat(ra, rb, outMat);
    return outMat;
}
