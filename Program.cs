using Emgu.CV.Structure;
using OpenCvSharp;

// ============================================================
// 第一课：图像的本质 —— 像素、通道与灰度化
// ============================================================
// 理论核心：
// 1. 数字图像在计算机中就是一个"数字矩阵"（二维数组）
// 2. 灰度图：单通道矩阵，每个像素是一个 0~255 的数（0=黑，255=白）
// 3. 彩色图：三通道矩阵（BGR），每个像素由蓝、绿、红三个数共同表示
//    注意：OpenCV 默认通道顺序是 BGR，不是 RGB，这是历史原因
// 4. 灰度化公式：Gray = 0.299*R + 0.587*G + 0.114*B
//    三个权重来自人眼感光特性——人眼对绿色最敏感，对蓝色最不敏感
// ============================================================

// ---------- 1. 读取图像：把磁盘文件解码成数字矩阵 ----------
// Mat 是 OpenCV 最核心的类，代表一个矩阵（图像就是矩阵）
// ImreadModes.Color 强制按彩色读取（即使原图是灰度图也会转成 3 通道）
Mat src = Cv2.ImRead(@"3.jpg", ImreadModes.Color);

// 健壮性检查：文件不存在或路径错误时 src 会是空的，直接使用会崩溃
if (src.Empty())
{
    Console.WriteLine("读取失败：请确认 3.jpg 在项目输出目录（bin/Debug/net8.0）中");
    return;
}

// ---------- 2. 观察图像的"数字属性" ----------
// 这一步很重要：从"看图"思维切换到"看矩阵"思维
Console.WriteLine($"图像尺寸: {src.Width} x {src.Height} 像素");
Console.WriteLine($"通道数: {src.Channels()}（彩色图=3，灰度图=1）");
Console.WriteLine($"数据类型: {src.Type()}（CV_8UC3 = 8位无符号整数 x 3通道）");
Console.WriteLine($"总像素数: {src.Width * src.Height}");
Console.WriteLine();

// ---------- 3. 直接访问像素：亲眼看到"图像就是数字" ----------
// Mat 的像素访问用索引器：mat[y, x] —— 注意顺序是 [行, 列] 即 [y, x]
// 取图像中心的一个像素
int cy = src.Height / 2;
int cx = src.Width / 2;
Vec3b centerPixel = src.At<Vec3b>(cy, cx); // Vec3b = 3个 byte 组成的向量
Console.WriteLine($"中心像素({cx},{cy})的 BGR 值: B={centerPixel.Item0} G={centerPixel.Item1} R={centerPixel.Item2}");
Console.WriteLine();

// ---------- 4. 手写灰度化：理解公式，不调用现成函数 ----------
// 创建一个单通道 8 位图像存放结果（尺寸与原图相同）
Mat manualGray = new Mat(src.Height, src.Width, MatType.CV_8UC1);

// 遍历所有像素（注意：这种逐像素方式慢，仅用于学习原理，后面会讲高效写法）
// 性能要点：Height/Width 是 P/Invoke 调用（每次都跨到 C++ 原生层取值），
// 写在循环条件里会被调用上百万次，必须先缓存成局部变量
int height = src.Height;
int width = src.Width;
for (int y = 0; y < height; y++)
{
    for (int x = 0; x < width; x++)
    {
        Vec3b pixel = src.At<Vec3b>(y, x);
        // 灰度化加权公式：人眼敏感度加权 G > R > B
        byte gray = (byte)(0.299 * pixel.Item2   // R
                         + 0.587 * pixel.Item1   // G
                         + 0.114 * pixel.Item0); // B
        gray = (byte)(gray > 127 ? 255 : 0);
        manualGray.Set(y, x, gray);
    }
}
Console.WriteLine("手写灰度化完成");

// ---------- 5. 调用 OpenCV 内置函数做同样的件事 ----------
// Cv2.CvtColor 是颜色空间转换函数，BGR2GRAY 内部也用同样的加权公式
// 但它是优化过的实现（SIMD 指令），比逐像素循环快几个数量级
Mat cvtGray = new Mat();
Cv2.CvtColor(src, cvtGray, ColorConversionCodes.BGR2GRAY);
Cv2.Threshold(cvtGray, cvtGray, 127, 255, ThresholdTypes.Binary);

// 验证两种方法结果是否一致：Cv2.Absdiff 计算两图差的绝对值
Mat diff = new Mat();
Cv2.Absdiff(manualGray, cvtGray, diff);
// CountNonZero 统计非零像素数，即"结果不同的像素个数"
// 由于浮点转整数的舍入方式差异，允许极少量像素相差 1
int differentPixels = Cv2.CountNonZero(diff);
Console.WriteLine($"手写与内置函数结果不同的像素数: {differentPixels}（应为0或极少）");
Console.WriteLine();

// ---------- 6. 像素级操作实战：手动调亮图像 ----------
// 理论：亮度 = 每个像素值加同一个常数；对比度 = 像素值乘同一个系数
// 注意 byte 溢出问题：200 + 100 = 300 > 255，必须截断到 255
Mat bright = new Mat(src.Height, src.Width, MatType.CV_8UC3);
for (int y = 0; y < height; y++)
{
    for (int x = 0; x < width; x++)
    {
        Vec3b p = src.At<Vec3b>(y, x);
        byte nb = (byte)Math.Min(p.Item0 + 60, 255); // 截断防溢出
        byte ng = (byte)Math.Min(p.Item1 + 60, 255);
        byte nr = (byte)Math.Min(p.Item2 + 60, 255);
        bright.Set(y, x, new Vec3b(nb, ng, nr));
    }
}
Console.WriteLine("加亮完成（每像素 +60，超过255截断）");

// ---------- 7. 展示结果 ----------
Cv2.ImShow("1-原图(BGR三通道)", src);
Cv2.ImShow("2-手写二值化", manualGray);
Cv2.ImShow("3-内置函数二值化", cvtGray);
Cv2.ImShow("4-加亮图(+60)", bright);
Cv2.WaitKey(0);
Cv2.DestroyAllWindows();

// ============================================================
// 本课小结：
// 1. 图像 = 矩阵，灰度图是单通道 2D 数组，彩色图是三通道数组
// 2. OpenCV 通道顺序是 BGR；像素访问 mat[y, x]
// 3. 灰度化公式 Gray = 0.299R + 0.587G + 0.114B 源自人眼特性
// 4. byte 运算注意 0~255 溢出截断
// 5. 逐像素循环慢但适合理解原理，实际项目用内置函数
// ============================================================
