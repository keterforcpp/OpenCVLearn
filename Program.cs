using OpenCvSharp;

// ============================================================
// 第二课：滤波与卷积 —— 图像处理的"万能地基"
// ============================================================
// 理论核心：
// 1. 卷积：用一个小矩阵（卷积核/滤波器）在图像上滑动，
//    每个位置：核的每个数 × 对应像素，全部加起来 = 该点新值
// 2. 卷积核的"形状"决定效果：
//    - 全是 1/n 的核（均值） → 模糊（邻域平均，抹平差异）
//    - 中心高四周低的钟形核（高斯）→ 更自然的模糊（近邻权重大）
//    - 中心为正、周围为负的核    → 锐化（放大与邻域的差异）
// 3. 边界问题：核滑到图像边缘会"越界"，OpenCV 自动补边处理
// ============================================================

// ---------- 1. 读取并灰度化（第一课的知识：滤波通常在灰度图上做） ----------
Mat src = Cv2.ImRead(@"3.jpg", ImreadModes.Color);
if (src.Empty())
{
    Console.WriteLine("读取失败：请确认 3.jpg 在项目输出目录（bin/Debug/net8.0）中");
    return;
}
Mat gray = new Mat();
Cv2.CvtColor(src, gray, ColorConversionCodes.BGR2GRAY);
Console.WriteLine($"图像: {src.Width}x{src.Height}，已灰度化");

int height = gray.Height;
int width = gray.Width;

// ---------- 2. 手写 3x3 均值滤波：亲手理解卷积过程 ----------
// 卷积核：       [1 1 1]
//               [1 1 1]  ÷ 9    每个像素 = 自己和周围8个邻居的平均值
//               [1 1 1]
Mat manualBlur = new Mat(height, width, MatType.CV_8UC1);

// 注意循环从 1 开始到 -1 结束：最外圈 1 像素是"边界"，邻居不全，
// 简单起见保持原值不处理（OpenCV 内部会用复制边界等方式补齐）
for (int y = 1; y < height - 1; y++)
{
    for (int x = 1; x < width - 1; x++)
    {
        int sum = 0;
        // 遍历 3x3 邻域：dy/dx 是相对中心点的偏移量
        for (int dy = -1; dy <= 1; dy++)
        {
            for (int dx = -1; dx <= 1; dx++)
            {
                sum += gray.At<byte>(y + dy, x + dx);
            }
        }
        manualBlur.Set(y, x, (byte)(sum / 9));
    }
}
// 边界一圈复制原值（简单处理，让图完整）
for (int x = 0; x < width; x++)
{
    manualBlur.Set(0, x, gray.At<byte>(0, x));
    manualBlur.Set(height - 1, x, gray.At<byte>(height - 1, x));
}
for (int y = 0; y < height; y++)
{
    manualBlur.Set(y, 0, gray.At<byte>(y, 0));
    manualBlur.Set(y, width - 1, gray.At<byte>(y, width - 1));
}
Console.WriteLine("手写 3x3 均值滤波完成");

// ---------- 3. 内置均值滤波：一行搞定同样的事 ----------
Mat blur3 = new Mat();
Cv2.Blur(gray, blur3, new Size(3, 3));   // 3x3 均值
Mat blur15 = new Mat();
Cv2.Blur(gray, blur15, new Size(15, 15)); // 15x15：核越大越模糊
Console.WriteLine("内置均值滤波完成（对比 3x3 与 15x15 核大小的差异）");

// ---------- 4. 高斯滤波：加权平均的模糊 ----------
// 与均值滤波的区别：核里的权重不是平均分配，而是二维高斯分布（钟形）
// 中心像素权重最大，越远权重越小 → 模糊效果更自然，边缘残留更少
Mat gaussian = new Mat();
Cv2.GaussianBlur(gray, gaussian, new Size(15, 15), 0);
// 第三个参数：核大小（宽高必须为奇数，保证有唯一的中心点）
// 第四个参数：标准差 sigma，0 表示由核大小自动推算（核越大 sigma 越大）
Console.WriteLine("高斯滤波完成（对比与同尺寸均值滤波的差异）");

// ---------- 5. 锐化：负权重核 ----------
// 锐化核：       [ 0 -1  0]
//               [-1  5 -1]   中心 5 倍强调自己，四周减去邻居
//               [ 0 -1  0]
// 原理：输出 = 5×自己 - 上下左右邻居 → 差异被放大，边缘更"锐"
// 核内所有数之和 = 1（5-4），保证整体亮度不变；若和为 0 输出会全黑
Mat sharpened = new Mat();
InputArray kernel = InputArray.Create<float>(new float[,]
{
    {  0, -1,  0 },
    { -1,  5, -1 },
    {  0, -1,  0 }
});
Cv2.Filter2D(gray, sharpened, -1, kernel, anchor: new Point(-1, -1));
// Filter2D：通用卷积函数，任何自定义核都用它执行
// depth 参数 -1：输出深度与输入相同（8位）
// anchor：核的锚点，(-1,-1) 表示核中心对准当前像素（默认值）

// ---------- 6. 添加噪点 + 中值滤波（去椒盐噪声专用） ----------
// 实验设计：故意撒黑白噪点，看哪种滤波能救回来
Mat noisy = gray.Clone(); // Clone：完整复制一份，两图互不影响
Random rand = new Random(42);
for (int i = 0; i < 5000; i++)
{
    int y = rand.Next(height);
    int x = rand.Next(width);
    noisy.Set(y, x, (byte)(rand.Next(2) * 255)); // 随机撒纯黑或纯白点
}
// 先用均值滤波试（会被噪声"拖累"，越滤越脏）
Mat noisyBlur = new Mat();
Cv2.Blur(noisy, noisyBlur, new Size(5, 5));
// 再用中值滤波试：取邻域所有值"排序后的中间值"，极端黑白点直接被排掉
Mat noisyMedian = new Mat();
Cv2.MedianBlur(noisy, noisyMedian, 5);
Console.WriteLine("噪点实验完成：对比均值 vs 中值的去噪效果");

// ---------- 7. 展示全部结果 ----------
Cv2.ImShow("1-灰度原图", gray);
Cv2.ImShow("2-手写3x3均值", manualBlur);
Cv2.ImShow("3-内置3x3均值", blur3);
Cv2.ImShow("4-内置15x15均值(核越大越糊)", blur15);
Cv2.ImShow("5-高斯15x15(更自然)", gaussian);
Cv2.ImShow("6-锐化", sharpened);
Cv2.ImShow("7-加了噪点的图", noisy);
Cv2.ImShow("8-均值去噪(不行)", noisyBlur);
Cv2.ImShow("9-中值去噪(干净)", noisyMedian);
Cv2.WaitKey(0);
Cv2.DestroyAllWindows();

// ============================================================
// 本课小结：
// 1. 卷积 = 小核矩阵滑过全图，加权求和得到新像素
// 2. 核的内容决定效果：全正平均→模糊；中心负权重→锐化
// 3. 核越大模糊越强；高斯权重比平均权重更自然
// 4. 中值滤波是非线性"排序"操作，对椒盐噪声特效
//    （均值/高斯是线性"加权"操作，对椒盐噪声无能为力）
// 5. Filter2D 是执行任意自定义核的通用接口
// ============================================================
