using OpenCvSharp;

// ============================================================
// 第三课：边缘检测 —— 找出图像中"亮度突变"的位置
// ============================================================
// 理论核心：
// 1. 边缘 = 亮度发生剧烈变化的地方（物体轮廓、纹理分界）
// 2. 数学工具：梯度（导数）。一维信号里变化最快的地方导数最大；
//    二维图像里用梯度模长衡量"变化强度"，方向指明"变化朝向"
// 3. 数字图像是离散的，导数用"差分"近似：
//    水平梯度 ≈ 右边像素 - 左边像素（Sobel 算子）
// 4. Canny 是经典流水线：高斯去噪 → 求梯度 → 非极大值抑制 → 双阈值筛选
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

// ---------- 2. 手写水平差分：最原始的"边缘检测" ----------
// 一维视角：f(x+1) - f(x) 变化越大，说明这里越可能是竖直边缘
// 对应卷积核 [ -1  1 ]（水平方向差分，检测竖直边缘）
int height = gray.Height;
int width = gray.Width;
Mat manualDx = new Mat(height, width, MatType.CV_8UC1);
for (int y = 0; y < height; y++)
{
    for (int x = 0; x < width - 1; x++)
    {
        int diff = gray.At<byte>(y, x + 1) - gray.At<byte>(y, x); // 右 - 左
        // diff 可能为负（从亮变暗），取绝对值表示"变化强度"，不看方向
        manualDx.Set(y, x, (byte)Math.Min(Math.Abs(diff), 255));
    }
}
Console.WriteLine("手写水平差分完成（只对竖直边缘敏感）");

// ---------- 3. Sobel 算子：3x3 的梯度近似 ----------
// Sobel 水平核 Gx：         垂直核 Gy：
//   [-1 0 1]                  [-1 -2 -1]
//   [-2 0 2]                  [ 0  0  0]
//   [-1 0 1]                  [ 1  2  1]
// 本质还是差分（右边减左边），但上下加了权重
// → 中间行权重 2，抗噪更好；同时兼顾上下邻居
Mat sobelX = new Mat(); // 水平梯度：对竖直边缘响应强
Mat sobelY = new Mat(); // 垂直梯度：对水平边缘响应强
Cv2.Sobel(gray, sobelX, MatType.CV_16S, 1, 0, 3); // dx=1, dy=0：求水平梯度
Cv2.Sobel(gray, sobelY, MatType.CV_16S, 0, 1, 3); // dx=0, dy=1：求垂直梯度
// 注意 depth 用 CV_16S：梯度有正有负（方向信息），8位装不下负数！

// 梯度模长 = sqrt(Gx² + Gy²)，衡量"变化强度"（OpenCV 用近似公式 |Gx|+|Gy| 提速）
Mat magnitude = new Mat();
Cv2.AddWeighted(sobelX, 0.5, sobelY, 0.5, 0, magnitude);   // 简化：加权求和
// 把 16 位结果转回 8 位便于显示（负值已在上一步被组合抵消大半，这里做线性缩放）
Mat magnitude8 = new Mat();
Cv2.ConvertScaleAbs(magnitude, magnitude8); // |x| 后线性压到 0~255
Console.WriteLine("Sobel 梯度计算完成");

// ---------- 4. Laplacian 算子：二阶导数找边缘 ----------
// 拉普拉斯核：          [ 0 -1  0]
//                      [-1  4 -1]
//                      [ 0 -1  0]
// 一阶导数（Sobel）在斜坡上是平台，二阶导数在边缘处是"尖峰"→ 定位更准
// 缺点：对噪声极其敏感（导数放大噪声），通常先高斯模糊再用
Mat blurred = new Mat();
Cv2.GaussianBlur(gray, blurred, new Size(5, 5), 0); // 先去噪（LoG 思想的雏形）
Mat laplacian = new Mat();
Cv2.Laplacian(blurred, laplacian, MatType.CV_16S, 3);
Mat laplacian8 = new Mat();
Cv2.ConvertScaleAbs(laplacian, laplacian8);
Console.WriteLine("Laplacian 完成（对比 Sobel：细边缘多但噪点也多）");

// ---------- 5. Canny：工程上最好用的边缘检测器 ----------
// Canny 内部流水线（理解它 = 理解所有前人的积累）：
//   ① 高斯滤波去噪（第二课）
//   ② Sobel 求梯度（本课第3节）
//   ③ 非极大值抑制：梯度图上的"粗边缘"削成 1 像素细线
//      （沿梯度方向只保留变化最强的那个像素，其余抹掉）
//   ④ 双阈值筛选：高于高阈值=强边缘(保留)；低于低阈值=噪声(丢弃)；
//      两者之间=弱边缘，只有连着强边缘才保留（滞后阈值，抗断线）
Mat canny = new Mat();
Cv2.Canny(gray, canny, 100, 200); // 低阈值100，高阈值200
Console.WriteLine("Canny 完成（推荐 2:1 ~ 3:1 的阈值比）");

// ---------- 6. 阈值参数实验：感受双阈值的作用 ----------
Mat cannyLow = new Mat();
Cv2.Canny(gray, cannyLow, 30, 60);   // 阈值低 → 边缘多而杂（噪声也被当边缘）
Mat cannyHigh = new Mat();
Cv2.Canny(gray, cannyHigh, 180, 360); // 阈值高 → 只剩最强烈的边缘
Console.WriteLine("阈值实验完成");

// ---------- 7. 展示全部结果 ----------
Cv2.ImShow("1-灰度原图", gray);
Cv2.ImShow("2-手写水平差分(只测竖直边)", manualDx);
Cv2.ImShow("3-Sobel梯度模长", magnitude8);
Cv2.ImShow("4-Laplacian(先模糊)", laplacian8);
Cv2.ImShow("5-Canny(100,200)经典", canny);
Cv2.ImShow("6-Canny低阈值(30,60)", cannyLow);
Cv2.ImShow("7-Canny高阈值(180,360)", cannyHigh);
Cv2.WaitKey(0);
Cv2.DestroyAllWindows();

// ============================================================
// 本课小结：
// 1. 边缘 = 亮度突变；检测边缘 = 求导数（梯度）
// 2. 一阶导数：Sobel（带抗噪加权的差分），输出有正负 → 用 16S
// 3. 二阶导数：Laplacian，定位准但对噪声敏感，先模糊再用
// 4. Canny = 去噪+梯度+非极大值抑制+双阈值的完整流水线，
//    工程首选；两个阈值按 2:1~3:1 配
// 5. 手写差分核 [-1 1] 是一切边缘检测的种子原型
// ============================================================
