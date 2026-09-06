using OpenCvSharp;

// ============================================================
// 第十一课：霍夫变换 —— 到"参数空间"去投票找直线和圆
// ============================================================
// 核心思想（和第九课分水岭同族: 换个空间看问题）:
//   在原图找直线很难（像素零散、有噪声、有断裂）
//   但换个视角: "一条直线" 由参数(θ, ρ)唯一确定
//     x·cosθ + y·sinθ = ρ     （原点到直线的垂线: 角θ、长度ρ）
//   → 图上每个白色边缘点，都能列出"经过我的所有直线"的参数方程
//     在(θ,ρ)参数空间里，这是一条曲线
//   → 多个点共线 = 它们的曲线在参数空间交于一点
//   → 投票: 每个边缘点给"所有可能经过自己的直线"各投一票
//     得票高的格子 = 真实存在的直线 —— 共线点的共识
//
// 一句话: 原图里"点共线"这个几何关系，翻译成参数空间里"曲线共点"
//        找直线 = 找曲线的交点 = 找票数峰值（局部极大值）
//
// 两代实现:
//   标准 HoughLines   : 返回无限长直线（数学直线），需自己延伸画线
//   概率 HoughLinesP  : 只在边缘点的子集上投票(快) + 返回线段端点(实用)
//   工程首选 P 版 —— 本课两者都演示，HoughCircles 找圆同理投票
// ============================================================

// ---------- 0. 数字实例: 一个点对应参数空间一条曲线 ----------
// 边缘点 (3, 4)，"经过我的直线"有无穷多条，每条一组(θ,ρ):
//   θ=0°:   ρ = 3·1 + 4·0 = 3      （竖直线 x=3）
//   θ=90°:  ρ = 3·0 + 4·1 = 4      （水平线 y=4）
//   θ=45°:  ρ = 3·0.707 + 4·0.707 ≈ 4.95
//   → 点(3,4)在(θ,ρ)空间里画出一条起伏的曲线（正弦形）
// 两个点 (3,4) 和 (6,8) 共线（都在 y = 4x/3 上）:
//   θ=53.13°(atan(4/3)) 时两点的 ρ 都是 0 —— 两条曲线在此相交!
//   相交格子得 2 票; 加第三个共线点 → 3 票... 票数 = 共线点数
Console.WriteLine("参数空间: 每个边缘点画出一条(θ,ρ)曲线");
Console.WriteLine("曲线相交处 = 共线共识 = 得票峰值 = 检出直线\n");

// ---------- 1. 读取 + Canny 边缘（第三课回收: 霍夫的输入是边缘图） ----------
Mat src = Cv2.ImRead(@"3.jpg", ImreadModes.Color);
if (src.Empty())
{
    Console.WriteLine("读取失败：请确认 3.jpg 在项目输出目录（bin/Debug/net8.0）中");
    return;
}
Mat gray = new Mat();
Cv2.CvtColor(src, gray, ColorConversionCodes.BGR2GRAY);
Mat edges = new Mat();
Cv2.Canny(gray, edges, 50, 150);   // 1:3 双阈值比例（第三课规矩）
Console.WriteLine($"Canny 边缘完成（霍夫的投票人 = 白色边缘点）");

// ---------- 2. 标准霍夫 HoughLines: 返回(ρ,θ)数学直线 ----------
// 参数: (边缘图, 输出数组, rho分辨率=1像素, theta分辨率=1°, threshold=票数门槛)
//   threshold=100: 参数空间某格子至少 100 票才算直线（相对阈值思想:
//   图越大/边缘点越多，门槛该越高）
LineSegmentPolar[] lines = Cv2.HoughLines(edges, 1, Math.PI / 180, 100);
Console.WriteLine($"\n标准霍夫: 检出 {lines.Length} 条直线（无限长，只有(ρ,θ)参数）");

// 画线: (ρ,θ) → 找直线上两个远端点连线（数学直线的可视化套路）
//   直线方向 = (−sinθ, cosθ)（与法向(cosθ,sinθ)垂直）
Mat stdDraw = src.Clone();
foreach (LineSegmentPolar l in lines.Take(50))   // 最多画50条防花屏
{
    double rho = l.Rho, theta = l.Theta;
    double a = Math.Cos(theta), b = Math.Sin(theta);
    double x0 = a * rho, y0 = b * rho;                    // 垂足
    Point p1 = new Point((int)(x0 + 1000 * -b), (int)(y0 + 1000 * a));
    Point p2 = new Point((int)(x0 - 1000 * -b), (int)(y0 - 1000 * a));
    Cv2.Line(stdDraw, p1, p2, new Scalar(0, 255, 0), 1);
}
Console.WriteLine("标准版问题: 同一条边的多个(ρ,θ)近似解全被检出 → 画出来是粗粗一坨");

// ---------- 3. 概率霍夫 HoughLinesP: 子集投票 + 返回线段 ----------
// "概率"= 随机抽取部分边缘点投票（够票就提前收工）→ 快很多
// 新参数:
//   minLineLength=40: 短于此的线段不要（过滤碎线）
//   maxLineGap=15:    同一直线上断口≤15像素的线段合并（桥接断裂）
// 返回 LineSegmentP[]: 每条是(x1,y1)-(x2,y2)的实打实线段
LineSegmentPoint[] segs = Cv2.HoughLinesP(edges, 1, Math.PI / 180, 80, 40, 15);
Mat pDraw = src.Clone();
foreach (LineSegmentPoint s in segs)
    Cv2.Line(pDraw, s.P1, s.P2, new Scalar(0, 0, 255), 2);
Console.WriteLine($"\n概率霍夫: {segs.Length} 条线段（红），自带长度过滤和断裂合并");

// ---------- 4. 手写"投票共识"迷你版: 验证"共线点得高票" ----------
// 不重写整个霍夫（累计器+峰值检测代码量大），只验证核心机制:
//   造 5 个精确共线的点，统计"过它们的直线"哪个(θ,ρ)得票最高
//   最高票格子的(θ,ρ)应正好是那条直线的参数
Point2f[] pts = { new(10, 20), new(20, 40), new(30, 60), new(40, 80), new(50, 100) };
// 这 5 个点在直线 y=2x 上 → 直线参数: θ=atan(1/2)≈63.43°方向... 法向角 θ 满足
//   x·cosθ + y·sinθ = ρ 恒定。y=2x → 斜率2 → 方向角63.43° → 法向角 153.43°-90°=...
// 直接扫描: θ 取 0~180° 每 1°，算 5 点的 ρ，找"5 个 ρ 几乎相等"的 θ
double bestTheta = 0, bestSpread = double.MaxValue;
for (int t = 0; t < 180; t++)
{
    double th = t * Math.PI / 180;
    double[] rhos = pts.Select(p => p.X * Math.Cos(th) + p.Y * Math.Sin(th)).ToArray();
    double spread = rhos.Max() - rhos.Min();     // 共线 ⇔ 某个θ下ρ全部相等(离散=0)
    if (spread < bestSpread) { bestSpread = spread; bestTheta = th; }
}
double bestRho = pts.Select(p => p.X * Math.Cos(bestTheta) + p.Y * Math.Sin(bestTheta)).Average();
Console.WriteLine($"\n手写投票验证: 5个共线点(y=2x上)");
Console.WriteLine($"  扫描θ找到 ρ 离散度最小的方向: θ={bestTheta * 180 / Math.PI:F1}°, ρ={bestRho:F2}");
Console.WriteLine($"  理论值: 直线 y=2x 的法向 θ=atan2(1,2)... 即 {Math.Atan2(1, 2) * 180 / Math.PI:F1}°（核对: 应一致）");
Console.WriteLine("  → 共线点在参数空间曲线相交、交点得满票 —— 霍夫的心脏");

// ---------- 5. HoughCircles: 圆的投票（3参数: 圆心x,y + 半径r） ----------
// 圆要 3 个参数 → 参数空间是三维(θ不再适用)，投票更贵
// 实现取巧: 先用梯度方向定位圆心(2D投票)，再沿半径投票定 r —— 两步降维
// 参数: (输入, 方法HOUGH_GRADIENT, dp=累加器分辨率倒数, minDist=圆心最小间距,
//        param1=Canny高阈值(内部自己跑Canny), param2=圆心票数门槛,
//        minRadius, maxRadius)
// minDist: 两个圆心靠太近只留票高的（防同一圆检出多个圆心）
// param2 越小 → 检出越多但误检越多（又一个纯度/完整度权衡，同第7课H容差）
Mat circlesImg = src.Clone();
Cv2.GaussianBlur(gray, gray, new Size(9, 9), 2);   // 预模糊: 平滑边缘,投票更稳
CircleSegment[] circles = Cv2.HoughCircles(gray, HoughModes.Gradient, 1.5,
                                           gray.Rows / 8, 100, 60, 20, 120);
foreach (CircleSegment c in circles)
{
    Cv2.Circle(circlesImg, (Point)c.Center, (int)c.Radius, new Scalar(0, 255, 0), 2);
    Cv2.Circle(circlesImg, (Point)c.Center, 3, new Scalar(0, 0, 255), -1);  // 圆心标记
}
Console.WriteLine($"\n霍夫圆: 检出 {circles.Length} 个（绿圈+红心）");
Console.WriteLine("参数提示: minRadius/maxRadius 卡住预期尺寸范围 = 最有效的降噪手段");

// ---------- 6. 参数实验: HoughLinesP 的 threshold 与 maxLineGap ----------
// threshold 80→150: 票数门槛提高 → 只要"更明显的直线" → 线变少但更可靠
// maxLineGap 15→2:  断口桥接变弱 → 虚线/断裂边缘拆成碎段
LineSegmentPoint[] segsStrict = Cv2.HoughLinesP(edges, 1, Math.PI / 180, 150, 60, 2);
Mat strictDraw = src.Clone();
foreach (LineSegmentPoint s in segsStrict)
    Cv2.Line(strictDraw, s.P1, s.P2, new Scalar(255, 0, 0), 2);
Console.WriteLine($"\n参数实验: threshold 80→150 + gap 15→2: {segs.Length} → {segsStrict.Length} 条");
Console.WriteLine("  （门槛高+不桥接 = 检出少而硬，取舍同 Canny 双阈值/面积过滤）");

// ---------- 7. 展示 ----------
Cv2.ImShow("1-Canny边缘(投票人)", edges);
Cv2.ImShow("2-标准霍夫(无限长线,粗坨)", stdDraw);
Cv2.ImShow("3-概率霍夫P(红线段)", pDraw);
Cv2.ImShow("4-霍夫圆", circlesImg);
Cv2.ImShow("5-严格参数(th150,gap2)", strictDraw);
Cv2.WaitKey(0);
Cv2.DestroyAllWindows();

// ============================================================
// 本课小结：
// 1. 霍夫 = 参数空间投票: 边缘点→(θ,ρ)曲线, 曲线交点=共线共识=峰值
// 2. 标准版返回无限长(ρ,θ)直线; 概率版P随机子集投票(快)+返回线段(实用)
// 3. minLineLength/maxLineGap: 线段级过滤（太短的踢、断口的接）
// 4. 圆=3参数投票, HoughCircles 用梯度先定圆心再定半径（两步降维）
// 5. threshold/param2 是纯度-完整度旋钮（同 Canny 双阈值、面积5%门槛）
// 6. 输入永远是 Canny 边缘图 —— 垃圾边缘进, 垃圾直线出
// 练习建议: 换一张有明显直线结构的图(建筑/表格/跑道), 对比窗口2/3/5
// ============================================================
