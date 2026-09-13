using OpenCvSharp;

// ============================================================
// 第十五课：相机标定 —— 教相机认识毫米（路线 D：标定与测量 开篇）
// ============================================================
// 理论核心：针孔相机模型 + 径向畸变 —— 上一课生成图片的"上帝管线"就是它:
//   世界点(X,Y,Z) --R,t 外参--> 相机系 --÷Z 透视除法--> 归一化平面
//   --×(1+k1r²+k2r⁴) 径向畸变--> --×内参K--> 像素(u,v)
//
// 内参 K = | fx 0 cx |    "镜头的身份证": fx,fy=焦距(像素单位), cx,cy=光心
// 畸变 D = [k1 k2 p1 p2 k3] "镜头的病历": k1<0 桶形(直线外鼓) k1>0 枕形(内弯)
//
// 上一课的 Proj 函数是这条管线的"手写正向版"; 标定=解它的逆:
//   输入一堆"世界点↔像素"对应关系, 反解 K/D —— LM 迭代优化, 太复杂交给 API
//
// 本课是"开卷考试": 15 张图由已知参数的虚拟相机生成
//   真值(ground_truth.txt): fx=800 fy=800 cx=320 cy=240 k1=-0.15 k2=0.05
//   标定只看图盲猜 → 与真值对比 → 亲眼验证标定到底准不准
// ============================================================

// ---------- 0. 数字实例: 内参与畸变到底在说什么 ----------
// ① fx=800 的直觉: 光心(320,240)正前方 Z=420mm、右偏 X=50mm 的点
//    u = fx·X/Z + cx = 800×50/420 + 320 ≈ 415.2 → 落在光心右侧 95px
//    同点移远到 Z=840: u ≈ 347.6 → 只偏 28px
//    → "近大远小"的定量版: 像素偏移 ∝ 1/Z, 焦距就是这条曲线的斜率
// ② k1=-0.15 的直觉: 距光心 r=300px 的点, 归一化 r²=(300/800)²≈0.141
//    缩放 = 1+(-0.15)(0.141)+0.05(0.141²) ≈ 0.980 → 拉回 6px
//    而 r=100px 只拉回 0.2px → 边缘弯得多中心弯得少 → 直线外鼓(桶形)
Console.WriteLine("fx=近大远小的斜率(800×50/420≈95px), k1=边缘拉回量(300px处拉6px)\n");

// ---------- 1. 检测棋盘角点: 标定的原料 ----------
// 棋盘=自带答案的考题: 格子边长 20mm(模拟世界无打印误差),
// 角点(i,j) 的真实坐标自动已知 = (i·20, j·20, 0) → "世界点↔像素"对应关系
string dir = Path.Combine(AppContext.BaseDirectory, "routeD_img");
Size pattern = new(9, 6);   // 9x6 内角点(10x7 个格子)
double sq = 20.0;           // 每格边长 mm

List<Mat> imgs = new();
List<Point2f[]> cornersList = new();
double shiftSum = 0;        // 亚像素精化的平均修正量(15 张平均)
for (int i = 1; i <= 15; i++)
{
    Mat img = Cv2.ImRead(Path.Combine(dir, $"calib{i:00}.png"), ImreadModes.Color);
    if (img.Empty()) { Console.WriteLine($"读取失败: routeD_img/calib{i:00}.png"); return; }
    Mat gray = new();
    Cv2.CvtColor(img, gray, ColorConversionCodes.BGR2GRAY);
    if (!Cv2.FindChessboardCorners(gray, pattern, out Point2f[] cs))
    { Console.WriteLine($"calib{i:00}: 角点检测失败, 这张不能用"); return; }

    Point2f[] rough = (Point2f[])cs.Clone();     // 精化前(约整数级精度)
    // 亚像素精化: 在角点邻域按灰度梯度加权, 把坐标磨到 0.001px 级
    cs = Cv2.CornerSubPix(gray, cs, new Size(11, 11), new Size(-1, -1),
        new TermCriteria(CriteriaTypes.MaxIter | CriteriaTypes.Eps, 30, 0.001));
    for (int k = 0; k < cs.Length; k++)
        shiftSum += Math.Sqrt(Math.Pow(cs[k].X - rough[k].X, 2) + Math.Pow(cs[k].Y - rough[k].Y, 2)) / cs.Length / 15;

    imgs.Add(img);
    cornersList.Add(cs);
}
Console.WriteLine($"15 张图 × 54 角点 = {15 * 54} 组「世界点↔像素」对应关系(原料备齐)");
Console.WriteLine($"CornerSubPix 亚像素精化: 平均每角点修正 {shiftSum:F3} px");

// ---------- 2. 世界坐标 + CalibrateCamera: 盲猜内参 ----------
// 棋盘平面取 Z=0(平面棋盘是张正友标定法的数学前提 → 打印一张纸就能标定的根源)
List<Point3f> board = new();
for (int j = 0; j < 6; j++)
    for (int i = 0; i < 9; i++)
        board.Add(new Point3f((float)(i * sq), (float)(j * sq), 0));

Mat boardM = Mat<Point3f>.FromArray(board.ToArray());   // 世界点(15 张图共享: 同一块棋盘)
List<Mat> objMat = new(), imgMat = new();
foreach (var cs in cornersList) { objMat.Add(boardM); imgMat.Add(Mat<Point2f>.FromArray(cs)); }

Mat K = new(3, 3, MatType.CV_64F, Scalar.All(0));  // 待解内参(传 0 → OpenCV 自动给初值)
Mat D = new(5, 1, MatType.CV_64F, Scalar.All(0));  // 待解畸变 k1 k2 p1 p2 k3
// FixK3: 固定 k3=0 —— k3 与 k1/k2 高度相关, 不固定会"互相补偿"一起漂移
// (工程铁律: 数据覆盖的 r 范围内多项式系数不可辨识; 实测不加此标志 k1 标到 -0.21, 加了才回 -0.15)
double rms = Cv2.CalibrateCamera(objMat, imgMat, imgs[0].Size(), K, D,
    out Mat[] rvecs, out Mat[] tvecs, CalibrationFlags.FixK3);
// 返回值 = RMS 重投影误差(第 4 节展开); rvecs/tvecs = 每张图的棋盘姿态(外参, 顺带产物)

// ---------- 3. 开卷对答案: 标出来的 vs 藏起来的真值 ----------
double fx = K.At<double>(0, 0), fy = K.At<double>(1, 1);
double cx = K.At<double>(0, 2), cy = K.At<double>(1, 2);
double k1 = D.At<double>(0, 0), k2 = D.At<double>(1, 0);
double p1 = D.At<double>(2, 0), p2 = D.At<double>(3, 0), k3 = D.At<double>(4, 0);
Console.WriteLine("\n开卷对答案 (标定值 vs 真值):");
Console.WriteLine($"  fx = {fx,8:F3}  (真值 800)     fy = {fy,8:F3}  (真值 800)");
Console.WriteLine($"  cx = {cx,8:F3}  (真值 320)     cy = {cy,8:F3}  (真值 240)");
Console.WriteLine($"  k1 = {k1,8:F4}  (真值 -0.15)   k2 = {k2,8:F4}  (真值 0.05)");
Console.WriteLine($"  p1 = {p1,8:F4}  (真值 0)       p2 = {p2,8:F4}  (真值 0)     k3 = {k3,8:F4}  (真值 0)");

// ---------- 4. 重投影误差: 标定质量的官方考分 ----------
// 用标出的 K/D/R/t 把世界点「重新拍」回照片, 与实测角点比距离
// 意义: 54 点 × 15 图互相投票, 任何参数错都无处藏身; 工业标准 <0.5px 可用
Console.WriteLine($"\nRMS 重投影误差 = {rms:F4} px");
double worstE = 0; int worstI = 0;
Point2f[] reproj0 = Array.Empty<Point2f>();
for (int i = 0; i < 15; i++)
{
    Mat rep = new(), jac = new();    // jac=雅可比矩阵(中间导数), 此处不要, 占位
    Cv2.ProjectPoints(boardM, rvecs[i], tvecs[i], K, D, rep, jac, 0);
    rep.GetArray(out Point2f[] rp);
    if (i == 0) reproj0 = rp;
    double e = 0;
    for (int k = 0; k < rp.Length; k++)
        e += Math.Sqrt(Math.Pow(rp[k].X - cornersList[i][k].X, 2) + Math.Pow(rp[k].Y - cornersList[i][k].Y, 2));
    e /= rp.Length;
    if (e > worstE) { worstE = e; worstI = i; }
}
Console.WriteLine($"最差一张 calib{worstI + 1:00}: 平均误差 {worstE:F4} px —— 全员亚像素级");

// ---------- 5. 畸变矫正 Undistort: 把弯的镜头拉直 ----------
// 原理(回收第十课): 内部 = InitUndistortRectifyMap(生成"矫正图每像素←原图哪取色"映射表)
//                  + Remap 反向映射重采样 —— 与旋转扩画布同一家族, 只是映射来自 K/D
// 注意: 矫正只去"镜头畸变", 不改"视角" —— 斜拍的图矫正后依旧斜(那是透视不是畸变)
Mat und = new();
Cv2.Undistort(imgs[0], und, K, D);

// 差值图(回收第八课"留差"思想): |原−矫正| 提亮 8 倍 = 畸变强度场
// 预期: 中心黑(畸变≈0) 越往边角越亮(r 越大拉回越多) —— 与第 0 节数字呼应
Mat diff = new();
Cv2.Absdiff(imgs[0], und, diff);
Cv2.CvtColor(diff, diff, ColorConversionCodes.BGR2GRAY);
Cv2.ConvertScaleAbs(diff, diff, 8, 0);

// ---------- 6. 参数实验: 标定照片数量 = 约束数量 ----------
// 呼应"为啥要拍十几张": 照片少 → 方程少 → "镜头畸变"和"棋盘拿歪了"分不开
// 观察点: fx 漂多远? (真值 800) —— 注意 N=1 恰好是正面照 calib01, 对焦距约束最弱
Console.WriteLine("\n参数实验: 只用前 N 张标定");
foreach (int n in new[] { 1, 2, 3, 5, 8, 15 })
{
    Mat Kn = new(3, 3, MatType.CV_64F, Scalar.All(0));
    Mat Dn = new(5, 1, MatType.CV_64F, Scalar.All(0));
    try
    {
        double r = Cv2.CalibrateCamera(objMat.Take(n), imgMat.Take(n), imgs[0].Size(), Kn, Dn, out _, out _, CalibrationFlags.FixK3);
        Console.WriteLine($"  N={n,2}: fx={Kn.At<double>(0, 0),8:F2} (漂 {Kn.At<double>(0, 0) - 800,+7:F2})  k1={Dn.At<double>(0, 0),8:F4}  RMS={r:F3}");
    }
    catch (Exception)
    {
        Console.WriteLine($"  N={n,2}: OpenCV 直接报错拒绝 —— 约束不足, 数学上解不出");
    }
}
Console.WriteLine("  → 照片越多参数钉得越死; 单张正面照连焦距都锁不住");

// ---------- 7. 展示 ----------
Mat drawCorner = imgs[0].Clone();
Cv2.DrawChessboardCorners(drawCorner, pattern, cornersList[0], true);  // 一行画 54 角点+连线

Mat drawRe = imgs[0].Clone();
foreach (var p in cornersList[0]) Cv2.Circle(drawRe, (Point)p, 4, new Scalar(0, 200, 0), -1);          // 检测: 绿点
foreach (var p in reproj0) Cv2.DrawMarker(drawRe, (Point)p, new Scalar(0, 0, 255), MarkerTypes.Cross, 14, 1); // 重投影: 红十字
// 绿点红十字完全重叠 = 标得准(若肉眼能看出分离, 说明参数有偏)

Cv2.ImShow("1-棋盘角点检测", drawCorner);
Cv2.ImShow("2-检测(绿) vs 重投影(红): 重叠=准", drawRe);
Cv2.ImShow("3-矫正前(边缘直线微弯)", imgs[0]);
Cv2.ImShow("4-Undistort 矫正后", und);
Cv2.ImShow("5-|原-矫正|×8 = 畸变场", diff);
Cv2.WaitKey(0);
Cv2.DestroyAllWindows();

// ============================================================
// 本课小结：
// 1. 成像管线: 世界点→R,t→÷Z→畸变→×K→像素; 上一课 Proj 是正向, 本课标定是求逆(LM 优化)
// 2. 棋盘=自带答案的考题(格子尺寸量好→54 个世界点坐标自动已知), 平面棋盘是张正友标定法前提
// 3. CornerSubPix 亚像素精化: 整数级→0.001px 级, 修正量看似微小, 却是高精度标定的地基
// 4. 重投影误差 RMS 是标定的官方考分(<0.5px 可用): 54点×15图互相投票, 参数错无处藏
// 5. Undistort 只去镜头畸变不改视角; 畸变场(差值图)中心暗边缘亮, 印证 k1 作用 ∝ r²
// 6. 照片数量=约束数量: 单张正面照连焦距都锁不住(畸变 vs 姿态分不开), 15 张全参数归位
// 练习建议: 第 6 节加 N=10 看是否已收敛; 把 calib01 从标定集中剔除再标, 观察 RMS 变化
// ============================================================
