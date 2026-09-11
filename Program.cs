using OpenCvSharp;

// ============================================================
// 第十四课：模板匹配的边界与实战 —— 三大天敌与 OK/NG 质检流水线
// ============================================================
// 第十二课找"最像的一个", 第十三课找"所有目标" —— 都是理想条件
// 本课问题: 模板匹配什么时候会失灵? 失灵了工程上怎么办?
//
// 三大天敌: ① 旋转 ② 尺度(第十三课已对策: 多尺度) ③ 非线性光照(阴影)
// 一个反直觉: 整体调亮/调暗居然不伤得分! 原因在得分公式:
//   CCoeffNormed = 相关系数 = 模板与窗口"各自去均值后"夹角的 cos
//   来料线性变换 I' = a·I + b: 减均值吃掉 b, 除模长(归一化)吃掉 a
//   → 向量方向不变 → cos 不变 → 免疫 (第0节数字验证)
//   而旋转/缩放/阴影改变的是像素排布本身(向量方向) → cos 变 → 掉分
//
// 工业对策: 治本 = 环境可控(固定角度/距离/光源, 让天敌不出现)
//           治标 = 多角度模板库(来料歪了就换对应角度的模板再试)
// ============================================================

// ---------- 0. 数字实例: 线性调光为什么不伤得分 ----------
// 模板 3 像素 [10,20,30] → 去均值 [-10,0,10], 模长 √200 ≈ 14.1
// 来料调暗 a=0.5,b=5 → [10,15,20] → 去均值 [-5,0,5], 模长 √50 ≈ 7.1
// cos = (50+0+50)/(14.1×7.1) = 100/100 = 1 → 依旧完全匹配!
// 反相 a=-1: 去均值向量反向 → cos = -1 → 得分 -1(负相关, 完全不像)
Console.WriteLine("得分 = 相关系数 = cos(夹角): 线性调光只改模长不改方向 → 免疫\n");

// ---------- 1. 读取 + 用"首件良品"生成标准模板 ----------
Mat src = Cv2.ImRead(@"yb.jpg", ImreadModes.Color);
Mat tplOld = Cv2.ImRead(@"ybmb.png", ImreadModes.Color);
if (src.Empty() || tplOld.Empty())
{
    Console.WriteLine("读取失败：请确认 yb.jpg / ybmb.png 在项目输出目录（bin/Debug/net8.0）中");
    return;
}
int h = src.Height, w = src.Width;

// 匹配工具(回收第十三课): 返回全局最高得分与位置
double TryMatch(Mat image, Mat template, out int bx, out int by)
{
    Mat res = new Mat();
    Cv2.MatchTemplate(image, template, res, TemplateMatchModes.CCoeffNormed);
    int[] minL = new int[2], maxL = new int[2];
    Cv2.MinMaxIdx(res, out _, out double maxV, minL, maxL);
    bx = maxL[1]; by = maxL[0];      // MinMaxIdx 下标 [y,x] 与 (x,y) 相反(十三课的坑)
    return maxV;
}
// 掩膜版匹配: 只返回得分(掩膜让"圆外背景"不参与比对)
double MaskScore(Mat image, Mat template, Mat mask)
{
    Mat res = new Mat();
    Cv2.MatchTemplate(image, template, res, TemplateMatchModes.CCoeffNormed, mask);
    Cv2.MinMaxIdx(res, out _, out double maxV);
    return maxV;
}

// 旧模板尺度未必与大图一致 → 多尺度搜索定位一枚硬币(回收十三课)
double bestS = 0; int bx = 0, by = 0; Mat tplBest = tplOld;
for (double k = 0.5; k <= 1.5; k += 0.1)
{
    Mat scaled = new Mat();
    Cv2.Resize(tplOld, scaled, new Size(0, 0), k, k, InterpolationFlags.Area);
    if (scaled.Width >= w || scaled.Height >= h) continue;
    int sx, sy; double sc = TryMatch(src, scaled, out sx, out sy);
    if (sc > bestS) { bestS = sc; bx = sx; by = sy; tplBest = scaled; }
}
// 本图的 ybmb 恰好是从 yb.jpg 裁出的(得分 1.0); 现实中跨照片模板得分会明显低
Console.WriteLine($"旧模板多尺度定位: 得分 {bestS:F3} @({bx},{by})");

// 工业做法: 定位到的"首件良品"直接裁出来当标准模板 → 基准得分 1.0
int tw = tplBest.Width, th = tplBest.Height;
Rect roi = new Rect(bx, by, tw, th) & new Rect(0, 0, w, h);   // & = 交集, 防越界
Mat tpl = src[roi].Clone();       // SubMat 是视图必须 Clone(第十二课)
Console.WriteLine($"标准件模板 {tpl.Width}x{tpl.Height}: 基准得分 {TryMatch(src, tpl, out _, out _):F3}");

// 工位视角: 相机视野只拍一个工件(单工位常态, 第 4/5 节都用它)
int mg = 15;      // 裁剪边距: 来料图 = 硬币 + 一圈背景
Rect partRoi = new Rect(bx - mg, by - mg, tw + 2 * mg, th + 2 * mg) & new Rect(0, 0, w, h);
int ox = bx - partRoi.X, oy = by - partRoi.Y;    // 硬币在来料图中的位置
Mat partOk = src[partRoi].Clone();
Console.WriteLine();

// ---------- 2. 天敌①旋转: 模板歪一点, 得分掉多少 ----------
// 旋转带来"双重伤害": ① 像素排布转了(真损伤) ② 方形画布旋转露出黑角(假损伤)
//   黑角 vs 浅色桌面 = 巨大像素差 → 无掩膜得分被黑角拖垮
// 对策: 圆形掩膜(mask)只让硬币本体参与比对 → 测出"纯旋转损伤"
// 注: 掩膜最早只支持 SqDiff/CCorrNormed, 新版 OpenCV 才扩展到 CCoeffNormed
Mat RotateExpand(Mat m, double deg)   // 旋转扩画布(回收第十课)
{
    Point2f c = new(m.Width / 2f, m.Height / 2f);
    Mat rot = Cv2.GetRotationMatrix2D(c, deg, 1.0);
    double rad = deg * Math.PI / 180.0;
    double ca = Math.Abs(Math.Cos(rad)), sa = Math.Abs(Math.Sin(rad));
    int nw = (int)(m.Width * ca + m.Height * sa);      // 旋转后外接矩形
    int nh = (int)(m.Width * sa + m.Height * ca);
    rot.Set(0, 2, rot.At<double>(0, 2) + nw / 2.0 - c.X);   // 中心平移修正
    rot.Set(1, 2, rot.At<double>(1, 2) + nh / 2.0 - c.Y);
    Mat dst = new Mat();
    Cv2.WarpAffine(m, dst, rot, new Size(nw, nh));
    return dst;
}
Mat CircleMask(Mat m, int r)           // 圆内 255=参与比对, 圆外 0=忽略
{
    Mat mk = new Mat(m.Size(), m.Type(), Scalar.All(0));   // 与模板同类型(官方要求)
    Cv2.Circle(mk, new Point(m.Width / 2, m.Height / 2), r, Scalar.All(255), -1);
    return mk;
}
int rMask = Math.Min(tpl.Width, tpl.Height) / 2;    // 掩膜半径 = 硬币半径
Console.WriteLine("实验一 旋转: 模板转 θ 后匹配原图");
Console.WriteLine("  角度 | 无掩膜(黑角+旋转) | 圆掩膜(纯旋转)");
foreach (double a in new double[] { 0, 3, 5, 10, 15 })
{
    Mat rt = a == 0 ? tpl : RotateExpand(tpl, a);
    double sA = TryMatch(src, rt, out _, out _);
    double sB = MaskScore(src, rt, CircleMask(rt, rMask));
    Console.WriteLine($"  转{a,3:F0}° |     {sA:F3}        |    {sB:F3}");
}
Console.WriteLine("  → 无掩膜列崩得快, 相当一部分是黑角的锅; 掩膜列才是纯旋转损伤(整体下行)");
Console.WriteLine("  → 掩膜能救黑角, 救不了旋转本身 —— 旋转还得靠下一招\n");

// ---------- 3. 天敌②光照: 线性免疫, 阴影致命 ----------
// (a) 均匀调暗 0.6x+10 —— 第 0 节公式预测: 得分不变
Mat dark = new Mat();
src.ConvertTo(dark, -1, 0.6, 10);
Console.WriteLine("实验二 光照:");
Console.WriteLine($"  均匀调暗(0.6x+10): 得分 {TryMatch(dark, tpl, out _, out _):F3} ≈ 1.0, 免疫!");

// (b) 反相 = 线性 a=-1, b=255 —— 预测: 目标位置得分 ≈ -1(负相关)
//     必须改用"定点观察": 反相图里全局峰值会跑到别处, 不能只看 Max
Mat inv = new Mat();
Cv2.BitwiseNot(src, inv);
Mat resInv = new Mat();
Cv2.MatchTemplate(inv, tpl, resInv, TemplateMatchModes.CCoeffNormed);
Console.WriteLine($"  反相(-1x+255):    目标位置得分 {resInv.At<float>(by, bx):F3} ≈ -1, 负相关");

// (c) 半边阴影: 与(a)同一个乘法, 但只作用于"硬币右半边的竖条" —— 大跌
//     同一操作换了作用范围, 结论反转: 全局=线性(免疫), 局部=空间上非线性(伤)
//     模板眼里"左右两半的 a 不同", 减均值吸收不了"变化的 a"
Mat shadow = src.Clone();
int shx = bx + tw / 2;                      // 阴影边界 = 硬币中心 x
Rect strip = new Rect(shx, 0, w - shx, h);  // 覆盖右半图的竖条
Mat tmpS = new Mat();
shadow[strip].ConvertTo(tmpS, -1, 0.3, 0);  // 这一条 ×0.3(回收第一课 aI+b)
tmpS.CopyTo(shadow[strip]);                 // 写回 ROI(写入视图=写入原图内存)
Mat resSh = new Mat();
Cv2.MatchTemplate(shadow, tpl, resSh, TemplateMatchModes.CCoeffNormed);
Cv2.MinMaxIdx(resSh, out _, out double gmax);
Console.WriteLine($"  半边阴影:         目标位置得分 {resSh.At<float>(by, bx):F3} (大跌), 全局最高 {gmax:F3} (逃去别的硬币)");
Console.WriteLine("  → 治本靠打光均匀: 阴影是'位置相关'的变换, 每个像素的 a 都不同\n");

// ---------- 4. 治标对策: 多角度模板库(掩膜版) ----------
// 模拟"来料放歪 30°": 0°模板得分大跌 → 库里逐角度试模板, 取最高分
Mat partRot = RotateExpand(partOk, 30);
Console.WriteLine($"实验三 多角度库: 来料歪 30°, 0°模板得分 {MaskScore(partRot, tpl, CircleMask(tpl, rMask)):F3}");
double libBest = 0, libAngle = 0;
foreach (double a in new double[] { -30, -20, -10, 0, 10, 20, 30 })
{
    Mat rt = RotateExpand(tpl, a);
    double s = MaskScore(partRot, rt, CircleMask(rt, rMask));
    Console.WriteLine($"    库[{a,3:F0}°] = {s:F3}");
    if (s > libBest) { libBest = s; libAngle = a; }
}
Console.WriteLine($"  → 最佳 {libAngle:F0}° 得分 {libBest:F3}: 找回! 库越密找得越准(代价是越慢)");
Console.WriteLine("  → 治本仍是工装夹具固定来料姿态, 库只是兜底\n");

// ---------- 5. 实战: 定位 + 得分判 OK/NG 的小型质检流水线 ----------
const double GATE = 0.90;    // 质检门槛: 得分 ≥ GATE 判 OK, 否则 NG
Mat partScr = partOk.Clone();                    // ② 划痕件: 拉三道深划痕
Cv2.Line(partScr, new Point(ox + tw / 6, oy + th / 6), new Point(ox + tw * 5 / 6, oy + th / 2), new Scalar(0, 0, 0), 4);
Cv2.Line(partScr, new Point(ox + tw / 6, oy + th / 2), new Point(ox + tw * 5 / 6, oy + th * 5 / 6), new Scalar(0, 0, 0), 4);
Cv2.Line(partScr, new Point(ox + tw / 2, oy + th / 6), new Point(ox + tw / 2, oy + th * 5 / 6), new Scalar(0, 0, 0), 4);
Mat partInv = new Mat();
Cv2.BitwiseNot(partOk, partInv);                 // ③ 来料根本不对(比如放错了料)

double sOk = TryMatch(partOk, tpl, out int pOkx, out int pOky);   // ① 良品
double sScr = TryMatch(partScr, tpl, out int pScx, out int pScy);
double sInv = TryMatch(partInv, tpl, out _, out _);
Console.WriteLine($"实验四 质检(门槛 {GATE}): 良品 {sOk:F3} | 划痕 {sScr:F3} | 反相 {sInv:F3}");

// 关键认知: MinMaxIdx 永远返回一个"最佳位置"(反相件也会给框!)
//   → 没有得分门槛的检测器 = "永远 OK"的假检测器, 门槛才是质检的判官
Mat drawOk = partOk.Clone(), drawScr = partScr.Clone(), drawInv = partInv.Clone();
Cv2.Rectangle(drawOk, new Rect(pOkx, pOky, tw, th), new Scalar(0, 255, 0), 2);
Cv2.PutText(drawOk, $"OK {sOk:F2}", new Point(pOkx, pOky - 4), HersheyFonts.HersheySimplex, 0.5, new Scalar(0, 255, 0), 1);
Cv2.Rectangle(drawScr, new Rect(pScx, pScy, tw, th), new Scalar(0, 0, 255), 2);
Cv2.PutText(drawScr, $"NG {sScr:F2}", new Point(pScx, pScy - 4), HersheyFonts.HersheySimplex, 0.5, new Scalar(0, 0, 255), 1);
Cv2.PutText(drawInv, $"NG {sInv:F2}", new Point(2, 16), HersheyFonts.HersheySimplex, 0.5, new Scalar(0, 0, 255), 1);

// ---------- 6. 参数实验: 门槛 = 漏检/误杀的旋钮 ----------
// (纯度-完整度权衡家族又一员: Canny 双阈值 / HSV 容差 / NMS 阈值)
Console.WriteLine("\n参数实验: 改 GATE 看三种来料的判定变化");
foreach (double g in new double[] { 0.80, 0.90, 0.97 })
{
    string j1 = sOk >= g ? "OK" : "NG", j2 = sScr >= g ? "OK" : "NG", j3 = sInv >= g ? "OK" : "NG";
    Console.WriteLine($"  GATE={g:F2}: 良品{j1} 划痕{j2} 反相{j3}");
}
Console.WriteLine("  → 门槛低: 缺陷漏放(划痕也 OK); 门槛高: 良品误杀 —— 现场靠良品/缺陷样本得分分布定");

// ---------- 7. 展示 ----------
Mat drawBase = src.Clone();
Cv2.Rectangle(drawBase, roi, new Scalar(0, 255, 0), 2);
Cv2.PutText(drawBase, "base 1.00", new Point(roi.X, roi.Y - 4), HersheyFonts.HersheySimplex, 0.5, new Scalar(0, 255, 0), 1);
Cv2.ImShow("1-基准定位(标准件模板)", drawBase);
Cv2.ImShow("2-半边阴影(非线性光照)", shadow);
Cv2.ImShow("3-来料歪30°(多角度库可救)", partRot);
Cv2.ImShow($"4-良品 {sOk:F2} → OK", drawOk);
Cv2.ImShow($"5-划痕 {sScr:F2} → NG", drawScr);
Cv2.ImShow($"6-反相 {sInv:F2} → NG", drawInv);
Cv2.WaitKey(0);
Cv2.DestroyAllWindows();

// ============================================================
// 本课小结：
// 1. 得分 = 相关系数 = 去均值后夹角 cos: 线性调光(aI+b)免疫, 阴影(空间非线性)大跌
// 2. 反相是 a=-1 的线性: 目标位置得分 ≈ -1(负相关), 印证 cos 模型
// 3. 旋转是"双重伤害": 像素排布转(真伤) + 方形画布露黑角(假伤)
// 4. 圆掩膜剔除黑角/背景干扰, 只让目标本体参与比对(新版才支持 CCoeffNormed+mask)
// 5. 多角度模板库治标(来料歪→逐角度试), 环境可控(夹具/光源)治本
// 6. MinMaxIdx 永远给"最佳位置" → 质检必须加得分门槛, 否则=永远 OK 的假检测器
// 练习建议: 划痕画粗/画细观察 sScr 变化; GATE 提到 0.97 看良品是否误杀
// ============================================================
