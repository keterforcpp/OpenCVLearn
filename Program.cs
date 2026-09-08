using OpenCvSharp;

// ============================================================
// 第十二课：模板匹配 —— 拿一小块图当"模板"去大图里找它
// ============================================================
// 核心思想（第二课卷积的近亲）:
//   卷积: 小核在图上滑动，每位置算"加权求和"
//   模板匹配: 模板在图上滑动，每位置算"这块和模板像不像"
//   区别: 卷积核是几x几的小数字, 模板是一张真正的小图;
//         卷积输出通道图, 匹配输出"相似度地图"(结果图)
//
// 结果图的读法（本课最重要认知）:
//   结果图比原图小(宽-w+1, 高-h+1), 每个格子 = "模板左上角放这时的得分"
//   得分最高的格子位置 = 模板在大图中的位置
//   → 模板定位 = 找结果图的峰值 = MinMaxLoc 一行搞定
//
// 六种方法分三族:
//   SQDIFF  差的平方和   越小越像(谷底找最小)
//   CCORR   互相关       越大越像(但受亮度影响: 都很亮时即使不像分也高)
//   CCOEFF  去均值相关   越大越像(先减各自均值再相关 → 抗整体亮暗)
//   各带 _NORMED 后缀 = 归一化到 [-1,1] 或 [0,1] → 跨图可比, 工程标配
// 工程默认: TM_CCOEFF_NORMED (1=完美, 0=无关, -1=负相关)
// ============================================================

// ---------- 0. 数字实例: "像不像"怎么算成一个数 ----------
// 模板 3 像素 [10, 20, 30], 图上两个窗口:
//   窗口A [10, 20, 30]: 差的平方和 = 0+0+0 = 0      ← 一模一样
//   窗口B [10, 20, 40]: 差的平方和 = 0+0+100 = 100  ← 差一点
//   窗口C [50, 60, 70]: 差的平方和 = 1600+1600+1600 = 4800 ← 完全不像
// → "像不像"被压成一个数, 数值可比较 → 扫全图取最优
// CCOEFF 的改进: 模板均值20, 窗口B均值23.3, 先各自减均值再比
//   → 整体偏亮/偏暗不影响判断(只比"形状"不比"亮度")
Console.WriteLine("匹配 = 滑窗逐位置算'相似度', 相似度地图的峰值 = 目标位置\n");

// ---------- 1. 读取 + 自动截模板 ----------
// 教学技巧: 从原图中央裁一块当模板 → 任何图都能演示,
// 且"模板一定在图中存在"(得分必然接近 1)
Mat src = Cv2.ImRead(@"3.jpg", ImreadModes.Color);
if (src.Empty())
{
    Console.WriteLine("读取失败：请确认 3.jpg 在项目输出目录（bin/Debug/net8.0）中");
    return;
}
int h = src.Height, w = src.Width;
int tw = w / 5, th = h / 5;                        // 模板 1/5 尺寸
Rect tplRect = new Rect(w / 3, h / 3, tw, th);     // 取图中央偏左上一块
Mat tpl = src.SubMat(tplRect).Clone();             // Clone! SubMat 是视图不拥有数据
Console.WriteLine($"原图 {w}x{h}, 模板 {tw}x{th}（截自 ({tplRect.X},{tplRect.Y})）");

// ---------- 2. 手写暴力匹配（缩小图上验证原理） ----------
// 对全图手写滑窗太慢(百万级窗口×每个窗口几万次运算),
// 教学惯例: 缩到 1/4 尺寸做, 逻辑与全图完全一致
Mat smallSrc = new Mat(), smallTpl = new Mat();
Cv2.Resize(src, smallSrc, new Size(w / 4, h / 4), 0, 0, InterpolationFlags.Area);
Cv2.Resize(tpl, smallTpl, new Size(tw / 4, th / 4), 0, 0, InterpolationFlags.Area);
int sh = smallSrc.Height, sw = smallSrc.Width, sth = smallTpl.Height, stw = smallTpl.Width;
if (!smallSrc.GetArray(out Vec3b[] sPx) || !smallTpl.GetArray(out Vec3b[] tPx))
{
    Console.WriteLine("GetArray 失败");
    return;
}
// 手写 SQDIFF(差的平方和): 每个窗口位置, 累加所有像素所有通道的差的平方
double bestScore = double.MaxValue; int bestX = -1, bestY = -1;
for (int y = 0; y <= sh - sth; y++)                // 滑窗: 模板左上角的所有可能位置
{
    for (int x = 0; x <= sw - stw; x++)
    {
        long sq = 0;
        for (int dy = 0; dy < sth; dy++)
        {
            int sRow = (y + dy) * sw, tRow = dy * stw;
            for (int dx = 0; dx < stw; dx++)
            {
                Vec3b s = sPx[sRow + x + dx];
                Vec3b t = tPx[tRow + dx];
                sq += (long)(s.Item0 - t.Item0) * (s.Item0 - t.Item0)
                    + (long)(s.Item1 - t.Item1) * (s.Item1 - t.Item1)
                    + (long)(s.Item2 - t.Item2) * (s.Item2 - t.Item2);
            }
        }
        if (sq < bestScore) { bestScore = sq; bestX = x; bestY = y; }
    }
}
// API 对照(缩小图上同尺度比较)
Mat resultSmall = new Mat();
Cv2.MatchTemplate(smallSrc, smallTpl, resultSmall, TemplateMatchModes.SqDiff);
// 坑: MinMaxIdx 的 minIdx/maxIdx 不是 out 参数, 要传预分配数组进去填充;
//      且两个数组都必须给(null 会被运行时拒绝), 不关心的也塞个占位数组
int[] apiMinLoc = new int[2], apiMaxLoc = new int[2];
Cv2.MinMaxIdx(resultSmall, out double apiMin, out _, apiMinLoc, apiMaxLoc);
Console.WriteLine($"\n手写 SQDIFF: 最优 ({bestX},{bestY}) 得分 {bestScore}");
Console.WriteLine($"API  SqDiff: 最优 ({apiMinLoc[0]},{apiMinLoc[1]}) 得分 {apiMin:F0}");
Console.WriteLine($"位置一致: {bestX == apiMinLoc[0] && bestY == apiMinLoc[1]}（应为 True）");

// ---------- 3. 正式匹配: 全图 + TM_CCOEFF_NORMED ----------
Mat result = new Mat();
Cv2.MatchTemplate(src, tpl, result, TemplateMatchModes.CCoeffNormed);
// 结果图尺寸 = (w-tw+1) x (h-th+1) —— 左上角可放置位置的个数
// 每格 = 模板放该处时的相关系数(1=完美 0=无关 -1=负相关)
int[] minLoc = new int[2], maxLoc = new int[2];   // 预分配, MinMaxIdx 填充
Cv2.MinMaxIdx(result, out double minV, out double maxV, minLoc, maxLoc);
Console.WriteLine($"\n全图匹配(TMQ_CCOEFF_NORMED): 结果图 {result.Width}x{result.Height}");
Console.WriteLine($"  最优位置 ({maxLoc[0]},{maxLoc[1]}) 得分 {maxV:F4}（模板截自原图, 应≈1）");
Console.WriteLine($"  最差得分 {minV:F4}（最不像的地方, 负相关=明暗相反）");

// 画框定位: 结果图峰值位置 = 模板左上角位置, 框的尺寸 = 模板尺寸
Mat matchDraw = src.Clone();
Rect found = new Rect(maxLoc[0], maxLoc[1], tw, th);
Cv2.Rectangle(matchDraw, found, new Scalar(0, 255, 0), 3);
Cv2.PutText(matchDraw, $"score={maxV:F3}", new Point(found.X, found.Y - 8),
            HersheyFonts.HersheySimplex, 0.7, new Scalar(0, 255, 0), 2);
// 机械自查: 模板出处画红框, 匹配结果画绿框 —— 两框应完全重叠(只看得见绿)
Rect truthRect = new Rect(tplRect.X, tplRect.Y, tw, th);
Cv2.Rectangle(matchDraw, truthRect, new Scalar(0, 0, 255), 1);
Console.WriteLine($"  [自查] 模板出处=({tplRect.X},{tplRect.Y}), 匹配框=({found.X},{found.Y}), "
                  + $"重合={(found.X == truthRect.X && found.Y == truthRect.Y)}（红框若露出=真错了）");

// 结果图可视化: 32F 值域[-1,1] → Normalize 到 0~255 才能看(第九课显示套路)
Mat resultShow = new Mat();
Cv2.Normalize(result, resultShow, 0, 255, NormTypes.MinMax);
Mat result8u = new Mat();
Cv2.ConvertScaleAbs(resultShow, result8u);
Cv2.Circle(result8u, new Point(maxLoc[0], maxLoc[1]), 8, new Scalar(255), 2); // 峰值标记

// ---------- 4. 参数实验: 三族方法的结果图对比 ----------
// SqDiff:   谷底=目标(越小越像), 显示时是"暗点"
// CCorr:    峰值=目标, 但整体偏亮(亮度干扰) → 目标峰不突出
// CCoeff:   峰值=目标, 去均值后对比强烈 → 峰最锐利
// 结论: 工程永远用带 _NORMED 的, 三族里 CCOEFF_NORMED 最稳
Mat resSq = new Mat(), resCc = new Mat();
Cv2.MatchTemplate(src, tpl, resSq, TemplateMatchModes.SqDiffNormed);
Cv2.MatchTemplate(src, tpl, resCc, TemplateMatchModes.CCorrNormed);
Mat showSq = new Mat(), showCc = new Mat();
Cv2.Normalize(resSq, showSq, 0, 255, NormTypes.MinMax);
Cv2.Normalize(resCc, showCc, 0, 255, NormTypes.MinMax);
Mat showSq8 = new Mat(), showCc8 = new Mat();
Cv2.ConvertScaleAbs(showSq, showSq8);
Cv2.ConvertScaleAbs(showCc, showCc8);
int[] sqMinLoc = new int[2], sqMaxLoc = new int[2];
Cv2.MinMaxIdx(resSq, out _, out _, sqMinLoc, sqMaxLoc);   // SqDiff 反着: 最小才像
Cv2.Circle(showSq8, new Point(sqMinLoc[0], sqMinLoc[1]), 8, new Scalar(255), 2);
Console.WriteLine("\n参数实验: 三族结果图对比(都归一化显示)");
Console.WriteLine("  SqDiffNormed: 目标=最暗点(注意'最小'才是答案)");
Console.WriteLine("  CCorrNormed:  目标=亮点, 但被亮度背景糊住");
Console.WriteLine("  CCoeffNormed: 目标=最锐利的亮点 ← 工程首选");

// ---------- 5. 阈值判定: 匹配得分的工程意义 ----------
// 工程上 maxV 不只是"找位置"—— 它是质量分数:
//   ≥0.95 几乎完美 / 0.8~0.9 大概率是 / <0.6 很可疑
// 第十四课的 OK/NG 判定就建立在这条分数带上
Console.WriteLine($"\n得分解读: {maxV:F3} ≥ 0.95 → 模板与该区域高度一致");

// ---------- 6. 展示 ----------
Cv2.ImShow("1-模板(截自原图)", tpl);
//Cv2.ImShow("2-手写验证(缩小图)", smallSrc.Clone(new Rect(bestX, bestY, stw, sth)));
Cv2.ImShow("3-匹配定位", matchDraw);
Cv2.ImShow("4-结果图CCoeffNormed(峰=目标)", result8u);
//Cv2.ImShow("5-对比SqDiffNormed(谷=目标)", showSq8);
//Cv2.ImShow("6-对比CCorrNormed(峰糊)", showCc8);
Cv2.WaitKey(0);
Cv2.DestroyAllWindows();

// ============================================================
// 本课小结：
// 1. 模板匹配 = 滑窗算相似度(卷积的近亲), 输出相似度地图(结果图)
// 2. 结果图每格 = "模板左上角放这的得分"; 定位 = 找峰值 MinMaxLoc
// 3. 三族方法: SqDiff(小=像)/CCorr(大=像,怕亮)/CCoeff(去均值,最稳)
// 4. _NORMED 后缀归一化到固定区间 → 分数跨图可比, 工程必带
// 5. CCoeffNormed 得分: 1=完美 0=无关 -1=负相关; ≥0.95 视为命中
// 6. SubMat 是视图不拥有数据, 要独立保存必须 Clone
// 练习建议: 把模板改成图中另一个物体的截图, 再改半透明物体的, 看得分变化
// ============================================================
