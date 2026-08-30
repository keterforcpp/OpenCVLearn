using OpenCvSharp;
//using System.Drawing;

// 1. 读取原始图像
Mat src = Cv2.ImRead(@"3.jpg", ImreadModes.Color);

// 2. 灰度化：将彩色图转为灰度图
Mat gray = new Mat();
Cv2.CvtColor(src, gray, ColorConversionCodes.BGR2GRAY);

// 3. 高斯模糊：减少图像噪声
Mat blurred = new Mat();
Cv2.GaussianBlur(gray, blurred, new Size(5, 5), 0);

// 4. Canny边缘检测：提取图像中的边缘
Mat blurrededges = new Mat();
Mat grayedge = new Mat();
Mat edges = new Mat();
Cv2.Canny(blurred, blurrededges, 100, 200);
Cv2.Canny(gray,grayedge,100,200);
Cv2.Canny(src, edges, 100, 200);

// 5. 分别显示原图和结果图
Cv2.ImShow("原始图像", src);
//Cv2.ImShow("灰度化 ", gray);
//Cv2.ImShow("高斯模糊",blurred);
Cv2.ImShow("gaosi边缘检测结果", blurrededges);
Cv2.ImShow("灰度化边缘检测",grayedge);
Cv2.ImShow("边缘检测结果", edges);

Cv2.WaitKey(0);
Cv2.DestroyAllWindows();