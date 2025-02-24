#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

int main()
{
     // 固定大小矩阵
     Matrix3d m1 = Matrix3d::Zero();     // 零矩阵
     Matrix3d m2 = Matrix3d::Identity(); // 单位矩阵
     Matrix3d m3;                        // 未初始化矩阵
     m3 << 1, 2, 3,
         4, 5, 6,
         7, 8, 9;

     // 动态大小矩阵
     MatrixXd m4(2, 3); // 2x3矩阵
     m4 << 1, 2, 3,
         4, 5, 6;

     cout << "零矩阵:\n"
          << m1 << "\n\n";
     cout << "单位矩阵:\n"
          << m2 << "\n\n";
     cout << "自定义矩阵:\n"
          << m3 << "\n\n";
     cout << "动态大小矩阵:\n"
          << m4 << "\n\n";

     // 元素访问
     cout << "m3(0,0) = " << m3(0, 0) << "\n";
     cout << "m3(1,2) = " << m3(1, 2) << "\n\n";

     // 矩阵块操作
     Block<Matrix3d, 2, 2> block = m3.block<2, 2>(0, 0);
     cout << "m3的左上2x2块:\n"
          << block << "\n\n";

     return 0;
}