#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

int main()
{
     // 矩阵块操作
     Matrix4d m = Matrix4d::Random();
     cout << "原始4x4矩阵:\n"
          << m << "\n\n";

     // 提取2x2块
     cout << "左上2x2块:\n"
          << m.block<2, 2>(0, 0) << "\n\n";

     // 行和列操作
     cout << "第一行:\n"
          << m.row(0) << "\n\n";
     cout << "第一列:\n"
          << m.col(0) << "\n\n";

     // 广播操作
     MatrixXd mat(3, 4);
     mat.setConstant(1.0);
     VectorXd vec(4);
     vec << 1, 2, 3, 4;

     // 按行广播
     cout << "按行广播前:\n"
          << mat << "\n\n";
     mat.rowwise() += vec.transpose();
     cout << "按行广播后:\n"
          << mat << "\n\n";

     // 数组操作
     ArrayXXd arr1 = ArrayXXd::Random(3, 3);
     ArrayXXd arr2 = ArrayXXd::Random(3, 3);
     cout << "数组乘法:\n"
          << arr1 * arr2 << "\n\n";
     cout << "数组最大值:\n"
          << arr1.max(arr2) << "\n\n";

     return 0;
}