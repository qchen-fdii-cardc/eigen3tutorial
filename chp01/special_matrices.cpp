#include <iostream>
#include <iomanip> // Add this for output formatting
#include <Eigen/Dense>
#include <Eigen/Householder>

using namespace Eigen;
using namespace std;

int main()
{
     // 对角矩阵
     Vector3d diag(1, 2, 3);
     auto diagonal = diag.asDiagonal().toDenseMatrix();
     cout << "对角矩阵:\n"
          << diagonal << "\n\n";

     // 三角矩阵
     Matrix3d m = Matrix3d::Random();
     auto upper = m.triangularView<Upper>().toDenseMatrix();
     auto lower = m.triangularView<Lower>().toDenseMatrix();
     cout << "上三角矩阵:\n"
          << upper << "\n\n";
     cout << "下三角矩阵:\n"
          << lower << "\n\n";

     // 置换矩阵
     PermutationMatrix<3> perm;
     perm.setIdentity();
     perm.indices()(0) = 2;
     perm.indices()(2) = 0;
     auto P = perm.toDenseMatrix();
     cout << "置换矩阵:\n"
          << P << "\n\n";

     // 带状矩阵
     Matrix3d band;
     band << 1, 2, 0,
         2, 3, 4,
         0, 4, 5;
     cout << "带状矩阵:\n"
          << band << "\n\n";

     // Householder变换
     Vector3d v(1, 2, 3);
     v.normalize();
     auto H = Matrix3d::Identity() - 2 * v * v.transpose();
     cout << "Householder矩阵:\n"
          << H << "\n\n";

     return 0;
}