#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

int main()
{
     Matrix3d A;
     Vector3d b;

     A << 2, -1, 0,
         -1, 2, -1,
         0, -1, 2;

     b << 1, 2, 3;

     cout << "系数矩阵 A:\n"
          << A << "\n\n";
     cout << "常数向量 b:\n"
          << b << "\n\n";

     // 直接求解
     Vector3d x1 = A.colPivHouseholderQr().solve(b);
     cout << "QR分解求解:\n"
          << x1 << "\n\n";

     // LU分解求解
     Vector3d x2 = A.lu().solve(b);
     cout << "LU分解求解:\n"
          << x2 << "\n\n";

     // Cholesky分解求解（仅适用于对称正定矩阵）
     Vector3d x3 = A.llt().solve(b);
     cout << "Cholesky分解求解:\n"
          << x3 << "\n\n";

     // 验证结果
     cout << "验证 Ax = b:\n"
          << A * x1 << "\n\n";
     cout << "残差: " << (A * x1 - b).norm() << "\n";

     return 0;
}