#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

int main()
{
     auto A = Matrix3d();
     auto b = Vector3d();

     A << 2, -1, 0,
         -1, 2, -1,
         0, -1, 2;

     b << 1, 2, 3;

     cout << "系数矩阵 A:\n"
          << A << "\n\n";
     cout << "常数向量 b:\n"
          << b << "\n\n";

     // 直接求解
     auto qr_solver = A.colPivHouseholderQr();
     auto x1 = qr_solver.solve(b);
     cout << "QR分解求解:\n"
          << x1 << "\n\n";

     // LU分解求解
     auto lu_solver = A.lu();
     auto x2 = lu_solver.solve(b);
     cout << "LU分解求解:\n"
          << x2 << "\n\n";

     // Cholesky分解求解（仅适用于对称正定矩阵）
     auto llt_solver = A.llt();
     auto x3 = llt_solver.solve(b);
     cout << "Cholesky分解求解:\n"
          << x3 << "\n\n";

     // 验证结果
     auto residual = A * x1 - b;
     cout << "验证 Ax = b:\n"
          << A * x1 << "\n\n";
     cout << "残差: " << residual.norm() << "\n";

     return 0;
}