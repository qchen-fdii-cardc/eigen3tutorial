#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

int main()
{
     Matrix3d A;
     A << 1, 2, 3,
         2, 4, 5,
         3, 5, 6;

     cout << "矩阵 A:\n"
          << A << "\n\n";

     // 一般特征值问题
     EigenSolver<Matrix3d> solver(A);
     cout << "特征值:\n"
          << solver.eigenvalues() << "\n\n";
     cout << "特征向量:\n"
          << solver.eigenvectors() << "\n\n";

     // 实对称矩阵的特征值
     SelfAdjointEigenSolver<Matrix3d> symm_solver(A);
     cout << "实对称矩阵的特征值:\n"
          << symm_solver.eigenvalues() << "\n\n";
     cout << "实对称矩阵的特征向量:\n"
          << symm_solver.eigenvectors() << "\n\n";

     // 验证特征值和特征向量
     Vector3cd eigenval = solver.eigenvalues();
     Matrix3cd eigenvec = solver.eigenvectors();
     for (int i = 0; i < 3; ++i)
     {
          cout << "验证第" << i + 1 << "个特征值和特征向量:\n";
          cout << "λv =\n"
               << eigenval(i) * eigenvec.col(i) << "\n";
          cout << "Av =\n"
               << A * eigenvec.col(i) << "\n\n";
     }

     return 0;
}