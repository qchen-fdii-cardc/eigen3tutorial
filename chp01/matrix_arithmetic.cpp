#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

int main()
{
     auto a = Matrix3d::Random();
     auto b = Matrix3d::Random();
     double scalar = 2.0;

     cout << "矩阵 a:\n"
          << a << "\n\n";
     cout << "矩阵 b:\n"
          << b << "\n\n";

     // 基本运算
     cout << "矩阵加法 a + b:\n"
          << a + b << "\n\n";
     cout << "矩阵减法 a - b:\n"
          << a - b << "\n\n";
     cout << "矩阵乘法 a * b:\n"
          << a * b << "\n\n";
     cout << "标量乘法 " << scalar << " * a:\n"
          << scalar * a << "\n\n";

     // 元素级运算
     cout << "元素级乘法 a.cwiseProduct(b):\n"
          << a.cwiseProduct(b) << "\n\n";

     cout << "元素级除法 a.cwiseQuotient(b):\n"
          << a.cwiseQuotient(b) << "\n\n";

     // 矩阵函数
     cout << "矩阵 a 的转置:\n"
          << a.transpose() << "\n\n";
     cout << "矩阵 a 的共轭:\n"
          << a.conjugate() << "\n\n";
     cout << "矩阵 a 的伴随:\n"
          << a.adjoint() << "\n\n";

     return 0;
}