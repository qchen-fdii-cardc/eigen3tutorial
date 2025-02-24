#include <iostream>
#include <eigen3/Eigen/Dense>

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
     auto sum = a + b;
     auto product = a * b;
     auto scaled = scalar * a;
     cout << "A + B =\n"
          << sum << "\n\n";
     cout << "A * B =\n"
          << product << "\n\n";
     cout << "A 的转置 =\n"
          << a.transpose() << "\n\n";

     // 高级运算
     cout << "A 的行列式 = " << a.determinant() << "\n";
     cout << "A 的迹 = " << a.trace() << "\n";
     cout << "A 的范数 = " << a.norm() << "\n";
     cout << "A 的秩 = " << a.fullPivLu().rank() << "\n\n";

     return 0;
}