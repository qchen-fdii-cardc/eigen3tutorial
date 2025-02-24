#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

int main()
{
     // 创建超定方程组
     auto A = MatrixXd(4, 2);
     auto b = VectorXd(4);

     A << 1, 1,
         1, 2,
         1, 3,
         1, 4;

     b << 2, 4, 6, 8;

     cout << "系数矩阵 A:\n"
          << A << "\n\n";
     cout << "常数向量 b:\n"
          << b << "\n\n";

     // 标准最小二乘
     auto svd = A.jacobiSvd(ComputeThinU | ComputeThinV);
     auto x = svd.solve(b);
     cout << "最小二乘解:\n"
          << x << "\n\n";

     // 加权最小二乘
     auto weights = VectorXd(4);
     weights << 1, 0.5, 0.5, 1;
     auto W = weights.asDiagonal();
     auto x_weighted = (W * A).jacobiSvd(ComputeThinU | ComputeThinV).solve(W * b);
     cout << "加权最小二乘解:\n"
          << x_weighted << "\n\n";

     // 计算残差
     auto residual = A * x - b;
     auto weighted_residual = A * x_weighted - b;
     cout << "标准最小二乘残差: " << residual.norm() << "\n";
     cout << "加权最小二乘残差: " << weighted_residual.norm() << "\n";

     return 0;
}