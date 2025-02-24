#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

int main()
{
     auto v1 = Vector3d(1, 2, 3);
     auto v2 = Vector3d(4, 5, 6);

     cout << "向量 v1:\n"
          << v1 << "\n\n";
     cout << "向量 v2:\n"
          << v2 << "\n\n";

     // 基本运算
     auto sum = v1 + v2;
     auto diff = v1 - v2;
     auto scaled = 2 * v1;
     cout << "v1 + v2 =\n"
          << sum << "\n\n";
     cout << "v1 - v2 =\n"
          << diff << "\n\n";
     cout << "2 * v1 =\n"
          << scaled << "\n\n";

     // 点积和叉积
     auto dot_product = v1.dot(v2);
     auto cross_product = v1.cross(v2);
     cout << "点积 v1·v2 = " << dot_product << "\n\n";
     cout << "叉积 v1×v2 =\n"
          << cross_product << "\n\n";

     // 范数计算
     cout << "v1的欧几里得范数: " << v1.norm() << "\n";
     cout << "v1的平方范数: " << v1.squaredNorm() << "\n";
     cout << "v1的L1范数: " << v1.lpNorm<1>() << "\n";
     cout << "v1的无穷范数: " << v1.lpNorm<Infinity>() << "\n\n";

     return 0;
}