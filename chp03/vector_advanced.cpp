#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

int main()
{
     auto v1 = Vector3d(1, 2, 3);
     auto v2 = Vector3d(4, 5, 6);

     // 向量投影
     auto proj = v2 * (v1.dot(v2) / v2.squaredNorm());
     cout << "v1在v2上的投影:\n"
          << proj << "\n\n";

     // 向量旋转
     double angle = M_PI / 4; // 45度
     auto rotation = AngleAxisd(angle, Vector3d::UnitZ());
     auto R = rotation.matrix();
     auto rotated = R * v1;
     cout << "绕Z轴旋转45度后的v1:\n"
          << rotated << "\n\n";

     // 向量归一化
     cout << "v1归一化:\n"
          << v1.normalized() << "\n\n";

     // 向量插值
     double t = 0.5;
     auto interpolated = (1 - t) * v1 + t * v2;
     cout << "v1和v2的线性插值(t=0.5):\n"
          << interpolated << "\n\n";

     return 0;
}