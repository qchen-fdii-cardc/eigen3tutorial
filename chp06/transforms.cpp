#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Geometry>

using namespace Eigen;
using namespace std;

int main()
{
     // 创建一个点
     Vector3d point(1, 0, 0);
     cout << "原始点:\n"
          << point << "\n\n";

     // 平移变换
     Translation3d translation(1, 2, 3);
     cout << "平移后:\n"
          << translation * point << "\n\n";

     // 旋转变换
     double angle = M_PI / 4; // 45度
     AngleAxisd rotation(angle, Vector3d::UnitZ());
     cout << "旋转后:\n"
          << rotation * point << "\n\n";

     // 缩放变换
     DiagonalMatrix<double, 3> scale(2, 2, 2);
     cout << "缩放后:\n"
          << scale * point << "\n\n";

     // 组合变换
     Affine3d transform = translation * rotation;
     transform.scale(2.0);
     cout << "组合变换后:\n"
          << transform * point << "\n\n";

     // 变换矩阵
     cout << "变换矩阵:\n"
          << transform.matrix() << "\n\n";

     return 0;
}