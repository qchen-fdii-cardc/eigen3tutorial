#include <iostream>
#include <Eigen/Dense>    // 密集矩阵运算
#include <Eigen/Geometry> // 几何变换相关功能

using namespace Eigen;
using namespace std;

int main()
{
    // 定义欧拉角（弧度制）
    double yaw = M_PI / 4;    // 绕Z轴旋转45度
    double pitch = M_PI / 6;  // 绕Y轴旋转30度
    double roll = M_PI / 3;   // 绕X轴旋转60度

    // 从欧拉角创建旋转矩阵
    Matrix3d R = (AngleAxisd(yaw, Vector3d::UnitZ()) * 
                  AngleAxisd(pitch, Vector3d::UnitY()) * 
                  AngleAxisd(roll, Vector3d::UnitX())).matrix();
    cout << "旋转矩阵:\n" << R << "\n\n";

    // 从旋转矩阵提取欧拉角
    Vector3d euler = R.eulerAngles(2, 1, 0); // ZYX顺序
    cout << "欧拉角 (ZYX):\n" 
         << euler * 180 / M_PI << " 度\n\n";

    // 四元数表示
    Quaterniond q(R);  // 从旋转矩阵构造四元数
    cout << "四元数:\n"
         << "w: " << q.w() << "\n"  // 实部
         << "x: " << q.x() << "\n"  // 虚部i
         << "y: " << q.y() << "\n"  // 虚部j
         << "z: " << q.z() << "\n\n"; // 虚部k

    // 轴角表示
    AngleAxisd aa(R);  // 从旋转矩阵构造轴角
    cout << "轴角表示:\n"
         << "角度: " << aa.angle() * 180 / M_PI << " 度\n"
         << "轴: " << aa.axis().transpose() << "\n\n";

    return 0;
}

/*
重要的类和对象：

1. Matrix3d
   - 3x3 双精度矩阵
   - 用于表示旋转矩阵
   - 成员函数：
     * eulerAngles(i,j,k): 提取欧拉角，i,j,k指定轴的顺序

2. AngleAxisd
   - 轴角表示法
   - 构造函数：AngleAxisd(angle, axis)
   - 成员函数：
     * angle(): 获取旋转角度
     * axis(): 获取旋转轴
     * matrix(): 转换为矩阵形式
   - 可以直接相乘组合多个旋转

3. Quaterniond
   - 四元数表示法
   - 构造函数：
     * Quaterniond(w,x,y,z): 直接指定四元数分量
     * Quaterniond(Matrix3d): 从旋转矩阵构造
     * Quaterniond(AngleAxisd): 从轴角构造
   - 成员函数：
     * w(),x(),y(),z(): 获取四元数分量
     * matrix(): 转换为矩阵形式
     * normalized(): 归一化

4. Vector3d
   - 3维向量
   - 静态成员：
     * UnitX(): (1,0,0)
     * UnitY(): (0,1,0)
     * UnitZ(): (0,0,1)
   - 成员函数：
     * transpose(): 转置
     * normalized(): 归一化

注意事项：
1. 旋转顺序很重要，不同的顺序会得到不同的结果
2. 欧拉角可能存在万向节死锁问题
3. 四元数提供了最稳定的旋转插值
4. 所有角度计算都使用弧度制
5. 组合多个旋转时要注意顺序，旋转不是可交换的
*/