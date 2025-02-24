# Eigen 教程

# 第三章：向量操作

## 类层次结构

> 本章介绍向量运算相关的类。相关内容请参见：
> - 几何变换：[第六章](chp06.md#类层次结构)
> - 矩阵运算：[第二章](chp02.md#类层次结构)
> - 稀疏向量：[第七章](chp07.md#类层次结构)

```mermaid
classDiagram
    class VectorBase~Derived~ {
        <<interface>>
        +size() int
        +norm() Scalar
        +normalize() void
    }
    
    class Vector~Scalar,Size~ {
        +Vector()
        +Vector(int)
        +dot(Vector) Scalar
        +cross(Vector) Vector
    }
    
    class ArrayVector~Scalar,Size~ {
        +ArrayVector()
        +ArrayVector(int)
        +array() Array
        +matrix() Vector
    }
    
    class Quaternion~Scalar~ {
        +w() Scalar
        +x() Scalar
        +y() Scalar
        +z() Scalar
        +normalize() void
    }
    
    VectorBase <|-- Vector
    VectorBase <|-- ArrayVector
    VectorBase <|-- Quaternion
}

### 类说明
1. VectorBase：向量基类
   - 定义了基本的向量操作接口
   - 使用 CRTP 模式实现静态多态
   - 继承自[第一章](chp01.md#类层次结构)的 MatrixBase

2. Vector：标准向量类
   - 支持固定大小和动态大小
   - 提供点积、叉积等运算
   - 详细几何运算见[第六章](chp06.md#基本变换)

3. ArrayVector：数组式向量
   - 支持元素级运算
   - 可与标准向量互相转换
   - 适用于数值计算和信号处理

4. Quaternion：四元数
   - 用于表示3D旋转
   - 提供旋转相关操作
   - 详细旋转操作见[第六章](chp06.md#三维旋转)

## 3.1 向量的基本操作
### 向量的创建和初始化
```cpp
Vector3d v1(1, 2, 3);           // 固定大小向量
VectorXd v2(5);                 // 动态大小向量
v2 << 1, 2, 3, 4, 5;           // 使用逗号初始化
Vector3d v3 = Vector3d::Zero(); // 零向量
Vector3d v4 = Vector3d::Ones(); // 全1向量
Vector3d v5 = Vector3d::Random(); // 随机向量
```

### 向量的访问
```cpp
double x = v1(0);    // 访问第一个元素
v1[1] = 4;          // 修改第二个元素
auto head = v2.head(3);  // 获取前3个元素
auto tail = v2.tail(2);  // 获取后2个元素
```

### 3.1.1 向量代数运算
- 加减法：O(n) 复杂度
- 点积：O(n) 复杂度
- 叉积：O(1) 仅适用于3D向量
- 范数计算：O(n) 复杂度

## 3.2 向量运算
### 基本运算
- 向量加减：`v1 + v2`, `v1 - v2`
- 标量乘除：`2 * v1`, `v1 / 2`
- 点积：`v1.dot(v2)` 或 `v1.transpose() * v2`
- 叉积：`v1.cross(v2)`（仅适用于3D向量）

### 范数计算
```cpp
double norm = v1.norm();        // 欧几里得范数（L2范数）
double squaredNorm = v1.squaredNorm();  // 范数的平方
double normL1 = v1.lpNorm<1>(); // L1范数
double normInf = v1.lpNorm<Infinity>(); // 无穷范数
```

### 向量归一化
```cpp
Vector3d normalized = v1.normalized(); // 返回归一化后的向量
v1.normalize();                        // 原地归一化
```

## 3.3 高级操作
### 向量投影
```cpp
// v1 在 v2 上的投影
Vector3d proj = v2 * (v1.dot(v2) / v2.squaredNorm());
```

### 旋转向量
```cpp
// 使用旋转矩阵
Matrix3d R = AngleAxisd(M_PI/4, Vector3d::UnitZ()).matrix();
Vector3d rotated = R * v1;
```

### 向量插值
```cpp
// 线性插值
double t = 0.5;  // 插值参数 [0,1]
Vector3d interpolated = (1-t) * v1 + t * v2;
```

## 3.4 代码示例说明
### vector_operations.cpp
```cpp
#include <Eigen/Dense>
using namespace Eigen;

int main() {
    // 1. 向量创建和初始化
    Vector3d v1(1.0, 2.0, 3.0);           // 直接初始化
    Vector3d v2 = Vector3d::Random();      // 随机向量
    Vector3d v3 = Vector3d::Zero();        // 零向量
    Vector3d v4 = Vector3d::Ones();        // 全1向量
    Vector3d v5 = Vector3d::UnitX();       // 单位向量 [1,0,0]
    
    // 2. 基本运算
    Vector3d sum = v1 + v2;                // 向量加法
    Vector3d diff = v1 - v2;               // 向量减法
    Vector3d scaled = 2.0 * v1;            // 标量乘法
    
    // 3. 点积和叉积
    double dot_product = v1.dot(v2);       // 点积
    Vector3d cross_product = v1.cross(v2); // 叉积（仅适用于3D向量）
    
    // 4. 范数计算
    double norm = v1.norm();               // 欧几里得范数（L2范数）
    double squaredNorm = v1.squaredNorm(); // 范数的平方
    double manhattan = v1.lpNorm<1>();     // L1范数（曼哈顿距离）
    
    return 0;
}
```

代码分析：
1. 向量创建和初始化：
   - Vector3d 是最常用的3D向量类型
   - 提供多种静态工厂方法创建特殊向量
   - 支持直接构造和赋值初始化

2. 基本运算：
   - 操作符重载使向量运算直观自然
   - 编译期进行维度检查，避免运行时错误
   - 自动优化计算，避免临时对象创建

3. 积运算：
   - dot()：计算向量内积，返回标量
   - cross()：计算向量外积，返回新向量
   - 3D向量特有的操作，其他维度不支持

4. 范数计算：
   - norm()：计算欧几里得范数，开销较大
   - squaredNorm()：避免开方，性能更好
   - lpNorm<p>()：支持不同的范数定义

5. 性能考虑：
   - 固定大小向量性能更好
   - 避免不必要的拷贝和临时对象
   - 使用表达式模板优化计算

### vector_advanced.cpp
```cpp
#include <Eigen/Dense>
#include <Eigen/Geometry>
using namespace Eigen;

int main() {
    // 1. 向量投影
    Vector3d v1(1.0, 2.0, 3.0);
    Vector3d v2(0.0, 0.0, 1.0);
    Vector3d proj = v2 * (v1.dot(v2) / v2.squaredNorm());
    
    // 2. 向量旋转
    // 使用旋转矩阵
    Matrix3d R = AngleAxisd(M_PI/4, Vector3d::UnitZ()).matrix();
    Vector3d rotated1 = R * v1;
    
    // 使用四元数
    Quaterniond q(AngleAxisd(M_PI/4, Vector3d::UnitZ()));
    Vector3d rotated2 = q * v1;
    
    // 3. 向量插值
    double t = 0.5;  // 插值参数 [0,1]
    Vector3d lerp = (1-t) * v1 + t * v2;  // 线性插值
    Vector3d slerp = q.slerp(t, q).toRotationMatrix() * v1;  // 球面插值
    
    // 4. 向量归一化
    Vector3d normalized = v1.normalized();  // 返回归一化的向量
    v1.normalize();                         // 原地归一化
    
    return 0;
}
```

代码分析：
1. 向量投影：
   - 实现了向量v1到v2的正交投影
   - 使用点积和范数计算投影长度
   - 数值计算中需要注意除零问题

2. 向量旋转：
   - 提供两种旋转实现方式
   - 旋转矩阵：直观但内存占用大
   - 四元数：更紧凑，避免万向节死锁

3. 向量插值：
   - lerp：简单线性插值，计算快速
   - slerp：球面线性插值，保持角速度均匀
   - 参数t控制插值位置，范围[0,1]

4. 向量归一化：
   - normalized()：返回新的归一化向量
   - normalize()：原地归一化，更高效
   - 自动处理数值稳定性问题

5. 应用建议：
   - 根据精度要求选择合适的插值方法
   - 注意旋转表示方法的特点和限制
   - 考虑数值稳定性和计算效率

### 常见问题和解决方案
1. 数值精度：
   - 使用 double 而不是 float 提高精度
   - 归一化前检查向量长度避免除零
   - 使用稳定的算法计算几何特征

2. 性能优化：
   - 避免重复计算范数
   - 使用 squaredNorm() 代替 norm()
   - 合理使用表达式模板

3. 内存管理：
   - 使用固定大小向量避免动态分配
   - 避免不必要的向量拷贝
   - 利用表达式模板延迟求值

## 3.5 应用场景
1. 计算几何
   - 点的表示
   - 方向向量
   - 法向量

2. 物理模拟
   - 速度和加速度
   - 力和力矩
   - 动量和角动量

3. 图形学
   - 顶点位置
   - 纹理坐标
   - 颜色向量

### linear_equations.cpp
```cpp
#include <Eigen/Dense>
using namespace Eigen;

int main() {
    // 1. 解线性方程 Ax = b
    Matrix3d A;
    Vector3d b;
    
    // 初始化矩阵和向量
    A << 2, -1, 0,
        -1, 2, -1,
         0, -1, 2;
    b << 1, 0, 1;
    
    // 直接求解
    Vector3d x1 = A.colPivHouseholderQr().solve(b);
    
    // 使用LU分解求解
    Vector3d x2 = A.lu().solve(b);
    
    // 对称正定矩阵可以使用Cholesky分解
    Vector3d x3 = A.llt().solve(b);
    
    // 检查求解精度
    double relative_error = (A * x1 - b).norm() / b.norm();
    
    return 0;
}
```

代码分析：
1. 方程设置：
   - 构造三对角矩阵系统
   - 使用逗号初始化器设置值
   - 确保系统有唯一解

2. 求解方法：
   - colPivHouseholderQr(): 通用稳定求解器
   - lu(): LU分解，适用于方阵
   - llt(): Cholesky分解，仅适用于对称正定矩阵

3. 精度验证：
   - 计算相对残差
   - 使用向量范数评估误差
   - 验证解的质量

4. 性能考虑：
   - 选择合适的求解器很重要
   - 对称正定系统优先使用 llt()
   - 大型系统考虑使用迭代法

> 注：更多求解器相关内容请参见[第四章](chp04.md#类层次结构) 