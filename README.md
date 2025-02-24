# Eigen 教程

这是一个 Eigen C++ 线性代数库的教程项目，包含了从基础到高级的示例代码。

## 环境要求

- C++11 或更高版本
- CMake 3.10 或更高版本
- Eigen 3.4 或更高版本

### 查看 Eigen 版本
```bash
# 使用 pkg-config
pkg-config --modversion eigen3

# 或者查看头文件
grep "#define EIGEN_.*_VERSION" /usr/include/eigen3/Eigen/src/Core/util/Macros.h
```

文档附带的源代码中包括了一个`eigen3version.cpp`文件，正常能够编译运行的话， 会在终端打印Eigen 的版本。这一方面可以确认版本，另一方面可以确认Eigen 和相应工具链的配置是正确的。

## 安装 Eigen

### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install libeigen3-dev
```

### macOS
```bash
brew install eigen
```

### Windows
1. 使用 vcpkg:
```bash
vcpkg install eigen3:x64-windows
```

2. 或者手动下载:
- 访问 [Eigen 官网](https://eigen.tuxfamily.org/)
- 下载最新版本
- 解压到合适的位置

### 配置环境变量

安装之后，需要配置环境变量，否则在编译时会找不到Eigen 的头文件。默认采用`apt`安装的Eigen 头文件位置为`/usr/include/eigen3`。因此，我们在编译时可以直接使用`-I/usr/include/eigen3`指定头文件位置。

```bash
g++ -I/usr/include/eigen3 eigen3version.cpp -o eigen3version
./eigen3version
```

如果能够正常打印Eigen 的版本，则说明Eigen 安装成功。

如果不希望每次编译都需要指定头文件位置，可以配置环境变量。打开`.bashrc`文件，添加以下内容：

```bash
vim ~/.bashrc
```

在文件末尾添加以下内容：

```bash
export EIGEN_ROOT=/usr/include/eigen3
export CPLUS_INCLUDE_PATH=$EIGEN_ROOT:$CPLUS_INCLUDE_PATH
```

保存文件并退出。然后重新加载`.bashrc`文件：

```bash
source ~/.bashrc
```

这样我们就可以在编译时直接使用`-I/usr/include/eigen3`指定头文件位置，而不需要每次都指定。当然，Eigen最好的地方（个人观点）也就是，它只有头文件，没有库文件，所以不需要链接和设置链接选项。

## 项目结构

```
eigen-tutorial/
├── CMakeLists.txt
├── eigen3version.cpp
├── README.md
├── chp01/          # 基础知识
│   ├── basic_matrix.cpp
│   ├── matrix_arithmetic.cpp
│   └── special_matrices.cpp
├── chp02/          # 矩阵操作
│   ├── matrix_operations.cpp
│   └── matrix_decompositions.cpp
├── chp03/          # 向量操作
│   ├── vector_operations.cpp
│   └── vector_advanced.cpp
├── chp04/          # 线性方程
│   ├── linear_equations.cpp
│   └── least_squares.cpp
├── chp05/          # 特征值
│   ├── eigenvalues.cpp
│   └── pca_example.cpp
├── chp06/          # 几何变换
│   ├── transforms.cpp
│   └── rotations.cpp
└── chp07/          # 高级特性
    ├── sparse_matrix.cpp
    └── advanced_features.cpp
```

## 编译和运行

1. 创建构建目录：
```bash
mkdir build
cd build
```

2. 配置 CMake：
```bash
cmake ..
```

如果 Eigen 安装在非标准位置，需要指定 Eigen3_DIR：
```bash
cmake .. -DEigen3_DIR=/path/to/eigen3/share/eigen3/cmake
```

3. 编译：
```bash
cmake --build .
```

4. 运行示例：
```bash
./basic_matrix
./matrix_arithmetic
# ... 其他示例
```

## 文档结构

- `chp01.md`: 基础知识，包括数据类型和基本操作
- `chp02.md`: 矩阵操作和分解
- `chp03.md`: 向量操作和高级特性
- `chp04.md`: 线性方程和最小二乘
- `chp05.md`: 特征值和主成分分析
- `chp06.md`: 几何变换和旋转
- `chp07.md`: 高级特性和稀疏矩阵

## 注意事项

1. 包含路径：
   - 如果使用系统安装的 Eigen：`#include <Eigen/...>`
   - 如果使用自定义路径：可能需要调整 CMakeLists.txt 中的 include_directories

2. 编译优化：
   - 建议开启 -O2 或 -O3 优化
   - 可以使用 -march=native 开启 SIMD 优化

3. 内存对齐：
   - Eigen 默认使用 16 字节对齐
   - 在类中使用固定大小的 Eigen 对象时，需要使用 EIGEN_MAKE_ALIGNED_OPERATOR_NEW

## 许可

本教程采用 MIT 许可证。

## 参考资料

- [Eigen 官方文档](https://eigen.tuxfamily.org/dox/)
- [Eigen 用户手册](https://eigen.tuxfamily.org/dox/UserManual_Index.html)
- [Eigen 快速参考](https://eigen.tuxfamily.org/dox/QuickReference_Index.html)