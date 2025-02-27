---
layout: default
title: Eigen 教程
permalink: /
---

# Eigen3 教程 （施工中）

这是一个 Eigen C++ 线性代数库的教程项目，包含了从基础到高级的示例代码。

这个教程与[Eigen 官方文档](https://eigen.tuxfamily.org/dox/) 的区别是什么？

- 官方文档是英文的，这个教程是中文的。
- 官方文档是按照Eigen 的源码顺序组织的，这个教程是按照实际应用的顺序组织的。
- 官方文档是面向Eigen 的开发者或者想要深入了解Eigen 的实现原理的读者，这个教程是面向需要使用Eigen 库解决实际问题的读者。
- 应用中的实际问题和解决方案都采用实际的代码进行测试和说明，例如POD、Kalman滤波等实际代码都是有具体的背景和应用场景，本教程尽量做到与实际应用相结合。


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
- 访问 [Eigen 官网下载页面](https://eigen.tuxfamily.org/index.php?title=Main_Page#Download)
- 下载[最新版本](https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip)
- 解压到合适的位置，例如`D:\Eigen3`

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

### Windows 下微软编译器

采用微软编译器编译Eigen 时，同样需要指定头文件位置，运行`vcvarsall.bat`脚本设置环境变量，然后调用`cl`命令编译。

```powershell
vcvarsall.bat
cl eigen3version.cpp /I "D:\Eigen3\eigen-3.4.0"
```

因为我们采用`cmake`编译，所以需要修改`CMakeLists.txt`文件，所以在Windows下面基本上也都是一样一样的。


## 项目总体结构


```
eigen3tutorial/
├── chapters/
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


本项目的结构设计旨在帮助用户快速找到所需的示例代码和文档。项目的根目录包含以下主要部分：

- `chapters/`: 包含每个章节的详细内容和示例代码。
- `CMakeLists.txt`: 项目的 CMake 配置文件，用于构建和编译示例代码。
- `eigen3version.cpp`: 一个简单的程序，用于检查 Eigen 库的版本。
- `README.md`: 项目的主要文档，提供了安装、配置和使用的指南。
- `chp01/` 至 `chp07/`: 每个子目录对应一个特定的主题或功能模块，包含相关的示例代码文件。

每个章节目录（如 `chp01/`）中包含多个 C++ 源文件，这些文件展示了如何使用 Eigen 库解决特定的线性代数问题。通过这种结构，用户可以轻松地导航到感兴趣的主题并查看相关的代码示例。

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


## 其他注意事项

1. 包含路径：
   - 如果使用系统安装的 Eigen：`#include <Eigen/...>`
   - 如果使用自定义路径：可能需要调整 CMakeLists.txt 中的 include_directories

2. 编译优化：
   - 建议开启 -O2 或 -O3 优化
   - SIMD 优化选项：
     * -march=native （自动检测并启用当前CPU支持的所有指令集）
     * -msse4.2 （启用 SSE4.2 指令集）
     * -mavx （启用 AVX 指令集）
     * -mavx2 （启用 AVX2 指令集）
     * -mfma （启用 FMA 指令集）

3. 内存对齐：
   - Eigen 默认使用 16 字节对齐
   - 在类中使用固定大小的 Eigen 对象时，需要使用 EIGEN_MAKE_ALIGNED_OPERATOR_NEW

## 许可

本项目采用 MIT 许可证。详细信息请查看 [LICENSE](LICENSE) 文件。

您可以自由地：
- 使用、复制、修改、合并、出版发行、散布、再授权及贩售本软件的副本。
- 不需要为修改后的软件做特别标注。

但必须：
- 在软件和软件的所有副本中都必须包含以上版权声明和本许可声明。

## 参考资料

- [Eigen 官方文档](https://eigen.tuxfamily.org/dox/)
