# StudyNoteBook
记录我的一些学习笔记

# 2025年7月8日 CUDA-SIFT和老代码的使用
最近在做基于无人机俯拍图像的汽车违停检测。考虑到无人机位置和速度数据2s才能返回一次，且可能与图像存在时间不对齐，因此考虑采用纯视觉对前后帧中地面的位移进行估计。
具体来说，使用特征匹配的方法，对地面特征点的像素位移进行估计。
我测试了ORB算子和SIFT算子，发现SIFT效果好于ORB，具体而言，是特征点的数量和匹配精度要高很多。但SIFT的缺点在于，使用python-opencv库时，即使用了高性能CPU，特征提取和匹配的速度依然很慢，多帧平均耗时0.77s，是一个准实时应用（每秒10帧）不可接受的速度。
因此，考虑使用之前用过的基于GPU的SIFT（在2022年做长短焦相机图像匹配时用过），可以实现非常高的计算速度。资源地址为https://github.com/Celebrandil/CudaSift
该项目虽然是纯C++构建的项目，但应该也可以使用Python加插件的方式使用。这次是不得不做这件事（之前觉得很高端，但又懒得去用，因为没有刚需）。
之前也看过几个类似的基于GPU的SIFT资源，但这些资源都有一些问题，就是年代比较早，都是几年前甚至十几年前的，使用了python2和一些非常旧的库，而现在（2025年）已经是CUDA12.2和gcc-11，显卡也是4090（sm_89架构），所以兼容性非常不好，出现很多“可以cmake ..,但make阶段报错”的情况。
此次遇到的这个CUDA-SIFT是2022年做其他项目时在RTX3090显卡（CUDA版本我忘记了）成功运行过的，效果还可以，因此考虑使用这个资源来构建应用。
在cmake ..阶段没有出问题，但在make阶段遇到两个编译错误，一个是“sm_35”不兼容的问题，一个是“gcc-6”不兼容的问题，但都可以通过在CMakeLists.txt中修改来规避错误。

pypopsift资源，地址https://github.com/OpenDroneMap/pypopsift
该资源的使用也存在一些问题，在2025.07.08上午解决，目前可以正常编译通过。
问题有以下几点：
1. 4年前的资源，最多只能针对sm_86架构，即RTX30x0系列显卡，其原生代码对sm_89的显卡不支持。处理方法是重新写资源的顶层CMakeLists.txt，让其可以兼容sm_89架构。
2. 资源自带了一个pybind11，但这个pybind11是2.5版本，远远落后于现在的2.13（现在甚至有3.0.0版本），因此其针对的python版本也较低，无法支持python3.12/3.13这样的新版本，因此需要下载pybind11的新的release版本源码，替换其初始的低版本pybind11。
3. 部分代码存在bug，主要是thurst库的一些依赖没有显式包含，需要添加s_filtergrid.cu文件里面的#include内容。
经过以上几处修改，才能保证make成功。

经过几小时的折腾，pypopsift这个库最后还是没能正常工作。我打算尝试自己用pybind11从CUDASIFT资源制作可以被python调用的cuda包。
需要注意的是，在conda环境中，libstdc++.so.6 version `GLIBCXX_3.4.30' not found问题比较明显。因为c++版本的opencv会使用这个库，而C++版本的opencv是在非conda的本机环境使用的，所以需要让conda使用本机的libstdc++。这里使用一种方法。在$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh中增加export OLD_LD_PRELOAD="$LD_PRELOAD"和export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6:$LD_PRELOAD。在$CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh中增加export LD_PRELOAD="$OLD_LD_PRELOAD" unset OLD_LD_PRELOAD的方式，使得在使用conda activate和deactivate动作时，自动加入和去掉本机环境的libstdc++环境。

# 2025年7月11日 基于CUDASIFT的python动态库制作
经过一周的折腾，终于尘埃落定，使用https://github.com/Celebrandil/CudaSift的CUDA-SIFT代码作为基础，制作了一个没有封装成类的python动态库。（后续我会找时间把做好的新代码放进github仓库。）
1. 如果制作成类，可以实现显存的重复使用，但也可能涉及到构造、析构、内存释放等问题，比较复杂。虽然我很想把CUDA显存做最大化利用，少做开辟和回收的动作，但能力有限、时间有限，只能做成每次计算都开辟和回收显存的程序。
2. 下面介绍pybind11构建动态库，主要改动只有以下几方面：增加pybind11库、修改main.cpp文件、修改CMakeLists.txt文件，以及增加"xxx_wrapper.cpp"绑定文件。其他程序原有的头文件和源文件完全都没动，也不需要增加CUDA相关的设置。
3. 在原代码目录中增加了pybind11的文件夹，可以放进external目录里面提示其性质是外部库。
4. 修改CMakeLists.txt文件：在里面增加包含external/pybind11的内容，使编译器可以找到pybind11，修改生成目标文件类型，不生成可执行程序，而是生成链接库。构建的过程与构建纯C++的动态库类似，代码里不能存在main()主函数，只有被python绑定的函数。构建出来的文件是一个.so的动态库文件，这个文件可以被python导入。构建的动态库名称可能与函数名称不同，可以在被python代码import之后，采用object.function()的方式调用。
5. 在.cpp源文件目录中补充一个"xxx_wrapper.cpp"的文件，将一个C++函数绑定到一个python函数上实现，代码主要还是C++写的。里面需要用到一些数据类型转换的函数，比如将C++的cv::Mat转化为uint8，比如将C++ std::vectors转化为python list的函数。
如果绑定的python函数有多个返回值，可以在C++代码中使用std::tuple的类型来将多个变量合成一个元组。
6. 值得一提的是，C++暴露给pybind11的接口，最好使用比较简单的数据类型，如果使用cv::Vec2f等，会容易出错，比较麻烦，就不如使用vector<vector<float>>这种了。
7. 关于SIFT算法本身的实现，CUDASIFT更偏向底层，所有参数都暴露给用户，需要自己调整，比如各种阈值，看起来眼花缭乱的，与python-opencv那种开箱即用的傻瓜函数包完全不同。我使用1080p的图像，很多时候可以提取很多特征，但无法匹配成功，后来发现重新提取特征可能会提升匹配成功的概率，很玄学。估计是某些算法的初始值有一定的随机性，每次执行同一操作可能有不同的结果。
8. 将SIFT的图像分辨率调低之后，匹配点的数量和计算速度都有了一定的提升。目前效果比较好的是1080p的一半分辨率，即960x540,大多数情况可以特征匹配成功。调用函数耗时基本在30ms以内（30ms的情况是连续5次特征提取和匹配都失败的情况，否则可达到20ms以内）。
9. 这次工程的经历，让我体会到使用chatgpt和copilot的威力。chatgpt可以搞定很多一两百行的代码，也可以帮助理解很多编译和运行的报错，直接把报错代码贴进去，80%以上的情况可以找到问题。copilot则是可以提升重复性比较高的代码的编写。如果不使用这两个工具，我可能一个月也做不出来这个4天搞出来的工程。
