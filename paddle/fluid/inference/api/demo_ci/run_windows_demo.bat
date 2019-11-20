@echo off
setlocal
set source_path=%~dp0
set build_path=%~dp0\build

setlocal enabledelayedexpansion

rem set gpu_inference
rem 请确认是否使用GPU预测库(Y/N)，默认为N，使用GPU预测库需要下载对应版本的预测库
SET /P gpu_inference="Use GPU_inference_lib or not(Y/N), default: N   =======>"
IF /i "%gpu_inference%"=="y" (
  SET gpu_inference=Y
) else (
  SET gpu_inference=N
)

rem 请确认该预测库是否使用MKL（Y/N），默认为Y（默认使用MKL），需要下载对应版本的预测库
SET /P use_mkl="Use MKL or not (Y/N), default: Y   =======>"
if /i "%use_mkl%"=="N" (
  set use_mkl=N
) else (
  set use_mkl=Y
)

:set_paddle_infernece_lib
rem 请输入paddle预测库路径，例如D:\fluid_inference_install_dir
SET /P paddle_infernece_lib="Please input the path of paddle inference library, such as D:\fluid_inference_install_dir   =======>"

IF NOT EXIST "%paddle_infernece_lib%" (
echo "------------%paddle_infernece_lib% does not exist!!------------"
goto set_paddle_infernece_lib
)

IF "%use_mkl%"=="N" (
  IF NOT EXIST "%paddle_infernece_lib%\third_party\install\openblas" (
    echo "------------It's not a OpenBlas inference library------------"
    goto:eof
  )
) else (
  IF NOT EXIST "%paddle_infernece_lib%\third_party\install\mklml" (
    echo "------------It's not a MKL inference library------------"
    goto:eof
  )
)

:set_path_cuda
rem 请输入cuda libraries目录，例如D:\cuda\lib\x64
if /i "!gpu_inference!"=="Y" (
    SET /P cuda_lib_dir="Please input the path of cuda libraries, such as D:\cuda\lib\x64   =======>"
    IF NOT EXIST "!cuda_lib_dir!" (
        echo "------------%cuda_lib_dir% does not exist!!------------"
        goto set_path_cuda
    )
)

rem set_use_gpu
rem 请确认是否使用GPU进行预测，默认为N，使用GPU需要GPU的预测库
if /i "!gpu_inference!"=="Y" (
    SET /P use_gpu="Use GPU or not(Y/N), default: N   =======>"
)

if /i "%use_gpu%"=="Y" (
  set use_gpu=Y
) else (
  set use_gpu=N
)

rem set_path_vs_command_prompt 
rem 设置vs本机工作命令提示符的路径，建议使用x64
:set_vcvarsall_dir
SET /P vcvarsall_dir="Please input the path of visual studio command Prompt, such as C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat   =======>"
    
IF NOT EXIST "%vcvarsall_dir%" (
    echo "------------%vcvarsall_dir% does not exist!!------------"
    goto set_vcvarsall_dir
)

rem set_demo_name
rem 设置要编译的demo文件名前缀,默认为windows_mobilenet
:set_demo_name
SET /P demo_name="Please input the demo name, default: windows_mobilenet  =======>"
if   "%demo_name%"==""  set demo_name=windows_mobilenet
IF NOT EXIST "%source_path%\%demo_name%.cc" (
    echo "------------%source_path%\%demo_name%.cc does not exist!!------------"
    goto set_demo_name
)
if "%demo_name%"=="windows_mobilenet" set model_name=mobilenet
if "%demo_name%"=="vis_demo" set model_name=mobilenet
if "%demo_name%"=="simple_on_word2vec" set model_name=word2vec.inference.model
if "%demo_name%"=="trt_mobilenet_demo" (
  echo "The trt_mobilenet_demo need tensorRT inference library"
  if NOT exist "%paddle_infernece_lib%\third_party\install\tensorrt" (
    echo "------------It's not a tensorRT inference library------------" 
    goto:eof
  )
  set model_name=mobilenet
)
echo "=================================================================="
echo "use_gpu_inference=%gpu_inference%"
echo "use_mkl=%use_mkl%"
echo "use_gpu=%use_gpu%"

echo "paddle_infernece_lib=%paddle_infernece_lib%"
IF /i "%gpu_inference%"=="y" (
  echo "cuda_lib_dir=%cuda_lib_dir%"
)
echo "vs_vcvarsall_dir=%vcvarsall_dir%"
echo "demo_name=%demo_name%"
echo "===================================================================="
pause

rem download model
if NOT EXIST "%source_path%\%model_name%.tar.gz" (
  if "%model_name%"=="mobilenet" (
     call:download_model_mobilenet
  )
  if "%model_name%"=="word2vec.inference.model" (
     call:download_model_word2vec
  )
  if EXIST "%source_path%\%model_name%.tar.gz" (
  if NOT EXIST "%source_path%\%model_name%" (
    md %source_path%\%model_name%
    rem 请输入python.exe或者python3.exe的路径，例如C:\Python35\python.exe，默认会从系统的环境变量中寻找python.exe，
    rem 如果使用的为python3.exe，则需要设置具体路径。
    SET /P python_path="Please input the path of python.exe, such as C:\Python35\python.exe, C:\Python35\python3.exe   =======>"
    if "!python_path!"=="" (
      python  %source_path%\untar_model.py %source_path%\%model_name%.tar.gz %source_path%\%model_name%
    ) else (
      !python! %source_path%\untar_model.py %source_path%\%model_name%.tar.gz %source_path%\%model_name%
    )
  )
)
)

rem compile and run demo

if NOT EXIST "%build_path%" (
    md %build_path%
    cd %build_path%
) else (
   cd %build_path%
   rm -rf *
)

if /i "%use_mkl%"=="N" (
  set use_mkl=OFF
) else (
  set use_mkl=ON
)

if /i "%gpu_inference%"=="Y" (
    if  "%demo_name%"=="trt_mobilenet_demo" (
      cmake .. -G "Visual Studio 14 2015 Win64"  -T host=x64 -DWITH_GPU=ON ^
      -DWITH_MKL=%use_mkl% -DWITH_STATIC_LIB=ON -DCMAKE_BUILD_TYPE=Release -DDEMO_NAME=%demo_name% ^
      -DPADDLE_LIB="%paddle_infernece_lib%" -DMSVC_STATIC_CRT=ON -DCUDA_LIB="%cuda_lib_dir%" -DUSE_TENSORRT=ON
    ) else (
      cmake .. -G "Visual Studio 14 2015 Win64"  -T host=x64 -DWITH_GPU=ON ^
      -DWITH_MKL=%use_mkl% -DWITH_STATIC_LIB=ON -DCMAKE_BUILD_TYPE=Release -DDEMO_NAME=%demo_name% ^
      -DPADDLE_LIB="%paddle_infernece_lib%" -DMSVC_STATIC_CRT=ON -DCUDA_LIB="%cuda_lib_dir%"
    )
) else (
    cmake .. -G "Visual Studio 14 2015 Win64"  -T host=x64 -DWITH_GPU=OFF ^
    -DWITH_MKL=%use_mkl% -DWITH_STATIC_LIB=ON -DCMAKE_BUILD_TYPE=Release -DDEMO_NAME=%demo_name% ^
    -DPADDLE_LIB="%paddle_infernece_lib%" -DMSVC_STATIC_CRT=ON
)

call "%vcvarsall_dir%" amd64
msbuild /m /p:Configuration=Release %demo_name%.vcxproj

if /i "%use_gpu%"=="Y" (
  SET use_gpu=true
) else (
  SET use_gpu=false
)

if exist "%build_path%\Release\%demo_name%.exe" (
  cd %build_path%\Release 
  set GLOG_v=4
  if "%demo_name%"=="simple_on_word2vec" (
      %demo_name%.exe --dirname="%source_path%\%model_name%\%model_name%" --use_gpu="%use_gpu%"
  ) else (
    if "%demo_name%"=="windows_mobilenet" (
        %demo_name%.exe --modeldir="%source_path%\%model_name%\model" --use_gpu="%use_gpu%"
    ) else (
      if "%demo_name%"=="trt_mobilenet_demo" (
        %demo_name%.exe --modeldir="%source_path%\%model_name%\model" --data=%source_path%\%model_name%\data.txt ^
        --refer=%source_path%\%model_name%\result.txt
      ) else (
        %demo_name%.exe --modeldir="%source_path%\%model_name%\model" --data=%source_path%\%model_name%\data.txt ^
        --refer=%source_path%\%model_name%\result.txt --use_gpu="%use_gpu%"
      )
    )
  )
) else (
  echo "=========compilation fails!!=========="
)
echo.&pause&goto:eof

:download_model_mobilenet
powershell.exe (new-object System.Net.WebClient).DownloadFile('http://paddlemodels.bj.bcebos.com//inference-vis-demos/mobilenet.tar.gz', ^
'%source_path%\mobilenet.tar.gz')
goto:eof

:download_model_word2vec
powershell.exe (new-object System.Net.WebClient).DownloadFile('http://paddle-inference-dist.bj.bcebos.com/word2vec.inference.model.tar.gz', ^
'%source_path%\word2vec.inference.model.tar.gz')
goto:eof
