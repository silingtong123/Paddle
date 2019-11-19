import tarfile, os
import sys
def untar(fname, dirs):
    """
    解压tar.gz文件
    :param fname: 压缩文件名
    :param dirs: 解压后的存放路径
    :return: bool
    """
    try:
        t = tarfile.open(name = fname, mode = 'r:gz')
        t.extractall(path = dirs)
        return True
    except Exception as e:
        print(e)
        return False

untar(sys.argv[1], sys.argv[2])