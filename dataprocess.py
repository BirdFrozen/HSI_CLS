import numpy as np


def onehot(data, n):
    """将标记图(每个像素值代该位置像素点的类别)转换为onehot编码\\
    turn np array into onehot form
    """
    buf = np.zeros(data.shape + (n, ))
    nmsk = np.arange(data.size)*n + data.ravel()
    buf.ravel()[nmsk] = 1
    return buf


def GroundTruth2Mask(gt):
    gt = gt.convert('L')

    table = []  # 建立映射表
    threshold = 128
    for i in range(256):
        if i < threshold:
            table.append(0)
        else:
            table.append(1)

    # 图片二值化
    mask = gt.point(table, 'L')

    return mask
