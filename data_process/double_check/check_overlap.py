import numpy as np
from shapely.geometry import Polygon
from astropy.coordinates import SkyCoord
import astropy.units as u
import pdb
from tqdm import tqdm
def check_overlap(train_polygons, test_polygons, threshold=3 * u.arcsec):
    """
    检查训练集和测试集的 Polygon 对象是否有重叠。

    参数:
    - train_polygons: 训练集的 Polygon 对象列表
    - test_polygons: 测试集的 Polygon 对象列表
    - threshold: 角距离阈值，默认为 1 角秒

    返回:
    - 有重叠的 Polygon 对
    """
    # 将训练集和测试集的坐标转换为 SkyCoord 对象，避免重复计算
    train_coords_list = []
    for train_polygon in train_polygons:
        train_ra = np.array(train_polygon.exterior.coords.xy[0])
        train_dec = np.array(train_polygon.exterior.coords.xy[1])
        train_coords = SkyCoord(ra=train_ra * u.deg, dec=train_dec * u.deg, frame='icrs')
        train_coords_list.append(train_coords)

    test_coords_list = []
    for test_polygon in test_polygons:
        test_ra = np.array(test_polygon.exterior.coords.xy[0])
        test_dec = np.array(test_polygon.exterior.coords.xy[1])
        test_coords = SkyCoord(ra=test_ra * u.deg, dec=test_dec * u.deg, frame='icrs')
        test_coords_list.append(test_coords)

    overlaps = []
    for train_coords in tqdm(train_coords_list):
        for test_coords in test_coords_list:
            # 批量计算角距离
            sep_matrix = train_coords.separation(test_coords[:, np.newaxis])

            # 检查是否有角距离小于阈值的点
            if np.any(sep_matrix < threshold):
                print(f"Found overlapping polygons: {train_coords} and {test_coords}")
                overlaps.append((train_coords, test_coords))
    return overlaps

def main():
    # 加载保存的 Polygon 对象
    train_polygons = np.load("train_ra_dec.npy", allow_pickle=True)
    test_polygons = np.load("test_ra_dec.npy", allow_pickle=True)

    # 检查重叠
    overlaps = check_overlap(train_polygons, test_polygons, threshold=3 * u.arcsec)

    # 输出结果
    if overlaps:
        print(f"Found {len(overlaps)} overlapping pairs.")
        for i, (train_coords, test_coords) in enumerate(overlaps):
            print(f"Overlap {i + 1}:")
            print(f"Train Polygon: {train_coords}")
            print(f"Test Polygon: {test_coords}")
            print()
    else:
        print("No overlapping pairs found.")

if __name__ == "__main__":
    main()