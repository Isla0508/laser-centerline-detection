import cv2
import numpy as np
from skimage.morphology import skeletonize
import os


# ============================================================
# 1. 显示图片的工具函数
# ============================================================

def show_image(image, title, duration=2000):
    """
    用较大的窗口显示图片

    参数：
        image    : 要显示的图片
        title    : 窗口标题
        duration : 显示时间，单位 ms
                   例如 2000 = 2 秒
    """

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)

    # 设置较大的窗口
    cv2.resizeWindow(title, 1800, 1100)

    cv2.imshow(title, image)

    # 自动等待指定时间
    cv2.waitKey(duration)

    cv2.destroyWindow(title)


# ============================================================
# 2. 创建标题画面
# ============================================================

def create_title_screen(title, subtitle=""):
    """
    创建项目演示标题画面
    """

    canvas = np.ones((800, 1400, 3), dtype=np.uint8) * 255

    # 主标题
    cv2.putText(
        canvas,
        title,
        (120, 330),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.0,
        (40, 80, 140),
        4,
        cv2.LINE_AA
    )

    # 副标题
    if subtitle != "":
        cv2.putText(
            canvas,
            subtitle,
            (120, 430),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (80, 80, 80),
            2,
            cv2.LINE_AA
        )

    return canvas


# ============================================================
# 3. 核心算法
# ============================================================

def process_image(image_path, csv_path, result_path):
    """
    对一张图片进行激光中心线提取

    理论图和真实图都使用这一份函数，
    因此两阶段使用完全相同的算法和参数。
    """

    print("\n" + "=" * 60)
    print("正在处理：", image_path)
    print("=" * 60)

    # --------------------------------------------------------
    # 读取图片
    # --------------------------------------------------------

    img = cv2.imread(image_path)

    if img is None:
        print("❌ 无法读取图片：", image_path)
        return None

    # 保存原始图片
    original = img.copy()

    # --------------------------------------------------------
    # BGR → HSV
    # --------------------------------------------------------

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # --------------------------------------------------------
    # 提取红色区域
    #
    # 红色在 HSV 中可能出现在：
    # 0°附近
    # 180°附近
    # --------------------------------------------------------

    lower_red_1 = np.array([0, 80, 80])
    upper_red_1 = np.array([10, 255, 255])

    lower_red_2 = np.array([170, 80, 80])
    upper_red_2 = np.array([180, 255, 255])

    mask_red_1 = cv2.inRange(
        hsv,
        lower_red_1,
        upper_red_1
    )

    mask_red_2 = cv2.inRange(
        hsv,
        lower_red_2,
        upper_red_2
    )

    # 合并两个红色范围
    mask_red = cv2.bitwise_or(
        mask_red_1,
        mask_red_2
    )

    # --------------------------------------------------------
    # 分离 B、G、R 通道
    # --------------------------------------------------------

    b, g, r = cv2.split(img)

    # --------------------------------------------------------
    # 计算红色优势：
    #
    # R - max(G, B)
    #
    # 数值越大，说明该像素越偏红
    # --------------------------------------------------------

    max_gb = np.maximum(g, b)

    red_difference = (
        r.astype(np.int16)
        -
        max_gb.astype(np.int16)
    )

    # --------------------------------------------------------
    # 获取亮红色激光区域
    #
    # 条件：
    # 1. 必须属于红色区域
    # 2. R 通道 > 180
    # 3. 红色明显强于 G/B
    # --------------------------------------------------------

    bright_red = (
        (mask_red > 0)
        &
        (r > 180)
        &
        (red_difference > 80)
    )

    laser_core = (
        bright_red.astype(np.uint8) * 255
    )

    # --------------------------------------------------------
    # 形态学闭运算
    #
    # 作用：
    # 填补激光区域中的一些小断裂
    # --------------------------------------------------------

    kernel = np.ones((3, 3), np.uint8)

    laser_core = cv2.morphologyEx(
        laser_core,
        cv2.MORPH_CLOSE,
        kernel
    )

    # ========================================================
    # 4. 连通区域分析
    # ========================================================

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            laser_core,
            connectivity=8
        )
    )

    # --------------------------------------------------------
    # 如果没有检测到任何区域
    # --------------------------------------------------------

    if num_labels <= 1:

        print("❌ 没有检测到有效的激光区域")

        empty_skeleton = np.zeros_like(laser_core)

        return {
            "original": original,
            "mask_red": mask_red,
            "laser_core": laser_core,
            "main_laser": laser_core,
            "skeleton_mask": empty_skeleton,
            "final_result": original.copy(),
            "coordinates_x": [],
            "coordinates_y": []
        }

    # --------------------------------------------------------
    # 找面积最大的连通区域
    # --------------------------------------------------------

    largest_label = 1

    largest_area = stats[
        1,
        cv2.CC_STAT_AREA
    ]

    for i in range(2, num_labels):

        area = stats[
            i,
            cv2.CC_STAT_AREA
        ]

        if area > largest_area:

            largest_area = area
            largest_label = i

    # --------------------------------------------------------
    # 提取最大连通区域
    # --------------------------------------------------------

    main_laser = np.zeros_like(
        laser_core
    )

    main_laser[
        labels == largest_label
    ] = 255

    print(
        "检测到连通区域数量：",
        num_labels - 1
    )

    print(
        "最大激光区域面积：",
        largest_area
    )

    # ========================================================
    # 5. 骨架化
    # ========================================================

    # 转换成布尔图
    binary = main_laser > 0

    # 骨架化
    skeleton = skeletonize(
        binary
    )

    # 转换成 OpenCV 图像
    skeleton_mask = (
        skeleton.astype(np.uint8)
        * 255
    )

    # 让骨架更加连续
    skeleton_mask = cv2.morphologyEx(
        skeleton_mask,
        cv2.MORPH_CLOSE,
        np.ones((2, 2), np.uint8)
    )

    # ========================================================
    # 6. 提取中心线坐标
    # ========================================================

    y_coords, x_coords = np.where(
        skeleton_mask > 0
    )

    print(
        "中心线坐标数量：",
        len(x_coords)
    )

    # --------------------------------------------------------
    # 保存 CSV
    #
    # 不使用 pandas
    # --------------------------------------------------------

    with open(
        csv_path,
        "w",
        encoding="utf-8-sig"
    ) as f:

        f.write("x,y\n")

        for x, y in zip(
            x_coords,
            y_coords
        ):

            f.write(
                f"{x},{y}\n"
            )

    print(
        "中心线数据已保存：",
        csv_path
    )

    # ========================================================
    # 7. 生成最终结果图
    # ========================================================

    final_result = original.copy()

    # --------------------------------------------------------
    # 注意：
    #
    # 这里仅仅是为了“显示”时让绿色中心线更加明显。
    #
    # 实际的 skeleton_mask 没有被修改。
    # CSV 保存的仍然是真实骨架坐标。
    # --------------------------------------------------------

    display_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)
    )

    thick_skeleton = cv2.dilate(
        skeleton_mask,
        display_kernel
    )

    # --------------------------------------------------------
    # 把中心线显示成绿色
    # --------------------------------------------------------

    green_mask = (
        thick_skeleton > 0
    )

    final_result[
        green_mask
    ] = (
        0,
        255,
        0
    )

    # ========================================================
    # 8. 保存最终图片
    # ========================================================

    cv2.imwrite(
        result_path,
        final_result
    )

    print(
        "最终结果已保存：",
        result_path
    )

    # ========================================================
    # 9. 返回结果
    # ========================================================

    return {
        "original": original,
        "mask_red": mask_red,
        "laser_core": laser_core,
        "main_laser": main_laser,
        "skeleton_mask": skeleton_mask,
        "final_result": final_result,
        "coordinates_x": x_coords,
        "coordinates_y": y_coords
    }


# ============================================================
# 10. 播放一个阶段
# ============================================================

def play_stage(result, stage_name):
    """
    播放某个阶段的检测过程
    """

    if result is None:
        return

    # --------------------------------------------------------
    # 1. 原始图片
    # --------------------------------------------------------

    show_image(
        result["original"],
        stage_name + " - Original Image",
        duration=1800
    )

    # --------------------------------------------------------
    # 2. 红色区域
    # --------------------------------------------------------

    show_image(
        result["mask_red"],
        stage_name + " - Red Mask",
        duration=1800
    )

    # --------------------------------------------------------
    # 3. 激光核心
    # --------------------------------------------------------

    show_image(
        result["laser_core"],
        stage_name + " - Laser Core",
        duration=1800
    )

    # --------------------------------------------------------
    # 4. 最终中心线
    # --------------------------------------------------------

    show_image(
        result["final_result"],
        stage_name + " - Final Result",
        duration=3000
    )


# ============================================================
# 11. 创建过渡页面
# ============================================================

def create_transition_screen(
    title,
    subtitle=""
):

    canvas = np.ones(
        (800, 1400, 3),
        dtype=np.uint8
    ) * 255

    cv2.putText(
        canvas,
        title,
        (120, 330),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.8,
        (40, 80, 140),
        3,
        cv2.LINE_AA
    )

    if subtitle != "":

        cv2.putText(
            canvas,
            subtitle,
            (120, 430),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            (80, 80, 80),
            2,
            cv2.LINE_AA
        )

    return canvas


# ============================================================
# 12. 最终总结页面
# ============================================================

def create_final_summary():

    canvas = np.ones(
        (800, 1400, 3),
        dtype=np.uint8
    ) * 255

    # --------------------------------------------------------
    # 标题
    # --------------------------------------------------------

    cv2.putText(
        canvas,
        "Laser Centerline Detection Completed",
        (120, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.55,
        (40, 80, 140),
        3,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # 第一步
    # --------------------------------------------------------

    cv2.putText(
        canvas,
        "1. Extract red laser region",
        (150, 320),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (70, 70, 70),
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # 第二步
    # --------------------------------------------------------

    cv2.putText(
        canvas,
        "2. Keep the main connected laser region",
        (150, 390),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (70, 70, 70),
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # 第三步
    # --------------------------------------------------------

    cv2.putText(
        canvas,
        "3. Skeletonize the laser region",
        (150, 460),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (70, 70, 70),
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # 第四步
    # --------------------------------------------------------

    cv2.putText(
        canvas,
        "4. Obtain the laser centerline coordinates",
        (150, 530),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (70, 70, 70),
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # 最终强调
    # --------------------------------------------------------

    cv2.putText(
        canvas,
        "Same algorithm -> Test Image -> Real Laser Image",
        (150, 650),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        (40, 130, 70),
        2,
        cv2.LINE_AA
    )

    return canvas


# ============================================================
# 13. 主程序
# ============================================================

def main():

    print()
    print("=" * 70)
    print("       基于计算机视觉的激光条纹中心线提取演示")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # 图片路径
    # --------------------------------------------------------

    theory_path = "images/test.jpg"
    real_path = "images/real.jpg"

    # --------------------------------------------------------
    # 检查理论图
    # --------------------------------------------------------

    if not os.path.exists(
        theory_path
    ):

        print(
            "❌ 找不到理论测试图：",
            theory_path
        )

        print(
            "请确认 images 文件夹中存在 test.jpg"
        )

        return

    # --------------------------------------------------------
    # 检查真实图
    # --------------------------------------------------------

    if not os.path.exists(
        real_path
    ):

        print(
            "❌ 找不到真实激光图：",
            real_path
        )

        print(
            "请确认 images 文件夹中存在 real.jpg"
        )

        return

    # ========================================================
    # 开场
    # ========================================================

    intro = create_title_screen(
        "Laser Centerline Detection",
        "Computer Vision Based Structured-Light Processing"
    )

    show_image(
        intro,
        "Project Introduction",
        duration=2500
    )

    # ========================================================
    # 第一阶段：理论 / 测试图
    # ========================================================

    print()
    print("【第一阶段】理论 / 测试图")
    print("开始检测...")

    theory_result = process_image(
        theory_path,
        "theory_centerline.csv",
        "theory_result.jpg"
    )

    # 播放理论图检测过程
    play_stage(
        theory_result,
        "Theory Image"
    )

    # ========================================================
    # 过渡
    # ========================================================

    transition = create_transition_screen(
        "Stage 1 Completed",
        "Applying the SAME algorithm to a real laser image..."
    )

    show_image(
        transition,
        "Transition",
        duration=2200
    )

    # ========================================================
    # 第二阶段：真实激光图
    #
    # 这里仍然调用完全相同的 process_image()
    # ========================================================

    print()
    print("【第二阶段】真实激光图")
    print("开始检测...")

    real_result = process_image(
        real_path,
        "real_centerline.csv",
        "real_result.jpg"
    )

    # 播放真实图检测过程
    play_stage(
        real_result,
        "Real Laser Image"
    )

    # ========================================================
    # 最终总结
    # ========================================================

    final_summary = create_final_summary()

    cv2.namedWindow(
        "Final Summary",
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        "Final Summary",
        1800,
        1100
    )

    cv2.imshow(
        "Final Summary",
        final_summary
    )

    # --------------------------------------------------------
    # 控制台输出
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("演示完成！")
    print("=" * 70)

    print(
        "理论图结果：theory_result.jpg"
    )

    print(
        "理论图中心线：theory_centerline.csv"
    )

    print(
        "真实图结果：real_result.jpg"
    )

    print(
        "真实图中心线：real_centerline.csv"
    )

    print()
    print("按任意键退出程序...")

    # 最后停住
    cv2.waitKey(0)

    # 关闭所有窗口
    cv2.destroyAllWindows()


# ============================================================
# 14. 程序入口
# ============================================================

if __name__ == "__main__":
    main()