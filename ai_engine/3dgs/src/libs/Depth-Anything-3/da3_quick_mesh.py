#!/usr/bin/env python3
"""
从 Depth Anything 3 的稠密点云快速生成 Mesh（无需训练）

使用 Open3D 的 Poisson Surface Reconstruction，直接从 DA3 输出的
稠密点云生成三角网格。整个过程只需要 10-30 秒。

适用场景：
- 快速预览 3D 重建效果
- DA3 点云质量已经足够好，不需要 SuGaR 的复杂训练
- 需要快速迭代测试不同参数

用法:
    python da3_quick_mesh.py --input output/sugar_streaming/pcd/combined_pcd.ply
    python da3_quick_mesh.py --input output/sugar_streaming/pcd/combined_pcd.ply --depth 10 --density_threshold 0.01
"""

import os
import sys
import argparse
import time
import numpy as np

def quick_mesh_from_pointcloud(
    input_ply: str,
    output_dir: str = None,
    poisson_depth: int = 9,
    density_threshold_quantile: float = 0.02,
    simplify_target: int = None,
    estimate_normals: bool = True,
    normal_radius: float = 0.05,
    normal_max_nn: int = 30,
):
    """
    从点云快速生成 Mesh

    Args:
        input_ply: 输入点云 PLY 文件路径
        output_dir: 输出目录（默认与输入同目录）
        poisson_depth: Poisson 重建深度（6=粗糙快速, 9=标准, 11=精细慢速）
        density_threshold_quantile: 密度阈值分位数，去除低密度区域（0.01~0.1）
        simplify_target: 简化到目标三角形数量（None=不简化）
        estimate_normals: 是否估计法线（对 DA3 点云通常需要）
        normal_radius: 法线估计搜索半径
        normal_max_nn: 法线估计最大近邻数
    """
    import open3d as o3d

    t0 = time.time()

    if output_dir is None:
        output_dir = os.path.dirname(input_ply)
    os.makedirs(output_dir, exist_ok=True)

    # ==================== 加载点云 ====================
    print(f"加载点云: {input_ply}")
    pcd = o3d.io.read_point_cloud(input_ply)
    print(f"  点数: {len(pcd.points):,}")
    print(f"  有颜色: {pcd.has_colors()}")
    print(f"  有法线: {pcd.has_normals()}")

    if len(pcd.points) == 0:
        print("❌ 错误: 点云为空")
        return None

    # ==================== 预处理 ====================
    # 统计离群点移除
    print("\n去除离群点...")
    pcd_clean, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    removed = len(pcd.points) - len(pcd_clean.points)
    print(f"  移除了 {removed:,} 个离群点 ({removed/len(pcd.points)*100:.1f}%)")
    print(f"  剩余: {len(pcd_clean.points):,} 个点")

    # ==================== 法线估计 ====================
    if estimate_normals or not pcd_clean.has_normals():
        print("\n估计法线...")
        pcd_clean.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=normal_radius, max_nn=normal_max_nn
            )
        )
        # 法线方向一致化
        pcd_clean.orient_normals_consistent_tangent_plane(k=15)
        print("  ✅ 法线估计完成")

    # ==================== Poisson 重建 ====================
    print(f"\nPoisson 表面重建 (depth={poisson_depth})...")
    t1 = time.time()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_clean, depth=poisson_depth, width=0, scale=1.1, linear_fit=False
    )
    t_poisson = time.time() - t1
    print(f"  重建耗时: {t_poisson:.1f} 秒")
    print(f"  三角形数: {len(mesh.triangles):,}")
    print(f"  顶点数: {len(mesh.vertices):,}")

    # ==================== 密度过滤 ====================
    # 移除低密度区域（Poisson 重建在边缘会产生不可靠的三角形）
    print(f"\n密度过滤 (分位数阈值={density_threshold_quantile})...")
    densities = np.asarray(densities)
    density_threshold = np.quantile(densities, density_threshold_quantile)
    vertices_to_remove = densities < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)
    print(f"  过滤后三角形数: {len(mesh.triangles):,}")
    print(f"  过滤后顶点数: {len(mesh.vertices):,}")

    # ==================== 简化（可选）====================
    if simplify_target is not None and len(mesh.triangles) > simplify_target:
        print(f"\n简化网格到 {simplify_target:,} 个三角形...")
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=simplify_target)
        print(f"  简化后三角形数: {len(mesh.triangles):,}")

    # ==================== 后处理 ====================
    # 去除非连通组件（只保留最大的）
    print("\n去除小碎片...")
    triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    largest_cluster = np.argmax(cluster_n_triangles)
    triangles_to_remove = triangle_clusters != largest_cluster
    mesh.remove_triangles_by_mask(triangles_to_remove)
    mesh.remove_unreferenced_vertices()
    removed_clusters = len(cluster_n_triangles) - 1
    if removed_clusters > 0:
        print(f"  移除了 {removed_clusters} 个小碎片")

    # 平滑
    print("平滑处理...")
    mesh = mesh.filter_smooth_laplacian(number_of_iterations=3)
    mesh.compute_vertex_normals()

    # ==================== 保存 ====================
    base_name = os.path.splitext(os.path.basename(input_ply))[0]

    # 保存 PLY
    ply_output = os.path.join(output_dir, f"{base_name}_mesh.ply")
    o3d.io.write_triangle_mesh(ply_output, mesh)
    ply_size = os.path.getsize(ply_output) / (1024 * 1024)
    print(f"\n✅ PLY 网格: {ply_output} ({ply_size:.1f} MB)")

    # 保存 OBJ
    obj_output = os.path.join(output_dir, f"{base_name}_mesh.obj")
    o3d.io.write_triangle_mesh(obj_output, mesh)
    obj_size = os.path.getsize(obj_output) / (1024 * 1024)
    print(f"✅ OBJ 网格: {obj_output} ({obj_size:.1f} MB)")

    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"✨ 完成! 总耗时: {total_time:.1f} 秒")
    print(f"{'='*60}")
    print(f"  顶点数: {len(mesh.vertices):,}")
    print(f"  三角形: {len(mesh.triangles):,}")
    print(f"  输出:")
    print(f"    - {ply_output}")
    print(f"    - {obj_output}")
    print()
    print(f"查看方法:")
    print(f"  1. MeshLab:  meshlab {ply_output}")
    print(f"  2. Blender:  导入 {obj_output}")
    print(f"  3. Open3D:   python -c \"import open3d as o3d; o3d.visualization.draw_geometries([o3d.io.read_triangle_mesh('{ply_output}')])\"")

    return mesh


def main():
    parser = argparse.ArgumentParser(
        description="从 DA3 点云快速生成 Mesh（无需训练，10-30秒）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 标准质量（推荐）
  python da3_quick_mesh.py --input output/sugar_streaming/pcd/combined_pcd.ply

  # 高质量（更慢）
  python da3_quick_mesh.py --input output/sugar_streaming/pcd/combined_pcd.ply --depth 11

  # 快速预览
  python da3_quick_mesh.py --input output/sugar_streaming/pcd/combined_pcd.ply --depth 7

  # 指定输出目录
  python da3_quick_mesh.py --input output/sugar_streaming/pcd/combined_pcd.ply --output_dir output/mesh

质量对比:
  depth=7:  粗糙，~5秒，适合快速预览
  depth=9:  标准，~15秒，推荐日常使用
  depth=11: 精细，~60秒，最高质量

  对比 SuGaR (dn_consistency long true false): ~2-3小时
        """
    )
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="输入点云 PLY 文件路径")
    parser.add_argument("--output_dir", "-o", type=str, default=None,
                        help="输出目录（默认与输入同目录）")
    parser.add_argument("--depth", "-d", type=int, default=9,
                        help="Poisson 重建深度: 7(快) 9(标准) 11(精细)，默认 9")
    parser.add_argument("--density_threshold", type=float, default=0.02,
                        help="密度过滤分位数阈值（0.01~0.1），越大过滤越多，默认 0.02")
    parser.add_argument("--simplify", "-s", type=int, default=None,
                        help="简化到目标三角形数量（默认不简化）")
    parser.add_argument("--normal_radius", type=float, default=0.05,
                        help="法线估计搜索半径（默认 0.05）")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ 错误: 文件不存在: {args.input}")
        sys.exit(1)

    quick_mesh_from_pointcloud(
        input_ply=args.input,
        output_dir=args.output_dir,
        poisson_depth=args.depth,
        density_threshold_quantile=args.density_threshold,
        simplify_target=args.simplify,
        normal_radius=args.normal_radius,
    )


if __name__ == "__main__":
    main()
