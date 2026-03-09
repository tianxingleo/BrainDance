from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class AlignmentResult:
    matrix: List[float]
    score: float
    method: str
    error: Optional[str] = None


class TimePeelingAligner:
    """Compute a transform that maps current capture to base capture coordinate system.

    The implementation tries Open3D feature matching + ICP first. If Open3D is unavailable
    or registration fails, it returns identity transform with score 0.
    """

    def __init__(self, score_threshold: float = 0.6):
        self.score_threshold = score_threshold

    @staticmethod
    def identity() -> List[float]:
        return [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]

    def align(self, source_ply: str, target_ply: str) -> AlignmentResult:
        try:
            import open3d as o3d  # type: ignore
        except Exception as e:
            return AlignmentResult(
                matrix=self.identity(),
                score=0.0,
                method="identity",
                error=f"open3d unavailable: {e}",
            )

        try:
            source = o3d.io.read_point_cloud(source_ply)
            target = o3d.io.read_point_cloud(target_ply)

            if len(source.points) < 50 or len(target.points) < 50:
                return AlignmentResult(
                    matrix=self.identity(),
                    score=0.0,
                    method="identity",
                    error="too few points",
                )

            src_bbox = source.get_axis_aligned_bounding_box()
            tgt_bbox = target.get_axis_aligned_bounding_box()
            scene_scale = max(
                float(np.max(src_bbox.get_extent())),
                float(np.max(tgt_bbox.get_extent())),
                1e-3,
            )

            voxel = max(scene_scale / 120.0, 0.01)
            src_down = source.voxel_down_sample(voxel)
            tgt_down = target.voxel_down_sample(voxel)

            src_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.0, max_nn=30)
            )
            tgt_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.0, max_nn=30)
            )

            src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                src_down,
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5.0, max_nn=100),
            )
            tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                tgt_down,
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5.0, max_nn=100),
            )

            coarse = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                src_down,
                tgt_down,
                src_fpfh,
                tgt_fpfh,
                mutual_filter=True,
                max_correspondence_distance=voxel * 2.5,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                ransac_n=4,
                checkers=[
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel * 2.5),
                ],
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
            )

            src_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.0, max_nn=30)
            )
            tgt_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.0, max_nn=30)
            )

            fine = o3d.pipelines.registration.registration_icp(
                src_down,
                tgt_down,
                max_correspondence_distance=voxel * 1.5,
                init=coarse.transformation,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            )

            transform = np.asarray(fine.transformation, dtype=float)
            score = float(fine.fitness)

            return AlignmentResult(
                matrix=transform.reshape(-1).tolist(),
                score=score,
                method="ransac_icp",
            )
        except Exception as e:
            return AlignmentResult(
                matrix=self.identity(),
                score=0.0,
                method="identity",
                error=f"alignment failed: {e}",
            )
