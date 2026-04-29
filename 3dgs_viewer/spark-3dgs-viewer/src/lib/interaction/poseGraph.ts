import * as THREE from 'three';

interface PoseNode {
  index: number;
  position: THREE.Vector3;
  forward: THREE.Vector3;
  matrix: number[];
  imageUrl: string;
  tag: string;
  neighbors: number[];
}

function normalizeMatrix(input: unknown): number[] {
  if (Array.isArray(input) && input.length === 16) {
    return input.map((v: unknown) => Number(v));
  }
  return new Array(16).fill(0) as number[];
}

/**
 * Builds an adjacency graph from capture poses.
 * Each node is a camera pose; edges connect spatially nearby poses.
 * Used for guided navigation (walkthrough mode) and pose-graph navigation.
 */
export class PoseGraph {
  nodes: PoseNode[] = [];

  private readonly kNearest: number;
  private readonly maxNeighborDist: number;

  constructor(sceneRadius: number, kNearest = 5) {
    this.kNearest = kNearest;
    this.maxNeighborDist = sceneRadius * 1.5;
  }

  buildFromPoses(
    poses: Array<{ matrix: unknown; image_url?: string; tag?: string }>,
  ) {
    this.nodes = poses.map((pose, index) => {
      const matrixValues = normalizeMatrix(pose.matrix);
      const m = new THREE.Matrix4().fromArray(matrixValues);
      const position = new THREE.Vector3();
      const quaternion = new THREE.Quaternion();
      const scale = new THREE.Vector3();
      m.decompose(position, quaternion, scale);

      const forward = new THREE.Vector3(0, 0, -1)
        .applyQuaternion(quaternion)
        .normalize();

      return {
        index,
        position,
        forward,
        matrix: matrixValues,
        imageUrl: pose.image_url || '',
        tag: pose.tag || '',
        neighbors: [] as number[],
      };
    });

    // Build k-nearest-neighbors adjacency within maxNeighborDist
    for (let i = 0; i < this.nodes.length; i++) {
      const distances: { index: number; dist: number }[] = [];

      for (let j = 0; j < this.nodes.length; j++) {
        if (i === j) continue;
        const dist = this.nodes[i].position.distanceTo(this.nodes[j].position);
        if (dist <= this.maxNeighborDist) {
          distances.push({ index: j, dist });
        }
      }

      distances.sort((a, b) => a.dist - b.dist);
      this.nodes[i].neighbors = distances
        .slice(0, this.kNearest)
        .map((d) => d.index);
    }
  }

  /** Find the node whose camera position is closest to `worldPosition`. */
  findNearestNode(worldPosition: THREE.Vector3): number {
    let best = 0;
    let bestDist = Infinity;

    for (let i = 0; i < this.nodes.length; i++) {
      const d = worldPosition.distanceTo(this.nodes[i].position);
      if (d < bestDist) {
        bestDist = d;
        best = i;
      }
    }
    return best;
  }

  /**
   * Get the next node along the camera-forward axis.
   * `direction`: 'forward' follows the camera gaze; 'backward' goes against it.
   */
  getNextAlongPath(
    currentIndex: number,
    direction: 'forward' | 'backward',
  ): number | null {
    const node = this.nodes[currentIndex];
    if (!node || node.neighbors.length === 0) return null;

    let bestIdx: number | null = null;
    let bestScore = -Infinity;

    for (const ni of node.neighbors) {
      const toNeighbor = this.nodes[ni].position
        .clone()
        .sub(node.position)
        .normalize();
      const dot = toNeighbor.dot(node.forward);
      const score = direction === 'forward' ? dot : -dot;
      if (score > bestScore) {
        bestScore = score;
        bestIdx = ni;
      }
    }

    return bestScore > 0 ? bestIdx : null;
  }

  getNode(index: number): PoseNode | null {
    return this.nodes[index] ?? null;
  }

  getNodeMatrix(index: number): number[] | null {
    return this.nodes[index]?.matrix ?? null;
  }

  get size(): number {
    return this.nodes.length;
  }
}
