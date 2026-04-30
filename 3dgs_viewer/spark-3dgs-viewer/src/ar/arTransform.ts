import type { ArTransform } from '../types/ar'
import * as THREE from 'three'

export function applyArTransform(target: THREE.Object3D, transform: ArTransform) {
  target.scale.setScalar(transform.scale)
  target.rotation.set(transform.rotation[0], transform.rotation[1], transform.rotation[2])
  target.position.set(transform.offset[0], transform.offset[1], transform.offset[2])
}

export function cloneArTransform(transform: ArTransform): ArTransform {
  return {
    scale: transform.scale,
    rotation: [...transform.rotation] as [number, number, number],
    offset: [...transform.offset] as [number, number, number],
  }
}

