import { WGPUTensor } from "../src/js/wgpu-tensor.js";
import { range } from "../src/js/tensor-utils.js";

const size = 1000;
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();

while (true) {
	const x = new WGPUTensor({ shape: [size, size], values: [...range(size * size)], label: "x", device });
	const w = new WGPUTensor({ shape: [size, size], values: [...range(size * size)], label: "w", device });
	const xw = await x.mul(w);
	const i = await xw.sum({ dimensionToReduce: 0 });
}