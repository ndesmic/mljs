import { Tensor } from "../src/js/tensor.js";
import { CUDATensor } from "../src/js/cuda-tensor.js";
import { WGPUTensor } from "../src/js/wgpu-tensor.js";
import { range } from "../src/js/tensor-utils.js";

for(let i = 1; i < 12; i++){
	const size = 2 ** i;
	Deno.bench(`CPU Perceptron ${size}x${size}`, { group: `Perceptron ${size}x${size}`, baseline: true }, () => {
		const x = new Tensor({ shape: [size, size], values: [...range(size * size)], label: "x" });
		const w = new Tensor({ shape: [size, size], values: [...range(size * size)], label: "w" });
		const b = new Tensor({ shape: [size, 1], values:[...range(size)], label: "b" });

		const xw = x.mul(w);
		const i = xw.sum({ dimensionToReduce: 0 });
		const t = i.add(b);
		const o = t.tanh();
	});

	Deno.bench(`CUDA Perceptron ${size}x${size}`, { group: `Perceptron ${size}x${size}` }, () => {
		const x = new CUDATensor({ shape: [size, size], values: [...range(size * size)], label: "x" });
		const w = new CUDATensor({ shape: [size, size], values: [...range(size * size)], label: "w" });
		const b = new CUDATensor({ shape: [size, 1], values: [...range(size)], label: "b" });

		const xw = x.mul(w);
		const i = xw.sum({ dimensionToReduce: 0 });
		const t = i.add(b);
		const o = t.tanh();
	});
	const adapter = await navigator.gpu.requestAdapter();
	const device = await adapter.requestDevice();
	Deno.bench(`WGPU Perceptron ${size}x${size}`, { group: `Perceptron ${size}x${size}` }, async (bench) => {
		const x = new WGPUTensor({ shape: [size, size], values: [...range(size * size)], label: "x", device });
		const w = new WGPUTensor({ shape: [size, size], values: [...range(size * size)], label: "w", device });
		const b = new WGPUTensor({ shape: [size, 1], values: [...range(size)], label: "b", device });

		const xw = await x.mul(w);
		const i = await xw.sum({ dimensionToReduce: 0 });
		const t = await i.add(b);
		const o = await t.tanh();
	});
}