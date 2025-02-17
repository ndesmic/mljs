import { Tensor } from "../src/js/tensor.js";
import { CUDATensor } from "../src/js/cuda-tensor.js";
import { WGPUTensor } from "../src/js/wgpu-tensor.js";
import { range } from "../src/js/tensor-utils.js";

for (let i = 1; i < 9; i++) {
	const size = 2 ** i;
	Deno.bench(`CPU add ${size}x${size}x${size}`, { group: `Add ${size}x${size}x${size}`, baseline: true }, () => {
		const a = new Tensor({ shape: [size, size, size], values: [...range(size * size * size)], label: "a" });
		const b = new Tensor({ shape: [size, size, size], values: [...range(size * size * size)], label: "b" });

		const r = a.add(b);
	});

	Deno.bench(`CUDA add ${size}x${size}x${size}`, { group: `Add ${size}x${size}x${size}` }, () => {
		const a = new CUDATensor({ shape: [size, size, size], values: [...range(size * size * size)], label: "a" });
		const b = new CUDATensor({ shape: [size, size, size], values: [...range(size * size * size)], label: "b" });

		const r = a.add(b);
	});
	const adapter = await navigator.gpu.requestAdapter();
	const device = await adapter.requestDevice();
	Deno.bench(`WGPU add ${size}x${size}x${size}`, { group: `Add ${size}x${size}x${size}` }, async () => {
		const a = new WGPUTensor({ shape: [size, size, size], values: [...range(size * size * size)], label: "a", device });
		const b = new WGPUTensor({ shape: [size, size, size], values: [...range(size * size * size)], label: "b", device });
	
		const r = await a.add(b);
	});
}