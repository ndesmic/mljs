import { Tensor } from "../src/js/tensor.js";
import { CUDATensor } from "../src/js/cuda-tensor.js"
import { range } from "../src/js/tensor-utils.js";

for(let i = 1; i < 10; i++){
	const size = 2 ** i;
	Deno.bench(`CPU Perceptron ${size}x${size}`, { group: `Perceptron ${size}x${size}`, baseline: true }, () => {
		const x = new Tensor({ shape: [size, size], values: [...range(size * size)], label: "x" });
		const w = new Tensor({ shape: [size, size], values: [...range(size * size)], label: "w" });
		const b = new Tensor({ shape: [size, 1], values:[...range(size)], label: "b" });

		const xw = x.mul(w);
		const i = xw.sum({ dimension: 0 });
		const t = i.add(b);
		const o = t.tanh();
	});

	Deno.bench(`CUDA Perceptron ${size}x${size}`, { group: `Perceptron ${size}x${size}` }, () => {
		const x = new CUDATensor({ shape: [size, size], values: [...range(size * size)], label: "x" });
		const w = new CUDATensor({ shape: [size, size], values: [...range(size * size)], label: "w" });
		const b = new CUDATensor({ shape: [size, 1], values: [...range(size)], label: "b" });

		const xw = x.mul(w);
		const i = xw.sum({ dimension: 0 });
		const t = i.add(b);
		const o = t.tanh();
	});
}