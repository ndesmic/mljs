import { WGPUTensor } from "../src/js/wgpu-tensor.js";
import { Tensor } from "../src/js/tensor.js";

const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();

const tensor = new WGPUTensor({
	shape: [3, 3, 3],
	values: [
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,

		10, 11, 12,
		13, 14, 15,
		16, 17, 18,

		19, 20, 21,
		22, 23, 24,
		25, 26, 27
	],
	device
});
const tensor2 = new WGPUTensor({
	shape: [3, 3, 3],
	values: [
		2, 2, 3,
		2, 2, 3,
		2, 2, 3,

		2, 2, 3,
		2, 2, 3,
		2, 2, 3,

		2, 2, 3,
		2, 2, 3,
		2, 2, 3
	],
	device
});

const result = await (await tensor.mul(tensor2)).sum({ dimensionToReduce: 0 });
await result.backward();

console.log(tensor.gradient.toString());

// console.log("\n\n");
// const t2 = new Tensor({
// 	shape: [4, 3],
// 	values: [
// 		1, 2, 3, 4,
// 		5, 6, 7, 8,
// 		9, 10, 11, 12
// 	],
// });
// const result2 = await t2.sum({ dimensionToReduce: 0 });

// result2.backward();

// console.log(result2.gradient.toString());