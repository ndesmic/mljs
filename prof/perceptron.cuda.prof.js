import { CUDATensor } from "../src/js/cuda-tensor.js";
import { range } from "../src/js/tensor-utils.js";

const size = 1000;

while(true){
	const x = new CUDATensor({ shape: [size, size], values: [...range(size * size)], label: "x" });
	const w = new CUDATensor({ shape: [size, size], values: [...range(size * size)], label: "w" });
	const xw = x.mul(w);
	const i = xw.sum({ dimensionToReduce: 0 });
}