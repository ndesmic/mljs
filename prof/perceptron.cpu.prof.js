import { Tensor } from "../src/js/tensor.js";
import { range } from "../src/js/tensor-utils.js";

const size = 1000;

while(true){
	const x = new Tensor({ shape: [size, size], values: [...range(size * size)], label: "x" });
	const w = new Tensor({ shape: [size, size], values: [...range(size * size)], label: "w" });
	const xw = x.mul(w);
	const i = xw.sum({ dimensionToReduce: 0 });
}