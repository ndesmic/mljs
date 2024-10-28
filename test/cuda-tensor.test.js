import { assertEquals } from "@std/assert";
import { CUDATensor } from "../src/js/cuda-tensor.js";
import { assertArrayAlmostEquals } from "./test-utils.js";

Deno.test("CUDATensor should add", () => {
	const t1 = new CUDATensor({ shape: [2, 2], values: [1, 2, 3, 4] });
	const t2 = new CUDATensor({ shape: [2, 2], values: [5, 6, 7, 8] });
	const result = t1.add(t2);

	assertEquals(result.shape, [2, 2]);
	assertEquals(result.values, new Float32Array([6, 8, 10, 12]));
});
Deno.test("CUDATensor should backprop gradient through add", () => {
	const t1 = new CUDATensor({ shape: [2, 2], values: [1, 2, 3, 4] });
	const t2 = new CUDATensor({ shape: [2, 2], values: [5, 6, 7, 8] });
	const result = t1.add(t2);

	assertEquals(result.values, new Float32Array([6, 8, 10, 12]));
	result.backward();
	assertEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
	assertEquals(t1.gradient, new Float32Array([1, 1, 1, 1]));
	assertEquals(t2.gradient, new Float32Array([1, 1, 1, 1]));
});
Deno.test("CUDATensor should add same node", () => {
	const t1 = new CUDATensor({ shape: [2, 2], values: [1, 2, 3, 4] });
	const result = t1.add(t1);

	assertEquals(result.values, new Float32Array([2, 4, 6, 8]));
	result.backward();
	assertEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
	assertEquals(t1.gradient, new Float32Array([2, 2, 2, 2]));
});

Deno.test("CUDATensor should subtract", () => {
	const t1 = new CUDATensor({ shape: [2, 2], values: [11, 22, 33, 44] });
	const t2 = new CUDATensor({ shape: [2, 2], values: [5, 6, 7, 8] })
	const result = t1.sub(t2);

	assertEquals(result.shape, [2, 2]);
	assertEquals(result.values, new Float32Array([6, 16, 26, 36]));
});
Deno.test("CUDATensor should backprop gradient through sub", () => {
	const t1 = new CUDATensor({ shape: [2, 2], values: [11, 22, 33, 44] });
	const t2 = new CUDATensor({ shape: [2, 2], values: [5, 6, 7, 8] });
	const result = t1.sub(t2);

	assertEquals(result.values, new Float32Array([6, 16, 26, 36]));
	result.backward();
	assertEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
	assertEquals(t1.gradient, new Float32Array([1, 1, 1, 1]));
	assertEquals(t2.gradient, new Float32Array([-1, -1, -1, -1]));
});
Deno.test("CUDATensor should subtract same node", () => {
	const tensor = new CUDATensor({ shape: [2, 2], values: [11, 22, 33, 44] });
	const result = tensor.sub(tensor);

	assertEquals(result.values, new Float32Array([0, 0, 0, 0]));
	result.backward();
	assertEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
	assertEquals(tensor.gradient, new Float32Array([0, 0, 0, 0]));
});

Deno.test("CUDATensor should multiply", () => {
	const t1 = new CUDATensor({ shape: [2, 2], values: [1, 2, 3, 4] });
	const t2 = new CUDATensor({ shape: [2, 2], values: [5, 6, 7, 8] });
	const result = t1.mul(t2);

	assertEquals(result.shape, [2, 2]);
	assertEquals(result.values, new Float32Array([5, 12, 21, 32]));
});
Deno.test("CUDATensor should backprop gradient through multiply", () => {
	const t1 = new CUDATensor({ shape: [2, 2], values: [1, 2, 3, 4] });
	const t2 = new CUDATensor({ shape: [2, 2], values: [5, 6, 7, 8] });
	const result = t1.mul(t2);

	assertEquals(result.values, new Float32Array([5, 12, 21, 32]));
	result.backward();
	assertEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
	assertEquals(t1.gradient, new Float32Array([5, 6, 7, 8]));
	assertEquals(t2.gradient, new Float32Array([1, 2, 3, 4]));
});
Deno.test("CUDATensor should multiply same node", () => {
	const t1 = new CUDATensor({ shape: [2, 2], values: [1, 2, 3, 4] });
	const result = t1.mul(t1);

	assertEquals(result.shape, [2, 2]);
	assertEquals(result.values, new Float32Array([1, 4, 9, 16]));
	result.backward();
	assertEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
	assertEquals(t1.gradient, new Float32Array([2, 4, 6, 8]));
});

Deno.test("CUDATensor should divide", () => {
	const t1 = new CUDATensor({ shape: [2, 2], values: [10, 24, 21, 4] });
	const t2 = new CUDATensor({ shape: [2, 2], values: [5, 6, 7, 8] })
	const result = t1.div(t2);

	assertEquals(result.shape, [2, 2]);
	assertEquals(result.values, new Float32Array([2, 4, 3, 0.5]));
});
Deno.test("CUDATensor should backprop gradient through div", () => {
	const t1 = new CUDATensor({ shape: [2, 2], values: [10, 24, 21, 4] });
	const t2 = new CUDATensor({ shape: [2, 2], values: [5, 6, 7, 8] })
	const result = t1.div(t2);

	assertEquals(result.values, new Float32Array([2, 4, 3, 0.5]));
	result.backward();
	assertEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
	assertEquals(t1.gradient, new Float32Array([1 / 5, 1 / 6, 1 / 7, 1 / 8]));
	assertEquals(t2.gradient, new Float32Array([-10 / 5 ** 2, -24 / 6 ** 2, -21 / 7 ** 2, -4 / 8 ** 2]));
});
Deno.test("CUDATensor should divide same node", () => {
	const t1 = new CUDATensor({ shape: [2, 2], values: [10, 24, 21, 4] });
	const result = t1.div(t1);

	assertEquals(result.shape, [2, 2]);
	assertEquals(result.values, new Float32Array([1, 1, 1, 1]));
	result.backward();
	assertEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
	assertArrayAlmostEquals(t1.gradient, new Float32Array([ //These are actually zero but dealing with precision loss
		1.4901161415892261e-9,
		1.2417634698280722e-9,
		8.869738832295582e-10,
		0
	]));
});
Deno.test("CUDATensor raises to power", () => {
	const t1 = new CUDATensor({ shape: [2, 2], values: [2, 2, 2, 2] });
	const t2 = new CUDATensor({ shape: [2, 2], values: [1, 2, 3, 4] });
	const result = t1.pow(t2);

	assertEquals(result.shape, [2, 2]);
	assertEquals(result.values, new Float32Array([2, 4, 8, 16]));
});
Deno.test("CUDATensor should backprop gradient through pow", () => {
	const t1 = new CUDATensor({ shape: [2, 2], values: [2, 2, 2, 2] });
	const t2 = new CUDATensor({ shape: [2, 2], values: [1, 2, 3, 4] });
	const result = t1.pow(t2);

	assertEquals(result.values, new Float32Array([2, 4, 8, 16]));
	result.backward();
	assertEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
	assertEquals(t1.gradient, new Float32Array([1, 4, 12, 32]));
	assertEquals(t2.gradient, new Float32Array([
		Math.log(2) * 2 ** 1,
		Math.log(2) * 2 ** 2,
		Math.log(2) * 2 ** 3,
		Math.log(2) * 2 ** 4
	]));
});
Deno.test("CUDATensor should pow same node", () => {
	const t1 = new CUDATensor({ shape: [2, 2], values: [1, 2, 3, 4] });
	const result = t1.pow(t1);

	assertEquals(result.values, new Float32Array([1, 4, 27, 256]));
	result.backward();
	assertEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
	assertArrayAlmostEquals(t1.gradient, new Float32Array([1.0000, 6.7726, 56.6625, 610.8914]));
});

Deno.test("CUDATensor should negate", () => {
	const t1 = new CUDATensor({ shape: [2, 2], values: [10, 24, 21, 4] });
	const result = t1.neg();

	assertEquals(result.shape, [2, 2]);
	assertEquals(result.values, new Float32Array([-10, -24, -21, -4]));
});
Deno.test("CUDATensor should backprop gradient through neg", () => {
	const t1 = new CUDATensor({ shape: [2, 2], values: [10, 24, 21, 4] });
	const result = t1.neg();

	assertEquals(result.values, new Float32Array([-10, -24, -21, -4]));
	result.backward();
	assertEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
	assertEquals(t1.gradient, new Float32Array([-1, -1, -1, -1]));
});
Deno.test("CUDATensor exponentiates", () => {
	const t1 = new CUDATensor({ shape: [2, 2], values: [1, 2, 3, 4] });
	const result = t1.exp();

	assertEquals(result.shape, [2, 2]);
	assertEquals(result.values, new Float32Array([Math.exp(1), Math.exp(2), Math.exp(3), Math.exp(4)]));
});
Deno.test("CUDATensor should backprop gradient through exp", () => {
	const t1 = new CUDATensor({ shape: [2, 2], values: [1, 2, 3, 4] });
	const result = t1.exp();

	assertEquals(result.values, new Float32Array([Math.exp(1), Math.exp(2), Math.exp(3), Math.exp(4)]));
	result.backward();
	assertEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
	assertEquals(t1.gradient, new Float32Array([Math.exp(1), Math.exp(2), Math.exp(3), Math.exp(4)]));
});
Deno.test("CUDATensor applies hyperbolic tangent", () => {
	const t1 = new CUDATensor({ shape: [2, 2], values: [1, 2, 3, 4] });
	const result = t1.tanh();

	assertEquals(result.shape, [2, 2]);
	assertArrayAlmostEquals(result.values, new Float32Array([Math.tanh(1), Math.tanh(2), Math.tanh(3), Math.tanh(4)]));
});
Deno.test("CUDATensor should backprop gradient through tanh", () => {
	const t1 = new CUDATensor({ shape: [2, 2], values: [1, 2, 3, 4] });
	const result = t1.tanh();

	assertArrayAlmostEquals(result.values, new Float32Array([Math.tanh(1), Math.tanh(2), Math.tanh(3), Math.tanh(4)]));
	result.backward();
	assertArrayAlmostEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
	assertArrayAlmostEquals(t1.gradient, new Float32Array([
		1 - Math.tanh(1) ** 2,
		1 - Math.tanh(2) ** 2,
		1 - Math.tanh(3) ** 2,
		1 - Math.tanh(4) ** 2
	]));
});

Deno.test("CUDATensor should sum a 4x3 across row", () => {
	const tensor = new CUDATensor({ 
		shape: [4,3],
		values: [
			1, 2, 3, 4,
			5, 6, 7, 8,
			9, 10, 11, 12
		]
	});

	const result = tensor.sum({ dimensionToReduce : 0 });
	assertEquals(result.shape, [3]);
	assertEquals(result.values, new Float32Array([10, 26, 42]));
});
Deno.test("CUDATensor should sum a 4x3 across cols", () => {
	const tensor = new CUDATensor({
		shape: [4, 3],
		values: [
			1, 2, 3, 4,
			5, 6, 7, 8,
			9, 10, 11, 12
		]
	});

	const result = tensor.sum({ dimensionToReduce: 1 });
	assertEquals(result.shape, [4]);
	assertEquals(result.values, new Float32Array([15, 18, 21, 24]));
});
Deno.test("CUDATensor should sum a 3x3x3 across rows", () => {
	const tensor = new CUDATensor({
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
		]
	});

	const result = tensor.sum({ dimensionToReduce: 0 });
	assertEquals(result.shape, [3, 3]);
	assertEquals(result.values, new Float32Array([
		6, 15, 24,
		33, 42, 51,
		60, 69, 78
	]));
});
Deno.test("CUDATensor should sum a 3x3x3 across cols", () => {
	const tensor = new CUDATensor({
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
		]
	});

	const result = tensor.sum({ dimensionToReduce: 1 });
	assertEquals(result.shape, [3, 3]);
	assertEquals(result.values, new Float32Array([
		12, 15, 18,
		39, 42, 45,
		66, 69, 72
	]));
});
Deno.test("CUDATensor should sum a 3x3x3 across depths", () => {
	const tensor = new CUDATensor({
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
		]
	});

	const result = tensor.sum({ dimensionToReduce: 2 });
	assertEquals(result.shape, [3, 3]);
	assertEquals(result.values, new Float32Array([
		30, 33, 36,
		39, 42, 45,
		48, 51, 54
	]));
});
// Deno.test("Tensor should backprop across 3x3x3 tensor rows", () => {
// 	const tensor = new CUDATensor({
// 		shape: [3, 3, 3],
// 		values: [
// 			1, 2, 3,
// 			4, 5, 6,
// 			7, 8, 9,

// 			10, 11, 12,
// 			13, 14, 15,
// 			16, 17, 18,

// 			19, 20, 21,
// 			22, 23, 24,
// 			25, 26, 27
// 		]
// 	});
// 	const tensor2 = new CUDATensor({
// 		shape: [3, 3, 3],
// 		values: [
// 			2, 2, 3,
// 			2, 2, 3,
// 			2, 2, 3,

// 			2, 2, 3,
// 			2, 2, 3,
// 			2, 2, 3,

// 			2, 2, 3,
// 			2, 2, 3,
// 			2, 2, 3
// 		]
// 	});

// 	const result = tensor.mul(tensor2).sum({ dimensionToReduce: 0 });
// 	assertEquals(result.shape, [3, 3]);
// 	assertEquals(result.values, new Float32Array([
// 		15,
// 		36,
// 		57,
// 		78,
// 		99,
// 		120,
// 		141,
// 		162,
// 		183
// 	]));
// 	result.backward();
// 	assertEquals(tensor.gradient, new Float32Array([
// 		2, 2, 3,
// 		2, 2, 3,
// 		2, 2, 3,

// 		2, 2, 3,
// 		2, 2, 3,
// 		2, 2, 3,

// 		2, 2, 3,
// 		2, 2, 3,
// 		2, 2, 3
// 	]));
// });