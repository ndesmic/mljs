import { assertEquals, assertAlmostEquals } from "https://deno.land/std/assert/mod.ts";
import { Tensor } from "../src/tensor.js";

Deno.test("Tensor should add", () => {
	const t1 = new Tensor({ shape: [2,2], values: [1,2,3,4] });
	const t2 = new Tensor({ shape: [2, 2], values: [5, 6, 7, 8] });
	const result = t1.add(t2);

	assertEquals(result.shape, [2,2]);
	assertEquals(result.values, new Float32Array([6,8,10,12]));
});
Deno.test("Tensor should backprop gradient through add", () => {
	const t1 = new Tensor({ shape: [2, 2], values: [1, 2, 3, 4] });
	const t2 = new Tensor({ shape: [2, 2], values: [5, 6, 7, 8] });
	const result = t1.add(t2);

	assertEquals(result.values, new Float32Array([6, 8, 10, 12]));
	result.backward();
	assertEquals(result.gradient, new Float32Array([1,1,1,1]));
	assertEquals(t1.gradient, new Float32Array([1, 1, 1, 1]));
	assertEquals(t2.gradient, new Float32Array([1, 1, 1, 1]));
});
Deno.test("Tensor should add same node", () => {
	const t1 = new Tensor({ shape: [2, 2], values: [1, 2, 3, 4] });
	const result = t1.add(t1);

	assertEquals(result.values, new Float32Array([2, 4, 6, 8]));
	result.backward();
	assertEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
	assertEquals(t1.gradient, new Float32Array([2, 2, 2, 2]));
});

Deno.test("Tensor should subtract", () => {
	const t1 = new Tensor({ shape: [2, 2], values: [11, 22, 33, 44] });
	const t2 = new Tensor({ shape: [2, 2], values: [5, 6, 7, 8] })
	const result = t1.sub(t2);

	assertEquals(result.shape, [2, 2]);
	assertEquals(result.values, new Float32Array([6, 16, 26, 36]));
});
Deno.test("Tensor should backprop gradient through sub", () => {
	const t1 = new Tensor({ shape: [2, 2], values: [11, 22, 33, 44] });
	const t2 = new Tensor({ shape: [2, 2], values: [5, 6, 7, 8] });
	const result = t1.sub(t2);

	assertEquals(result.values, new Float32Array([6, 16, 26, 36]));
	result.backward();
	assertEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
	assertEquals(t1.gradient, new Float32Array([1, 1, 1, 1]));
	assertEquals(t2.gradient, new Float32Array([-1, -1, -1, -1]));
});
Deno.test("Tensor should subtract same node", () => {
	const tensor = new Tensor({ shape: [2, 2], values: [11, 22, 33, 44] });
	const result = tensor.sub(tensor);

	assertEquals(result.values, new Float32Array([0, 0, 0, 0]));
	result.backward();
	assertEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
	assertEquals(tensor.gradient, new Float32Array([0, 0, 0, 0]));
});


Deno.test("Tensor should multiply", () => {
	const t1 = new Tensor({ shape: [2, 2], values: [1, 2, 3, 4] });
	const t2 = new Tensor({ shape: [2, 2], values: [5, 6, 7, 8] });
	const result = t1.mul(t2);

	assertEquals(result.shape, [2, 2]);
	assertEquals(result.values, new Float32Array([5, 12, 21, 32]));
});
Deno.test("Tensor should backprop gradient through multiply", () => {
	const t1 = new Tensor({ shape: [2, 2], values: [1, 2, 3, 4] });
	const t2 = new Tensor({ shape: [2, 2], values: [5, 6, 7, 8] });
	const result = t1.mul(t2);

	assertEquals(result.values, new Float32Array([5, 12, 21, 32]));
	result.backward();
	assertEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
	assertEquals(t1.gradient, new Float32Array([5, 6, 7, 8]));
	assertEquals(t2.gradient, new Float32Array([1, 2, 3, 4]));
});
Deno.test("Tensor should multiply same node", () => {
	const t1 = new Tensor({ shape: [2, 2], values: [1, 2, 3, 4] });
	const result = t1.mul(t1);

	assertEquals(result.shape, [2, 2]);
	assertEquals(result.values, new Float32Array([1, 4, 9, 16]));
	result.backward();
	assertEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
	assertEquals(t1.gradient, new Float32Array([2, 4, 6, 8]));
});

Deno.test("Tensor should divide", () => {
	const t1 = new Tensor({ shape: [2, 2], values: [10, 24, 21, 4] });
	const t2 = new Tensor({ shape: [2, 2], values: [5, 6, 7, 8] })
	const result = t1.div(t2);

	assertEquals(result.shape, [2, 2]);
	assertEquals(result.values, new Float32Array([2, 4, 3, 0.5]));
});
Deno.test("Tensor should backprop gradient through div", () => {
	const t1 = new Tensor({ shape: [2, 2], values: [10, 24, 21, 4] });
	const t2 = new Tensor({ shape: [2, 2], values: [5, 6, 7, 8] })
	const result = t1.div(t2);

	assertEquals(result.values, new Float32Array([2, 4, 3, 0.5]));
	result.backward();
	assertEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
	assertEquals(t1.gradient, new Float32Array([1/5, 1/6, 1/7, 1/8]));
	assertEquals(t2.gradient, new Float32Array([-10/5**2, -24/6**2, -21/7**2, -4/8**2]));
});
Deno.test("Tensor should divide same node", () => {
	const t1 = new Tensor({ shape: [2, 2], values: [10, 24, 21, 4] });
	const result = t1.div(t1);

	assertEquals(result.shape, [2, 2]);
	assertEquals(result.values, new Float32Array([1, 1, 1, 1]));
	result.backward();
	assertEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
	assertEquals(t1.gradient, new Float32Array([ //These are actually zero but dealing with precision loss
		1.4901161415892261e-9, 
		1.2417634698280722e-9, 
		8.869738832295582e-10,
		0
	]));
});

Deno.test("Tensor negate", () => {
	const t1 = new Tensor({ shape: [2, 2], values: [10, 24, 21, 4] });
	const result = t1.neg();

	assertEquals(result.shape, [2, 2]);
	assertEquals(result.values, new Float32Array([-10, -24, -21, -4]));
});
Deno.test("Tensor should backprop gradient through neg", () => {
	const t1 = new Tensor({ shape: [2, 2], values: [10, 24, 21, 4] });
	const result = t1.neg();

	assertEquals(result.values, new Float32Array([-10, -24, -21, -4]));
	result.backward();
	assertEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
	assertEquals(t1.gradient, new Float32Array([-1, -1, -1, -1]));
});

Deno.test("Tensor raises to power", () => {
	const t1 = new Tensor({ shape: [2, 2], values: [2, 2, 2, 2] });
	const t2 = new Tensor({ shape: [2, 2], values: [1, 2, 3, 4] });
	const result = t1.pow(t2);

	assertEquals(result.shape, [2, 2]);
	assertEquals(result.values, new Float32Array([2, 4, 8, 16]));
});
Deno.test("Tensor should backprop gradient through pow", () => {
	const t1 = new Tensor({ shape: [2, 2], values: [2, 2, 2, 2] });
	const t2 = new Tensor({ shape: [2, 2], values: [1, 2, 3, 4] });
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

Deno.test("Tensor exponentiates", () => {
	const t1 = new Tensor({ shape: [2, 2], values: [1, 2, 3, 4] });
	const result = t1.exp();

	assertEquals(result.shape, [2, 2]);
	assertEquals(result.values, new Float32Array([Math.exp(1), Math.exp(2), Math.exp(3), Math.exp(4)]));
});
Deno.test("Tensor should backprop gradient through exp", () => {
	const t1 = new Tensor({ shape: [2, 2], values: [1, 2, 3, 4] });
	const result = t1.exp();

	assertEquals(result.values, new Float32Array([Math.exp(1), Math.exp(2), Math.exp(3), Math.exp(4)]));
	result.backward();
	assertEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
	assertEquals(t1.gradient, new Float32Array([Math.exp(1), Math.exp(2), Math.exp(3), Math.exp(4)]));
});

Deno.test("Tensor applies hyperbolic tangent", () => {
	const t1 = new Tensor({ shape: [2, 2], values: [1, 2, 3, 4] });
	const result = t1.tanh();

	assertEquals(result.shape, [2, 2]);
	assertEquals(result.values, new Float32Array([Math.tanh(1), Math.tanh(2), Math.tanh(3), Math.tanh(4)]));
});
Deno.test("Tensor should backprop gradient through tanh", () => {
	const t1 = new Tensor({ shape: [2, 2], values: [1, 2, 3, 4] });
	const result = t1.tanh();

	assertEquals(result.values, new Float32Array([Math.tanh(1), Math.tanh(2), Math.tanh(3), Math.tanh(4)]));
	result.backward();
	assertEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
	assertEquals(t1.gradient, new Float32Array([
		1 - Math.tanh(1) ** 2, 
		1 - Math.tanh(2) ** 2, 
		1 - Math.tanh(3) ** 2, 
		1 - Math.tanh(4) ** 2
	]));
});

Deno.test("Tensor should sum a 4x3 across row", () => {
	const tensor = new Tensor({ 
		shape: [4,3],
		values: [
			1, 2, 3, 4,
			5, 6, 7, 8,
			9, 10, 11, 12
		]
	});

	const result = tensor.sum({ dimension : 0 });
	assertEquals(result.shape, [3]);
	assertEquals(result.values, new Float32Array([10, 26, 42]));
});
Deno.test("Tensor should sum a 4x3 across cols", () => {
	const tensor = new Tensor({
		shape: [4, 3],
		values: [
			1, 2, 3, 4,
			5, 6, 7, 8,
			9, 10, 11, 12
		]
	});

	const result = tensor.sum({ dimension: 1 });
	assertEquals(result.shape, [4]);
	assertEquals(result.values, new Float32Array([15, 18, 21, 24]));
});
Deno.test("Tensor should sum a 3x3x3 across rows", () => {
	const tensor = new Tensor({
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

	const result = tensor.sum({ dimension: 0 });
	assertEquals(result.shape, [3, 3]);
	assertEquals(result.values, new Float32Array([
		6, 15, 24,
		33, 42, 51,
		60, 69, 78
	]));
});
Deno.test("Tensor should sum a 3x3x3 across cols", () => {
	const tensor = new Tensor({
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

	const result = tensor.sum({ dimension: 1 });
	assertEquals(result.shape, [3, 3]);
	assertEquals(result.values, new Float32Array([
		12, 15, 18,
		39, 42, 45,
		66, 69, 72
	]));
});
Deno.test("Tensor should sum a 3x3x3 across depths", () => {
	const tensor = new Tensor({
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

	const result = tensor.sum({ dimension: 2 });
	assertEquals(result.shape, [3, 3]);
	assertEquals(result.values, new Float32Array([
		30, 33, 36,
		39, 42, 45,
		48, 51, 54
	]));
});

Deno.test("Tensor should have a nice string", () => {
	const tensor = new Tensor({ shape: [2, 2], values: [10, 24, 21, 4] });

	assertEquals(tensor.toString(), "<10, 24, 21, 4>");
});
Deno.test("Tensor should have a nice string with label", () => {
	const tensor = new Tensor({ shape: [2, 2], values: [10, 24, 21, 4], label: "t1" });

	assertEquals(tensor.toString(), "<t1:10, 24, 21, 4>");
});

Deno.test("Should work on perceptron example (singles)", () => {
	const x1 = new Tensor({ shape: [1,1], values: [2], label: "x1" }); 
	const x2 = new Tensor({ shape: [1,1], values: [0], label: "x2" }); 
	const w1 = new Tensor({ shape: [1,1], values: [-3], label: "w1" }); 
	const w2 = new Tensor({ shape: [1,1], values: [1], label: "w2" });
	const b = new Tensor({ shape: [1,1], values: [6.881373587019432], label: "b" });

	const x1w1 = x1.mul(w1);
	const x2w2 = x2.mul(w2);

	const i1 = x1w1.add(x2w2);
	const t1 = i1.add(b);
	const o1 = t1.tanh();

	assertAlmostEquals(o1.values[0], 0.70710, 1e-5);
	assertAlmostEquals(t1.values[0], 0.88137, 1e-5);
	assertAlmostEquals(i1.values[0], -6, 1e-5);
	o1.backward();
	assertAlmostEquals(o1.gradient[0], 1);
	assertAlmostEquals(t1.gradient[0], 0.5, 1e-5);
	assertAlmostEquals(b.gradient[0], 0.5, 1e-5);
	assertAlmostEquals(i1.gradient[0], 0.5, 1e-5);
	assertAlmostEquals(x1w1.gradient[0], 0.5, 1e-5);
	assertAlmostEquals(x2w2.gradient[0], 0.5, 1e-5);

	assertAlmostEquals(x1.gradient[0], -1.5, 1e-5);
	assertAlmostEquals(x2.gradient[0], 0.5, 1e-5);
	assertAlmostEquals(w1.gradient[0], 1, 1e-5);
	assertAlmostEquals(w2.gradient[0], 0, 1e-5);
});
Deno.test("Should work on perceptron example (parallel)", () => {
	const x = new Tensor({ shape: [2, 1], values: [2,0], label: "x" });
	const w = new Tensor({ shape: [2, 1], values: [-3, 1], label: "w" });
	const b = new Tensor({ shape: [1, 1], values: [6.881373587019432], label: "b" });

	const xw = x.mul(w);
	const i = xw.sum({ dimension: 0 });
	const t = i.add(b);
	const o = t.tanh();

	assertAlmostEquals(o.values[0], 0.70710, 1e-5);
	assertAlmostEquals(t.values[0], 0.88137, 1e-5);
	assertAlmostEquals(i.values[0], -6, 1e-5);
	o.backward();
	assertAlmostEquals(o.gradient[0], 1);
	assertAlmostEquals(t.gradient[0], 0.5, 1e-5);
	assertAlmostEquals(b.gradient[0], 0.5, 1e-5);
	assertAlmostEquals(i.gradient[0], 0.5, 1e-5);
	assertAlmostEquals(xw.gradient[0], 0.5, 1e-5);
	assertAlmostEquals(xw.gradient[1], 0.5, 1e-5);

	assertAlmostEquals(x.gradient[0], -1.5, 1e-5);
	assertAlmostEquals(x.gradient[1], 0.5, 1e-5);
	assertAlmostEquals(w.gradient[0], 1, 1e-5);
	assertAlmostEquals(w.gradient[1], 0, 1e-5);
});