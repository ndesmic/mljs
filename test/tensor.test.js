import { assertEquals } from "@std/assert";
import { Tensor } from "../src/js/tensor.js";
import { assertArrayAlmostEquals } from "./test-utils.js";
import { describe, it } from "@std/testing/bdd";

describe("Tensor", () => {
	describe("add", () => {
		it("should forward pass", () => {
			const t1 = new Tensor({ shape: [2, 2], values: [1, 2, 3, 4] });
			const t2 = new Tensor({ shape: [2, 2], values: [5, 6, 7, 8] });
			const result = t1.add(t2);

			assertEquals(result.shape, [2, 2]);
			assertArrayAlmostEquals(result.values, new Float32Array([6, 8, 10, 12]));
		});
		it("should backprop", () => {
			const t1 = new Tensor({ shape: [2, 2], values: [1, 2, 3, 4] });
			const t2 = new Tensor({ shape: [2, 2], values: [5, 6, 7, 8] });
			const result = t1.add(t2);

			assertArrayAlmostEquals(result.values, new Float32Array([6, 8, 10, 12]));
			result.backward();
			assertArrayAlmostEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
			assertArrayAlmostEquals(t1.gradient, new Float32Array([1, 1, 1, 1]));
			assertArrayAlmostEquals(t2.gradient, new Float32Array([1, 1, 1, 1]));
		});
		it("should allow same node", () => {
			const t1 = new Tensor({ shape: [2, 2], values: [1, 2, 3, 4] });
			const result = t1.add(t1);

			assertArrayAlmostEquals(result.values, new Float32Array([2, 4, 6, 8]));
			result.backward();
			assertArrayAlmostEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
			assertArrayAlmostEquals(t1.gradient, new Float32Array([2, 2, 2, 2]));
		});
	});
	describe("subtract", () => {
		it("should forward pass", () => {
			const t1 = new Tensor({ shape: [2, 2], values: [11, 22, 33, 44] });
			const t2 = new Tensor({ shape: [2, 2], values: [5, 6, 7, 8] })
			const result = t1.sub(t2);

			assertEquals(result.shape, [2, 2]);
			assertArrayAlmostEquals(result.values, new Float32Array([6, 16, 26, 36]));
		});
		it("should backprop", () => {
			const t1 = new Tensor({ shape: [2, 2], values: [11, 22, 33, 44] });
			const t2 = new Tensor({ shape: [2, 2], values: [5, 6, 7, 8] });
			const result = t1.sub(t2);

			assertArrayAlmostEquals(result.values, new Float32Array([6, 16, 26, 36]));
			result.backward();
			assertArrayAlmostEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
			assertArrayAlmostEquals(t1.gradient, new Float32Array([1, 1, 1, 1]));
			assertArrayAlmostEquals(t2.gradient, new Float32Array([-1, -1, -1, -1]));
		});
		it("should allow same node", () => {
			const tensor = new Tensor({ shape: [2, 2], values: [11, 22, 33, 44] });
			const result = tensor.sub(tensor);

			assertArrayAlmostEquals(result.values, new Float32Array([0, 0, 0, 0]));
			result.backward();
			assertArrayAlmostEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
			assertArrayAlmostEquals(tensor.gradient, new Float32Array([0, 0, 0, 0]));
		});
	});
	describe("multiply", () => {
		it("should forward pass", () => {
			const t1 = new Tensor({ shape: [2, 2], values: [1, 2, 3, 4] });
			const t2 = new Tensor({ shape: [2, 2], values: [5, 6, 7, 8] });
			const result = t1.mul(t2);

			assertEquals(result.shape, [2, 2]);
			assertArrayAlmostEquals(result.values, new Float32Array([5, 12, 21, 32]));
		});
		it("should backprop", () => {
			const t1 = new Tensor({ shape: [2, 2], values: [1, 2, 3, 4] });
			const t2 = new Tensor({ shape: [2, 2], values: [5, 6, 7, 8] });
			const result = t1.mul(t2);

			assertEquals(result.values, new Float32Array([5, 12, 21, 32]));
			result.backward();
			assertArrayAlmostEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
			assertArrayAlmostEquals(t1.gradient, new Float32Array([5, 6, 7, 8]));
			assertArrayAlmostEquals(t2.gradient, new Float32Array([1, 2, 3, 4]));
		});
		it("should allow same node", () => {
			const t1 = new Tensor({ shape: [2, 2], values: [1, 2, 3, 4] });
			const result = t1.mul(t1);

			assertEquals(result.shape, [2, 2]);
			assertArrayAlmostEquals(result.values, new Float32Array([1, 4, 9, 16]));
			result.backward();
			assertArrayAlmostEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
			assertArrayAlmostEquals(t1.gradient, new Float32Array([2, 4, 6, 8]));
		});
	});
	describe("divide", () => {
		it("should forward pass", () => {
			const t1 = new Tensor({ shape: [2, 2], values: [10, 24, 21, 4] });
			const t2 = new Tensor({ shape: [2, 2], values: [5, 6, 7, 8] })
			const result = t1.div(t2);

			assertEquals(result.shape, [2, 2]);
			assertArrayAlmostEquals(result.values, new Float32Array([2, 4, 3, 0.5]));
		});
		it("should backprop", () => {
			const t1 = new Tensor({ shape: [2, 2], values: [10, 24, 21, 4] });
			const t2 = new Tensor({ shape: [2, 2], values: [5, 6, 7, 8] })
			const result = t1.div(t2);

			assertArrayAlmostEquals(result.values, new Float32Array([2, 4, 3, 0.5]));
			result.backward();
			assertArrayAlmostEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
			assertArrayAlmostEquals(t1.gradient, new Float32Array([1 / 5, 1 / 6, 1 / 7, 1 / 8]));
			assertArrayAlmostEquals(t2.gradient, new Float32Array([-10 / 5 ** 2, -24 / 6 ** 2, -21 / 7 ** 2, -4 / 8 ** 2]));
		});
		it("should allow same node", () => {
			const t1 = new Tensor({ shape: [2, 2], values: [10, 24, 21, 4] });
			const result = t1.div(t1);

			assertEquals(result.shape, [2, 2]);
			assertArrayAlmostEquals(result.values, new Float32Array([1, 1, 1, 1]));
			result.backward();
			assertArrayAlmostEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
			assertArrayAlmostEquals(t1.gradient, new Float32Array([ //These are actually zero but dealing with precision loss
				1.4901161415892261e-9,
				1.2417634698280722e-9,
				8.869738832295582e-10,
				0
			]));
		});
	});
	describe("power", () => {
		it("should forward pass", () => {
			const t1 = new Tensor({ shape: [2, 2], values: [2, 2, 2, 2] });
			const t2 = new Tensor({ shape: [2, 2], values: [1, 2, 3, 4] });
			const result = t1.pow(t2);

			assertEquals(result.shape, [2, 2]);
			assertArrayAlmostEquals(result.values, new Float32Array([2, 4, 8, 16]));
		});
		it("should backprop", () => {
			const t1 = new Tensor({ shape: [2, 2], values: [2, 2, 2, 2] });
			const t2 = new Tensor({ shape: [2, 2], values: [1, 2, 3, 4] });
			const result = t1.pow(t2);

			assertArrayAlmostEquals(result.values, new Float32Array([2, 4, 8, 16]));
			result.backward();
			assertArrayAlmostEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
			assertArrayAlmostEquals(t1.gradient, new Float32Array([1, 4, 12, 32]));
			assertArrayAlmostEquals(t2.gradient, new Float32Array([
				Math.log(2) * 2 ** 1,
				Math.log(2) * 2 ** 2,
				Math.log(2) * 2 ** 3,
				Math.log(2) * 2 ** 4
			]));
		});
		it("should allow same node", () => {
			const t1 = new Tensor({ shape: [2, 2], values: [1, 2, 3, 4] });
			const result = t1.pow(t1);

			assertEquals(result.values, new Float32Array([1, 4, 27, 256]));
			result.backward();
			assertArrayAlmostEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
			assertArrayAlmostEquals(t1.gradient, new Float32Array([1.0000, 6.7726, 56.6625, 610.8914]));
		});
	});
	describe("negate", () => {
		it("should forward pass", () => {
			const t1 = new Tensor({ shape: [2, 2], values: [10, 24, 21, 4] });
			const result = t1.neg();

			assertEquals(result.shape, [2, 2]);
			assertArrayAlmostEquals(result.values, new Float32Array([-10, -24, -21, -4]));
		});
		it("should backprop", () => {
			const t1 = new Tensor({ shape: [2, 2], values: [10, 24, 21, 4] });
			const result = t1.neg();

			assertEquals(result.values, new Float32Array([-10, -24, -21, -4]));
			result.backward();
			assertArrayAlmostEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
			assertArrayAlmostEquals(t1.gradient, new Float32Array([-1, -1, -1, -1]));
		});
	});
	describe("exponential", () => {
		it("should forward pass", () => {
			const t1 = new Tensor({ shape: [2, 2], values: [1, 2, 3, 4] });
			const result = t1.exp();

			assertEquals(result.shape, [2, 2]);
			assertArrayAlmostEquals(result.values, new Float32Array([Math.exp(1), Math.exp(2), Math.exp(3), Math.exp(4)]));
		});
		it("should backprop", () => {
			const t1 = new Tensor({ shape: [2, 2], values: [1, 2, 3, 4] });
			const result = t1.exp();

			assertEquals(result.values, new Float32Array([Math.exp(1), Math.exp(2), Math.exp(3), Math.exp(4)]));
			result.backward();
			assertArrayAlmostEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
			assertArrayAlmostEquals(t1.gradient, new Float32Array([Math.exp(1), Math.exp(2), Math.exp(3), Math.exp(4)]));
		});
	});
	describe("hyperbolic tangent", () => {
		it("should forward pass", () => {
			const t1 = new Tensor({ shape: [2, 2], values: [1, 2, 3, 4] });
			const result = t1.tanh();

			assertEquals(result.shape, [2, 2]);
			assertArrayAlmostEquals(result.values, new Float32Array([Math.tanh(1), Math.tanh(2), Math.tanh(3), Math.tanh(4)]));
		});
		it("should backprop", () => {
			const t1 = new Tensor({ shape: [2, 2], values: [1, 2, 3, 4] });
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
	});
	describe("sum", () => {
		it("should sum a 4x3 across rows", () => {
			const tensor = new Tensor({
				shape: [4, 3],
				values: [
					1, 2, 3, 4,
					5, 6, 7, 8,
					9, 10, 11, 12
				]
			});

			const result = tensor.sum({ dimensionToReduce: 0 });
			assertEquals(result.shape, [3]);
			assertArrayAlmostEquals(result.values, new Float32Array([10, 26, 42]));
		});
		it("should sum a 4x3 across cols", () => {
			const tensor = new Tensor({
				shape: [4, 3],
				values: [
					1, 2, 3, 4,
					5, 6, 7, 8,
					9, 10, 11, 12
				]
			});

			const result = tensor.sum({ dimensionToReduce: 1 });
			assertEquals(result.shape, [4]);
			assertArrayAlmostEquals(result.values, new Float32Array([15, 18, 21, 24]));
		});
		it("should sum a 3x3x3 across rows", () => {
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

			const result = tensor.sum({ dimensionToReduce: 0 });
			assertEquals(result.shape, [3, 3]);
			assertArrayAlmostEquals(result.values, new Float32Array([
				6, 15, 24,
				33, 42, 51,
				60, 69, 78
			]));
		});
		it("should sum a 3x3x3 across cols", () => {
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

			const result = tensor.sum({ dimensionToReduce: 1 });
			assertEquals(result.shape, [3, 3]);
			assertArrayAlmostEquals(result.values, new Float32Array([
				12, 15, 18,
				39, 42, 45,
				66, 69, 72
			]));
		});
		it("should sum a 3x3x3 across depths", () => {
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

			const result = tensor.sum({ dimensionToReduce: 2 });
			assertEquals(result.shape, [3, 3]);
			assertArrayAlmostEquals(result.values, new Float32Array([
				30, 33, 36,
				39, 42, 45,
				48, 51, 54
			]));
		});
		it("should backprop", () => {
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
			const tensor2 = new Tensor({
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
				]
			});

			const result = tensor.mul(tensor2).sum({ dimensionToReduce: 0 });
			assertEquals(result.shape, [3, 3]);
			assertEquals(result.values, new Float32Array([
				15,
				36,
				57,
				78,
				99,
				120,
				141,
				162,
				183
			]));
			result.backward();
			assertEquals(tensor.gradient, new Float32Array([
				2, 2, 3,
				2, 2, 3,
				2, 2, 3,

				2, 2, 3,
				2, 2, 3,
				2, 2, 3,

				2, 2, 3,
				2, 2, 3,
				2, 2, 3
			]));
		});
	});

	describe("toString", () => {
		it("should have a nice string", () => {
			const tensor = new Tensor({ shape: [2, 2], values: [10, 24, 21, 4] });

			assertEquals(tensor.toString(), "<10, 24, 21, 4>");
		});
		it("should have a nice string with label", () => {
			const tensor = new Tensor({ shape: [2, 2], values: [10, 24, 21, 4], label: "t1" });

			assertEquals(tensor.toString(), "<t1:10, 24, 21, 4>");
		});
	});
	describe("perceptron example (single)", () => {
		it("should forward and backward pass correctly", () => {
			const x1 = new Tensor({ shape: [1, 1], values: [2], label: "x1" });
			const x2 = new Tensor({ shape: [1, 1], values: [0], label: "x2" });
			const w1 = new Tensor({ shape: [1, 1], values: [-3], label: "w1" });
			const w2 = new Tensor({ shape: [1, 1], values: [1], label: "w2" });
			const b = new Tensor({ shape: [1, 1], values: [6.881373587019432], label: "b" });

			const x1w1 = x1.mul(w1);
			const x2w2 = x2.mul(w2);

			const i1 = x1w1.add(x2w2);
			const t1 = i1.add(b);
			const o1 = t1.tanh();

			assertArrayAlmostEquals(o1.values[0], 0.70710, 1e-5);
			assertArrayAlmostEquals(t1.values[0], 0.88137, 1e-5);
			assertArrayAlmostEquals(i1.values[0], -6, 1e-5);
			o1.backward();
			assertArrayAlmostEquals(o1.gradient[0], 1);
			assertArrayAlmostEquals(t1.gradient[0], 0.5, 1e-5);
			assertArrayAlmostEquals(b.gradient[0], 0.5, 1e-5);
			assertArrayAlmostEquals(i1.gradient[0], 0.5, 1e-5);
			assertArrayAlmostEquals(x1w1.gradient[0], 0.5, 1e-5);
			assertArrayAlmostEquals(x2w2.gradient[0], 0.5, 1e-5);

			assertArrayAlmostEquals(x1.gradient[0], -1.5, 1e-5);
			assertArrayAlmostEquals(x2.gradient[0], 0.5, 1e-5);
			assertArrayAlmostEquals(w1.gradient[0], 1, 1e-5);
			assertArrayAlmostEquals(w2.gradient[0], 0, 1e-5);
		});
	});
	describe("perception example (2x2)", () => {
		it("should forward and backward pass correctly", () => {
			const x = new Tensor({ shape: [2, 1], values: [2, 0], label: "x" });
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
	});
	describe("filled", () => {
		it("should give back ones", () => {
			const tensor = Tensor.filled(1, [2, 3]);
			assertEquals(tensor.shape, [2, 3]);
			assertEquals(tensor.values, new Float32Array([1, 1, 1, 1, 1, 1]));
		});
		it("should give back zeros", () => {
			const tensor = Tensor.filled(0, [3, 2]);
			assertEquals(tensor.shape, [3, 2]);
			assertEquals(tensor.values, new Float32Array([0, 0, 0, 0, 0, 0]));
		});
	});
	describe("getLinearSpace", () => {
		it("should get linear spacing values", () => {
			const t1 = Tensor.getLinearSpace(0, 100, 5);
			assertEquals(t1.shape, [5]);
			assertEquals(t1.values, new Float32Array([0, 25, 50, 75, 100]));

			const t2 = Tensor.getLinearSpace(3, 9, 4);
			assertEquals(t2.shape, [4]);
			assertEquals(t2.values, new Float32Array([3, 5, 7, 9]));

			const t3 = Tensor.getLinearSpace(0, 10, 2);
			assertEquals(t3.shape, [2]);
			assertEquals(t3.values, new Float32Array([0, 10]));
		});
	});
});