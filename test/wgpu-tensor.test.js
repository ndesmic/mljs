import { assertEquals } from "@std/assert";
import { describe, it, beforeEach, afterEach } from "@std/testing/bdd";
import { WGPUTensor } from "../src/js/wgpu-tensor.js";
import { assertArrayAlmostEquals } from "./test-utils.js";
import { getRandom } from "../src/js/tensor-utils.js";

describe("WGPUTensor", () => {
	let adapter;
	let device;
	beforeEach(async () => {
		adapter = await navigator.gpu.requestAdapter();
		device = await adapter.requestDevice();
	});
	afterEach(() => {
		device.destroy();
	});
	describe("add", () => {
		it("should forward pass", async () => {
			const t1 = new WGPUTensor({ shape: [2, 2], values: [1, 2, 3, 4], device });
			const t2 = new WGPUTensor({ shape: [2, 2], values: [5, 6, 7, 8], device });
			const result = await t1.add(t2);

			assertEquals(result.shape, [2, 2]);
			assertArrayAlmostEquals(result.values, new Float32Array([6, 8, 10, 12]));
		});
		it("should backprop", async () => {
			const t1 = new WGPUTensor({ shape: [2, 2], values: [1, 2, 3, 4], device });
			const t2 = new WGPUTensor({ shape: [2, 2], values: [5, 6, 7, 8], device });
			const result = await t1.add(t2);

			assertArrayAlmostEquals(result.values, new Float32Array([6, 8, 10, 12]));
			await result.backward();
			assertArrayAlmostEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
			assertArrayAlmostEquals(t1.gradient, new Float32Array([1, 1, 1, 1]));
			assertArrayAlmostEquals(t2.gradient, new Float32Array([1, 1, 1, 1]));
		});
		it("should allow same node", async () => {
			const t1 = new WGPUTensor({ shape: [2, 2], values: [1, 2, 3, 4], device });
			const result = await t1.add(t1);

			assertArrayAlmostEquals(result.values, new Float32Array([2, 4, 6, 8]));
			await result.backward();
			assertArrayAlmostEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
			assertArrayAlmostEquals(t1.gradient, new Float32Array([2, 2, 2, 2]));
		});
	});
	describe("subtract", () => {
		it("should forward pass", async () => {
			const t1 = new WGPUTensor({ shape: [2, 2], values: [11, 22, 33, 44], device });
			const t2 = new WGPUTensor({ shape: [2, 2], values: [5, 6, 7, 8], device })
			const result = await t1.sub(t2);

			assertEquals(result.shape, [2, 2]);
			assertArrayAlmostEquals(result.values, new Float32Array([6, 16, 26, 36]));
		});
		it("should backprop", async () => {
			const t1 = new WGPUTensor({ shape: [2, 2], values: [11, 22, 33, 44], device });
			const t2 = new WGPUTensor({ shape: [2, 2], values: [5, 6, 7, 8], device });
			const result = await t1.sub(t2);

			assertArrayAlmostEquals(result.values, new Float32Array([6, 16, 26, 36]));
			await result.backward();
			assertArrayAlmostEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
			assertArrayAlmostEquals(t1.gradient, new Float32Array([1, 1, 1, 1]));
			assertArrayAlmostEquals(t2.gradient, new Float32Array([-1, -1, -1, -1]));
		});
		it("should allow same node", async () => {
			const tensor = new WGPUTensor({ shape: [2, 2], values: [11, 22, 33, 44], device });
			const result = await tensor.sub(tensor);

			assertArrayAlmostEquals(result.values, new Float32Array([0, 0, 0, 0]));
			await result.backward();
			assertArrayAlmostEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
			assertArrayAlmostEquals(tensor.gradient, new Float32Array([0, 0, 0, 0]));
		});
	});
	describe("multiply", () => {
		it("should forward pass", async () => {
			const t1 = new WGPUTensor({ shape: [2, 2], values: [1, 2, 3, 4], device });
			const t2 = new WGPUTensor({ shape: [2, 2], values: [5, 6, 7, 8], device });
			const result = await t1.mul(t2);

			assertEquals(result.shape, [2, 2]);
			assertArrayAlmostEquals(result.values, new Float32Array([5, 12, 21, 32]));
		});
		it("should backprop", async () => {
			const t1 = new WGPUTensor({ shape: [2, 2], values: [1, 2, 3, 4], device });
			const t2 = new WGPUTensor({ shape: [2, 2], values: [5, 6, 7, 8], device });
			const result = await t1.mul(t2);

			assertEquals(result.values, new Float32Array([5, 12, 21, 32]));
			await result.backward();
			assertArrayAlmostEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
			assertArrayAlmostEquals(t1.gradient, new Float32Array([5, 6, 7, 8]));
			assertArrayAlmostEquals(t2.gradient, new Float32Array([1, 2, 3, 4]));
		});
		it("should allow same node", async () => {
			const t1 = new WGPUTensor({ shape: [2, 2], values: [1, 2, 3, 4], device });
			const result = await t1.mul(t1);

			assertEquals(result.shape, [2, 2]);
			assertArrayAlmostEquals(result.values, new Float32Array([1, 4, 9, 16]));
			await result.backward();
			assertArrayAlmostEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
			assertArrayAlmostEquals(t1.gradient, new Float32Array([2, 4, 6, 8]));
		});
	});
	describe("divide", () => {
		it("should forward pass", async () => {
			const t1 = new WGPUTensor({ shape: [2, 2], values: [10, 24, 21, 4], device });
			const t2 = new WGPUTensor({ shape: [2, 2], values: [5, 6, 7, 8], device })
			const result = await t1.div(t2);

			assertEquals(result.shape, [2, 2]);
			assertArrayAlmostEquals(result.values, new Float32Array([2, 4, 3, 0.5]));
		});
		it("should backprop", async () => {
			const t1 = new WGPUTensor({ shape: [2, 2], values: [10, 24, 21, 4], device });
			const t2 = new WGPUTensor({ shape: [2, 2], values: [5, 6, 7, 8], device })
			const result = await t1.div(t2);

			assertArrayAlmostEquals(result.values, new Float32Array([2, 4, 3, 0.5]));
			await result.backward();
			assertArrayAlmostEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
			assertArrayAlmostEquals(t1.gradient, new Float32Array([1 / 5, 1 / 6, 1 / 7, 1 / 8]));
			assertArrayAlmostEquals(t2.gradient, new Float32Array([-10 / 5 ** 2, -24 / 6 ** 2, -21 / 7 ** 2, -4 / 8 ** 2]));
		});
		it("should allow same node", async () => {
			const t1 = new WGPUTensor({ shape: [2, 2], values: [10, 24, 21, 4], device });
			const result = await t1.div(t1);

			assertEquals(result.shape, [2, 2]);
			assertArrayAlmostEquals(result.values, new Float32Array([1, 1, 1, 1]));
			await result.backward();
			assertArrayAlmostEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
			assertArrayAlmostEquals(t1.gradient, new Float32Array([0, 0, 0, 0]), 1e-7);
		});
	});
	describe("power", () => {
		it("should forward pass", async () => {
			const t1 = new WGPUTensor({ shape: [2, 2], values: [2, 2, 2, 2], device });
			const t2 = new WGPUTensor({ shape: [2, 2], values: [1, 2, 3, 4], device });
			const result = await t1.pow(t2);

			assertEquals(result.shape, [2, 2]);
			assertArrayAlmostEquals(result.values, new Float32Array([2, 4, 8, 16]));
		});
		it("should backprop", async () => {
			const t1 = new WGPUTensor({ shape: [2, 2], values: [2, 2, 2, 2], device });
			const t2 = new WGPUTensor({ shape: [2, 2], values: [1, 2, 3, 4], device });
			const result = await t1.pow(t2);

			assertArrayAlmostEquals(result.values, new Float32Array([2, 4, 8, 16]));
			await result.backward();
			assertArrayAlmostEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
			assertArrayAlmostEquals(t1.gradient, new Float32Array([1, 4, 12, 32]));
			assertArrayAlmostEquals(t2.gradient, new Float32Array([
				Math.log(2) * 2 ** 1,
				Math.log(2) * 2 ** 2,
				Math.log(2) * 2 ** 3,
				Math.log(2) * 2 ** 4
			]));
		});
		it("should allow same node", async () => {
			const t1 = new WGPUTensor({ shape: [2, 2], values: [1, 2, 3, 4], device });
			const result = await t1.pow(t1);

			assertArrayAlmostEquals(result.values, new Float32Array([1, 4, 27, 256]));
			await result.backward();
			assertArrayAlmostEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
			assertArrayAlmostEquals(t1.gradient, new Float32Array([1.0000, 6.7726, 56.6625, 610.8914]), 1e-4);
		});
	});
	describe("negate", () => {
		it("should forward pass", async () => {
			const t1 = new WGPUTensor({ shape: [2, 2], values: [10, 24, 21, 4], device });
			const result = await t1.neg();

			assertEquals(result.shape, [2, 2]);
			assertArrayAlmostEquals(result.values, new Float32Array([-10, -24, -21, -4]));
		});
		it("should backprop", async () => {
			const t1 = new WGPUTensor({ shape: [2, 2], values: [10, 24, 21, 4], device });
			const result = await t1.neg();

			assertEquals(result.values, new Float32Array([-10, -24, -21, -4]));
			await result.backward();
			assertArrayAlmostEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
			assertArrayAlmostEquals(t1.gradient, new Float32Array([-1, -1, -1, -1]));
		});
	});
	describe("exponential", () => {
		it("should forward pass", async () => {
			const t1 = new WGPUTensor({ shape: [2, 2], values: [1, 2, 3, 4], device });
			const result = await t1.exp();

			assertEquals(result.shape, [2, 2]);
			assertArrayAlmostEquals(result.values, new Float32Array([Math.exp(1), Math.exp(2), Math.exp(3), Math.exp(4)]));
		});
		it("should backprop", async () => {
			const t1 = new WGPUTensor({ shape: [2, 2], values: [1, 2, 3, 4], device });
			const result = await t1.exp();

			assertArrayAlmostEquals(result.values, new Float32Array([Math.exp(1), Math.exp(2), Math.exp(3), Math.exp(4)]));
			await result.backward();
			assertArrayAlmostEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
			assertArrayAlmostEquals(t1.gradient, new Float32Array([Math.exp(1), Math.exp(2), Math.exp(3), Math.exp(4)]));
		});
	});
	describe("hyperbolic tangent", () => {
		it("should forward pass", async () => {
			const t1 = new WGPUTensor({ shape: [2, 2], values: [1, 2, 3, 4], device });
			const result = await t1.tanh();

			assertEquals(result.shape, [2, 2]);
			assertArrayAlmostEquals(result.values, new Float32Array([Math.tanh(1), Math.tanh(2), Math.tanh(3), Math.tanh(4)]));
		});
		it("should backprop", async () => {
			const t1 = new WGPUTensor({ shape: [2, 2], values: [1, 2, 3, 4], device });
			const result = await t1.tanh();

			assertArrayAlmostEquals(result.values, new Float32Array([Math.tanh(1), Math.tanh(2), Math.tanh(3), Math.tanh(4)]));
			await result.backward();
			assertArrayAlmostEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
			assertArrayAlmostEquals(t1.gradient, new Float32Array([
				1 - Math.tanh(1) ** 2,
				1 - Math.tanh(2) ** 2,
				1 - Math.tanh(3) ** 2,
				1 - Math.tanh(4) ** 2
			]), 1e-7);
		});
	});
	describe("sum", () => {
		it("should sum a 4x3 across rows", async () => {
			const tensor = new WGPUTensor({
				shape: [4, 3],
				values: [
					1, 2, 3, 4,
					5, 6, 7, 8,
					9, 10, 11, 12
				],
				device
			});

			const result = await tensor.sum({ dimensionToReduce: 0 });
			assertEquals(result.shape, [3]);
			assertArrayAlmostEquals(result.values, new Float32Array([10, 26, 42]));
		});
		it("should sum a 4x3 across cols", async () => {
			const tensor = new WGPUTensor({
				shape: [4, 3],
				values: [
					1, 2, 3, 4,
					5, 6, 7, 8,
					9, 10, 11, 12
				],
				device
			});

			const result = await tensor.sum({ dimensionToReduce: 1 });
			assertEquals(result.shape, [4]);
			assertArrayAlmostEquals(result.values, new Float32Array([15, 18, 21, 24]));
		});
		it("should sum a 3x3x3 across rows", async () => {
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

			const result = await tensor.sum({ dimensionToReduce: 0 });
			assertEquals(result.shape, [3, 3]);
			assertArrayAlmostEquals(result.values, new Float32Array([
				6, 15, 24,
				33, 42, 51,
				60, 69, 78
			]));
		});
		it("should sum a 3x3x3 across cols", async () => {
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

			const result = await tensor.sum({ dimensionToReduce: 1 });
			assertEquals(result.shape, [3, 3]);
			assertArrayAlmostEquals(result.values, new Float32Array([
				12, 15, 18,
				39, 42, 45,
				66, 69, 72
			]));
		});
		it("should sum a 3x3x3 across depths", async () => {
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

			const result = await tensor.sum({ dimensionToReduce: 2 });
			assertEquals(result.shape, [3, 3]);
			assertArrayAlmostEquals(result.values, new Float32Array([
				30, 33, 36,
				39, 42, 45,
				48, 51, 54
			]));
		});
		it("should backprop", async () => {
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

			const result = await tensor.sum({ dimensionToReduce: 0 });
			assertEquals(result.shape, [3, 3]);
			assertEquals(result.values, new Float32Array([
				6, 15, 24,
				33, 42, 51,
				60, 69, 78
			]));
			await result.backward();
			assertEquals(tensor.gradient, new Float32Array([
				1, 1, 1,
				1, 1, 1,
				1, 1, 1,

				1, 1, 1,
				1, 1, 1,
				1, 1, 1,

				1, 1, 1,
				1, 1, 1,
				1, 1, 1
			]));
		});
		it("should backprop 2", async () => {
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
			await result.backward();
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
			const tensor = new WGPUTensor({ shape: [2, 2], values: [10, 24, 21, 4], device });

			assertEquals(tensor.toString(), "<10, 24, 21, 4>");
		});
		it("should have a nice string with label", () => {
			const tensor = new WGPUTensor({ shape: [2, 2], values: [10, 24, 21, 4], label: "t1", device });

			assertEquals(tensor.toString(), "<t1:10, 24, 21, 4>");
		});
	});
	describe("perceptron example (single)", () => {
		it("should forward and backward pass correctly", async () => {
			const x1 = new WGPUTensor({ shape: [1, 1], values: [2], label: "x1", device });
			const x2 = new WGPUTensor({ shape: [1, 1], values: [0], label: "x2", device });
			const w1 = new WGPUTensor({ shape: [1, 1], values: [-3], label: "w1", device });
			const w2 = new WGPUTensor({ shape: [1, 1], values: [1], label: "w2", device });
			const b = new WGPUTensor({ shape: [1, 1], values: [6.881373587019432], label: "b", device });

			const x1w1 = await x1.mul(w1);
			const x2w2 = await x2.mul(w2);

			const i1 = await x1w1.add(x2w2);
			const t1 = await i1.add(b);
			const o1 = await t1.tanh();

			assertArrayAlmostEquals(o1.values[0], 0.70710, 1e-5);
			assertArrayAlmostEquals(t1.values[0], 0.88137, 1e-5);
			assertArrayAlmostEquals(i1.values[0], -6, 1e-5);
			await o1.backward();
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
		it("should forward and backward pass correctly", async () => {
			const x = new WGPUTensor({ shape: [2, 2], values: [2, 0, 6, 9], label: "x", device });
			const w = new WGPUTensor({ shape: [2, 2], values: [-3, 1, -1, 2], label: "w", device });
			const b = new WGPUTensor({ shape: [2], values: [4, 5], label: "b", device });

			const xw = await x.mul(w);
			const i = await xw.sum({ dimensionToReduce: 0 });
			const t = await i.add(b);
			const o = await t.tanh();

			assertArrayAlmostEquals(xw.values, [-6, 0, -6, 18]);
			assertArrayAlmostEquals(i.values, [-6, 12]);
			assertArrayAlmostEquals(t.values, [-2, 17]);
			assertArrayAlmostEquals(o.values, new Float32Array([-0.9640, 1]), 1e-4)

			await o.backward();

			assertArrayAlmostEquals(o.gradient, [1, 1]);
			assertArrayAlmostEquals(t.gradient, [0.0707, 0], 1e-4);
			assertArrayAlmostEquals(i.gradient, [0.0707, 0], 1e-4);
			assertArrayAlmostEquals(xw.gradient, [0.0707, 0.0707, 0, 0], 1e-4);
			assertArrayAlmostEquals(b.gradient, [0.0707, 0], 1e-4);
			assertArrayAlmostEquals(w.gradient, [0.1413, 0, 0, 0], 1e-4);
			assertArrayAlmostEquals(x.gradient, [-0.212, 0.0707, 0, 0], 1e-4);
		});
	});
	describe("filled", () => {
		it("should give back ones", () => {
			const tensor = WGPUTensor.filled(1, [2, 3], { device });
			assertEquals(tensor.shape, [2, 3]);
			assertEquals(tensor.values, new Float32Array([1, 1, 1, 1, 1, 1]));
		});
		it("should give back zeros", () => {
			const tensor = WGPUTensor.filled(0, [3, 2], { device });
			assertEquals(tensor.shape, [3, 2]);
			assertEquals(tensor.values, new Float32Array([0, 0, 0, 0, 0, 0]));
		});
	});
	describe("random", () => {
		it("should give back random values (seeded)", () => {
			const generator = getRandom(0, 10, 77);
			const tensor = WGPUTensor.random([3, 3], { generator, device });
			assertEquals(tensor.shape, [3, 3]);
			assertArrayAlmostEquals(tensor.values, [
				0.006026304326951504,
				1.2840968370437622,
				1.8160980939865112,
				3.1606016159057617,
				0.23077280819416046,
				8.598573684692383,
				6.224354267120361,
				2.725831985473633,
				3.058232545852661,
			]);
		});
	});
	describe("getLinearSpace", () => {
		it("should get linear spacing values", () => {
			const t1 = WGPUTensor.getLinearSpace(0, 100, 5, { device });
			assertEquals(t1.shape, [5]);
			assertEquals(t1.values, new Float32Array([0, 25, 50, 75, 100]));

			const t2 = WGPUTensor.getLinearSpace(3, 9, 4, { device });
			assertEquals(t2.shape, [4]);
			assertEquals(t2.values, new Float32Array([3, 5, 7, 9]));

			const t3 = WGPUTensor.getLinearSpace(0, 10, 2, { device });
			assertEquals(t3.shape, [2]);
			assertEquals(t3.values, new Float32Array([0, 10]));
		});
	});
});