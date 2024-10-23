import { assertEquals, assertAlmostEquals } from "@std/assert";
import { Tensor } from "../src/gpu-tensor.js";
import { assertArrayAlmostEquals } from "./test-utils.js";

Deno.test("GPUTensor should add", async () => {
	const adapter = await navigator.gpu.requestAdapter();
	const device = await adapter.requestDevice();
	const t1 = new Tensor({ shape: [2, 2], values: [1, 2, 3, 4], device });
	const t2 = new Tensor({ shape: [2, 2], values: [5, 6, 7, 8], device });
	const result = await t1.add(t2);

	assertEquals(result.shape, [2, 2]);
	assertEquals(result.values, new Float32Array([6, 8, 10, 12]));
	device.destroy();
});
Deno.test("GPUTensor should backprop gradient through add", async () => {
	const adapter = await navigator.gpu.requestAdapter();
	const device = await adapter.requestDevice();
	const t1 = new Tensor({ shape: [2, 2], values: [1, 2, 3, 4], device });
	const t2 = new Tensor({ shape: [2, 2], values: [5, 6, 7, 8], device });
	const result = await t1.add(t2);

	assertEquals(result.values, new Float32Array([6, 8, 10, 12]));
	await result.backward();
	assertEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
	assertEquals(t1.gradient, new Float32Array([1, 1, 1, 1]));
	assertEquals(t2.gradient, new Float32Array([1, 1, 1, 1]));
	device.destroy();
});
Deno.test("GPUTensor should add same node", async () => {
	const adapter = await navigator.gpu.requestAdapter();
	const device = await adapter.requestDevice();
	const t1 = new Tensor({ shape: [2, 2], values: [1, 2, 3, 4], device });
	const result = await t1.add(t1);

	assertEquals(result.values, new Float32Array([2, 4, 6, 8]));
	await result.backward();
	assertEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
	assertEquals(t1.gradient, new Float32Array([2, 2, 2, 2]));
	device.destroy();
});
Deno.test("GPUTensor should subtract", async () => {
	const adapter = await navigator.gpu.requestAdapter();
	const device = await adapter.requestDevice();
	const t1 = new Tensor({ shape: [2, 2], values: [11, 22, 33, 44], device });
	const t2 = new Tensor({ shape: [2, 2], values: [5, 6, 7, 8], device })
	const result = await t1.sub(t2);

	assertEquals(result.shape, [2, 2]);
	assertEquals(result.values, new Float32Array([6, 16, 26, 36]));
	device.destroy();
});
Deno.test("GPUTensor should backprop gradient through sub", async () => {
	const adapter = await navigator.gpu.requestAdapter();
	const device = await adapter.requestDevice();
	const t1 = new Tensor({ shape: [2, 2], values: [11, 22, 33, 44], device });
	const t2 = new Tensor({ shape: [2, 2], values: [5, 6, 7, 8], device });
	const result = await t1.sub(t2);

	assertEquals(result.values, new Float32Array([6, 16, 26, 36]));
	await result.backward();
	assertEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
	assertEquals(t1.gradient, new Float32Array([1, 1, 1, 1]));
	assertEquals(t2.gradient, new Float32Array([-1, -1, -1, -1]));
	device.destroy();
});
Deno.test("GPUTensor should subtract same node", async () => {
	const adapter = await navigator.gpu.requestAdapter();
	const device = await adapter.requestDevice();
	const tensor = new Tensor({ shape: [2, 2], values: [11, 22, 33, 44], device });
	const result = await tensor.sub(tensor);

	assertEquals(result.values, new Float32Array([0, 0, 0, 0]));
	await result.backward();
	assertEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
	assertEquals(tensor.gradient, new Float32Array([0, 0, 0, 0]));
	device.destroy();
});
Deno.test("GPUTensor should multiply", async () => {
	const adapter = await navigator.gpu.requestAdapter();
	const device = await adapter.requestDevice();
	const t1 = new Tensor({ shape: [2, 2], values: [1, 2, 3, 4], device });
	const t2 = new Tensor({ shape: [2, 2], values: [5, 6, 7, 8], device });
	const result = await t1.mul(t2);

	assertEquals(result.shape, [2, 2]);
	assertEquals(result.values, new Float32Array([5, 12, 21, 32]));
	device.destroy();
});
Deno.test("GPUTensor should backprop gradient through multiply", async () => {
	const adapter = await navigator.gpu.requestAdapter();
	const device = await adapter.requestDevice();
	const t1 = new Tensor({ shape: [2, 2], values: [1, 2, 3, 4], device });
	const t2 = new Tensor({ shape: [2, 2], values: [5, 6, 7, 8], device });
	const result = await t1.mul(t2);

	assertEquals(result.values, new Float32Array([5, 12, 21, 32]));
	await result.backward();
	assertEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
	assertEquals(t1.gradient, new Float32Array([5, 6, 7, 8]));
	assertEquals(t2.gradient, new Float32Array([1, 2, 3, 4]));
	device.destroy();
});
Deno.test("GPUTensor should multiply same node", async () => {
	const adapter = await navigator.gpu.requestAdapter();
	const device = await adapter.requestDevice();
	const t1 = new Tensor({ shape: [2, 2], values: [1, 2, 3, 4], device });
	const result = await t1.mul(t1);

	assertEquals(result.shape, [2, 2]);
	assertEquals(result.values, new Float32Array([1, 4, 9, 16]));
	await result.backward();
	assertEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
	assertEquals(t1.gradient, new Float32Array([2, 4, 6, 8]));
	device.destroy();
});
Deno.test("GPUTensor should divide", async () => {
	const adapter = await navigator.gpu.requestAdapter();
	const device = await adapter.requestDevice();
	const t1 = new Tensor({ shape: [2, 2], values: [10, 24, 21, 4], device });
	const t2 = new Tensor({ shape: [2, 2], values: [5, 6, 7, 8], device })
	const result = await t1.div(t2);

	assertEquals(result.shape, [2, 2]);
	assertArrayAlmostEquals(result.values, new Float32Array([2, 4, 3, 0.5]));
	device.destroy();
});
Deno.test("GPUTensor should backprop gradient through div", async () => {
	const adapter = await navigator.gpu.requestAdapter();
	const device = await adapter.requestDevice();
	const t1 = new Tensor({ shape: [2, 2], values: [10, 24, 21, 4], device });
	const t2 = new Tensor({ shape: [2, 2], values: [5, 6, 7, 8], device })
	const result = await t1.div(t2);

	assertArrayAlmostEquals(result.values, new Float32Array([2, 4, 3, 0.5]));
	await result.backward();
	assertArrayAlmostEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
	assertArrayAlmostEquals(t1.gradient, new Float32Array([1 / 5, 1 / 6, 1 / 7, 1 / 8]));
	assertArrayAlmostEquals(t2.gradient, new Float32Array([-10 / 5 ** 2, -24 / 6 ** 2, -21 / 7 ** 2, -4 / 8 ** 2]));
	device.destroy();
});
Deno.test("GPUTensor should divide same node", async () => {
	const adapter = await navigator.gpu.requestAdapter();
	const device = await adapter.requestDevice();
	const t1 = new Tensor({ shape: [2, 2], values: [10, 24, 21, 4], device });
	const result = await t1.div(t1);

	assertEquals(result.shape, [2, 2]);
	assertEquals(result.values, new Float32Array([1, 1, 1, 1]));
	await result.backward();
	assertEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
	assertArrayAlmostEquals(t1.gradient, new Float32Array([ //These are actually zero but dealing with precision loss
		0, 0, 0, 0
	]));
	device.destroy();
});

Deno.test("GPUTensor negate", async () => {
	const adapter = await navigator.gpu.requestAdapter();
	const device = await adapter.requestDevice();
	const t1 = new Tensor({ shape: [2, 2], values: [10, 24, 21, 4], device });
	const result = await t1.neg();

	assertEquals(result.shape, [2, 2]);
	assertEquals(result.values, new Float32Array([-10, -24, -21, -4]));
	device.destroy();
});
Deno.test("GPUTensor should backprop gradient through neg", async () => {
	const adapter = await navigator.gpu.requestAdapter();
	const device = await adapter.requestDevice();
	const t1 = new Tensor({ shape: [2, 2], values: [10, 24, 21, 4], device });
	const result = await t1.neg();

	assertEquals(result.values, new Float32Array([-10, -24, -21, -4]));
	await result.backward();
	assertEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
	assertEquals(t1.gradient, new Float32Array([-1, -1, -1, -1]));
	device.destroy();
});

Deno.test("GPUTensor raises to power", async () => {
	const adapter = await navigator.gpu.requestAdapter();
	const device = await adapter.requestDevice();
	const t1 = new Tensor({ shape: [2, 2], values: [2, 2, 2, 2], device });
	const t2 = new Tensor({ shape: [2, 2], values: [1, 2, 3, 4], device });
	const result = await t1.pow(t2);

	assertEquals(result.shape, [2, 2]);
	assertEquals(result.values, new Float32Array([2, 4, 8, 16]));
	device.destroy();
});
Deno.test("GPUTensor should backprop gradient through pow", async () => {
	const adapter = await navigator.gpu.requestAdapter();
	const device = await adapter.requestDevice();
	const t1 = new Tensor({ shape: [2, 2], values: [2, 2, 2, 2], device });
	const t2 = new Tensor({ shape: [2, 2], values: [1, 2, 3, 4], device });
	const result = await t1.pow(t2);

	assertEquals(result.values, new Float32Array([2, 4, 8, 16]));
	await result.backward();
	assertEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
	assertEquals(t1.gradient, new Float32Array([1, 4, 12, 32]));
	assertEquals(t2.gradient, new Float32Array([
		Math.log(2) * 2 ** 1,
		Math.log(2) * 2 ** 2,
		Math.log(2) * 2 ** 3,
		Math.log(2) * 2 ** 4
	]));
	device.destroy();
});
Deno.test("Tensor should pow same node", async () => {
	const adapter = await navigator.gpu.requestAdapter();
	const device = await adapter.requestDevice();
	const t1 = new Tensor({ shape: [2, 2], values: [1, 2, 3, 4], device });
	const result = await t1.pow(t1);

	assertArrayAlmostEquals(result.values, new Float32Array([1, 4, 27, 256]));
	await result.backward();
	assertEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
	assertArrayAlmostEquals(t1.gradient, new Float32Array([1.0000, 6.7726, 56.6625, 610.8914]));
	device.destroy();
});
Deno.test("Tensor exponentiates", async () => {
	const adapter = await navigator.gpu.requestAdapter();
	const device = await adapter.requestDevice();
	const t1 = new Tensor({ shape: [2, 2], values: [1, 2, 3, 4], device });
	const result = await t1.exp();

	assertEquals(result.shape, [2, 2]);
	assertArrayAlmostEquals(result.values, new Float32Array([Math.exp(1), Math.exp(2), Math.exp(3), Math.exp(4)]));
	device.destroy();
});
Deno.test("Tensor should backprop gradient through exp", async () => {
	const adapter = await navigator.gpu.requestAdapter();
	const device = await adapter.requestDevice();
	const t1 = new Tensor({ shape: [2, 2], values: [1, 2, 3, 4], device });
	const result = await t1.exp();

	assertArrayAlmostEquals(result.values, new Float32Array([Math.exp(1), Math.exp(2), Math.exp(3), Math.exp(4)]));
	await result.backward();
	assertEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
	assertArrayAlmostEquals(t1.gradient, new Float32Array([Math.exp(1), Math.exp(2), Math.exp(3), Math.exp(4)]));
	device.destroy();
});

Deno.test("Tensor applies hyperbolic tangent", async () => {
	const adapter = await navigator.gpu.requestAdapter();
	const device = await adapter.requestDevice();
	const t1 = new Tensor({ shape: [2, 2], values: [1, 2, 3, 4], device });
	const result = await t1.tanh();

	assertEquals(result.shape, [2, 2]);
	assertArrayAlmostEquals(result.values, new Float32Array([Math.tanh(1), Math.tanh(2), Math.tanh(3), Math.tanh(4)]));
	device.destroy();
});
Deno.test("Tensor should backprop gradient through tanh", async () => {
	const adapter = await navigator.gpu.requestAdapter();
	const device = await adapter.requestDevice();
	const t1 = new Tensor({ shape: [2, 2], values: [1, 2, 3, 4], device });
	const result = await t1.tanh();

	assertArrayAlmostEquals(result.values, new Float32Array([Math.tanh(1), Math.tanh(2), Math.tanh(3), Math.tanh(4)]));
	await result.backward();
	assertEquals(result.gradient, new Float32Array([1, 1, 1, 1]));
	assertArrayAlmostEquals(t1.gradient, new Float32Array([
		1 - Math.tanh(1) ** 2,
		1 - Math.tanh(2) ** 2,
		1 - Math.tanh(3) ** 2,
		1 - Math.tanh(4) ** 2
	]));
	device.destroy();
});