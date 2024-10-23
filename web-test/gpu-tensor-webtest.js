import { Tensor } from "../src/gpu-tensor.js";

const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();

const t1 = new Tensor({ shape: [2, 2], values: [1, 2, 3, 4], device });
const result = await t1.tanh()

await result.backward();

console.log(t1.gradient);