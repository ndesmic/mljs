/**
 * @typedef {{
 * 	direction: "in" | "out",
 *  name: string
 * }} Params
 * 
 * Compiles a webgpu kernel into a function
 * @param {{
 *   device: GPUDevice,
 *   code: string,
 *   bindGroupLayout: Params[],
 * }} options
 * @returns {({ inputs: Float32Array[], outputSize: number }) => Promise<Float32Array>}
 */
export function compileKernel({
	device,
	code,
	params,
}) {
	const module = device.createShaderModule({ code });
	const bindGroupLayout = getBindGroupLayoutForParams(device, params);

	const pipeline = device.createComputePipeline({
		layout: device.createPipelineLayout({
			bindGroupLayouts: [bindGroupLayout]
		}),
		compute: {
			module,
			entryPoint: "main"
		}
	});

	return async function ({ inputs, outputs }) {
		const bindGroupEntries = [];

		for (let i = 0; i < inputs.length; i++) {
			const inputBuffer = inputs[i];
			switch (params[i].type) {
				case "array": {
					const gpuBuffer = device.createBuffer({ size: inputBuffer.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
					bindGroupEntries.push({
						binding: i,
						resource: {
							buffer: gpuBuffer
						}
					});
					device.queue.writeBuffer(gpuBuffer, 0, inputBuffer);
					break;
				}
				case "boolean": {
					const gpuBuffer = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
					bindGroupEntries.push({
						binding: i,
						resource: {
							buffer: gpuBuffer
						}
					});
					device.queue.writeBuffer(gpuBuffer, 0, inputBuffer ? new Uint8Array([0xFF, 0xFF, 0xFF, 0xFF]) : new Uint8Array([0, 0, 0, 0]));
					break;
				}
			}
		}

		const outputBuffers = new Array(outputs.length);
		const stagingBuffers = new Array(outputs.length);
		for (let i = 0; i < outputs.length; i++) {
			outputBuffers[i] = device.createBuffer({
				size: outputs[i].byteLength,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
			});
			stagingBuffers[i] = device.createBuffer({
				size: outputs[i].byteLength,
				usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
			});
			bindGroupEntries.push({
				binding: bindGroupEntries.length,
				resource: {
					buffer: outputBuffers[i]
				}
			});
		}

		const bindGroup = device.createBindGroup({
			layout: bindGroupLayout,
			entries: bindGroupEntries
		});

		const commandEncoder = device.createCommandEncoder();
		const passEncoder = commandEncoder.beginComputePass();
		passEncoder.setPipeline(pipeline);
		passEncoder.setBindGroup(0, bindGroup);
		passEncoder.dispatchWorkgroups(Math.ceil(Math.max(...inputs.filter(b => b.byteLength).map(b => b.byteLength)) / 8));
		passEncoder.end();
		for(let i = 0; i < stagingBuffers.length; i++){
			commandEncoder.copyBufferToBuffer(outputBuffers[i], 0, stagingBuffers[i], 0, outputs[i].byteLength);
		}
		const commands = commandEncoder.finish();
		device.queue.submit([commands]);

		const outputResults = new Array(outputs.length);
		for(let i = 0; i < outputs.length; i++){
			await stagingBuffers[i].mapAsync(GPUMapMode.READ, 0, outputs[i].byteLength);
			const copyArrayBuffer = stagingBuffers[i].getMappedRange(0, outputs[i].byteLength);

			const data = copyArrayBuffer.slice(0);
			stagingBuffers[i].unmap();
			outputResults[i] = new Float32Array(data);
		}

		return outputResults;
	}
}
export function compileOp(device, op){
	return {
		forward: compileKernel({ device, code: op.forward.code, params: op.forward.params }),
		backward: compileKernel({ device, code: op.backward.code, params: op.backward.params })
	}
}

function getBindGroupLayoutForParams(device, params){
	const entries = params.map((param, i) => {
		return {
			binding: i,
			visibility: GPUShaderStage.COMPUTE,
			buffer: {
				type: param.direction === "in" ? "read-only-storage" : "storage"
			}
		}
	});
	return device.createBindGroupLayout({
		entries
	})
}

const cache = new Map();
export function getDeviceKernels(device){
	if(cache.get(device)){
		return cache.get(device);
	}
	const kernels = {
		add: compileOp(device, addOp),
		sub: compileOp(device, subOp),
		mul: compileOp(device, mulOp),
		div: compileOp(device, divOp),
		neg: compileOp(device, negOp),
		pow: compileOp(device, powOp),
		exp: compileOp(device, expOp),
		tanh: compileOp(device, tanhOp)
	};
	cache.set(device, kernels);
	return kernels;
}

const addOp = {
	forward: {
		params: [
			{ name: "leftHandSide", direction: "in", type: "array" },
			{ name: "rightHandSide", direction: "in", type: "array" },
			{ name: "output", direction: "out", type: "array" }
		],
		code: `
			@group(0) @binding(0)
			var<storage, read> leftHandSide: array<f32>;

			@group(0) @binding(1)
			var<storage, read> rightHandSide: array<f32>;

			@group(0) @binding(2)
			var<storage, read_write> output: array<f32>;

			@compute @workgroup_size(8, 8)
			fn main(
				@builtin(global_invocation_id) global_id: vec3<u32>
			){
				if(global_id.x > arrayLength(&leftHandSide)) {
					return;
				}
				let idx = global_id.x;
				output[idx] = leftHandSide[idx] + rightHandSide[idx];
			}
			`
	},
	backward: {
		params: [
			{ name: "leftGradient", direction: "in", type: "array" },
			{ name: "rightGradient", direction: "in", type: "array" },
			{ name: "resultGradient", direction: "in", type: "array" },
			{ name: "isSame", direction: "in", type: "boolean" },
			{ name: "leftGradientOutput", direction: "out", type: "array" },
			{ name: "rightGradientOutput", direction: "out", type: "array" },
		],
		code: `
			@group(0) @binding(0)
			var<storage, read> leftGradient: array<f32>;

			@group(0) @binding(1)
			var<storage, read> rightGradient: array<f32>;

			@group(0) @binding(2)
			var<storage, read> resultGradient: array<f32>;

			@group(0) @binding(3)
			var<storage, read> is_same: u32;

			@group(0) @binding(4)
			var<storage, read_write> leftGradientOutput: array<f32>;

			@group(0) @binding(5)
			var<storage, read_write> rightGradientOutput: array<f32>;

			@compute @workgroup_size(8, 8)
			fn main(
				@builtin(global_invocation_id) global_id: vec3<u32>
			){
				if(global_id.x > arrayLength(&leftGradient)) { //buffers are the same size so pick one
					return;
				}
				let idx = global_id.x;

				if is_same == 0xFFFFFFFF {
					leftGradientOutput[idx] = leftGradient[idx] + resultGradient[idx] + rightGradient[idx] + resultGradient[idx];
					rightGradientOutput[idx] = rightGradient[idx] + resultGradient[idx] + leftGradient[idx] + resultGradient[idx];
				} else {
					leftGradientOutput[idx] = leftGradient[idx] + resultGradient[idx];
					rightGradientOutput[idx] = rightGradient[idx] + resultGradient[idx];
				}
			}
		`
	}
}

const subOp = {
	forward: {
		params: [
			{ name: "leftHandSide", direction: "in", type: "array" },
			{ name: "rightHandSide", direction: "in", type: "array" },
			{ name: "output", direction: "out", type: "array" }
		],
		code: `
			@group(0) @binding(0)
			var<storage, read> leftHandSide: array<f32>;

			@group(0) @binding(1)
			var<storage, read> rightHandSide: array<f32>;

			@group(0) @binding(2)
			var<storage, read_write> output: array<f32>;

			@compute @workgroup_size(8, 8)
			fn main(
				@builtin(global_invocation_id) global_id: vec3<u32>
			){
				if(global_id.x > arrayLength(&leftHandSide)) {
					return;
				}
				let idx = global_id.x;
				output[idx] = leftHandSide[idx] - rightHandSide[idx];
			}
			`
	},
	backward: {
		params: [
			{ name: "leftGradient", direction: "in", type: "array" },
			{ name: "rightGradient", direction: "in", type: "array" },
			{ name: "resultGradient", direction: "in", type: "array" },
			{ name: "isSame", direction: "in", type: "boolean" },
			{ name: "leftGradientOutput", direction: "out", type: "array" },
			{ name: "rightGradientOutput", direction: "out", type: "array" }
		],
		code: `
			@group(0) @binding(0)
			var<storage, read> leftGradient: array<f32>;

			@group(0) @binding(1)
			var<storage, read> rightGradient: array<f32>;

			@group(0) @binding(2)
			var<storage, read> resultGradient: array<f32>;

			@group(0) @binding(3)
			var<storage, read> is_same: u32;

			@group(0) @binding(4)
			var<storage, read_write> leftGradientOutput: array<f32>;

			@group(0) @binding(5)
			var<storage, read_write> rightGradientOutput: array<f32>;

			@compute @workgroup_size(8, 8)
			fn main(
				@builtin(global_invocation_id) global_id: vec3<u32>
			){
				if(global_id.x > arrayLength(&leftGradient)) { //buffers are the same size so pick one
					return;
				}
				let idx = global_id.x;

				if is_same == 0xFFFFFFFF {
					leftGradientOutput[idx] = leftGradient[idx] + resultGradient[idx] + (rightGradient[idx] - resultGradient[idx]);
					rightGradientOutput[idx] = (rightGradient[idx] - resultGradient[idx]) + leftGradient[idx] + resultGradient[idx];
				} else {
					leftGradientOutput[idx] = leftGradient[idx] + resultGradient[idx];
					rightGradientOutput[idx] = rightGradient[idx] - resultGradient[idx];
				}
			}
		`
	}
}

const mulOp = {
	forward: {
		params: [
			{ name: "leftHandSide", direction: "in", type: "array" },
			{ name: "rightHandSide", direction: "in", type: "array" },
			{ name: "output", direction: "out", type: "array" }
		],
		code: `
			@group(0) @binding(0)
			var<storage, read> leftHandSide: array<f32>;

			@group(0) @binding(1)
			var<storage, read> rightHandSide: array<f32>;

			@group(0) @binding(2)
			var<storage, read_write> output: array<f32>;

			@compute @workgroup_size(8, 8)
			fn main(
				@builtin(global_invocation_id) global_id: vec3<u32>
			){
				if(global_id.x > arrayLength(&leftHandSide)) {
					return;
				}
				let idx = global_id.x;
				output[idx] = leftHandSide[idx] * rightHandSide[idx];
			}
			`
	},
	backward: {
		params: [
			{ name: "leftValues", direction: "in", type: "array" },
			{ name: "rightValues", direction: "in", type: "array" },
			{ name: "leftGradient", direction: "in", type: "array" },
			{ name: "rightGradient", direction: "in", type: "array" },
			{ name: "resultGradient", direction: "in", type: "array" },
			{ name: "isSame", direction: "in", type: "boolean" },
			{ name: "leftGradientOutput", direction: "out", type: "array" },
			{ name: "rightGradientOutput", direction: "out", type: "array" },
		],
		code: `
			@group(0) @binding(0)
			var<storage, read> leftValues: array<f32>;

			@group(0) @binding(1)
			var<storage, read> rightValues: array<f32>;

			@group(0) @binding(2)
			var<storage, read> leftGradient: array<f32>;

			@group(0) @binding(3)
			var<storage, read> rightGradient: array<f32>;

			@group(0) @binding(4)
			var<storage, read> resultGradient: array<f32>;

			@group(0) @binding(5)
			var<storage, read> is_same: u32;

			@group(0) @binding(6)
			var<storage, read_write> leftGradientOutput: array<f32>;

			@group(0) @binding(7)
			var<storage, read_write> rightGradientOutput: array<f32>;

			@compute @workgroup_size(8, 8)
			fn main(
				@builtin(global_invocation_id) global_id: vec3<u32>
			){
				if(global_id.x > arrayLength(&leftGradient)) { //buffers are the same size so pick one
					return;
				}
				let idx = global_id.x;

				if is_same == 0xFFFFFFFF {
					leftGradientOutput[idx] = leftGradient[idx] + (rightValues[idx] * resultGradient[idx] + leftValues[idx] * resultGradient[idx]);
					rightGradientOutput[idx] = rightGradient[idx] + (leftValues[idx] * resultGradient[idx] + rightValues[idx] * resultGradient[idx]);
				} else {
					leftGradientOutput[idx] = leftGradient[idx] + rightValues[idx] * resultGradient[idx];
					rightGradientOutput[idx] = rightGradient[idx] + leftValues[idx] * resultGradient[idx];
				}
			}
		`
	}
}

const divOp = {
	forward: {
		params: [
			{ name: "leftHandSide", direction: "in", type: "array" },
			{ name: "rightHandSide", direction: "in", type: "array" },
			{ name: "output", direction: "out", type: "array" }
		],
		code: `
			@group(0) @binding(0)
			var<storage, read> leftHandSide: array<f32>;

			@group(0) @binding(1)
			var<storage, read> rightHandSide: array<f32>;

			@group(0) @binding(2)
			var<storage, read_write> output: array<f32>;

			@compute @workgroup_size(8, 8)
			fn main(
				@builtin(global_invocation_id) global_id: vec3<u32>
			){
				if(global_id.x > arrayLength(&leftHandSide)) {
					return;
				}
				let idx = global_id.x;
				output[idx] = leftHandSide[idx] / rightHandSide[idx];
			}
			`
	},
	backward: {
		params: [
			{ name: "leftValues", direction: "in", type: "array" },
			{ name: "rightValues", direction: "in", type: "array" },
			{ name: "leftGradient", direction: "in", type: "array" },
			{ name: "rightGradient", direction: "in", type: "array" },
			{ name: "resultGradient", direction: "in", type: "array" },
			{ name: "isSame", direction: "in", type: "boolean" },
			{ name: "leftGradientOutput", direction: "out", type: "array" },
			{ name: "rightGradientOutput", direction: "out", type: "array" },
		],
		code: `
			@group(0) @binding(0)
			var<storage, read> leftValues: array<f32>;

			@group(0) @binding(1)
			var<storage, read> rightValues: array<f32>;

			@group(0) @binding(2)
			var<storage, read> leftGradient: array<f32>;

			@group(0) @binding(3)
			var<storage, read> rightGradient: array<f32>;

			@group(0) @binding(4)
			var<storage, read> resultGradient: array<f32>;

			@group(0) @binding(5)
			var<storage, read> is_same: u32;

			@group(0) @binding(6)
			var<storage, read_write> leftGradientOutput: array<f32>;

			@group(0) @binding(7)
			var<storage, read_write> rightGradientOutput: array<f32>;

			@compute @workgroup_size(8, 8)
			fn main(
				@builtin(global_invocation_id) global_id: vec3<u32>
			){
				if(global_id.x > arrayLength(&leftGradient)) { //buffers are the same size so pick one
					return;
				}
				let idx = global_id.x;

				if is_same == 0xFFFFFFFF {
					leftGradientOutput[idx] = leftGradient[idx] + (leftGradient[idx] + 1 / rightValues[idx] * resultGradient[idx]) + (-(leftValues[idx] / rightValues[idx] * rightValues[idx]) * resultGradient[idx]);
					rightGradientOutput[idx] = rightGradient[idx] + (-(leftValues[idx] / rightValues[idx] * rightValues[idx]) * resultGradient[idx]) + (1 / rightValues[idx] * resultGradient[idx]);
				} else {
					leftGradientOutput[idx] = leftGradient[idx] + 1 / rightValues[idx] * resultGradient[idx];
					rightGradientOutput[idx] = -(leftValues[idx] / rightValues[idx] * rightValues[idx]) * resultGradient[idx];
				}
			}
		`
	}
}

const negOp = {
	forward: {
		params: [
			{ name: "input", direction: "in", type: "array" },
			{ name: "output", direction: "out", type: "array" }
		],
		code: `
			@group(0) @binding(0)
			var<storage, read> input: array<f32>;

			@group(0) @binding(1)
			var<storage, read_write> output: array<f32>;

			@compute @workgroup_size(8, 8)
			fn main(
				@builtin(global_invocation_id) global_id: vec3<u32>
			){
				if(global_id.x > arrayLength(&input)) {
					return;
				}
				let idx = global_id.x;
				output[idx] = -input[idx];
			}
			`
	},
	backward: {
		params: [
			{ name: "input", direction: "in", type: "array" },
			{ name: "output", direction: "out", type: "array" }
		],
		code: `
			@group(0) @binding(0)
			var<storage, read> input: array<f32>;

			@group(0) @binding(1)
			var<storage, read_write> output: array<f32>;

			@compute @workgroup_size(8, 8)
			fn main(
				@builtin(global_invocation_id) global_id: vec3<u32>
			){
				if(global_id.x > arrayLength(&input)) {
					return;
				}
				let idx = global_id.x;
				output[idx] = -input[idx];
			}
			`
	}
}

const powOp = {
	forward: {
		params: [
			{ name: "leftHandSide", direction: "in", type: "array" },
			{ name: "rightHandSide", direction: "in", type: "array" },
			{ name: "output", direction: "out", type: "array" }
		],
		code: `
			@group(0) @binding(0)
			var<storage, read> leftHandSide: array<f32>;

			@group(0) @binding(1)
			var<storage, read> rightHandSide: array<f32>;

			@group(0) @binding(2)
			var<storage, read_write> output: array<f32>;

			@compute @workgroup_size(8, 8)
			fn main(
				@builtin(global_invocation_id) global_id: vec3<u32>
			){
				if(global_id.x > arrayLength(&leftHandSide)) {
					return;
				}
				let idx = global_id.x;
				output[idx] = pow(leftHandSide[idx], rightHandSide[idx]);
			}
			`
	},
	backward: {
		params: [
			{ name: "leftValues", direction: "in", type: "array" },
			{ name: "rightValues", direction: "in", type: "array" },
			{ name: "leftGradient", direction: "in", type: "array" },
			{ name: "rightGradient", direction: "in", type: "array" },
			{ name: "resultGradient", direction: "in", type: "array" },
			{ name: "isSame", direction: "in", type: "boolean" },
			{ name: "leftGradientOutput", direction: "out", type: "array" },
			{ name: "rightGradientOutput", direction: "out", type: "array" },
		],
		code: `
			@group(0) @binding(0)
			var<storage, read> leftValues: array<f32>;

			@group(0) @binding(1)
			var<storage, read> rightValues: array<f32>;

			@group(0) @binding(2)
			var<storage, read> leftGradient: array<f32>;

			@group(0) @binding(3)
			var<storage, read> rightGradient: array<f32>;

			@group(0) @binding(4)
			var<storage, read> resultGradient: array<f32>;

			@group(0) @binding(5)
			var<storage, read> is_same: u32;

			@group(0) @binding(6)
			var<storage, read_write> leftGradientOutput: array<f32>;

			@group(0) @binding(7)
			var<storage, read_write> rightGradientOutput: array<f32>;

			@compute @workgroup_size(8, 8)
			fn main(
				@builtin(global_invocation_id) global_id: vec3<u32>
			){
				if(global_id.x > arrayLength(&leftGradient)) { //buffers are the same size so pick one
					return;
				}
				let idx = global_id.x;

				if is_same == 0xFFFFFFFF {
					leftGradientOutput[idx] = leftGradient[idx] + (leftGradient[idx] + 1 / rightValues[idx] * resultGradient[idx]) + (-(leftValues[idx] / rightValues[idx] * rightValues[idx]) * resultGradient[idx]);
					rightGradientOutput[idx] = rightGradient[idx] + (-(leftValues[idx] / rightValues[idx] * rightValues[idx]) * resultGradient[idx]) + (1 / rightValues[idx] * resultGradient[idx]);
				} else {
					leftGradientOutput[idx] = rightValues[idx] * pow(leftValues[idx], rightValues[idx] - 1) * resultGradient[idx];
					rightGradientOutput[idx] = log(leftValues[idx]) * pow(leftValues[idx], rightValues[idx]) * resultGradient[idx];
				}
			}
		`
	}
}

const expOp = {
	forward: {
		params: [
			{ name: "input", direction: "in", type: "array" },
			{ name: "output", direction: "out", type: "array" }
		],
		code: `
			@group(0) @binding(0)
			var<storage, read> input: array<f32>;

			@group(0) @binding(1)
			var<storage, read_write> output: array<f32>;

			@compute @workgroup_size(8, 8)
			fn main(
				@builtin(global_invocation_id) global_id: vec3<u32>
			){
				if(global_id.x > arrayLength(&input)) {
					return;
				}
				let idx = global_id.x;
				output[idx] = exp(input[idx]);
			}
			`
	},
	backward: {
		params: [
			{ name: "inputValues", direction: "in", type: "array" },
			{ name: "inputGradient", direction: "in", type: "array" },
			{ name: "resultGradient", direction: "in", type: "array" },
			{ name: "output", direction: "out", type: "array" }
		],
		code: `
			@group(0) @binding(0)
			var<storage, read> inputValues: array<f32>;

			@group(0) @binding(1)
			var<storage, read> inputGradient: array<f32>;

			@group(0) @binding(2)
			var<storage, read> resultGradient: array<f32>;

			@group(0) @binding(3)
			var<storage, read_write> output: array<f32>;

			@compute @workgroup_size(8, 8)
			fn main(
				@builtin(global_invocation_id) global_id: vec3<u32>
			){
				if(global_id.x > arrayLength(&inputGradient)) {
					return;
				}
				let idx = global_id.x;
				output[idx] = inputGradient[idx] + exp(inputValues[idx]) * resultGradient[idx];
			}
			`
	}
}

const tanhOp = {
	forward: {
		params: [
			{ name: "input", direction: "in", type: "array" },
			{ name: "output", direction: "out", type: "array" }
		],
		code: `
			@group(0) @binding(0)
			var<storage, read> input: array<f32>;

			@group(0) @binding(1)
			var<storage, read_write> output: array<f32>;

			@compute @workgroup_size(8, 8)
			fn main(
				@builtin(global_invocation_id) global_id: vec3<u32>
			){
				if(global_id.x > arrayLength(&input)) {
					return;
				}
				let idx = global_id.x;
				output[idx] = tanh(input[idx]);
			}
			`
	},
	backward: {
		params: [
			{ name: "inputValues", direction: "in", type: "array" },
			{ name: "inputGradient", direction: "in", type: "array" },
			{ name: "resultGradient", direction: "in", type: "array" },
			{ name: "output", direction: "out", type: "array" }
		],
		code: `
			@group(0) @binding(0)
			var<storage, read> inputValues: array<f32>;

			@group(0) @binding(1)
			var<storage, read> inputGradient: array<f32>;

			@group(0) @binding(2)
			var<storage, read> resultGradient: array<f32>;

			@group(0) @binding(3)
			var<storage, read_write> output: array<f32>;

			@compute @workgroup_size(8, 8)
			fn main(
				@builtin(global_invocation_id) global_id: vec3<u32>
			){
				if global_id.x > arrayLength(&inputGradient) {
					return;
				}
				let idx = global_id.x;
				let tanh_x = tanh(inputValues[idx]);
				let pow_two = tanh_x * tanh_x;
				let one_minus_pow = 1 - pow_two;
				output[idx] = inputGradient[idx] + (1 - one_minus_pow) * resultGradient[idx];
			}`
	}
}