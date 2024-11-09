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
			{ name: "valuesA", direction: "in", type: "array" },
			{ name: "valuesB", direction: "in", type: "array" },
			{ name: "output", direction: "out", type: "array" }
		],
		code: `
			@group(0) @binding(0)
			var<storage, read> valuesA: array<f32>;

			@group(0) @binding(1)
			var<storage, read> valuesB: array<f32>;

			@group(0) @binding(2)
			var<storage, read_write> output: array<f32>;

			@compute @workgroup_size(8, 8)
			fn main(
				@builtin(global_invocation_id) global_id: vec3<u32>
			){
				let idx = global_id.x;

				if global_id.x < arrayLength(&valuesA) {
					output[idx] = valuesA[idx] + valuesB[idx];
				}
			}
			`
	},
	backward: {
		params: [
			{ name: "gradA", direction: "in", type: "array" },
			{ name: "gradB", direction: "in", type: "array" },
			{ name: "gradResult", direction: "in", type: "array" },
			{ name: "gradAOut", direction: "out", type: "array" },
			{ name: "gradBOut", direction: "out", type: "array" },
		],
		code: `
			@group(0) @binding(0)
			var<storage, read> gradA: array<f32>;

			@group(0) @binding(1)
			var<storage, read> gradB: array<f32>;

			@group(0) @binding(2)
			var<storage, read> gradResult: array<f32>;

			@group(0) @binding(3)
			var<storage, read_write> gradAOut: array<f32>;

			@group(0) @binding(4)
			var<storage, read_write> gradBOut: array<f32>;

			@compute @workgroup_size(8, 8)
			fn main(
				@builtin(global_invocation_id) global_id: vec3<u32>
			){
				let idx = global_id.x;

				if global_id.x < arrayLength(&gradA) { //buffers are the same size so pick one
					gradAOut[idx] = gradA[idx] + gradResult[idx];
					gradBOut[idx] = gradB[idx] + gradResult[idx];
				}
			}
		`
	}
}

const subOp = {
	forward: {
		params: [
			{ name: "valuesA", direction: "in", type: "array" },
			{ name: "valuesB", direction: "in", type: "array" },
			{ name: "output", direction: "out", type: "array" }
		],
		code: `
			@group(0) @binding(0)
			var<storage, read> valuesA: array<f32>;

			@group(0) @binding(1)
			var<storage, read> valuesB: array<f32>;

			@group(0) @binding(2)
			var<storage, read_write> output: array<f32>;

			@compute @workgroup_size(8, 8)
			fn main(
				@builtin(global_invocation_id) global_id: vec3<u32>
			){
				let idx = global_id.x;

				if global_id.x < arrayLength(&valuesA) {
					output[idx] = valuesA[idx] - valuesB[idx];
				}
			}
			`
	},
	backward: {
		params: [
			{ name: "gradA", direction: "in", type: "array" },
			{ name: "gradB", direction: "in", type: "array" },
			{ name: "gradResult", direction: "in", type: "array" },
			{ name: "gradAOut", direction: "out", type: "array" },
			{ name: "gradBOut", direction: "out", type: "array" }
		],
		code: `
			@group(0) @binding(0)
			var<storage, read> gradA: array<f32>;

			@group(0) @binding(1)
			var<storage, read> gradB: array<f32>;

			@group(0) @binding(2)
			var<storage, read> gradResult: array<f32>;

			@group(0) @binding(3)
			var<storage, read_write> gradAOut: array<f32>;

			@group(0) @binding(4)
			var<storage, read_write> gradBOut: array<f32>;

			@compute @workgroup_size(8, 8)
			fn main(
				@builtin(global_invocation_id) global_id: vec3<u32>
			){
				let idx = global_id.x;

				if global_id.x < arrayLength(&gradA) { //buffers are the same size so pick one
					gradAOut[idx] = gradA[idx] + gradResult[idx];
					gradBOut[idx] = gradB[idx] - gradResult[idx];
				}
			}
		`
	}
}

const mulOp = {
	forward: {
		params: [
			{ name: "valuesA", direction: "in", type: "array" },
			{ name: "valuesB", direction: "in", type: "array" },
			{ name: "output", direction: "out", type: "array" }
		],
		code: `
			@group(0) @binding(0)
			var<storage, read> valuesA: array<f32>;

			@group(0) @binding(1)
			var<storage, read> valuesB: array<f32>;

			@group(0) @binding(2)
			var<storage, read_write> output: array<f32>;

			@compute @workgroup_size(8, 8)
			fn main(
				@builtin(global_invocation_id) global_id: vec3<u32>
			){
				let idx = global_id.x;

				if global_id.x < arrayLength(&valuesA) {
					output[idx] = valuesA[idx] * valuesB[idx];
				}
			}
			`
	},
	backward: {
		params: [
			{ name: "valuesA", direction: "in", type: "array" },
			{ name: "valuesB", direction: "in", type: "array" },
			{ name: "gradA", direction: "in", type: "array" },
			{ name: "gradB", direction: "in", type: "array" },
			{ name: "gradResult", direction: "in", type: "array" },
			{ name: "gradAOut", direction: "out", type: "array" },
			{ name: "gradBOut", direction: "out", type: "array" },
		],
		code: `
			@group(0) @binding(0)
			var<storage, read> gradA: array<f32>;

			@group(0) @binding(1)
			var<storage, read> gradB: array<f32>;

			@group(0) @binding(2)
			var<storage, read> gradResult: array<f32>;

			@group(0) @binding(3)
			var<storage, read> valuesA: array<f32>;

			@group(0) @binding(4)
			var<storage, read> valuesB: array<f32>;

			@group(0) @binding(5)
			var<storage, read_write> gradAOut: array<f32>;

			@group(0) @binding(6)
			var<storage, read_write> gradBOut: array<f32>;

			@compute @workgroup_size(8, 8)
			fn main(
				@builtin(global_invocation_id) global_id: vec3<u32>
			){
				let idx = global_id.x;

				if global_id.x < arrayLength(&gradA) { //buffers are the same size so pick one
					gradAOut[idx] = gradA[idx] + valuesB[idx] * gradResult[idx];
					gradBOut[idx] = gradB[idx] + valuesA[idx] * gradResult[idx];
				}
			}
		`
	}
}

const divOp = {
	forward: {
		params: [
			{ name: "valuesA", direction: "in", type: "array" },
			{ name: "valuesB", direction: "in", type: "array" },
			{ name: "output", direction: "out", type: "array" }
		],
		code: `
			@group(0) @binding(0)
			var<storage, read> valuesA: array<f32>;

			@group(0) @binding(1)
			var<storage, read> valuesB: array<f32>;

			@group(0) @binding(2)
			var<storage, read_write> output: array<f32>;

			@compute @workgroup_size(8, 8)
			fn main(
				@builtin(global_invocation_id) global_id: vec3<u32>
			){
				let idx = global_id.x;

				if idx < arrayLength(&valuesA) {
					
					output[idx] = valuesA[idx] / valuesB[idx];
				}
			}
			`
	},
	backward: {
		params: [
			{ name: "gradA", direction: "in", type: "array" },
			{ name: "gradB", direction: "in", type: "array" },
			{ name: "gradResult", direction: "in", type: "array" },
			{ name: "valuesA", direction: "in", type: "array" },
			{ name: "valuesB", direction: "in", type: "array" },
			{ name: "gradAOut", direction: "out", type: "array" },
			{ name: "gradBOut", direction: "out", type: "array" },
		],
		code: `
			@group(0) @binding(0)
			var<storage, read> gradA: array<f32>;

			@group(0) @binding(1)
			var<storage, read> gradB: array<f32>;

			@group(0) @binding(2)
			var<storage, read> gradResult: array<f32>;

			@group(0) @binding(3)
			var<storage, read> valuesA: array<f32>;

			@group(0) @binding(4)
			var<storage, read> valuesB: array<f32>;

			@group(0) @binding(5)
			var<storage, read_write> gradAOut: array<f32>;

			@group(0) @binding(6)
			var<storage, read_write> gradBOut: array<f32>;

			@compute @workgroup_size(8, 8)
			fn main(
				@builtin(global_invocation_id) global_id: vec3<u32>
			){
				let idx = global_id.x;

				if idx < arrayLength(&gradA) { //buffers are the same size so pick one
					gradAOut[idx] = gradA[idx] + (1 / valuesB[idx] * gradResult[idx]);
					gradBOut[idx] = gradB[idx] + -1 * valuesA[idx] / pow(valuesB[idx], 2.0f) * gradResult[idx];
				}
			}
		`
	}
}

const powOp = {
	forward: {
		params: [
			{ name: "valuesA", direction: "in", type: "array" },
			{ name: "valuesB", direction: "in", type: "array" },
			{ name: "output", direction: "out", type: "array" }
		],
		code: `
			@group(0) @binding(0)
			var<storage, read> valuesA: array<f32>;

			@group(0) @binding(1)
			var<storage, read> valuesB: array<f32>;

			@group(0) @binding(2)
			var<storage, read_write> output: array<f32>;

			@compute @workgroup_size(8, 8)
			fn main(
				@builtin(global_invocation_id) global_id: vec3<u32>
			){
				let idx = global_id.x;

				if idx < arrayLength(&valuesA) {
					output[idx] = pow(valuesA[idx], valuesB[idx]);
				}
			}
			`
	},
	backward: {
		params: [
			{ name: "gradA", direction: "in", type: "array" },
			{ name: "gradB", direction: "in", type: "array" },
			{ name: "gradResult", direction: "in", type: "array" },
			{ name: "valuesA", direction: "in", type: "array" },
			{ name: "valuesB", direction: "in", type: "array" },
			{ name: "gradAOut", direction: "out", type: "array" },
			{ name: "gradBOut", direction: "out", type: "array" },
		],
		code: `
			@group(0) @binding(0)
			var<storage, read> gradA: array<f32>;

			@group(0) @binding(1)
			var<storage, read> gradB: array<f32>;

			@group(0) @binding(2)
			var<storage, read> gradResult: array<f32>;

			@group(0) @binding(3)
			var<storage, read> valuesA: array<f32>;

			@group(0) @binding(4)
			var<storage, read> valuesB: array<f32>;

			@group(0) @binding(5)
			var<storage, read_write> gradAOut: array<f32>;

			@group(0) @binding(6)
			var<storage, read_write> gradBOut: array<f32>;

			@compute @workgroup_size(8, 8)
			fn main(
				@builtin(global_invocation_id) global_id: vec3<u32>
			){
				let idx = global_id.x;

				if idx < arrayLength(&gradA) { //buffers are the same size so pick one
					gradAOut[idx] = valuesB[idx] * pow(valuesA[idx], valuesB[idx] - 1) * gradResult[idx];
					gradBOut[idx] = log(valuesA[idx]) * pow(valuesA[idx], valuesB[idx]) * gradResult[idx];
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
				let idx = global_id.x;

				if idx < arrayLength(&input) {
					output[idx] = -input[idx];
				}
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
				let idx = global_id.x;

				if idx < arrayLength(&input) {
					output[idx] = -input[idx];
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
				let idx = global_id.x;

				if idx < arrayLength(&input) {
					output[idx] = exp(input[idx]);
				}
			}
			`
	},
	backward: {
		params: [
			{ name: "values", direction: "in", type: "array" },
			{ name: "grad", direction: "in", type: "array" },
			{ name: "gradResult", direction: "in", type: "array" },
			{ name: "gradOut", direction: "out", type: "array" }
		],
		code: `
			@group(0) @binding(0)
			var<storage, read> values: array<f32>;

			@group(0) @binding(1)
			var<storage, read> grad: array<f32>;

			@group(0) @binding(2)
			var<storage, read> gradResult: array<f32>;

			@group(0) @binding(3)
			var<storage, read_write> gradOut: array<f32>;

			@compute @workgroup_size(8, 8)
			fn main(
				@builtin(global_invocation_id) global_id: vec3<u32>
			){
				let idx = global_id.x;

				if idx < arrayLength(&grad) {
					gradOut[idx] = grad[idx] + exp(values[idx]) * gradResult[idx];
				}
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
				let idx = global_id.x;

				if global_id.x < arrayLength(&input) {
					output[idx] = tanh(input[idx]);
				}
			}
			`
	},
	backward: {
		params: [
			{ name: "values", direction: "in", type: "array" },
			{ name: "grad", direction: "in", type: "array" },
			{ name: "gradResult", direction: "in", type: "array" },
			{ name: "gradOut", direction: "out", type: "array" }
		],
		code: `
			@group(0) @binding(0)
			var<storage, read> values: array<f32>;

			@group(0) @binding(1)
			var<storage, read> grad: array<f32>;

			@group(0) @binding(2)
			var<storage, read> gradResult: array<f32>;

			@group(0) @binding(3)
			var<storage, read_write> gradOut: array<f32>;

			@compute @workgroup_size(8, 8)
			fn main(
				@builtin(global_invocation_id) global_id: vec3<u32>
			){
				let idx = global_id.x;

				if idx < arrayLength(&grad) {
					gradOut[idx] = grad[idx] + (1 - pow(tanh(values[idx]), 2.0)) * gradResult[idx];
				}
			}`
	}
}