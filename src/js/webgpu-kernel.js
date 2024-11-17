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

	return async function ({ memory, inputs, outputs }) {
		const bindGroupEntries = [];

		if(params.memory){
			for (let i = 0; i < params.memory.length; i++) {
				const size = memory[i];
				const param = params.memory[i];
				//only supports u32 for now...
				const gpuBuffer = device.createBuffer({ size: size * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
				bindGroupEntries.push({
					label: param.name,
					binding: bindGroupEntries.length,
					resource: {
						buffer: gpuBuffer
					},
				});
			}
		}

		for (let i = 0; i < params.inputs.length; i++) {
			const inputBuffer = inputs[i];
			const param = params.inputs[i];

			switch (param.type) {
				case "array": {
					const gpuBuffer = device.createBuffer({ size: inputBuffer.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
					bindGroupEntries.push({
						label: param.name,
						binding: bindGroupEntries.length,
						resource: {
							buffer: gpuBuffer
						}
					});
					device.queue.writeBuffer(gpuBuffer, 0, inputBuffer);
					break;
				}
				case "u32": {
					const gpuBuffer = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
					bindGroupEntries.push({
						label: param.name,
						binding: bindGroupEntries.length,
						resource: {
							buffer: gpuBuffer
						}
					});
					device.queue.writeBuffer(gpuBuffer, 0, new Uint32Array([inputBuffer]));
					break;
				}
			}
		}

		const outputBuffers = new Array(params.outputs.length);
		const stagingBuffers = new Array(params.outputs.length);

		for (let i = 0; i < params.outputs.length; i++) {
			outputBuffers[i] = device.createBuffer({
				label: params.outputs[i].name,
				size: outputs[i],
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
			});
			stagingBuffers[i] = device.createBuffer({
				label: `${params.outputs[i].name}-staging`,
				size: outputs[i],
				usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
			});
			bindGroupEntries.push({
				label: params.outputs[i].name,
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
			commandEncoder.copyBufferToBuffer(outputBuffers[i], 0, stagingBuffers[i], 0, outputs[i]);
		}
		const commands = commandEncoder.finish();
		device.queue.submit([commands]);

		const outputResults = new Array(outputs.length);
		for(let i = 0; i < outputs.length; i++){
			await stagingBuffers[i].mapAsync(GPUMapMode.READ, 0, outputs[i].byteLength);
			const copyArrayBuffer = stagingBuffers[i].getMappedRange(0, outputs[i]);

			const data = copyArrayBuffer.slice(0);
			stagingBuffers[i].unmap();

			switch(params.outputs[i].subtype){
				case "u32": {
					outputResults[i] = new Uint32Array(data);
					break;
				}
				default: {
					outputResults[i] = new Float32Array(data);
				}
			}
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
	const entries = [];
	if(params.memory){
		for (let i = 0; i < params.memory.length; i++) {
			entries.push({
				binding: entries.length,
				visibility: GPUShaderStage.COMPUTE,
				buffer: {
					type: "storage"
				}
			});
		}
	}
	if(params.inputs){
		for(let i = 0; i < params.inputs.length; i++){
			entries.push({
				binding: entries.length,
				visibility: GPUShaderStage.COMPUTE,
				buffer: {
					type:"read-only-storage"
				}
			});
		}
	}
	if(params.outputs){
		for(let i = 0; i < params.outputs.length; i++){
			entries.push({
				binding: entries.length,
				visibility: GPUShaderStage.COMPUTE,
				buffer: {
					type: "storage"
				}
			});
		}
	}

	return device.createBindGroupLayout({
		entries
	});
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
		pow: compileOp(device, powOp),
		neg: compileOp(device, negOp),
		exp: compileOp(device, expOp),
		tanh: compileOp(device, tanhOp),
		sum: compileOp(device, sumOp),
	};
	cache.set(device, kernels);
	return kernels;
}

const addOp = {
	forward: {
		params: {
			inputs: [
				{ name: "valuesA", type: "array" },
				{ name: "valuesB", type: "array" },
			],
			outputs: [
				{ name: "output", type: "array", subtype: "f32" }
			]
		},
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
		params: {
			inputs: [
				{ name: "gradA", direction: "in", type: "array" },
				{ name: "gradB", direction: "in", type: "array" },
				{ name: "gradResult", direction: "in", type: "array" }
			],
			outputs: [
				{ name: "gradAOut", direction: "out", type: "array", subtype: "f32" },
				{ name: "gradBOut", direction: "out", type: "array", subtype: "f32" },
			]
		},
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
		params: {
			inputs: [
				{ name: "valuesA", type: "array" },
				{ name: "valuesB", type: "array" },
			],
			outputs: [
				{ name: "output", type: "array", subtype: "f32" }
			]
		},
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
		params: {
			inputs: [
				{ name: "gradA", type: "array" },
				{ name: "gradB", type: "array" },
				{ name: "gradResult", type: "array" },
			],
			outputs: [
				{ name: "gradAOut", type: "array", subtype: "f32" },
				{ name: "gradBOut", type: "array", subtype: "f32" }
			]
		},
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
		params: {
			inputs: [
				{ name: "valuesA", type: "array" },
				{ name: "valuesB", type: "array" },
			],
			outputs: [
				{ name: "output", type: "array", subtype: "f32" }
			]
		},
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
		params: {
			inputs: [
				{ name: "valuesA", type: "array" },
				{ name: "valuesB", type: "array" },
				{ name: "gradA", type: "array" },
				{ name: "gradB", type: "array" },
				{ name: "gradResult",type: "array" },
			],
			outputs: [
				{ name: "gradAOut", type: "array", subtype: "f32" },
				{ name: "gradBOut", type: "array", subtype: "f32" },
			]
		},
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
		params: {
			inputs: [
				{ name: "valuesA", type: "array" },
				{ name: "valuesB", type: "array" },
			],
			outputs: [
				{ name: "output", type: "array", subtype: "f32" }
			]
		},
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
		params: {
			inputs: [
				{ name: "gradA", type: "array" },
				{ name: "gradB", type: "array" },
				{ name: "gradResult", type: "array" },
				{ name: "valuesA", type: "array" },
				{ name: "valuesB", type: "array" }
			],
			outputs: [
				{ name: "gradAOut", type: "array", subtype: "f32" },
				{ name: "gradBOut", type: "array", subtype: "f32" },
			]
		},
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
		params: {
			inputs: [
				{ name: "valuesA", type: "array" },
				{ name: "valuesB", type: "array" },
			],
			outputs: [
				{ name: "output", type: "array", subtype: "f32" }
			]
		},
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
		params: {
			inputs: [
				{ name: "gradA", type: "array" },
				{ name: "gradB", type: "array" },
				{ name: "gradResult", type: "array" },
				{ name: "valuesA", type: "array" },
				{ name: "valuesB", type: "array" },
			],
			outputs: [
				{ name: "gradAOut", type: "array", subtype: "f32" },
				{ name: "gradBOut", type: "array", subtype: "f32" },
			]
		},
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
		params: {
			inputs: [
				{ name: "input",  type: "array" },
			],
			outputs: [
				{ name: "output",  type: "array", subtype: "f32" }
			]
		},
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
		params: {
			inputs: [
				{ name: "input", type: "array" },
			],
			outputs: [
				{ name: "output", type: "array", subtype: "f32" }
			]
		},
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
		params: {
			inputs: [
				{ name: "input", type: "array" },
			],
			outputs: [
				{ name: "output", type: "array", subtype: "f32" }
			]
		},
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
		params: {
			inputs: [
				{ name: "values", type: "array" },
				{ name: "grad", type: "array" },
				{ name: "gradResult", type: "array" },
			],
			outputs: [
				{ name: "gradOut", type: "array", subtype: "f32" }
			]
		},
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
		params: {
			inputs: [
				{ name: "input", type: "array" },
			],
			outputs: [
				{ name: "output", type: "array", subtype: "f32" }
			]
		},
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
		params: {
			inputs: [
				{ name: "values", type: "array" },
				{ name: "grad",type: "array" },
				{ name: "gradResult",type: "array" },
			],
			outputs: [
				{ name: "gradOut", type: "array", subtype: "f32" }
			]
		},
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

const sumOp = {
	forward: {
		params: {
			memory: [
				{ name: "mem", type: "array", },
			],
			inputs: [
				{ name: "shape", type: "array" },
				{ name: "dimToReduce", type: "u32" },
				{ name: "values", type: "array" },
			],
			outputs: [
				{ name: "output", type: "array", subtype: "f32" }
			]
		},
		code: `
			@group(0) @binding(0)
			var<storage, read_write> mem: array<u32>; //memory arena because we can't allocate in the shader

			@group(0) @binding(1)
			var<storage, read> shape: array<u32>;

			struct U32 {
				value: u32
			}

			@group(0) @binding(2)
			var<storage, read> dim_to_reduce: U32;

			@group(0) @binding(3)
			var<storage, read> values: array<f32>;

			@group(0) @binding(4)
			var<storage, read_write> output: array<f32>;

			fn remove_at_index(source_ptr: u32, length: u32, destination_ptr: u32, index_to_remove: u32) {
				var source_index = 0u;
				var destination_index = 0u;

				while(destination_index < length - 1){
					if(source_index != index_to_remove){
						mem[destination_ptr + destination_index] = mem[source_ptr + source_index];
						source_index++;
						destination_index++;
					} else {
						source_index++;
					}
				}
			}

			fn insert_at_index(source_ptr: u32, length: u32, destination_ptr: u32, index_to_insert: u32, value: u32) {
				var source_index = 0u;
				var destination_index = 0u;

				while(destination_index < length + 1){
					if(destination_index != index_to_insert){
						mem[destination_ptr + destination_index] = mem[source_ptr + source_index];
						source_index++;
						destination_index++;
					} else {
						mem[destination_ptr + destination_index] = value;

						destination_index++;
					}
				}
			}

			fn get_dimensional_indices(flat_index: u32, shape_ptr: u32, shape_size: u32, destination_ptr: u32) {
				var destination_index = destination_ptr;
				var current_index = flat_index;

				for(var i = 0u; i < shape_size; i++){
					mem[destination_ptr + i] = current_index % mem[shape_ptr + i];
					current_index = current_index / mem[shape_ptr + i];
				}
			}

			fn get_flat_index(dimensional_index_ptr: u32, shape_ptr: u32, shape_size: u32) -> u32 {
				var index = 0u;

				for (var i = 0u; i < shape_size; i++)
				{
					index *= mem[shape_ptr + shape_size - 1 - i];
					index += mem[dimensional_index_ptr + shape_size - 1 - i];
				}
				return index;
			}

			@compute @workgroup_size(8, 8)
			fn main(
				@builtin(global_invocation_id) global_id: vec3<u32>
			){
				let idx = global_id.x;
				let shape_size = arrayLength(&shape);
				
				var out_size = 1u;
				for(var j = 0u; j < shape_size; j++){
					if(j != dim_to_reduce.value){
						out_size *= shape[j];
					}
				}

				if idx < out_size {
					let base_ptr = idx * ((shape_size * 4) - 2); //manual calc :/ 
					var mem_ptr = base_ptr;

					let shape_ptr = base_ptr;
					for(var i = 0u; i < shape_size; i++){ //write shape to mem
						mem[mem_ptr] = shape[i];
						mem_ptr++;
					}

					let out_shape_ptr = mem_ptr;
					remove_at_index(shape_ptr, shape_size, out_shape_ptr, dim_to_reduce.value);
					mem_ptr += shape_size - 1;

					let partial_dim_index_ptr = mem_ptr;
					get_dimensional_indices(idx, out_shape_ptr, shape_size - 1, partial_dim_index_ptr);
					mem_ptr += shape_size - 1;		

					for(var i = 0u; i < shape[dim_to_reduce.value]; i++){
						let dim_index_ptr = mem_ptr;
						insert_at_index(partial_dim_index_ptr, shape_size - 1, dim_index_ptr, dim_to_reduce.value, i);

						let flat_index = get_flat_index(dim_index_ptr, shape_ptr, shape_size);

						output[idx] += values[flat_index];
					} 
				}
			}
			`
	},
	backward: {
		params: {
			memory: [
				{ name: "mem", type: "array" }
			],
			inputs: [
				{ name: "shape", type: "array" },
				{ name: "dimToReduce", type: "u32" },
				{ name: "grad", type: "array" },
				{ name: "gradResult", type: "array" },
			],
			outputs: [
				{ name: "gradOut", type: "array", subtype: "f32" }
			]
		},
		code: `
			@group(0) @binding(0)
			var<storage, read_write> mem: array<u32>; //memory arena because we can't allocate in the shader

			@group(0) @binding(1)
			var<storage, read> shape: array<u32>;

			struct U32 {
				value: u32
			}

			@group(0) @binding(2)
			var<storage, read> dim_to_reduce: U32;

			@group(0) @binding(3)
			var<storage, read> grad: array<f32>;

			@group(0) @binding(4)
			var<storage, read> grad_result: array<f32>;

			@group(0) @binding(5)
			var<storage, read_write> grad_out: array<f32>;

			fn remove_at_index(source_ptr: u32, length: u32, destination_ptr: u32, index_to_remove: u32) {
				var source_index = 0u;
				var destination_index = 0u;

				while(destination_index < length - 1){
					if(source_index != index_to_remove){
						mem[destination_ptr + destination_index] = mem[source_ptr + source_index];
						source_index++;
						destination_index++;
					} else {
						source_index++;
					}
				}
			}

			fn get_dimensional_indices(flat_index: u32, shape_ptr: u32, shape_size: u32, destination_ptr: u32) {
				var destination_index = destination_ptr;
				var current_index = flat_index;

				for(var i = 0u; i < shape_size; i++){
					mem[destination_ptr + i] = current_index % mem[shape_ptr + i];
					current_index = current_index / mem[shape_ptr + i];
				}
			}

			fn get_flat_index(dimensional_index_ptr: u32, shape_ptr: u32, shape_size: u32) -> u32 {
				var index = 0u;

				for (var i = 0u; i < shape_size; i++)
				{
					index *= mem[shape_ptr + shape_size - 1 - i];
					index += mem[dimensional_index_ptr + shape_size - 1 - i];
				}
				return index;
			}

			@compute @workgroup_size(8, 8)
			fn main(
				@builtin(global_invocation_id) global_id: vec3<u32>
			){
				let idx = global_id.x;
				let shape_size = arrayLength(&shape);

				var size = 1u;
				for(var j = 0u; j < shape_size; j++){
					size *= shape[j];
				}

				if idx < size {
					let base_ptr = idx * ((shape_size * 4) - 2); //manual calc :/
					var mem_ptr = base_ptr;

					let shape_ptr = base_ptr;
					for(var i = 0u; i < shape_size; i++){ //write shape to mem
						mem[mem_ptr] = shape[i];
						mem_ptr++;
					}

					let out_shape_ptr = mem_ptr;
					remove_at_index(shape_ptr, shape_size, out_shape_ptr, dim_to_reduce.value);
					mem_ptr += shape_size - 1;

					let in_dim_index_ptr = mem_ptr;
					get_dimensional_indices(idx, shape_ptr, shape_size, in_dim_index_ptr);
					mem_ptr += shape_size;

					let out_dim_index_ptr = mem_ptr;
					remove_at_index(in_dim_index_ptr, shape_size, out_dim_index_ptr, dim_to_reduce.value);
					mem_ptr += shape_size - 1;

					let out_index = get_flat_index(out_dim_index_ptr, out_shape_ptr, shape_size - 1);

					grad_out[idx] = grad[idx] + grad_result[out_index];
				}
			}
		`
	}
}