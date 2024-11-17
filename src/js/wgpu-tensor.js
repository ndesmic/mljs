import { getDeviceKernels } from "../js/webgpu-kernel.js";
import { topologicalSort } from "./topological-sort.js";
import { getTotalLength } from "./tensor-utils.js";

const backward = Symbol("backward");

export class WGPUTensor {
	#shape;
	#values;
	#children;
	#op;
	#label;
	[backward] = () => { };
	#gradient = 0;
	#device;
	#kernels;

	constructor({ values, shape, children, op, label, device }) {
		const totalLength = getTotalLength(shape);

		this.#shape = shape;
		if (values) {
			if (values.length != totalLength) throw new Error(`Supplied values are the wrong length, need ${totalLength}`);
			if (values instanceof Float32Array) {
				this.#values = values;
			} else {
				this.#values = new Float32Array(values)
			}
		} else {
			this.#values = new Float32Array(totalLength);
		}

		this.#children = children ?? [];
		this.#gradient = new Float32Array(totalLength);
		this.#op = op ?? null;
		this.#label = label ?? null;
		this.#device = device;
		this.#kernels = getDeviceKernels(device);
	}
	get totalLength() {
		return this.#values.length;
	}
	get shape() {
		return this.#shape;
	}
	get values() {
		return this.#values;
	}
	get gradient() {
		return this.#gradient;
	}
	set gradient(val){
		this.#gradient = val;
	}
	get children() {
		return this.#children;
	}
	async backward() {
		this.#gradient = new Float32Array(this.totalLength).fill(1);
		const sortedDependencies = topologicalSort(this, x => x.children).reverse();
		for (const node of sortedDependencies) {
			await node[backward]();
		}
	}
	async add(other) {
		if (other.totalLength != this.totalLength) throw new Error(`Tensor not the right length, argument was ${other.totalLength}, needs to be ${this.totalLength}`);

		const [resultValues] = await this.#kernels.add.forward({ 
			inputs: [
				this.#values, 
				other.values
			],
			outputs: [
				this.#values.byteLength
			]
		});
		const result = new WGPUTensor({
			values: resultValues,
			shape: this.#shape,
			children: [this, other],
			op: "+",
			device: this.#device
		});

		result[backward] = async () => {
			const [thisGradient, otherGradient] = await this.#kernels.add.backward({ 
				inputs: [
					this.gradient, 
					other.gradient, 
					result.gradient
				],
				outputs: [
					this.gradient.byteLength,
					other.gradient.byteLength
				]
			});
			if (this === other) {
				const [combinedGrad] = await this.#kernels.add.forward({
					inputs: [
						thisGradient, 
						otherGradient
					],
					outputs: [
						this.#values.byteLength
					]
				});
				this.#gradient = combinedGrad;
			} else {
				this.gradient = thisGradient;
				other.gradient = otherGradient;
			}
		};

		return result;
	}
	async sub(other){
		if (other.totalLength != this.totalLength) throw new Error(`Tensor not the right length, argument was ${other.totalLength}, needs to be ${this.totalLength}`);
		const [resultValues] = await this.#kernels.sub.forward({
			inputs: [
				this.#values, 
				other.values
			],
			outputs: [
				this.#values.byteLength
			]
		});
		const result = new WGPUTensor({
			values: resultValues,
			shape: this.#shape,
			children: [this, other],
			op: "-",
			device: this.#device
		});

		result[backward] = async () => {
			const [thisGradient, otherGradient] = await this.#kernels.sub.backward({
				inputs: [
					this.gradient, 
					other.gradient, 
					result.gradient],
				outputs: [
					this.gradient.byteLength,
					other.gradient.byteLength
				]
			});
			if (this === other) {
				const [combinedGrad] = await this.#kernels.add.forward({
					inputs: [
						thisGradient, 
						otherGradient
					],
					outputs: [
						this.#values.byteLength
					]
				});
				this.#gradient = combinedGrad;
			} else {
				this.gradient = thisGradient;
				other.gradient = otherGradient;
			}
		};

		return result;
	}
	async mul(other) {
		if (other.totalLength != this.totalLength) throw new Error(`Tensor not the right length, argument was ${other.totalLength}, needs to be ${this.totalLength}`);

		const [resultValues] = await this.#kernels.mul.forward({
			inputs: [
				this.#values, 
				other.values
			],
			outputs: [
				this.#values.byteLength
			]
		});

		const result = new WGPUTensor({
			values: resultValues,
			shape: this.#shape,
			children: [this, other],
			op: "*",
			device: this.#device
		});

		result[backward] = async () => {
			const [thisGradient, otherGradient] = await this.#kernels.mul.backward({
				inputs: [
					this.gradient, 
					other.gradient, 
					result.gradient, 
					this.values, other.values
				],
				outputs: [
					this.gradient.byteLength,
					other.gradient.byteLength
				]
			});
			if (this === other) {
				const [combinedGrad] = await this.#kernels.add.forward({
					inputs: [
						thisGradient, 
						otherGradient
					],
					outputs: [
						this.#values.byteLength
					]
				});
				this.#gradient = combinedGrad;
			} else {
				this.gradient = thisGradient;
				other.gradient = otherGradient;
			}
		};

		return result;
	}
	async div(other) {
		if (other.totalLength != this.totalLength) throw new Error(`Tensor not the right length, argument was ${other.totalLength}, needs to be ${this.totalLength}`);

		const [resultValues] = await this.#kernels.div.forward({
			inputs: [
				this.#values, 
				other.values
			],
			outputs: [
				this.#values.byteLength
			]
		});

		const result = new WGPUTensor({
			values: resultValues,
			shape: this.#shape,
			children: [this, other],
			op: "/",
			device: this.#device
		});

		result[backward] = async () => {
			const [thisGradient, otherGradient] = await this.#kernels.div.backward({
				inputs: [
					this.gradient, other.gradient, 
					result.gradient, 
					this.values, 
					other.values
				],
				outputs: [
					this.gradient.byteLength,
					other.gradient.byteLength
				]
			});
			if (this === other){
				const [combinedGrad] = await this.#kernels.add.forward({
					inputs: [
						thisGradient, 
						otherGradient
					],
					outputs: [
						this.#values.byteLength
					]
				});
				this.gradient = combinedGrad;
			} else { 
				this.gradient = thisGradient;
				other.gradient = otherGradient;
			}
		};

		return result;
	}
	async pow(other) {
		if (other.totalLength != this.totalLength) throw new Error(`Tensor not the right length, argument was ${other.totalLength}, needs to be ${this.totalLength}`);

		const [resultValues] = await this.#kernels.pow.forward({
			inputs: [
				this.#values, 
				other.values
			],
			outputs: [
				this.#values.byteLength
			]
		});

		const result = new WGPUTensor({
			values: resultValues,
			shape: this.#shape,
			children: [this],
			op: `pow`,
			device: this.#device
		});

		result[backward] = async () => {
			const [thisGradient, otherGradient] = await this.#kernels.pow.backward({
				inputs: [
					this.gradient, 
					other.gradient, 
					result.gradient, 
					this.values, 
					other.values
				],
				outputs: [
					this.gradient.byteLength, 
					other.gradient.byteLength
				]
			});
			if (this === other) {
				const [combinedGrad] = await this.#kernels.add.forward({
					inputs: [
						thisGradient, 
						otherGradient
					],
					outputs: [
						this.#values.byteLength
					]
				});
				this.gradient = combinedGrad;
			} else {
				this.gradient = thisGradient;
				other.gradient = otherGradient;
			}
		}

		return result;
	}
	async neg() {
		const [resultValues] = await this.#kernels.neg.forward({
			inputs: [
				this.#values
			],
			outputs: [
				this.#values.byteLength
			]
		});

		const result = new WGPUTensor({
			values: resultValues,
			shape: this.#shape,
			children: [this],
			op: "neg",
			device: this.#device
		});

		result[backward] = async () => {
			const [thisGradient] = await this.#kernels.neg.backward({
				inputs: [
					result.gradient
				],
				outputs: [
					this.gradient.byteLength
				]
			});
			this.gradient = thisGradient;
		};

		return result;
	}
	async exp() {
		const [resultValues] = await this.#kernels.exp.forward({
			inputs: [
				this.#values
			],
			outputs: [
				this.#values.byteLength
			]
		});

		const result = new WGPUTensor({
			values: resultValues,
			shape: this.#shape,
			children: [this],
			op: "exp",
			device: this.#device
		});

		result[backward] = async () => {
			const [thisGradient] = await this.#kernels.exp.backward({
				inputs: [
					this.values, 
					this.gradient, 
					result.gradient
				],
				outputs: [
					this.gradient.byteLength
				]
			});
			this.gradient = thisGradient;
		};

		return result;
	}
	async tanh(){
		const [resultValues] = await this.#kernels.tanh.forward({
			inputs: [
				this.#values
			],
			outputs: [
				this.#values.byteLength
			]
		});

		const result = new WGPUTensor({
			values: resultValues,
			shape: this.#shape,
			children: [this],
			op: "tanh",
			device: this.#device
		});

		result[backward] = async () => {
			const [thisGradient] = await this.#kernels.tanh.backward({
				inputs: [
					this.values, 
					this.gradient, 
					result.gradient
				],
				outputs: [
					this.gradient.byteLength
				]
			});
			this.gradient = thisGradient;
		}

		return result;
	}

	//reduction
	async sum({ dimensionToReduce, keepDims }) {
		const outputLength = this.#shape.reduce((prod, x, idx) => {
			return idx !== dimensionToReduce ? prod * x : prod;
		}, 1);
		const outputByteLength = outputLength * 4;
		const newShape = this.#shape.filter((_, i) => i !== dimensionToReduce);
		const threadMemSize = ((this.#shape.length * 4) - 2) * 4; //manually calculated from shader :/
		const memSize = threadMemSize * outputLength;


		const [resultValues] = await this.#kernels.sum.forward({
			memory: [
				memSize
			],
			inputs: [
				new Uint32Array(this.#shape), 
				dimensionToReduce, 
				this.#values
			],
			outputs: [
				outputByteLength
			]
		});

		const result = new WGPUTensor({
			values: resultValues,
			shape: keepDims ? newShape.toSpliced(dimensionToReduce, 0, 1) : newShape,
			children: [this],
			op: "sum",
			device: this.#device
		});

		result[backward] = async () => {
			const [thisGradient] = await this.#kernels.sum.backward({
				memory: [
					memSize //same memSize as forward pass luckily
				],
				inputs: [
					new Uint32Array(this.#shape),
					dimensionToReduce,
					this.#gradient,
					result.gradient
					
				],
				outputs: [
					this.gradient.byteLength
				]
			});
			this.gradient = thisGradient;
		}

		return result;
	}

	toString() {
		return `<${this.#label ? `${this.#label}:` : ""}${Array.from(this.#values).join(", ")}>`;
	}

	[Symbol.for("Deno.customInspect")]() {
		return this.toString();
	}

	//statics

	static filled(value, shape, options) {
		return new WGPUTensor({ values: new Float32Array(getTotalLength(shape)).fill(value), shape, device: options.device });
	}
	static random(shape, options = {}) {
		const length = getTotalLength(shape);
		const values = new Float32Array(length);
		const generator = options.generator ?? getRandom(options.min, options.max, options.seed);
		values.set(generator.take(length).toArray(), 0);
		return new WGPUTensor({ values, shape, device: options.device });
	}
	static getLinearSpace(start, end, steps, options) {
		steps = steps - 1; //counting the spaces not the nodes
		const length = end - start;
		const partLength = length / steps;
		const array = new Array(steps);
		let current = start;
		for (let i = 0; i <= steps; i++) {
			array[i] = current;
			current += partLength
		}
		return new WGPUTensor({ values: array, shape: [array.length], device: options.device });
	}
}