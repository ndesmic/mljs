import { getDeviceKernels } from "../js/webgpu-kernel.js";
import { topologicalSort } from "./topological-sort.js";

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
		const totalLength = shape.reduce((s, x) => s * x);

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
			inputs: [this.#values, other.values], 
			outputs: [{ byteLength: this.#values.byteLength }] 
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
				inputs: [this.gradient, other.gradient, result.gradient],
				outputs: [{ byteLength: this.gradient.byteLength }, { byteLength: other.gradient.byteLength }]
			});
			if (this === other) {
				const [combinedGrad] = await this.#kernels.add.forward({
					inputs: [thisGradient, otherGradient],
					outputs: [{ byteLength: this.#values.byteLength }]
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
			inputs: [this.#values, other.values],
			outputs: [{ byteLength: this.#values.byteLength }]
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
				inputs: [this.gradient, other.gradient, result.gradient],
				outputs: [{ byteLength: this.gradient.byteLength }, { byteLength: other.gradient.byteLength }]
			});
			if (this === other) {
				const [combinedGrad] = await this.#kernels.add.forward({
					inputs: [thisGradient, otherGradient],
					outputs: [{ byteLength: this.#values.byteLength }]
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
			inputs: [this.#values, other.values],
			outputs: [{ byteLength: this.#values.byteLength }]
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
				inputs: [this.gradient, other.gradient, result.gradient, this.values, other.values],
				outputs: [{ byteLength: this.gradient.byteLength }, { byteLength: other.gradient.byteLength }]
			});
			if (this === other) {
				const [combinedGrad] = await this.#kernels.add.forward({
					inputs: [thisGradient, otherGradient],
					outputs: [{ byteLength: this.#values.byteLength }]
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
			inputs: [this.#values, other.values],
			outputs: [{ byteLength: this.#values.byteLength }]
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
				inputs: [this.gradient, other.gradient, result.gradient, this.values, other.values],
				outputs: [{ byteLength: this.gradient.byteLength }, { byteLength: other.gradient.byteLength }]
			});
			if (this === other){
				const [combinedGrad] = await this.#kernels.add.forward({
					inputs: [thisGradient, otherGradient],
					outputs: [{ byteLength: this.#values.byteLength }]
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
			inputs: [this.#values, other.values],
			outputs: [{ byteLength: this.#values.byteLength }]
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
				inputs: [this.gradient, other.gradient, result.gradient, this.values, other.values],
				outputs: [{ byteLength: this.gradient.byteLength }, { byteLength: other.gradient.byteLength }]
			});
			if (this === other) {
				const [combinedGrad] = await this.#kernels.add.forward({
					inputs: [thisGradient, otherGradient],
					outputs: [{ byteLength: this.#values.byteLength }]
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
			inputs: [this.#values],
			outputs: [{ byteLength: this.#values.byteLength }]
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
				inputs: [result.gradient],
				outputs: [{ byteLength: this.gradient.byteLength }]
			});
			this.gradient = thisGradient;
		};

		return result;
	}
	async exp() {
		const [resultValues] = await this.#kernels.exp.forward({
			inputs: [this.#values],
			outputs: [{ byteLength: this.#values.byteLength }]
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
				inputs: [this.values, this.gradient, result.gradient],
				outputs: [{ byteLength: this.gradient.byteLength }]
			});
			this.gradient = thisGradient;
		};

		return result;
	}
	async tanh(){
		const [resultValues] = await this.#kernels.tanh.forward({
			inputs: [this.#values],
			outputs: [{ byteLength: this.#values.byteLength }]
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
				inputs: [this.values, this.gradient, result.gradient],
				outputs: [{ byteLength: this.gradient.byteLength }]
			});
			this.gradient = thisGradient;
		}

		return result;
	}
}