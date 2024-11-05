import { getTotalLength } from "./tensor-utils.js";
import { topologicalSort } from "./topological-sort.js";

const kernel = Deno.dlopen("./dist/kernel.dll", {
	add_op: { parameters: ["i32", "buffer", "buffer"], result: "pointer" },
	addBackprop_op: { parameters: ["i32", "buffer", "buffer", "buffer"], result: "void" },
	sub_op: { parameters: ["i32", "buffer", "buffer"], result: "pointer" },
	subBackprop_op: { parameters: ["i32", "buffer", "buffer", "buffer"], result: "void" },
	mul_op: { parameters: ["i32", "buffer", "buffer"], result: "pointer" },
	mulBackprop_op: { parameters: ["i32", "buffer", "buffer", "buffer", "buffer", "buffer"], result: "void" },
	div_op: { parameters: ["i32", "buffer", "buffer"], result: "pointer" },
	divBackprop_op: { parameters: ["i32", "buffer", "buffer", "buffer", "buffer", "buffer"], result: "void" },
	pow_op: { parameters: ["i32", "buffer", "buffer"], result: "pointer" },
	powBackprop_op: { parameters: ["i32", "buffer", "buffer", "buffer", "buffer", "buffer"], result: "void" },
	neg_op: { parameters: ["i32", "buffer"], result: "pointer" },
	negBackprop_op: { parameters: ["i32", "buffer", "buffer"], result: "pointer" },
	exp_op: { parameters: ["i32", "buffer"], result: "pointer" },
	expBackprop_op: { parameters: ["i32", "buffer", "buffer", "buffer"], result: "pointer" },
	tanh_op: { parameters: ["i32", "buffer"], result: "pointer" },
	tanhBackprop_op: { parameters: ["i32", "buffer", "buffer", "buffer"], result: "pointer" },
	sum_op: { parameters: ["buffer", "i32", "i32", "buffer"], result: "pointer" },
	sumBackprop_op: { parameters: ["buffer", "i32", "i32", "buffer", "buffer"], result: "void" },
});

const backward = Symbol("backward");

export class CUDATensor {
	#shape;
	#values;
	#children;
	#op;
	#label;
	[backward] = () => { };
	#gradient = 0;

	constructor({ values, shape, children, op, label }) {
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
	get children() {
		return this.#children;
	}
	set label(val){
		this.#label = val;
	}
	add(other, options = {}) {
		if (other.totalLength != this.totalLength) throw new Error(`Tensor not the right length, argument was ${other.totalLength}, needs to be ${this.totalLength}`);

		const resultPointer = kernel.symbols.add_op(this.totalLength, this.#values, other.values);
		const outBuffer = Deno.UnsafePointerView.getArrayBuffer(resultPointer, this.totalLength * 4);
		const values = new Float32Array(outBuffer);

		const result = new CUDATensor({
			values,
			shape: this.#shape,
			children: [this, other],
			op: "+",
			label: options.label
		});

		result[backward] = () => {
			kernel.symbols.addBackprop_op(this.totalLength, this.#gradient, other.gradient, result.gradient);
		};

		return result;
	}
	sub(other, options = {}) {
		if (other.totalLength != this.totalLength) throw new Error(`Tensor not the right length, argument was ${other.totalLength}, needs to be ${this.totalLength}`);

		const resultPointer = kernel.symbols.sub_op(this.totalLength, this.#values, other.values);
		const outBuffer = Deno.UnsafePointerView.getArrayBuffer(resultPointer, this.totalLength * 4);
		const values = new Float32Array(outBuffer);

		const result = new CUDATensor({
			values,
			shape: this.#shape,
			children: [this, other],
			op: "-",
			label: options.label
		});

		result[backward] = () => {
			kernel.symbols.subBackprop_op(this.totalLength, this.#gradient, other.gradient, result.gradient);
		};

		return result;
	}
	mul(other, options = {}) {
		if (other.totalLength != this.totalLength) throw new Error(`Tensor not the right length, argument was ${other.totalLength}, needs to be ${this.totalLength}`);

		const resultPointer = kernel.symbols.mul_op(this.totalLength, this.#values, other.values);
		const outBuffer = Deno.UnsafePointerView.getArrayBuffer(resultPointer, this.totalLength * 4);
		const values = new Float32Array(outBuffer);

		const result = new CUDATensor({
			values,
			shape: this.#shape,
			children: [this, other],
			op: "*",
			label: options.label
		});

		result[backward] = () => {
			kernel.symbols.mulBackprop_op(this.totalLength, this.#gradient, other.gradient, result.gradient, this.#values, other.values);
		};

		return result;
	}
	div(other, options = {}) {
		if (other.totalLength != this.totalLength) throw new Error(`Tensor not the right length, argument was ${other.totalLength}, needs to be ${this.totalLength}`);

		const resultPointer = kernel.symbols.div_op(this.totalLength, this.#values, other.values);
		const outBuffer = Deno.UnsafePointerView.getArrayBuffer(resultPointer, this.totalLength * 4);
		const values = new Float32Array(outBuffer);

		const result = new CUDATensor({
			values,
			shape: this.#shape,
			children: [this, other],
			op: "/",
			label: options.label
		});

		result[backward] = () => {
			kernel.symbols.divBackprop_op(this.totalLength, this.#gradient, other.gradient, result.gradient, this.#values, other.values);
		};

		return result;
	}
	pow(other, options = {}) {
		if (other.totalLength != this.totalLength) throw new Error(`Tensor not the right length, argument was ${other.totalLength}, needs to be ${this.totalLength}`);

		const resultPointer = kernel.symbols.pow_op(this.totalLength, this.#values, other.values);
		const outBuffer = Deno.UnsafePointerView.getArrayBuffer(resultPointer, this.totalLength * 4);
		const values = new Float32Array(outBuffer);

		const result = new CUDATensor({
			values,
			shape: this.#shape,
			children: [this, other],
			op: "pow",
			label: options.label
		});

		result[backward] = () => {
			kernel.symbols.powBackprop_op(this.totalLength, this.#gradient, other.gradient, result.gradient, this.#values, other.values);
		};

		return result;
	}
	neg(options = {}) {
		const resultPointer = kernel.symbols.neg_op(this.totalLength, this.#values);
		const outBuffer = Deno.UnsafePointerView.getArrayBuffer(resultPointer, this.totalLength * 4);
		const values = new Float32Array(outBuffer);

		const result = new CUDATensor({
			values,
			shape: this.#shape,
			children: [this],
			op: `neg`,
			label: options.label
		});

		result[backward] = () => {
			kernel.symbols.negBackprop_op(this.totalLength, this.#gradient, result.gradient);
		}

		return result;
	}
	exp(options = {}) {
		const resultPointer = kernel.symbols.exp_op(this.totalLength, this.#values);
		const outBuffer = Deno.UnsafePointerView.getArrayBuffer(resultPointer, this.totalLength * 4);
		const values = new Float32Array(outBuffer);

		const result = new CUDATensor({
			values,
			shape: this.#shape,
			children: [this],
			op: `exp`,
			label: options.label
		});

		result[backward] = () => {
			kernel.symbols.expBackprop_op(this.totalLength, this.#gradient, result.gradient, this.#values);
		}

		return result;
	}
	tanh(options = {}) {
		const resultPointer = kernel.symbols.tanh_op(this.totalLength, this.#values);
		const outBuffer = Deno.UnsafePointerView.getArrayBuffer(resultPointer, this.totalLength * 4);
		const values = new Float32Array(outBuffer);

		const result = new CUDATensor({
			values,
			shape: this.#shape,
			children: [this],
			op: `tanh`,
			label: options.label
		});

		result[backward] = () => {
			kernel.symbols.tanhBackprop_op(this.totalLength, this.#gradient, result.gradient, this.#values);
		}

		return result;
	}
	sum({ dimensionToReduce, keepDims = false, label }){
		const outputLength = this.#shape.reduce((prod, x, idx) => {
			return idx !== dimensionToReduce ? prod * x : prod;
		}, 1);
		const resultPointer = kernel.symbols.sum_op(new Int32Array(this.#shape).buffer, this.#shape.length, dimensionToReduce, this.#values);
		const outBuffer = Deno.UnsafePointerView.getArrayBuffer(resultPointer, outputLength * 4);
		const output = new Float32Array(outBuffer);

		const newShape = keepDims
			? [...this.#shape].with(dimensionToReduce, 1)
			: this.#shape.filter((_,i) => i != dimensionToReduce);

		const result = new CUDATensor({
			values: output,
			shape: newShape,
			children: [this],
			op: "sum",
			label: label
		});

		result[backward] = () => {
			kernel.symbols.sumBackprop_op(new Int32Array(this.#shape).buffer, this.#shape.length,  dimensionToReduce, this.#gradient, result.gradient);
		}

		return result;
	}
	backward() {
		this.#gradient = new Float32Array(this.totalLength).fill(1);
		const sortedDependencies = topologicalSort(this, x => x.children).reverse();
		for (const node of sortedDependencies) {
			node[backward]();
		}
	}
	toString() {
		return `<${this.#label ? `${this.#label}:` : ""}${Array.from(this.#values).join(", ")}>`;
	}
	[Symbol.for("Deno.customInspect")]() {
		return this.toString();
	}

	//statics
	static filled(value, shape) {
		return new CUDATensor({ values: new Float32Array(getTotalLength(shape)).fill(value), shape });
	}
	static random(shape, options = {}) {
		const length = getTotalLength(shape);
		const values = new Float32Array(length);
		const generator = options.generator ?? getRandom(options.min, options.max, options.seed);
		values.set(generator.take(length).toArray(), 0);
		return new CUDATensor({ values, shape });
	}
	static getLinearSpace(start, end, steps) {
		steps = steps - 1; //counting the spaces not the nodes
		const length = end - start;
		const partLength = length / steps;
		const array = new Array(steps);
		let current = start;
		for (let i = 0; i <= steps; i++) {
			array[i] = current;
			current += partLength
		}
		return new CUDATensor({ values: array, shape: [array.length] });
	}
}