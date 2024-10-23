import { topologicalSort } from "./topological-sort.js";
import { getDimensionalIndices, getFlatIndex, getTotalLength } from "./tensor-utils.js";

const backward = Symbol("backward");

export class Tensor {
	#shape;
	#values;
	#children;
	#op;
	#label;
	[backward] = () => {};
	#gradient = 0;

	constructor({ values, shape, children, op, label }){
		const totalLength = getTotalLength(shape);

		this.#shape = shape;
		if(values){
			if(values.length != totalLength) throw new Error(`Supplied values are the wrong length, need ${totalLength}`);
			if(values instanceof Float32Array){
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
	get totalLength(){
		return this.#values.length;
	}
	get shape(){
		return this.#shape;
	}
	get values(){
		return this.#values;
	}
	get gradient(){
		return this.#gradient;
	}
	get children(){
		return this.#children;
	}
	add(other){
		if(other.totalLength != this.totalLength) throw new Error(`Tensor not the right length, argument was ${other.totalLength}, needs to be ${this.totalLength}`);

		const values = new Float32Array(this.totalLength);
		for(let i = 0; i < this.totalLength; i++){
			values[i] = this.#values[i] + other.values[i];
		}

		const result = new Tensor({
			values,
			shape: this.#shape,
			children: [this, other],
			op: "+"
		});

		result[backward] = () => {
			for (let i = 0; i < this.totalLength; i++) {
				this.gradient[i] += result.gradient[i];
				other.gradient[i] += result.gradient[i];
			}
		};

		return result;
	}
	sub(other){
		if (other.totalLength != this.totalLength) throw new Error(`Tensor not the right length, argument was ${other.totalLength}, needs to be ${this.totalLength}`);

		const values = new Float32Array(this.totalLength);
		for (let i = 0; i < this.totalLength; i++) {
			values[i] = this.#values[i] - other.values[i];
		}

		const result = new Tensor({
			values,
			shape: this.#shape,
			children: [this, other],
			op: "-"
		});

		result[backward] = () => {
			for(let i = 0; i < this.totalLength; i++){
				this.gradient[i] += result.gradient[i];
				other.gradient[i] += -result.gradient[i];
			}
		};

		return result;
	}
	mul(other) {
		if (other.totalLength != this.totalLength) throw new Error(`Tensor not the right length, argument was ${other.totalLength}, needs to be ${this.totalLength}`);

		const values = new Float32Array(this.totalLength);
		for (let i = 0; i < this.totalLength; i++) {
			values[i] = this.#values[i] * other.values[i];
		}

		const result = new Tensor({ 
			values, 
			shape: this.#shape, 
			children: [this, other], 
			op: "*" 
		});

		result[backward] = () => {
			for(let i = 0; i < this.totalLength; i++){
				this.gradient[i] += other.values[i] * result.gradient[i];
				other.gradient[i] += this.values[i] * result.gradient[i];
			}
		};

		return result;
	}
	div(other) {
		if (other.totalLength != this.totalLength) throw new Error(`Tensor not the right length, argument was ${other.totalLength}, needs to be ${this.totalLength}`);

		const values = new Float32Array(this.totalLength);
		for (let i = 0; i < this.totalLength; i++) {
			values[i] = this.#values[i] / other.values[i];
		}

		const result = new Tensor({
			values,
			shape: this.#shape,
			children: [this, other],
			label: "/"
		});

		result[backward] = () => {
			for (let i = 0; i < this.totalLength; i++) {
				this.gradient[i] += 1 / other.values[i] * result.gradient[i];
				other.gradient[i] += -(this.values[i] / other.values[i] ** 2) * result.gradient[i];
			}
		};

		return result;
	}
	neg() {
		const values = new Float32Array(this.totalLength);
		for (let i = 0; i < this.totalLength; i++) {
			values[i] = -this.#values[i];
		}

		const result = new Tensor({
			values,
			shape: this.#shape,
			children: [this],
			op: `neg`
		});

		result[backward] = () => {
			for (let i = 0; i < this.totalLength; i++) {
				this.gradient[i] += -1 * result.gradient[i];
			}
		}

		return result;
	}
	pow(other) {
		if (other.totalLength != this.totalLength) throw new Error(`Tensor not the right length, argument was ${other.totalLength}, needs to be ${this.totalLength}`);

		const values = new Float32Array(this.totalLength);
		for (let i = 0; i < this.totalLength; i++) {
			values[i] = Math.pow(this.#values[i], other.values[i]);
		}

		const result = new Tensor({
			values,
			shape: this.#shape,
			children: [this, other],
			op: `pow`
		});

		result[backward] = () => {
			for (let i = 0; i < this.totalLength; i++) {
				this.gradient[i] += other.values[i] * Math.pow(this.values[i], other.values[i] - 1) * result.gradient[i];
				other.gradient[i] += Math.log(this.values[i]) * Math.pow(this.values[i], other.values[i]) * result.gradient[i]; 
			}
		}

		return result;
	}
	exp() {
		const values = new Float32Array(this.totalLength);
		for (let i = 0; i < this.totalLength; i++) {
			values[i] = Math.exp(this.#values[i])
		}

		const result = new Tensor({
			values,
			shape: this.#shape,
			children: [this],
			op: "exp"
		});

		result[backward] = () => {
			for (let i = 0; i < this.totalLength; i++) {
				this.gradient[i] += Math.exp(this.values[i]) * result.gradient[i];
			}
		}

		return result;
	}
	tanh(){
		const values = new Float32Array(this.totalLength);
		for (let i = 0; i < this.totalLength; i++) {
			values[i] = Math.tanh(this.#values[i])
		}

		const result = new Tensor({
			values,
			shape: this.#shape,
			children: [this],
			op: "tanh"
		});

		result[backward] = () => {
			for (let i = 0; i < this.totalLength; i++) {
				this.gradient[i] += (1 - Math.tanh(this.values[i]) ** 2) * result.gradient[i];
			}
		}

		return result;
	}
	//reductions
	sum({ dimension, keepDims }){
		const newShape = this.#shape.filter((_, i) => i !== dimension);
		const outputLength = newShape.reduce((prod, x) => prod * x, 1);
		const output = new Array(outputLength).fill(0);

		for (let i = 0; i < output.length; i++) {
			const newIndices = getDimensionalIndices(i, newShape);
			for (let j = 0; j < this.#shape[dimension]; j++) {
				const oldIndices = newIndices.toSpliced(dimension, 0, j);
				const oldFlatIndex = getFlatIndex(oldIndices, this.#shape);

				output[i] += this.#values[oldFlatIndex]
			}
		}
		
		const result = new Tensor({
			values: output,
			shape: keepDims ? newShape.toSpliced(dimension, 0, 1) : newShape,
			children: [this],
			op: "sum"
		});

		result[backward] = () => {
			for (let i = 0; i < this.totalLength; i++) {
				const inputIndices = getDimensionalIndices(i, this.#shape);
				const outputIndices = inputIndices.toSpliced(dimension, 1);
				const outputFlatIndex = getFlatIndex(outputIndices, newShape);
				this.gradient[i] += result.gradient[outputFlatIndex];
			}
		}

		return result;
	}
	backward(){
		this.#gradient = new Float32Array(this.totalLength).fill(1);
		const sortedDependencies = topologicalSort(this, x => x.children).reverse();
		for(const node of sortedDependencies){
			node[backward]();
		}
	}
	toString(){
		return `<${this.#label ? `${this.#label}:` : ""}${Array.from(this.#values).join(", ")}>`;
	}
	[Symbol.for("Deno.customInspect")]() {
		return this.toString();
	}

	static filled(value, shape) {
		return new Tensor({ values: new Float32Array(getTotalLength(shape)).fill(value), shape });
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
		return new Tensor({ values: array, shape: [array.length] });
	}
}