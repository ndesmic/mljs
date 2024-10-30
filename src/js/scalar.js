import { topologicalSort } from "./topological-sort.js";

const backward = Symbol("backward");

export class Scalar {
	#value;
	#children;
	#op;
	#label;
	[backward] = () => { };
	#gradient = 0;

	constructor(options) {
		if (typeof (options) === "number") {
			options = { value: options };
		}

		this.#value = options.value ?? 0;
		this.#children = options.children ?? [];
		this.#op = options.op ?? "";
		this.#label = options.label ?? "";
	}
	add(other) {
		if (!(other instanceof Scalar)) other = new Scalar(other);
		const result = new Scalar({
			value: this.value + other.value,
			children: [this, other],
			op: "+"
		});
		result[backward] = () => {
			this.gradient += result.gradient;
			other.gradient += result.gradient;
		};
		return result;
	}
	sub(other) {
		if (!(other instanceof Scalar)) other = new Scalar(other);
		const result = new Scalar({
			value: this.value - other.value,
			children: [this, other],
			op: "-"
		});
		result[backward] = () => {
			this.gradient += result.gradient;
			other.gradient += -1 * result.gradient;
		};
		return result;
	}
	mul(other) {
		if (!(other instanceof Scalar)) other = new Scalar(other);
		const result = new Scalar({
			value: this.value * other.value,
			children: [this, other],
			op: "*"
		});
		result[backward] = () => {
			this.gradient += other.value * result.gradient;
			other.gradient += this.value * result.gradient;
		}
		return result;
	}
	div(other) {
		if (!(other instanceof Scalar)) other = new Scalar(other);
		const result = new Scalar({
			value: this.value / other.value,
			children: [this, other],
			op: "/"
		});
		result[backward] = () => {
			this.gradient += (1 / other.value) * result.gradient;
			other.gradient += (-this.value * (1 / other.value ** 2)) * result.gradient;
		}
		return result;
	}
	neg() {
		const result = new Scalar({
			value: -this.value,
			children: [this],
			op: `neg`
		});
		result[backward] = () => {
			this.gradient += -1 * result.gradient;
		}
		return result;
	}
	pow(other) {
		const result = new Scalar({
			value: Math.pow(this.value, other.value),
			children: [this],
			op: `pow`
		});
		result[backward] = () => {
			this.gradient += other.value * Math.pow(this.value, other.value - 1) * result.gradient;
			other.gradient += Math.log(this.value) * Math.pow(this.value, other.value) * result.gradient;
		}
		return result;
	}
	exp() {
		const result = new Scalar({
			value: Math.exp(this.value),
			children: [this],
			op: "exp"
		});
		result[backward] = () => {
			this.gradient += Math.exp(this.value) * result.gradient;
		}
		return result;
	}
	tanh() {
		const result = new Scalar({
			value: Math.tanh(this.value),
			children: [this],
			op: "tanh"
		});
		result[backward] = () => {
			this.gradient += (1 - Math.tanh(this.value) ** 2) * result.gradient;
		}
		return result;
	}
	backward() {
		this.#gradient = 1;
		const sortedDependencies = topologicalSort(this, x => x.children).reverse();
		for (const node of sortedDependencies) {
			node[backward]();
		}
	}
	toString() {
		return `<${this.#label ? `${this.#label}:` : ""}${this.#value.toString()}>`;
	}
	[Symbol.for("Deno.customInspect")]() {
		return this.toString();
	}
	set value(value) {
		this.#value = value;
	}
	get value() {
		return this.#value;
	}
	get children() {
		return this.#children;
	}
	get op() {
		return this.#op;
	}
	get gradient() {
		return this.#gradient;
	}
	set gradient(value) {
		this.#gradient = value;
	}
	set label(value) {
		this.#label = value;
	}
	get label() {
		return this.#label;
	}
}

export function asValues(numbers) {
	return numbers.map(n => new Scalar(n));
}