import { describe, it } from "@std/testing/bdd";
import { assertEquals, assertAlmostEquals } from "@std/assert";
import { Scalar } from "../src/js/scalar.js";

describe("Scalar", () => {
	describe("add", () => {
		it("should forward pass", () => {
			const s1 = new Scalar(2);
			const s2 = new Scalar(3);
			const result = s1.add(s2);

			assertEquals(result.value, 5);
		});
		it("should backprop", () => {
			const s1 = new Scalar(2);
			const s2 = new Scalar(3);
			const result = s1.add(s2);

			assertEquals(result.value, 5);
			result.backward();
			assertEquals(result.gradient, 1);
			assertEquals(s1.gradient, 1);
			assertEquals(s2.gradient, 1);
		});
		it("should allow same node", () => {
			const value = new Scalar(2);
			const result = value.add(value);

			assertEquals(result.value, 4);
			result.backward();
			assertEquals(result.gradient, 1);
			assertEquals(value.gradient, 2);
		});
	});
	describe("subtract", () => {
		it("should forward pass", () => {
			const s1 = new Scalar(2);
			const s2 = new Scalar(3);
			const result = s1.sub(s2);

			assertEquals(result.value, -1);
		});
		it("should backprop", () => {
			const s1 = new Scalar(2);
			const s2 = new Scalar(3);
			const result = s1.sub(s2);

			assertEquals(result.value, -1);
			result.backward();
			assertEquals(result.gradient, 1);
			assertEquals(s1.gradient, 1);
			assertEquals(s2.gradient, -1);
		});
		it("should allow same node", () => {
			const value = new Scalar(2);
			const result = value.sub(value);

			assertEquals(result.value, 0);
			result.backward();
			assertEquals(result.gradient, 1);
			assertEquals(value.gradient, 0);
		});
	});
	describe("multiply", () => {
		it("should forward pass", () => {
			const s1 = new Scalar(2);
			const s2 = new Scalar(3);
			const result = s1.mul(s2);

			assertEquals(result.value, 6);
		});
		it("should backprop", () => {
			const s1 = new Scalar(2);
			const s2 = new Scalar(3);
			const result = s1.mul(s2);

			assertEquals(result.value, 6);
			result.backward();
			assertEquals(result.gradient, 1);
			assertEquals(s1.gradient, 3);
			assertEquals(s2.gradient, 2);
		});
		it("should allow same node", () => {
			const value = new Scalar(2);
			const result = value.mul(value);

			assertEquals(result.value, 4);
			result.backward();
			assertEquals(result.gradient, 1);
			assertEquals(value.gradient, 4);
		});
	});
	describe("divide", () => {
		it("should forward pass", () => {
			const s1 = new Scalar(2);
			const s2 = new Scalar(3);
			const result = s1.div(s2);

			assertEquals(result.value, 2 / 3);
		});
		it("should backprop", () => {
			const s1 = new Scalar(2);
			const s2 = new Scalar(3);
			const result = s1.div(s2);

			assertEquals(result.value, 2 / 3);
			result.backward();
			assertEquals(result.gradient, 1);
			assertEquals(s1.gradient, 1 / 3);
			assertEquals(s2.gradient, -2 / 3 ** 2);
		});
		it("should allow same node", () => {
			const value = new Scalar(2);
			const result = value.div(value);

			assertEquals(result.value, 1);
			result.backward();
			assertEquals(result.gradient, 1);
			assertEquals(value.gradient, 0);
		});
	});
	describe("power", () => {
		it("should forward pass", () => {
			const s1 = new Scalar(2);
			const s2 = new Scalar(3);
			const result = s1.pow(s2);

			assertEquals(result.value, 8);
		});
		it("should backprop", () => {
			const s1 = new Scalar(2);
			const s2 = new Scalar(3);
			const result = s1.pow(s2);

			assertEquals(result.value, 8);
			result.backward();
			assertEquals(result.gradient, 1);
			assertEquals(s1.gradient, 3 * 2 ** 2);
			assertEquals(s2.gradient, Math.log(2) * 2 ** 3);
		});
		it("should allow same node", () => {
			const s1 = new Scalar(3);
			const result = s1.pow(s1);

			assertEquals(result.value, 27);
			result.backward();
			assertEquals(result.gradient, 1);
			assertAlmostEquals(s1.gradient, 5.6662531794038955e1);
		});
	});
	describe("negate", () => {
		it("should forward pass", () => {
			const value = new Scalar(2);
			const result = value.neg();

			assertEquals(result.value, -2);
		});
		it("should backprop", () => {
			const value = new Scalar(2);
			const result = value.neg();

			assertEquals(result.value, -2);
			result.backward();
			assertEquals(result.gradient, 1);
			assertEquals(value.gradient, -1);
		});
	});
	describe("exponential", () => {
		it("should forward pass", () => {
			const value = new Scalar(2);
			const result = value.exp();

			assertEquals(result.value, Math.exp(2));
		});
		it("should backprop", () => {
			const value = new Scalar(2);
			const result = value.exp();

			assertEquals(result.value, Math.exp(2));
			result.backward();
			assertEquals(result.gradient, 1);
			assertEquals(value.gradient, Math.exp(2));
		});
	});
	describe("hyperbolic tangent", () => {
		it("should forward pass", () => {
			const value = new Scalar(2);
			const result = value.tanh();

			assertEquals(result.value, Math.tanh(2));
		});
		it("should backprop", () => {
			const value = new Scalar(2);
			const result = value.tanh();

			assertEquals(result.value, Math.tanh(2));
			result.backward();
			assertEquals(result.gradient, 1);
			assertEquals(value.gradient, 1 - Math.tanh(2) ** 2);
		});
	});
	describe("toString", () => {
		it("should have a nice string", () => {
			const value = new Scalar(2);

			assertEquals(value.toString(), "<2>");
		});
		it("should have a nice string with label", () => {
			const value = new Scalar({ value: 2, label: "value1" });

			assertEquals(value.toString(), "<value1:2>");
		});
	});
	describe("perceptron example", () => {
		const x1 = new Scalar({ value: 2, label: "x1" });
		const x2 = new Scalar({ value: 0, label: "x2" });
		const w1 = new Scalar({ value: -3, label: "w1" });
		const w2 = new Scalar({ value: 1, label: "w2" });
		const b = new Scalar({ value: 6.881373587019432, label: "b" });

		const x1w1 = x1.mul(w1);
		const x2w2 = x2.mul(w2);

		const i1 = x1w1.add(x2w2);
		const t1 = i1.add(b);
		const o1 = t1.tanh();

		assertAlmostEquals(o1.value, 0.70710, 1e-5);
		assertAlmostEquals(t1.value, 0.88137, 1e-5);
		assertAlmostEquals(i1.value, -6, 1e-5);
		o1.backward();
		assertAlmostEquals(o1.gradient, 1);
		assertAlmostEquals(t1.gradient, 0.5, 1e-5);
		assertAlmostEquals(b.gradient, 0.5, 1e-5);
		assertAlmostEquals(i1.gradient, 0.5, 1e-5);
		assertAlmostEquals(x1w1.gradient, 0.5, 1e-5);
		assertAlmostEquals(x2w2.gradient, 0.5, 1e-5);

		assertAlmostEquals(x1.gradient, -1.5, 1e-5);
		assertAlmostEquals(x2.gradient, 0.5, 1e-5);
		assertAlmostEquals(w1.gradient, 1, 1e-5);
		assertAlmostEquals(w2.gradient, 0, 1e-5);
	});
});