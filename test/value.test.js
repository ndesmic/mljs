import { assertEquals, assertAlmostEquals } from "@std/assert";
import { Value } from "../src/js/value.js";

Deno.test("Value should add", () => {
	const v1 = new Value(2);
	const v2 = new Value(3);
	const result = v1.add(v2);

	assertEquals(result.value, 5);
});
Deno.test("Value should backprop gradient through add", () => {
	const v1 = new Value(2);
	const v2 = new Value(3);
	const result = v1.add(v2);

	assertEquals(result.value, 5);
	result.backward();
	assertEquals(result.gradient, 1);
	assertEquals(v1.gradient, 1);
	assertEquals(v2.gradient, 1);
});
Deno.test("Value should add same value", () => {
	const value = new Value(2);
	const result = value.add(value);

	assertEquals(result.value, 4);
	result.backward();
	assertEquals(result.gradient, 1);
	assertEquals(value.gradient, 2);
});


Deno.test("Value should subtract", () => {
	const v1 = new Value(2);
	const v2 = new Value(3);
	const result = v1.sub(v2);

	assertEquals(result.value, -1);
});
Deno.test("Value should backprop gradient through sub", () => {
	const v1 = new Value(2);
	const v2 = new Value(3);
	const result = v1.sub(v2);

	assertEquals(result.value, -1);
	result.backward();
	assertEquals(result.gradient, 1);
	assertEquals(v1.gradient, 1);
	assertEquals(v2.gradient, -1);
});
Deno.test("Value should subtract same node", () => {
	const value = new Value(2);
	const result = value.sub(value);

	assertEquals(result.value, 0);
	result.backward();
	assertEquals(result.gradient, 1);
	assertEquals(value.gradient, 0);
});

Deno.test("Value should multiply", () => {
	const v1 = new Value(2);
	const v2 = new Value(3);
	const result = v1.mul(v2);

	assertEquals(result.value, 6);
});
Deno.test("Value should backprop gradient through mul", () => {
	const v1 = new Value(2);
	const v2 = new Value(3);
	const result = v1.mul(v2);

	assertEquals(result.value, 6);
	result.backward();
	assertEquals(result.gradient, 1);
	assertEquals(v1.gradient, 3);
	assertEquals(v2.gradient, 2);
});
Deno.test("Value should multiply same value", () => {
	const value= new Value(2);
	const result = value.mul(value);

	assertEquals(result.value, 4);
	result.backward();
	assertEquals(result.gradient, 1);
	assertEquals(value.gradient, 4);
});

Deno.test("Value should divide", () => {
	const v1 = new Value(2);
	const v2 = new Value(3);
	const result = v1.div(v2);

	assertEquals(result.value, 2/3);
});
Deno.test("Value should backprop gradient through div", () => {
	const v1 = new Value(2);
	const v2 = new Value(3);
	const result = v1.div(v2);

	assertEquals(result.value, 2/3);
	result.backward();
	assertEquals(result.gradient, 1);
	assertEquals(v1.gradient, 1/3);
	assertEquals(v2.gradient, -2/3**2);
});
Deno.test("Value should backprop gradient through div", () => {
	const value = new Value(2);
	const result = value.div(value);

	assertEquals(result.value, 1);
	result.backward();
	assertEquals(result.gradient, 1);
	assertEquals(value.gradient, 0);
});

Deno.test("Value should negate", () => {
	const value = new Value(2);
	const result = value.neg();

	assertEquals(result.value, -2);
});
Deno.test("Value should backprop gradient through neg", () => {
	const value = new Value(2);
	const result = value.neg();

	assertEquals(result.value, -2);
	result.backward();
	assertEquals(result.gradient, 1);
	assertEquals(value.gradient, -1);
});

Deno.test("Value should raise to power", () => {
	const v1 = new Value(2);
	const v2 = new Value(3);
	const result = v1.pow(v2);

	assertEquals(result.value, 8);
});
Deno.test("Value should backprop gradient through pow", () => {
	const v1 = new Value(2);
	const v2 = new Value(3);
	const result = v1.pow(v2);

	assertEquals(result.value, 8);
	result.backward();
	assertEquals(result.gradient, 1);
	assertEquals(v1.gradient, 3 * 2 ** 2);
	assertEquals(v2.gradient, Math.log(2) * 2**3);
});

Deno.test("Value should exponentiate", () => {
	const value = new Value(2);
	const result = value.exp();

	assertEquals(result.value, Math.exp(2));
});
Deno.test("Value should backprop gradient through exp", () => {
	const value = new Value(2);
	const result = value.exp();

	assertEquals(result.value, Math.exp(2));
	result.backward();
	assertEquals(result.gradient, 1);
	assertEquals(value.gradient, Math.exp(2));
});

Deno.test("Value should apply hyperbolic tangent", () => {
	const value = new Value(2);
	const result = value.tanh();

	assertEquals(result.value, Math.tanh(2));
});
Deno.test("Value should backprop gradient through tanh", () => {
	const value = new Value(2);
	const result = value.tanh();

	assertEquals(result.value, Math.tanh(2));
	result.backward();
	assertEquals(result.gradient, 1);
	assertEquals(value.gradient, 1 - Math.tanh(2) ** 2);
});

Deno.test("Value should have a nice string", () => {
	const value = new Value(2);

	assertEquals(value.toString(), "<2>");
});
Deno.test("Value should have a nice string with label", () => {
	const value = new Value({ value: 2, label: "value1" });

	assertEquals(value.toString(), "<value1:2>");
});

Deno.test("Value should work on perceptron example", () => {
	const x1 = new Value({ value: 2, label: "x1" });
	const x2 = new Value({ value: 0, label: "x2" });
	const w1 = new Value({ value: -3, label: "w1" });
	const w2 = new Value({ value: 1, label: "w2" });
	const b = new Value({ value: 6.881373587019432, label: "b" });

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