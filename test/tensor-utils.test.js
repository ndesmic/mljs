import { assertEquals} from "@std/assert";
import { getFlatIndex, getDimensionalIndices } from "../src/tensor-utils.js";

Deno.test("getFlatIndex should get flat index", () => {
	const r1 = getFlatIndex([1,1,1], [3,3,3]);
	assertEquals(r1, 13);

	const r2 = getFlatIndex([0,0], [4, 3]);
	assertEquals(r2, 0);

	const r3 = getFlatIndex([3, 0], [4, 3]);
	assertEquals(r3, 3);

	const r4 = getFlatIndex([2,3,4], [5,5,5]);
	assertEquals(r4, 117);
});

Deno.test("getShapedIndex should get shaped index", () => {
	const r1 = getDimensionalIndices(13, [3, 3, 3]);
	assertEquals(r1, [1,1,1]);

	const r2 = getDimensionalIndices(0, [4, 3]);
	assertEquals(r2, [0,0]);

	const r3 = getDimensionalIndices(3, [4, 3]);
	assertEquals(r3, [3, 0]);

	const r4 = getDimensionalIndices(117, [5, 5, 5]);
	assertEquals(r4, [2,3,4]);
});