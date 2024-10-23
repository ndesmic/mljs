import { assertEquals } from "@std/assert";
import { topologicalSort } from "../src/topological-sort.js";

Deno.test("should sort", () => {
	const g = {
		name: "g",
		children: [
			{
				name: "h",
				children: []
			}
		]
	};
	const graph = {
		name: "a",
		children: [
			{
				name: "b",
				children: [
					{
						name: "c",
						children: []
					},
					{
						name: "d",
						children: [g]
					}
				]
			},
			{
				name: "e",
				children: [
					{
						name: "f",
						children: []
					}
				]
			},
			g
		]
	}

	const sortedGraph = topologicalSort(graph, x => x.children);

	assertEquals(sortedGraph.length, 8);
	assertEquals(sortedGraph[0].name, "c");
	assertEquals(sortedGraph[1].name, "h");
	assertEquals(sortedGraph[2].name, "g");
	assertEquals(sortedGraph[3].name, "d");
	assertEquals(sortedGraph[4].name, "b");
	assertEquals(sortedGraph[5].name, "f");
	assertEquals(sortedGraph[6].name, "e");
	assertEquals(sortedGraph[7].name, "a");
});