export function topologicalSort(startingNode, childAccessor) {
	const visited = new Set();
	const topology = [];

	function visit(node) {
		if (!visited.has(node)) {
			visited.add(node);
			const children = childAccessor(node);
			for (const child of children) {
				visit(child);
			}
			topology.push(node);
		}
	}
	visit(startingNode);
	return topology;
}