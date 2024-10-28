export function getFlatIndex(colMajorIndices, colMajorShape) {
	if (colMajorIndices.length != colMajorShape.length) throw new Error(`Indices count must match shape. indices length was ${colMajorIndices.length}, shape has length ${colMajorShape.length}.`);

	const rowMajorShape = colMajorShape.toReversed();
	const rowMajorIndices = colMajorIndices.toReversed();

	let index = 0;
	for (let i = 0; i < rowMajorShape.length; i++) {
		index *= rowMajorShape[i];
		index += rowMajorIndices[i];
	}

	return index;
}
export function getDimensionalIndices(flatIndex, colMajorShape) {
	const indices = [];
	for (const size of colMajorShape) {
		indices.push(flatIndex % size);
		flatIndex = Math.floor(flatIndex / size);
	}
	return indices;
}
export function getTotalLength(shape) {
	return shape.reduce((s, x) => s * x);
}
