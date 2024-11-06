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
export function* range(end = 1, start = 0, step = 1){
	for(let i = start; i < end; i += step){
		yield i;
	}
}
export function* getRandom(min = 0, max = 1, seed = undefined) {
	const mod = 0x7fffffff;
	
	//seeds must be less than mod
	seed = seed ?? Math.floor(Math.random() * (0x7fffffff - 1));
	let state = seed % mod;

	const length = (max - min) / mod;

	while (true) {
		state = (16807 * state) % mod;
		yield (length * state) + min;
	}
}