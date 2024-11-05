import { assertAlmostEquals } from "@std/assert";

export function assertArrayAlmostEquals(actual, expected, tolerence, message){
	for(let i = 0; i < actual.length; i++){
		assertAlmostEquals(actual[i], expected[i], tolerence, message);
	}
}